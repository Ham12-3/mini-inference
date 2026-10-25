"""KV caches: where attention keys and values are stored between steps.

Why a cache at all: a token's key and value vectors depend only on that token
and the ones before it, so once computed they never change. Without a cache
(see ``baselines/naive.py``) every decode step recomputes them for the whole
sequence. With a cache, a decode step computes k/v for the one new token,
appends them, and attends over the stored ones.

Each class here implements :class:`engine.model.AttentionContext`.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from engine.model import Qwen2Config, sdpa_attention


def offset_causal_mask(start: int, end: int, device: torch.device) -> torch.Tensor:
    """Causal mask for new tokens at positions [start, end) attending to keys [0, end).

    SDPA's ``is_causal`` aligns its mask to the top-left corner, which is only
    right when the queries start at position 0. Here new token i sits at
    position ``start + i`` and may see key j only if ``j <= start + i``.
    Returns a ``[end - start, end]`` boolean mask (True = may attend).
    """
    rows = torch.arange(start, end, device=device).unsqueeze(1)
    cols = torch.arange(end, device=device).unsqueeze(0)
    return cols <= rows


class NoCache:
    """No memory between calls: plain causal attention over the tokens of this forward pass.

    Used by the parity tests to compare a full forward pass against Hugging Face.
    """

    def attend(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return sdpa_attention(q, k, v, is_causal=q.shape[2] > 1)


class SingleSequenceKVCache:
    """A preallocated KV cache for one sequence (Stage 2).

    Layout: ``k`` and ``v`` are each ``[num_layers, num_kv_heads, max_len, head_dim]``.
    Slicing ``[layer, :, :length]`` gives exactly the ``[heads, seq, head_dim]``
    shape SDPA wants, with no copying.

    Usage: run the model with this as its context (prefill with the whole
    prompt, or decode with one token), then call :meth:`advance` with the
    number of tokens just processed. ``length`` only moves forward after all
    layers have written their k/v for the step.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_len: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        shape = (num_layers, num_kv_heads, max_len, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.max_len = max_len
        self.length = 0

    @classmethod
    def for_model(cls, cfg: Qwen2Config, max_len: int, dtype: torch.dtype, device: torch.device | str):
        return cls(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, max_len, dtype, device)

    def reset(self) -> None:
        self.length = 0  # old entries are simply overwritten later

    def advance(self, num_tokens: int) -> None:
        self.length += num_tokens

    def attend(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if q.shape[0] != 1:
            raise ValueError("SingleSequenceKVCache holds exactly one sequence")
        start, num_new = self.length, k.shape[2]
        end = start + num_new
        if end > self.max_len:
            raise ValueError(f"sequence length {end} exceeds cache size {self.max_len}")

        # Write the new tokens' k/v at positions [start, end), then attend over [0, end).
        self.k[layer_idx, :, start:end] = k[0]
        self.v[layer_idx, :, start:end] = v[0]
        keys = self.k[layer_idx, :, :end].unsqueeze(0)
        values = self.v[layer_idx, :, :end].unsqueeze(0)

        if num_new == 1:
            # Decode: the single new token may look at every stored token (and itself).
            return sdpa_attention(q, keys, values)
        if start == 0:
            # Prefill of a fresh sequence: ordinary square causal mask.
            return sdpa_attention(q, keys, values, is_causal=True)
        # Several new tokens on top of existing ones (e.g. chunked prefill).
        return sdpa_attention(q, keys, values, attn_mask=offset_causal_mask(start, end, q.device))


class SlotKVCache:
    """A preallocated KV cache with one fixed-size slot per running sequence (Stage 3).

    At start-up we allocate one big tensor each for keys and values:

        k, v: [num_layers, num_slots, num_kv_heads, max_len, head_dim]

    Every running sequence owns one slot (an index into dim 1), and
    ``lengths[slot]`` says how many of its positions hold real data. A
    sequence's token at position p always lives at ``[:, slot, :, p]``.

    (This is the brief's ``[layers, 2, slots, len, heads, head_dim]`` layout
    with k and v split into two tensors and heads moved before positions, so
    that ``k[layer, slot, :, :n]`` is already the ``[heads, seq, head_dim]``
    shape SDPA wants.)

    Why preallocate: allocating on the fly would fragment GPU memory and cost
    time every step. The price is waste: every slot reserves ``max_len``
    positions even if its sequence is short. Stage 6's paged cache fixes that.

    Each forward pass gets a small context object from :meth:`prefill_context`
    or :meth:`decode_context` that knows which slots and positions it touches.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_slots: int,
        max_len: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        shape = (num_layers, num_slots, num_kv_heads, max_len, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.num_slots = num_slots
        self.max_len = max_len
        self.lengths = [0] * num_slots
        # A stack of free slot ids; pop() hands out the lowest id first.
        self._free = list(range(num_slots - 1, -1, -1))

    @classmethod
    def for_model(
        cls, cfg: Qwen2Config, num_slots: int, max_len: int, dtype: torch.dtype, device: torch.device | str
    ) -> SlotKVCache:
        return cls(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, num_slots, max_len, dtype, device)

    @property
    def num_free_slots(self) -> int:
        return len(self._free)

    def allocate(self) -> int:
        """Hand out a free slot (its length starts at 0)."""
        if not self._free:
            raise RuntimeError("no free KV cache slots")
        slot = self._free.pop()
        self.lengths[slot] = 0
        return slot

    def free(self, slot: int) -> None:
        """Return a slot to the free list straight away.

        Nothing is zeroed: the old keys/values stay in memory, but the next
        owner starts at length 0, overwrites positions as it goes, and the
        attention mask never lets it read past its own length.
        """
        if slot in self._free:
            raise RuntimeError(f"slot {slot} freed twice")
        self.lengths[slot] = 0
        self._free.append(slot)

    def prefill_context(self, slot: int) -> SlotPrefillContext:
        return SlotPrefillContext(self, slot)

    def decode_context(self, slots: Sequence[int]) -> SlotDecodeContext:
        return SlotDecodeContext(self, slots)


class SlotPrefillContext:
    """Attention for one sequence's prompt tokens, written into its slot."""

    def __init__(self, cache: SlotKVCache, slot: int) -> None:
        self.cache = cache
        self.slot = slot
        self.start = cache.lengths[slot]

    def attend(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        c, slot = self.cache, self.slot
        start, end = self.start, self.start + k.shape[2]
        if end > c.max_len:
            raise ValueError(f"sequence length {end} exceeds cache slot size {c.max_len}")
        c.k[layer_idx, slot, :, start:end] = k[0]
        c.v[layer_idx, slot, :, start:end] = v[0]
        if start == 0:
            # A fresh prompt: attend over the new keys directly, no read-back needed.
            return sdpa_attention(q, k, v, is_causal=k.shape[2] > 1)
        keys = c.k[layer_idx, slot, :, :end].unsqueeze(0)
        values = c.v[layer_idx, slot, :, :end].unsqueeze(0)
        return sdpa_attention(q, keys, values, attn_mask=offset_causal_mask(start, end, q.device))

    def commit(self, num_tokens: int) -> None:
        """Call after the forward pass: the slot now holds ``num_tokens`` more positions."""
        self.cache.lengths[self.slot] = self.start + num_tokens


class SlotDecodeContext:
    """Attention for a batched decode step: B sequences, one new token each.

    The sequences have different lengths, which is the whole difficulty:

    * Position: sequence b's new token sits at position ``lengths[slot_b]``.
      That is both where its k/v are written and the position its RoPE uses
      (``positions`` is passed to the model for exactly that).
    * Attention: we gather every sequence's keys up to the *longest* length in
      the batch, ``kv_len``, so they fit in one ``[B, heads, kv_len, head_dim]``
      tensor. A shorter sequence's extra positions are empty or stale
      leftovers from the slot's previous owner, so a mask hides them: row b
      allows key j only if ``j <= positions[b]``.
    """

    def __init__(self, cache: SlotKVCache, slots: Sequence[int]) -> None:
        self.cache = cache
        self.slots = list(slots)
        device = cache.k.device
        pos = [cache.lengths[s] for s in self.slots]
        if max(pos) >= cache.max_len:
            raise ValueError(f"a sequence reached the cache slot size {cache.max_len}")
        self.slot_ids = torch.tensor(self.slots, device=device)
        self.positions = torch.tensor(pos, device=device)  # [B]: where each new token goes
        self.kv_len = max(pos) + 1
        # [B, 1, 1, kv_len]: broadcasts over heads and the single query position.
        self.mask = (torch.arange(self.kv_len, device=device) <= self.positions.unsqueeze(1))[:, None, None, :]

    def attend(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        k_layer, v_layer = self.cache.k[layer_idx], self.cache.v[layer_idx]  # [slots, heads, max_len, head_dim]
        # Scatter: one write per sequence at (its slot, its position). Two
        # index tensors around a slice select a [B, heads, head_dim] target.
        k_layer[self.slot_ids, :, self.positions] = k[:, :, 0]
        v_layer[self.slot_ids, :, self.positions] = v[:, :, 0]
        # Gather each sequence's first kv_len positions. This copies; paged
        # attention kernels avoid the copy (see Stage 6).
        keys = k_layer[self.slot_ids, :, : self.kv_len]
        values = v_layer[self.slot_ids, :, : self.kv_len]
        return sdpa_attention(q, keys, values, attn_mask=self.mask)

    def commit(self) -> None:
        """Call after the forward pass: every sequence in the batch grew by one token."""
        for s in self.slots:
            self.cache.lengths[s] += 1
