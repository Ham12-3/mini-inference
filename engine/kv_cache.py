"""KV caches: where attention keys and values are stored between steps.

Why a cache at all: a token's key and value vectors depend only on that token
and the ones before it, so once computed they never change. Without a cache
(see ``baselines/naive.py``) every decode step recomputes them for the whole
sequence. With a cache, a decode step computes k/v for the one new token,
appends them, and attends over the stored ones.

Each class here implements :class:`engine.model.AttentionContext`.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

import torch

from engine.model import Qwen2Config, decode_attention, sdpa_attention


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
            return decode_attention(q, keys, values)
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
        # One extra "scratch" slot that is never handed out: CUDA-graph
        # padding rows write there (see engine/cuda_graphs.py).
        shape = (num_layers, num_slots + 1, num_kv_heads, max_len, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.num_slots = num_slots
        self.scratch_slot = num_slots
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

    def can_allocate(self, num_tokens: int) -> bool:
        """Is there room for a new sequence of ``num_tokens``? (A free slot, and it fits in one.)"""
        return bool(self._free) and num_tokens <= self.max_len

    def allocate(self, num_tokens: int = 0, token_ids: Sequence[int] | None = None) -> int:
        """Hand out a free slot (its length starts at 0). Every slot already holds ``max_len`` tokens.

        ``token_ids`` is accepted for interface compatibility with the paged
        cache (which can reuse cached prefixes); the slot cache ignores it.
        """
        if not self.can_allocate(num_tokens):
            raise RuntimeError("no free KV cache slots" if not self._free else f"{num_tokens} tokens exceed max_len")
        slot = self._free.pop()
        self.lengths[slot] = 0
        return slot

    def ensure_capacity(self, slot: int, num_tokens: int) -> bool:
        """Can ``slot`` hold ``num_tokens`` tokens? A slot never grows, so this is a bounds check."""
        return num_tokens <= self.max_len

    def register_prefix(self, slot: int, token_ids: Sequence[int]) -> None:
        """No prefix caching in the slot cache (kept so both caches share one interface)."""

    def write_index(
        self, slots: Sequence[int], starts: Sequence[int], counts: Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Where tokens ``start..start+count-1`` of each slot live: ``(slot, position)`` per token."""
        device = self.k.device
        token_slots = torch.repeat_interleave(torch.tensor(list(slots), device=device),
                                              torch.tensor(list(counts), device=device))
        positions = torch.cat([torch.arange(a, a + n, device=device) for a, n in zip(starts, counts)])
        return token_slots, positions

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

    def packed_prefill_context(self, slots: Sequence[int], lengths: Sequence[int]) -> PackedPrefillContext:
        return PackedPrefillContext(self, slots, lengths)


class PackedPrefillContext:
    """Prefill several fresh prompts in ONE forward pass by packing them end to end.

    The prompts are concatenated into a single row of N tokens (no padding).
    Every token keeps its own position *within its prompt* for RoPE, and
    attention uses a block-diagonal causal mask: a token may only see earlier
    tokens of the same prompt.

        prompt A: a0 a1 a2    prompt B: b0 b1        mask (1 = may attend)
        packed:   a0 a1 a2 b0 b1                     a0: 1 . . . .
                                                     a1: 1 1 . . .
                                                     a2: 1 1 1 . .
                                                     b0: . . . 1 .
                                                     b1: . . . 1 1

    Why bother: each prefill forward pass costs a fixed ~25 ms of launch
    overhead on the T4, whatever its length. Prefilling 32 prompts one by
    one pays that 32 times. Packing pays it once, and the matmuls get bigger
    and more efficient. The mask is N x N, so callers keep N within the
    step's token budget.

    Works with both caches: each cache says where token i of a sequence
    lives (``write_index``): (slot, position) in the slot cache, (block,
    offset in block) in the paged cache.

    A sequence may already have tokens in the cache (a prefix-cache hit, see
    :class:`PagedKVCache`). Then only its *new* tokens are packed, at
    positions ``start..start+n-1``, and they must also see the cached prefix,
    which is not part of this forward pass. In that case we write the new
    k/v first, read every sequence's keys ``0..start+n-1`` back from the
    cache, and use a mask that lets each token see its own sequence's keys up
    to its own position.
    """

    def __init__(self, cache: SlotKVCache | PagedKVCache, slots: Sequence[int], lengths: Sequence[int]) -> None:
        self.cache = cache
        self.slots = list(slots)
        self.lengths = list(lengths)  # new tokens per sequence
        self.starts = [cache.lengths[s] for s in self.slots]  # tokens each sequence already has cached
        for s, start, n in zip(self.slots, self.starts, self.lengths):
            if not cache.ensure_capacity(s, start + n):
                raise ValueError(f"no room in the KV cache for {start + n} tokens")
        device = cache.k.device
        # Per packed token: which prompt it belongs to, and its position in that prompt.
        seg = torch.repeat_interleave(torch.arange(len(self.slots), device=device),
                                      torch.tensor(self.lengths, device=device))
        self.positions = torch.cat([torch.arange(start, start + n, device=device)
                                    for start, n in zip(self.starts, self.lengths)])
        self.write_at = cache.write_index(self.slots, self.starts, self.lengths)
        self.has_prefix = any(self.starts)
        if self.has_prefix:
            ends = [start + n for start, n in zip(self.starts, self.lengths)]
            self.read_at = cache.write_index(self.slots, [0] * len(self.slots), ends)  # keys 0..end-1 per sequence
            key_seg = torch.repeat_interleave(torch.arange(len(self.slots), device=device),
                                              torch.tensor(ends, device=device))
            key_pos = torch.cat([torch.arange(e, device=device) for e in ends])
            self.mask = (seg.unsqueeze(1) == key_seg.unsqueeze(0)) & (key_pos.unsqueeze(0) <= self.positions.unsqueeze(1))
        elif len(self.slots) == 1:
            self.mask = None  # one fresh prompt alone: plain causal attention
        else:
            index = torch.arange(len(seg), device=device)
            self.mask = (seg.unsqueeze(1) == seg.unsqueeze(0)) & (index.unsqueeze(0) <= index.unsqueeze(1))
        # Index of each prompt's last token: its logits predict the next output token.
        self.last_index = torch.tensor(self.lengths, device=device).cumsum(0) - 1

    def attend(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Scatter every packed token's k/v to where the cache keeps it, in one write.
        # k[0] is [heads, N, head_dim]; the indexed target is [N, heads, head_dim].
        k_layer, v_layer = self.cache.k[layer_idx], self.cache.v[layer_idx]
        where_0, where_2 = self.write_at
        k_layer[where_0, :, where_2] = k[0].transpose(0, 1)
        v_layer[where_0, :, where_2] = v[0].transpose(0, 1)
        if self.has_prefix:
            read_0, read_2 = self.read_at
            keys = k_layer[read_0, :, read_2].transpose(0, 1).unsqueeze(0)  # [1, heads, total keys, head_dim]
            values = v_layer[read_0, :, read_2].transpose(0, 1).unsqueeze(0)
            return sdpa_attention(q, keys, values, attn_mask=self.mask)
        if self.mask is None:
            return sdpa_attention(q, k, v, is_causal=True)
        return sdpa_attention(q, k, v, attn_mask=self.mask)

    def commit(self) -> None:
        for slot, start, n in zip(self.slots, self.starts, self.lengths):
            self.cache.lengths[slot] = start + n


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
        return decode_attention(q, keys, values, attn_mask=self.mask)

    def commit(self) -> None:
        """Call after the forward pass: every sequence in the batch grew by one token."""
        for s in self.slots:
            self.cache.lengths[s] += 1


class PagedKVCache:
    """A KV cache split into fixed-size blocks that are handed out as sequences grow (Stage 6).

    Memory is one pool of blocks, each holding ``block_size`` tokens:

        k, v: [num_layers, num_blocks, num_kv_heads, block_size, head_dim]

    A sequence owns a *block table*: the list of block ids holding its
    tokens, in order. Token p lives in block ``table[p // block_size]`` at
    offset ``p % block_size``. The blocks do not need to be next to each
    other, just like pages of virtual memory.

    Compared with the slot cache, a sequence only holds the blocks it needs
    (at most ``block_size - 1`` tokens wasted, in its last block) instead of a
    whole ``max_len`` slot. The same memory therefore fits far more sequences.
    When the pool runs out, the scheduler preempts a sequence (see
    ``Scheduler.schedule``).

    **Prefix caching** (``enable_prefix_caching``): a full block's keys and
    values depend only on the tokens in it and every token before it. So a
    full block can be named by its content: a key made of its tokens plus the
    key of the block before it. After a prompt is prefilled, its full blocks
    are registered under those keys. A later prompt that starts with the same
    tokens (a shared system prompt, say) finds the matching blocks and
    *shares* them instead of recomputing them: it only prefills the rest.
    Blocks are reference counted. When nobody uses a cached block any more it
    goes back to the free pool but keeps its contents and key, so it can still
    be reused, until the pool needs it for new data (the least recently freed
    blocks are reused first).

    Sequences are still identified by a "slot" id, so the engine and scheduler
    treat both caches the same way; here a slot id just names a block table.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_blocks: int,
        block_size: int,
        max_seqs: int,
        max_len: int,
        dtype: torch.dtype,
        device: torch.device | str,
        enable_prefix_caching: bool = False,
    ) -> None:
        # One extra "scratch" block outside the pool: CUDA-graph padding rows
        # point at it (see engine/cuda_graphs.py).
        shape = (num_layers, num_blocks + 1, num_kv_heads, block_size, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.num_blocks = num_blocks
        self.scratch_block = num_blocks
        self.block_size = block_size
        self.max_len = max_len
        self.enable_prefix_caching = enable_prefix_caching
        self.lengths: dict[int, int] = {}  # slot id -> tokens stored
        self.block_tables: dict[int, list[int]] = {}  # slot id -> its blocks, in token order
        self._free_ids = list(range(max_seqs - 1, -1, -1))
        # Free blocks in reuse order (oldest first). An ordered dict gives
        # O(1) "take the oldest" and O(1) "take back this particular one".
        self._free_blocks: OrderedDict[int, None] = OrderedDict((b, None) for b in range(num_blocks))
        self._ref_counts = [0] * num_blocks
        self._block_key: dict[int, tuple] = {}  # block -> its content key (cached blocks only)
        self._cached: dict[tuple, int] = {}  # content key -> block
        self.prefix_hit_tokens = 0  # total prompt tokens served from the cache

    @classmethod
    def for_model(cls, cfg: Qwen2Config, num_blocks: int, block_size: int, max_seqs: int, max_len: int,
                  dtype: torch.dtype, device: torch.device | str, enable_prefix_caching: bool = False) -> PagedKVCache:
        return cls(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, num_blocks, block_size, max_seqs, max_len,
                   dtype, device, enable_prefix_caching)

    @property
    def num_free_slots(self) -> int:
        return len(self._free_ids)

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    def blocks_for(self, num_tokens: int) -> int:
        return -(-num_tokens // self.block_size)  # ceiling division

    def can_allocate(self, num_tokens: int) -> bool:
        # Conservative: does not count blocks a prefix hit would share.
        return bool(self._free_ids) and num_tokens <= self.max_len and self.blocks_for(num_tokens) <= self.num_free_blocks

    # ------------------------------------------------------------------ blocks

    def _new_block(self) -> int:
        """Take the least recently freed block; if it held a cached prefix, forget that prefix."""
        block, _ = self._free_blocks.popitem(last=False)
        key = self._block_key.pop(block, None)
        if key is not None:
            del self._cached[key]
        self._ref_counts[block] = 1
        return block

    def _share(self, block: int) -> None:
        if self._ref_counts[block] == 0:
            del self._free_blocks[block]  # free but still cached: take it back out of the pool
        self._ref_counts[block] += 1

    def _release(self, block: int) -> None:
        self._ref_counts[block] -= 1
        if self._ref_counts[block] == 0:
            self._free_blocks[block] = None  # newest at the end: reused last
            if block not in self._block_key:
                # Nothing worth keeping in it, so hand it out before any cached block.
                self._free_blocks.move_to_end(block, last=False)

    @staticmethod
    def _chain_keys(token_ids: Sequence[int], block_size: int, num_blocks: int):
        """Content keys of the first ``num_blocks`` full blocks: (hash of the previous key, this block's tokens)."""
        parent = None
        for i in range(num_blocks):
            key = (parent, tuple(token_ids[i * block_size : (i + 1) * block_size]))
            yield key
            parent = hash(key)

    def _match_prefix(self, token_ids: Sequence[int]) -> list[int]:
        # Keep at least one token to compute: its logits pick the next token.
        usable = (len(token_ids) - 1) // self.block_size
        blocks = []
        for key in self._chain_keys(token_ids, self.block_size, usable):
            block = self._cached.get(key)
            if block is None:
                break
            blocks.append(block)
        return blocks

    # ------------------------------------------------------------------ sequences

    def allocate(self, num_tokens: int = 0, token_ids: Sequence[int] | None = None) -> int:
        """Start a new sequence with room for ``num_tokens`` tokens.

        With prefix caching and ``token_ids``, leading full blocks already in
        the cache are shared, and ``lengths[slot]`` starts at the number of
        tokens they hold, so the caller only prefills the rest.
        """
        if not self.can_allocate(num_tokens):
            raise RuntimeError("not enough free KV cache blocks")
        shared = self._match_prefix(token_ids) if self.enable_prefix_caching and token_ids else []
        for block in shared:
            self._share(block)
        slot = self._free_ids.pop()
        fresh = [self._new_block() for _ in range(self.blocks_for(num_tokens) - len(shared))]
        self.block_tables[slot] = shared + fresh
        self.lengths[slot] = len(shared) * self.block_size
        self.prefix_hit_tokens += self.lengths[slot]
        return slot

    def register_prefix(self, slot: int, token_ids: Sequence[int]) -> None:
        """Make ``slot``'s full blocks findable by later prompts (call after they are filled)."""
        if not self.enable_prefix_caching:
            return
        table = self.block_tables[slot]
        full_blocks = self.lengths[slot] // self.block_size
        for i, key in enumerate(self._chain_keys(token_ids, self.block_size, full_blocks)):
            if key not in self._cached and table[i] not in self._block_key:
                self._cached[key] = table[i]
                self._block_key[table[i]] = key

    def ensure_capacity(self, slot: int, num_tokens: int) -> bool:
        """Grow ``slot``'s block table to hold ``num_tokens`` tokens. False if the pool is out of blocks."""
        if num_tokens > self.max_len:
            return False
        table = self.block_tables[slot]
        missing = self.blocks_for(num_tokens) - len(table)
        if missing > self.num_free_blocks:
            return False
        table.extend(self._new_block() for _ in range(missing))
        return True

    def free(self, slot: int) -> None:
        """Release a sequence's blocks (blocks shared with other sequences stay in use)."""
        if slot not in self.block_tables:
            raise RuntimeError(f"slot {slot} freed twice")
        for block in reversed(self.block_tables.pop(slot)):
            self._release(block)
        del self.lengths[slot]
        self._free_ids.append(slot)

    def write_index(
        self, slots: Sequence[int], starts: Sequence[int], counts: Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Where tokens ``start..start+count-1`` of each sequence live: ``(block, offset in block)``."""
        device = self.k.device
        blocks, offsets = [], []
        for slot, a, n in zip(slots, starts, counts):
            pos = torch.arange(a, a + n, device=device)
            table = torch.tensor(self.block_tables[slot], device=device)
            blocks.append(table[pos // self.block_size])
            offsets.append(pos % self.block_size)
        return torch.cat(blocks), torch.cat(offsets)

    def packed_prefill_context(self, slots: Sequence[int], lengths: Sequence[int]) -> PackedPrefillContext:
        return PackedPrefillContext(self, slots, lengths)

    def decode_context(self, slots: Sequence[int]) -> PagedDecodeContext:
        return PagedDecodeContext(self, slots)


class PagedDecodeContext:
    """A batched decode step on the paged cache.

    Same idea as :class:`SlotDecodeContext` (each new token at its own
    position, a mask hides positions beyond each sequence's length), but keys
    and values are gathered block by block through each sequence's block
    table. Tables are padded to the same number of blocks with block 0; the
    mask hides whatever those padding blocks contain.
    """

    def __init__(self, cache: PagedKVCache, slots: Sequence[int]) -> None:
        self.cache = cache
        self.slots = list(slots)
        device = cache.k.device
        bs = cache.block_size
        pos = [cache.lengths[s] for s in self.slots]
        tables = [cache.block_tables[s] for s in self.slots]
        if any(p // bs >= len(t) for p, t in zip(pos, tables)):
            raise ValueError("a sequence has no block for its next token (call ensure_capacity first)")
        self.positions = torch.tensor(pos, device=device)  # [B]: where each new token goes
        # Where each new token is written: its block, and its offset inside that block.
        self.write_blocks = torch.tensor([t[p // bs] for t, p in zip(tables, pos)], device=device)
        self.write_offsets = self.positions % bs
        num_blocks = max(pos) // bs + 1  # blocks needed by the longest sequence
        padded = [t[:num_blocks] + [0] * (num_blocks - len(t[:num_blocks])) for t in tables]
        self.tables = torch.tensor(padded, device=device)  # [B, num_blocks]
        self.kv_len = num_blocks * bs
        self.mask = (torch.arange(self.kv_len, device=device) <= self.positions.unsqueeze(1))[:, None, None, :]

    def attend(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        k_layer, v_layer = self.cache.k[layer_idx], self.cache.v[layer_idx]  # [blocks, heads, block_size, head_dim]
        k_layer[self.write_blocks, :, self.write_offsets] = k[:, :, 0]
        v_layer[self.write_blocks, :, self.write_offsets] = v[:, :, 0]
        batch, heads = k.shape[0], k.shape[1]
        # Gather each sequence's blocks in table order, then lay them end to end:
        # [B, blocks, heads, block_size, d] -> [B, heads, blocks * block_size, d].
        keys = k_layer[self.tables].permute(0, 2, 1, 3, 4).reshape(batch, heads, self.kv_len, -1)
        values = v_layer[self.tables].permute(0, 2, 1, 3, 4).reshape(batch, heads, self.kv_len, -1)
        return decode_attention(q, keys, values, attn_mask=self.mask)

    def commit(self) -> None:
        for s in self.slots:
            self.cache.lengths[s] += 1
