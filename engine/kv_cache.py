"""KV caches: where attention keys and values are stored between steps.

Why a cache at all: a token's key and value vectors depend only on that token
and the ones before it, so once computed they never change. Without a cache
(see ``baselines/naive.py``) every decode step recomputes them for the whole
sequence. With a cache, a decode step computes k/v for the one new token,
appends them, and attends over the stored ones.

Each class here implements :class:`engine.model.AttentionContext`.
"""

from __future__ import annotations

import torch

from engine.model import Qwen2Config, sdpa_attention


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
        # SDPA's is_causal aligns the mask to the top-left corner, which is
        # wrong here, so build it: new token i sits at position start + i and
        # may see key j only if j <= start + i.
        rows = torch.arange(start, end, device=q.device).unsqueeze(1)
        cols = torch.arange(end, device=q.device).unsqueeze(0)
        return sdpa_attention(q, keys, values, attn_mask=cols <= rows)
