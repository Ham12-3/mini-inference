"""CUDA graphs for the decode step (Stage 6).

Why: a decode step on the T4 takes ~27 ms, but the GPU work in it is only a
few ms. The rest is the CPU launching hundreds of small kernels one by one
(24 layers x norms, matmuls, RoPE, attention, ...) through Python. A CUDA
graph records that whole sequence of kernels once; replaying it is a single
launch, so the per-step CPU cost almost disappears.

The catch: a graph replays exactly what was recorded, with the same tensor
shapes and the same memory addresses. So:

* Inputs live in *static buffers* (token ids, positions, slot ids or block
  tables) that we overwrite before every replay.
* Shapes must be fixed. We record one graph per (batch size, KV length)
  *bucket* and round each real step up to the nearest bucket (batch sizes
  1, 2, 4, 8, then every 16; KV lengths every 256 positions). Extra batch
  rows point at a scratch slot/block that no sequence owns, so their writes
  are harmless, and their outputs are ignored. Extra KV positions are hidden
  by the mask like any other padding.
* Anything that depends on the inputs (the attention mask, where each new
  token is written) is computed *inside* the graph from the static buffers.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from engine.kv_cache import PagedKVCache, SlotKVCache
from engine.model import Qwen2ForCausalLM, decode_attention


def _batch_buckets(limit: int) -> list[int]:
    """1, 2, 4, 8, then every 16 up to ``limit`` (always including ``limit``).

    Fine steps matter: a step with 65 sequences runs the 80-row graph, not a
    128-row one. Padding rows are real work once a step stops being
    dominated by launch overhead.
    """
    sizes = [b for b in (1, 2, 4, 8) if b < limit] + list(range(16, limit, 16))
    return sizes + [limit]


def _kv_buckets(limit: int, step: int = 256) -> list[int]:
    """Every ``step`` positions up to ``limit`` (always including ``limit``)."""
    return list(range(step, limit, step)) + [limit]


class _GraphDecodeContext:
    """Attention for a captured decode step; reads everything from the static buffers."""

    def __init__(self, owner: GraphedDecoder, batch: int, kv_len: int) -> None:
        self.cache = owner.cache
        self.paged = owner.paged
        self.batch, self.kv_len = batch, kv_len
        self.positions = owner.positions[:batch]
        if self.paged:
            self.tables = owner.tables[:batch, : kv_len // owner.cache.block_size]
        else:
            self.slots = owner.slots[:batch]
        self.device = owner.device

    def prepare(self) -> None:
        """Run inside the graph, once per step, before the layers.

        Everything here is (re)computed on every replay from the static
        buffers. A tensor made *outside* the graph and only referenced by it
        would be a bug waiting to happen: if Python freed it, the graph would
        keep reading whatever later reused that memory.
        """
        key_index = torch.arange(self.kv_len, device=self.device)
        self.mask = (key_index <= self.positions.unsqueeze(1))[:, None, None, :]
        if self.paged:
            bs = self.cache.block_size
            self.write_blocks = self.tables.gather(1, (self.positions // bs).unsqueeze(1)).squeeze(1)
            self.write_offsets = self.positions % bs

    def attend(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        k_layer, v_layer = self.cache.k[layer_idx], self.cache.v[layer_idx]
        if self.paged:
            k_layer[self.write_blocks, :, self.write_offsets] = k[:, :, 0]
            v_layer[self.write_blocks, :, self.write_offsets] = v[:, :, 0]
            heads = k.shape[1]
            keys = k_layer[self.tables].permute(0, 2, 1, 3, 4).reshape(self.batch, heads, self.kv_len, -1)
            values = v_layer[self.tables].permute(0, 2, 1, 3, 4).reshape(self.batch, heads, self.kv_len, -1)
        else:
            k_layer[self.slots, :, self.positions] = k[:, :, 0]
            v_layer[self.slots, :, self.positions] = v[:, :, 0]
            keys = k_layer[self.slots, :, : self.kv_len]
            values = v_layer[self.slots, :, : self.kv_len]
        return decode_attention(q, keys, values, attn_mask=self.mask)


class GraphedDecoder:
    """Captures the decode step as CUDA graphs and replays them.

    Use :meth:`capture` once at start-up, then :meth:`decode` each step. It
    returns ``None`` when no captured graph fits (the caller then runs the
    normal eager step).
    """

    def __init__(self, model: Qwen2ForCausalLM, cache: SlotKVCache | PagedKVCache, max_batch: int) -> None:
        self.model, self.cache = model, cache
        self.paged = isinstance(cache, PagedKVCache)
        self.device = cache.k.device
        self.batch_buckets = _batch_buckets(max_batch)
        self.kv_buckets = _kv_buckets(cache.max_len)
        if self.paged:
            # kv lengths must be whole blocks
            bs = cache.block_size
            self.kv_buckets = sorted({-(-kv // bs) * bs for kv in self.kv_buckets})
        top = self.batch_buckets[-1]
        self.ids = torch.zeros(top, 1, dtype=torch.long, device=self.device)
        self.positions = torch.zeros(top, dtype=torch.long, device=self.device)
        if self.paged:
            max_blocks = self.kv_buckets[-1] // cache.block_size
            self.tables = torch.full((top, max_blocks), cache.scratch_block, dtype=torch.long, device=self.device)
        else:
            self.slots = torch.full((top,), cache.scratch_slot, dtype=torch.long, device=self.device)
        # (batch, kv_len) -> (graph, its output logits, its context). The context
        # is kept alive with the graph on purpose: the graph reads its tensors.
        self.graphs: dict[tuple[int, int], tuple[torch.cuda.CUDAGraph, torch.Tensor, _GraphDecodeContext]] = {}
        self._pool = None
        self.replays = 0  # decode steps served by a graph (the rest fell back to eager)

    @torch.inference_mode()
    def capture(self) -> None:
        """Record one graph per (batch, kv length) bucket, largest first so smaller ones reuse its memory."""
        for batch in reversed(self.batch_buckets):
            for kv_len in reversed(self.kv_buckets):
                self._capture_one(batch, kv_len)
        torch.cuda.synchronize()

    def _capture_one(self, batch: int, kv_len: int) -> None:
        ctx = _GraphDecodeContext(self, batch, kv_len)

        def step() -> torch.Tensor:
            ctx.prepare()
            hidden = self.model(self.ids[:batch], self.positions[:batch].unsqueeze(1), ctx)
            return self.model.compute_logits(hidden[:, -1]).float()

        # Point every row at the scratch slot/block while recording.
        self.positions.zero_()
        if self.paged:
            self.tables.fill_(self.cache.scratch_block)
        else:
            self.slots.fill_(self.cache.scratch_slot)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):  # warm-up runs (kernel selection, allocator) outside the graph
            for _ in range(2):
                step()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            logits = step()
        self._pool = graph.pool()
        self.graphs[(batch, kv_len)] = (graph, logits, ctx)

    def decode(self, slots: Sequence[int], last_tokens: Sequence[int]) -> torch.Tensor | None:
        """One decode step via a captured graph; returns logits ``[len(slots), vocab]`` or None."""
        n = len(slots)
        positions = [self.cache.lengths[s] for s in slots]
        batch = next((b for b in self.batch_buckets if b >= n), None)
        kv_len = next((k for k in self.kv_buckets if k > max(positions)), None)
        if batch is None or kv_len is None or (batch, kv_len) not in self.graphs:
            return None
        graph, logits, _ = self.graphs[(batch, kv_len)]

        # Fill the static inputs; padding rows use the scratch slot/block.
        self.ids[:n, 0] = torch.tensor(list(last_tokens), device=self.device)
        self.ids[n:batch] = 0
        self.positions[:n] = torch.tensor(positions, device=self.device)
        self.positions[n:batch] = 0
        if self.paged:
            nb = kv_len // self.cache.block_size
            rows = [(t + [self.cache.scratch_block] * nb)[:nb] for t in (self.cache.block_tables[s] for s in slots)]
            self.tables[:n, :nb] = torch.tensor(rows, device=self.device)
            self.tables[n:batch, :nb] = self.cache.scratch_block
        else:
            self.slots[:n] = torch.tensor(list(slots), device=self.device)
            self.slots[n:batch] = self.cache.scratch_slot

        graph.replay()
        self.replays += 1
        for s in slots:
            self.cache.lengths[s] += 1
        return logits[:n]
