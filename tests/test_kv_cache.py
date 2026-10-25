"""Slot KV cache and static batching (Stage 3).

* Slot bookkeeping: allocate, free, reuse, no double free.
* Batched decode attention (no model, random tensors): every sequence in a
  mixed-length batch must get exactly the attention it would get alone.
* Static batching with the real model: batched greedy outputs must equal
  running each prompt alone with the Stage 2 single-sequence path.
"""

import pytest
import torch
from conftest import DEVICE, requires_gpu

from engine.engine import ModelRunner, generate_greedy, generate_static
from engine.kv_cache import SingleSequenceKVCache, SlotKVCache
from engine.loader import load_eos_token_ids, load_model
from engine.model import sdpa_attention


def tiny_cache(num_slots: int = 4, max_len: int = 16) -> SlotKVCache:
    return SlotKVCache(num_layers=2, num_kv_heads=2, head_dim=8, num_slots=num_slots, max_len=max_len,
                       dtype=torch.float32, device="cpu")


def test_slot_allocation_and_reuse():
    cache = tiny_cache(num_slots=3)
    slots = [cache.allocate() for _ in range(3)]
    assert slots == [0, 1, 2]
    with pytest.raises(RuntimeError):
        cache.allocate()
    cache.lengths[1] = 7
    cache.free(1)
    assert cache.num_free_slots == 1
    assert cache.allocate() == 1 and cache.lengths[1] == 0  # reused, starts empty
    with pytest.raises(RuntimeError):
        cache.free(0)
        cache.free(0)  # double free


@torch.inference_mode()
def test_batched_decode_attention_matches_per_sequence():
    torch.manual_seed(0)
    cache = tiny_cache(num_slots=4, max_len=16)
    heads, kv_heads, dim = 4, 2, 8

    # Fill three slots with prompts of different lengths, plus stale junk in
    # slot 1 from a "previous owner" that must never be read.
    cache.k[:, 1].normal_()
    cache.v[:, 1].normal_()
    lengths = {2: 5, 0: 11, 1: 3}
    for slot, n in lengths.items():
        ctx = cache.prefill_context(slot)
        for layer in range(2):
            ctx.attend(layer, torch.randn(1, heads, n, dim), torch.randn(1, kv_heads, n, dim), torch.randn(1, kv_heads, n, dim))
        ctx.commit(n)

    slots = [2, 0, 1]  # deliberately not in slot order
    q = torch.randn(3, heads, 1, dim)
    k = torch.randn(3, kv_heads, 1, dim)
    v = torch.randn(3, kv_heads, 1, dim)
    ctx = cache.decode_context(slots)
    assert ctx.positions.tolist() == [5, 11, 3]  # each new token's own position
    out = ctx.attend(0, q, k, v)

    for b, slot in enumerate(slots):
        n = lengths[slot]
        # The new k/v were written at position n of the right slot...
        assert torch.equal(cache.k[0, slot, :, n], k[b, :, 0])
        # ...and attention equals attending over just this sequence's n + 1 keys.
        ref = sdpa_attention(q[b : b + 1], cache.k[0, slot, :, : n + 1][None], cache.v[0, slot, :, : n + 1][None])
        torch.testing.assert_close(out[b : b + 1], ref, rtol=1e-5, atol=1e-6)

    ctx.commit()
    assert [cache.lengths[s] for s in slots] == [6, 12, 4]


def run_static_vs_single(model_dir, chat_prompt_ids, dtype, device, num_tokens):
    model = load_model(str(model_dir), dtype, device)
    eos = load_eos_token_ids(model_dir)
    single = SingleSequenceKVCache.for_model(model.cfg, max_len=1024, dtype=dtype, device=device)
    expected = [generate_greedy(model, p, num_tokens, single, eos_token_ids=eos) for p in chat_prompt_ids]

    cache = SlotKVCache.for_model(model.cfg, num_slots=8, max_len=1024, dtype=dtype, device=device)
    runner = ModelRunner(model, cache)
    got = generate_static(runner, chat_prompt_ids, num_tokens, eos_token_ids=eos)
    assert cache.num_free_slots == 8  # every slot handed back
    return expected, got


def test_static_batch_matches_single_fp32(model_dir, chat_prompt_ids):
    """fp32 on whatever device is available: batched and single outputs must be identical."""
    expected, got = run_static_vs_single(model_dir, chat_prompt_ids, torch.float32, DEVICE, num_tokens=24)
    for i, (e, g) in enumerate(zip(expected, got)):
        assert g == e, f"prompt {i}: batched output differs from single-sequence output"


@requires_gpu
def test_static_batch_matches_single_fp16(model_dir, chat_prompt_ids):
    expected, got = run_static_vs_single(model_dir, chat_prompt_ids, torch.float16, "cuda", num_tokens=50)
    for i, (e, g) in enumerate(zip(expected, got)):
        assert g == e, f"prompt {i}: batched output differs from single-sequence output"
