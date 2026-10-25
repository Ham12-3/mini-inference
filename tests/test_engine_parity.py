"""Continuous batching must not change what a request generates.

Requests arrive at different times, queue when the engine is full, join and
leave the running batch at different steps. With greedy sampling, each
request's output must still be exactly what it generates when run alone.
"""

import pytest
from conftest import DEVICE, requires_gpu

from benchmarks.workload import PROMPT_POOL
from benchmarks.workload import chat_prompt_ids as encode_chat
from engine.config import EngineConfig
from engine.engine import LLMEngine, generate_greedy
from engine.kv_cache import SingleSequenceKVCache
from engine.sequence import SamplingParams


def run_staggered(engine: LLMEngine, prompts, max_tokens, arrive_every: int):
    """Two requests start; then one more arrives every ``arrive_every`` steps."""
    pending = list(zip(prompts, max_tokens))
    ids = []

    def arrive():
        prompt, n = pending.pop(0)
        ids.append(engine.add_request(prompt, SamplingParams(max_tokens=n)))

    arrive()
    arrive()
    step = 0
    while pending or engine.has_unfinished_requests():
        if pending and step % arrive_every == 0:
            arrive()
        if engine.has_unfinished_requests():
            engine.step()
        step += 1
    return [engine.get_output(i).output_token_ids for i in ids]


def check_engine_matches_single(engine, prompts, max_tokens, arrive_every):
    got = run_staggered(engine, prompts, max_tokens, arrive_every)

    single = SingleSequenceKVCache.for_model(engine.model.cfg, max_len=1024, dtype=engine.config.torch_dtype,
                                             device=engine.config.device)
    expected = [generate_greedy(engine.model, p, n, single, eos_token_ids=engine.eos_token_ids)
                for p, n in zip(prompts, max_tokens)]
    for i, (e, g) in enumerate(zip(expected, got)):
        assert g == e, f"request {i}: engine output differs from running it alone"

    # Make sure the run actually exercised continuous batching.
    stats = engine.step_stats
    assert max(s.num_decodes for s in stats) > 1, "never decoded more than one sequence at once"
    assert any(s.num_prefills and s.num_decodes for s in stats), "no request ever joined a running batch"
    assert any(s.num_waiting for s in stats), "the queue was never used"
    assert engine.cache.num_free_slots == engine.config.max_num_seqs


@pytest.fixture(scope="module")
def prompts(tokenizer, chat_prompt_ids):
    """Eight prompts: the five parity prompts plus three from the benchmark pool."""
    return chat_prompt_ids + encode_chat(tokenizer, PROMPT_POOL[2:5])


def test_engine_matches_single_fp32(prompts):
    cfg = EngineConfig(device=DEVICE, dtype="float32", max_num_seqs=3, max_model_len=512, max_num_batched_tokens=1024)
    engine = LLMEngine(cfg)
    max_tokens = [8, 20, 12, 16, 6, 18, 10, 14]
    check_engine_matches_single(engine, prompts, max_tokens, arrive_every=3)


@requires_gpu
def test_engine_matches_single_fp16(prompts):
    cfg = EngineConfig(device="cuda", dtype="float16", max_num_seqs=4, max_model_len=1024, max_num_batched_tokens=2048)
    engine = LLMEngine(cfg)
    max_tokens = [50, 20, 64, 35, 50, 28, 60, 40]
    check_engine_matches_single(engine, prompts, max_tokens, arrive_every=7)


def test_engine_paged_matches_single_fp32(prompts):
    cfg = EngineConfig(device=DEVICE, dtype="float32", max_num_seqs=3, max_model_len=512, max_num_batched_tokens=1024,
                       kv_cache="paged", block_size=16)
    engine = LLMEngine(cfg)
    check_engine_matches_single(engine, prompts, [8, 20, 12, 16, 6, 18, 10, 14], arrive_every=3)


def test_engine_paged_with_preemption_matches_single(prompts):
    """A pool of only 7 blocks forces preemption; recomputed requests must still match.

    Two ~45-token prompts are admitted with 3 blocks each; growing to ~65
    tokens they need 4 each, 8 in total, so one of them must be preempted.
    """
    cfg = EngineConfig(device=DEVICE, dtype="float32", max_num_seqs=4, max_model_len=112, max_num_batched_tokens=512,
                       kv_cache="paged", block_size=16, kv_cache_tokens=112)
    engine = LLMEngine(cfg)
    check_engine_matches_single(engine, prompts, [20, 20, 16, 20, 12, 18, 20, 14], arrive_every=2)
    assert engine.scheduler.num_preemptions > 0, "the test pool was meant to force preemption"
    assert engine.cache.num_free_blocks == 7


@requires_gpu
def test_engine_paged_matches_single_fp16(prompts):
    cfg = EngineConfig(device="cuda", dtype="float16", max_num_seqs=4, max_model_len=1024, max_num_batched_tokens=2048,
                       kv_cache="paged", block_size=16)
    engine = LLMEngine(cfg)
    check_engine_matches_single(engine, prompts, [50, 20, 64, 35, 50, 28, 60, 40], arrive_every=7)


def test_rejects_prompt_longer_than_max_model_len():
    cfg = EngineConfig(device=DEVICE, dtype="float32", max_num_seqs=2, max_model_len=64, max_num_batched_tokens=64)
    engine = LLMEngine(cfg)
    with pytest.raises(ValueError, match="max_model_len"):
        engine.add_request(list(range(100)))
    assert not engine.has_unfinished_requests()
