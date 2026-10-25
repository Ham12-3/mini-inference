"""CUDA graphs must change speed only, never outputs (GPU only)."""

import pytest
from conftest import requires_gpu
from test_engine_parity import check_engine_matches_single, run_staggered

from engine.config import EngineConfig
from engine.engine import LLMEngine

MAX_TOKENS = [20, 12, 24, 16, 20, 10, 18, 14]


@requires_gpu
@pytest.mark.parametrize("kv_cache", ["slot", "paged"])
def test_graphs_match_eager_fp32(prompts, kv_cache):
    def run(graphs: bool):
        cfg = EngineConfig(device="cuda", dtype="float32", max_num_seqs=4, max_model_len=512,
                           max_num_batched_tokens=1024, kv_cache=kv_cache, cuda_graphs=graphs)
        engine = LLMEngine(cfg)
        return engine, run_staggered(engine, prompts, MAX_TOKENS, arrive_every=3)

    graphed_engine, graphed = run(True)
    _, eager = run(False)
    assert graphed == eager
    assert graphed_engine.runner.graphs.replays > 0, "no decode step used a CUDA graph"


@requires_gpu
@pytest.mark.parametrize("kv_cache", ["slot", "paged"])
def test_graphs_fp16_match_single(prompts, kv_cache):
    cfg = EngineConfig(device="cuda", dtype="float16", max_num_seqs=4, max_model_len=1024,
                       max_num_batched_tokens=2048, kv_cache=kv_cache, cuda_graphs=True)
    engine = LLMEngine(cfg)
    check_engine_matches_single(engine, prompts, [50, 20, 64, 35, 50, 28, 60, 40], arrive_every=7)
    assert engine.runner.graphs.replays > 0
