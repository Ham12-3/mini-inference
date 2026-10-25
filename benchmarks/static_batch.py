"""Static batching throughput vs batch size (chart 2).

B different prompts start together and each generates exactly N tokens
(EOS ignored, so every batch does the same amount of work per request).
Throughput = B * N / wall time, including prefill.

    python -m benchmarks.static_batch --backend engine --batch-sizes 1 4 8 16 32
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable, Sequence

import torch

from benchmarks.report import save_results
from benchmarks.workload import PROMPT_POOL, chat_prompt_ids
from engine.loader import resolve_model_path

BatchFn = Callable[[Sequence[Sequence[int]], int], list[list[int]]]


def make_engine(model_path: str, dtype: torch.dtype, device: str, max_batch: int) -> BatchFn:
    from engine.engine import ModelRunner, generate_static
    from engine.kv_cache import SlotKVCache
    from engine.loader import load_model

    model = load_model(model_path, dtype, device)
    cache = SlotKVCache.for_model(model.cfg, num_slots=max_batch, max_len=1024, dtype=dtype, device=device)
    runner = ModelRunner(model, cache)
    return lambda prompts, n: generate_static(runner, prompts, n, eos_token_ids=())


def make_hf(model_path: str, dtype: torch.dtype, device: str, max_batch: int) -> BatchFn:
    from transformers import AutoTokenizer

    from baselines.hf_static import hf_static_generate, load_hf_greedy

    model = load_hf_greedy(model_path, dtype, device)
    pad = AutoTokenizer.from_pretrained(model_path).pad_token_id
    return lambda prompts, n: hf_static_generate(model, prompts, n, pad_token_id=pad, eos_token_ids=None)


BACKENDS = {"engine": make_engine, "hf": make_hf}


def timed(fn: BatchFn, prompts: Sequence[Sequence[int]], n: int) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn(prompts, n)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    assert all(len(o) == n for o in out), "every request must generate exactly n tokens"
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    model_path = str(resolve_model_path(args.model))
    tok = AutoTokenizer.from_pretrained(model_path)
    pool = chat_prompt_ids(tok, PROMPT_POOL)
    fn = BACKENDS[args.backend](model_path, torch.float16, "cuda", max(args.batch_sizes))

    timed(fn, pool[:4], 16)  # warm-up

    runs = []
    for b in args.batch_sizes:
        prompts = [pool[i % len(pool)] for i in range(b)]
        torch.cuda.reset_peak_memory_stats()
        times = [timed(fn, prompts, args.output_len) for _ in range(args.repeats)]
        median = statistics.median(times)
        runs.append(
            {
                "batch_size": b,
                "seconds": times,
                "median_seconds": median,
                "tokens_per_s": b * args.output_len / median,
                "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
            }
        )
        print(f"{args.backend:>6}  B={b:<3} {median:7.2f}s  {b * args.output_len / median:8.1f} tok/s")

    save_results(
        f"static_{args.backend}",
        {
            "backend": args.backend,
            "model": args.model,
            "dtype": "float16",
            "output_len": args.output_len,
            "prompt_lens": [len(p) for p in pool],
            "repeats": args.repeats,
            "runs": runs,
        },
    )


if __name__ == "__main__":
    main()
