"""Real vLLM on the same Stage 5 workload, for comparison (offline: all requests at t=0).

vLLM pins its own torch version, so on Colab it runs in a separate
virtualenv (see the notebook) and is never imported by the engine.

    PYTHONPATH=. /content/vllm-venv/bin/python benchmarks/vllm_bench.py

Throughput is measured exactly like ``run_bench --rate inf``: output tokens
divided by the wall time from submitting all requests until the last one
finishes. Per-request latency is reported when vLLM exposes request
timestamps (``RequestOutput.metrics``); otherwise only throughput is saved.
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    args = parser.parse_args()

    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from benchmarks.report import save_results
    from benchmarks.workload import make_workload

    tok = AutoTokenizer.from_pretrained(args.model)
    requests = make_workload(tok, num_requests=args.num_requests, seed=0, rate=None)

    # Same limits as my engine: fp16, 32 concurrent sequences, 1024-token context.
    llm = LLM(model=args.model, dtype="float16", max_num_seqs=args.max_num_seqs, max_model_len=1024,
              gpu_memory_utilization=args.gpu_memory_utilization, seed=0)
    prompts = [{"prompt_token_ids": r.prompt_ids} for r in requests]
    params = [SamplingParams(temperature=0.0, max_tokens=r.output_len, ignore_eos=True) for r in requests]

    llm.generate(prompts[:4], [SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)] * 4)  # warm-up
    start = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    duration = time.perf_counter() - start

    tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    metrics = {
        "num_requests": len(outputs),
        "output_tokens": tokens,
        "duration_s": round(duration, 3),
        "output_tokens_per_s": round(tokens / duration, 1),
        "requests_per_s": round(len(outputs) / duration, 3),
    }
    # Request timestamps, if this vLLM version records them.
    m = [o.metrics for o in outputs]
    if all(x is not None and getattr(x, "first_token_time", None) for x in m):
        ttft = np.array([x.first_token_time - x.arrival_time for x in m])
        e2e = np.array([x.finished_time - x.arrival_time for x in m])
        metrics.update(ttft_p50_s=round(float(np.percentile(ttft, 50)), 4), ttft_p95_s=round(float(np.percentile(ttft, 95)), 4),
                       e2e_p50_s=round(float(np.percentile(e2e, 50)), 4), e2e_p95_s=round(float(np.percentile(e2e, 95)), 4))
    print(" ".join(f"{k}={v}" for k, v in metrics.items()))

    save_results("bench_vllm_rateinf", {
        "backend": "vllm",
        "vllm_version": vllm.__version__,
        "vllm_torch": torch.__version__,
        "model": args.model,
        "dtype": "float16",
        "workload": {"num_requests": args.num_requests, "seed": 0, "rate": "inf", "ignore_eos": True},
        "max_num_seqs": args.max_num_seqs,
        "metrics": metrics,
    })


if __name__ == "__main__":
    main()
