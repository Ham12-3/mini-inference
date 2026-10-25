"""Demo: 100 requests of mixed lengths through the continuous-batching engine.

With ``--debug`` the engine prints one line per step: the decode batch size,
how many requests joined (prefilled) and finished that step, and the queue
length. Watch the batch stay full while requests come and go.

    python scripts/demo_continuous_batching.py --debug
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from benchmarks.workload import PROMPT_POOL, chat_prompt_ids
from engine.config import EngineConfig
from engine.engine import LLMEngine
from engine.loader import resolve_model_path
from engine.sequence import SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--debug", action="store_true", help="print the batch composition every step")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(resolve_model_path(args.model))
    pool = chat_prompt_ids(tok, PROMPT_POOL)

    engine = LLMEngine(EngineConfig(model=args.model, max_num_seqs=args.max_num_seqs, max_model_len=1024,
                                    max_num_batched_tokens=2048, seed=args.seed, debug=args.debug,
                                    device=args.device, dtype="float16" if args.device == "cuda" else "float32"))

    # Mixed workload: output caps from 16 to 256 tokens; every third request
    # samples (temperature 0.7, top-p 0.9), the rest are greedy.
    params = []
    for i in range(args.num_requests):
        max_tokens = rng.choice([16, 32, 64, 128, 256])
        params.append(SamplingParams(temperature=0.7, top_p=0.9, max_tokens=max_tokens) if i % 3 == 0
                      else SamplingParams(max_tokens=max_tokens))

    start = time.perf_counter()
    ids = [engine.add_request(pool[i % len(pool)], params[i], arrival_time=start) for i in range(args.num_requests)]
    while engine.has_unfinished_requests():
        engine.step()
    elapsed = time.perf_counter() - start

    outputs = [engine.get_output(i) for i in ids]
    total = sum(len(o.output_token_ids) for o in outputs)
    stats = engine.step_stats
    reasons = {}
    for o in outputs:
        reasons[o.finish_reason] = reasons.get(o.finish_reason, 0) + 1
    print(f"\n{args.num_requests} requests, {total} output tokens in {elapsed:.1f}s "
          f"= {total / elapsed:.0f} tok/s over {len(stats)} steps")
    print(f"largest decode batch: {max(s.num_decodes for s in stats)}, finish reasons: {reasons}")
    for o in outputs[:2]:
        print(f"\n--- request {o.request_id} ({len(o.output_token_ids)} tokens, {o.finish_reason}) ---\n{o.text[:300]}")

    path = Path(__file__).resolve().parents[1] / "benchmarks" / "results" / "demo_continuous_steps.json"
    path.write_text(json.dumps({"start": start, "steps": [vars(s) for s in stats]}) + "\n")
    print(f"\nstep log saved to {path}")


if __name__ == "__main__":
    main()
