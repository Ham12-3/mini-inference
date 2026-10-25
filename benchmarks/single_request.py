"""Single-request decode speed: tokens per second for one prompt at several output lengths.

This produces the data for chart 1 (naive vs KV cache). Every backend gets the
same chat-formatted prompt and must generate exactly N tokens (EOS is
ignored), so the work done is identical and only the method differs.

    python -m benchmarks.single_request --backend naive --lengths 64 128 256
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

import torch

from benchmarks.env_info import download_model
from benchmarks.report import save_results

PROMPT = "Explain, step by step, how a CPU cache works and why it makes programs faster."

# A backend takes prompt ids ([1, prompt_len] on the GPU) and a token count,
# and returns exactly that many generated token ids.
GenerateFn = Callable[[torch.Tensor, int], list[int]]


def make_naive_backend(model_path: str, dtype: torch.dtype, device: str) -> GenerateFn:
    from transformers import AutoModelForCausalLM

    from baselines.naive import naive_generate

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, attn_implementation="sdpa")
    model.to(device).eval()
    return lambda ids, n: naive_generate(model, ids, n, eos_token_id=None)


BACKENDS: dict[str, Callable[[str, torch.dtype, str], GenerateFn]] = {
    "naive": make_naive_backend,
}


def build_prompt_ids(model_path: str, device: str) -> torch.Tensor:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    text = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True
    )
    return tok(text, return_tensors="pt").input_ids.to(device)


def time_generate(generate: GenerateFn, ids: torch.Tensor, n: int, device: str) -> float:
    """Wall-clock seconds to generate ``n`` tokens, with GPU work fully finished on both ends."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    out = generate(ids, n)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    assert len(out) == n, f"backend returned {len(out)} tokens, expected {n}"
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--lengths", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--name", default=None, help="result file name (default: single_<backend>)")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    model_path = str(download_model(args.model))
    generate = BACKENDS[args.backend](model_path, dtype, args.device)
    ids = build_prompt_ids(model_path, args.device)

    # Warm-up: the first calls pay for CUDA context setup, kernel selection
    # and memory allocation. We do not want that in the numbers.
    for _ in range(2):
        time_generate(generate, ids, 16, args.device)

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    runs = []
    for n in args.lengths:
        times = [time_generate(generate, ids, n, args.device) for _ in range(args.repeats)]
        median = statistics.median(times)
        runs.append(
            {
                "output_len": n,
                "seconds": times,
                "median_seconds": median,
                "tokens_per_s": n / median,
            }
        )
        print(f"{args.backend:>8}  n={n:<4} median {median:7.3f}s  {n / median:8.1f} tok/s")

    save_results(
        args.name or f"single_{args.backend}",
        {
            "backend": args.backend,
            "model": args.model,
            "dtype": args.dtype,
            "device": args.device,
            "prompt_len": ids.shape[1],
            "repeats": args.repeats,
            "peak_memory_gb": (
                torch.cuda.max_memory_allocated() / 1e9 if args.device.startswith("cuda") else None
            ),
            "runs": runs,
        },
    )


if __name__ == "__main__":
    main()
