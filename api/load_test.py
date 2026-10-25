"""Load test: fire many streaming requests at the server at once and measure what clients see.

    uvicorn api.server:app --port 8000 &
    python -m api.load_test --url http://127.0.0.1:8000 --concurrency 64

Each client sends a request and reads the SSE stream. We record time to the
first token and to the last, so TTFT includes the time spent waiting in the
server's queue, exactly as a user would experience it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time

import httpx
import numpy as np

from benchmarks.workload import PROMPT_POOL


async def one_request(client: httpx.AsyncClient, url: str, prompt: str, max_tokens: int) -> dict:
    start = time.perf_counter()
    first = None
    tokens = 0
    async with client.stream("POST", f"{url}/generate", json={"prompt": prompt, "max_tokens": max_tokens,
                                                                "ignore_eos": True}) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[len("data: "):])
            if "error" in event:
                raise RuntimeError(event["error"])
            tokens += 1
            if first is None:
                first = time.perf_counter()
    end = time.perf_counter()
    return {"ttft": first - start, "latency": end - start, "tokens": tokens}


async def run(url: str, concurrency: int, seed: int) -> dict:
    rng = random.Random(seed)
    jobs = [(rng.choice(PROMPT_POOL), rng.choice([32, 64, 128, 256])) for _ in range(concurrency)]
    limits = httpx.Limits(max_connections=concurrency + 4)
    async with httpx.AsyncClient(timeout=None, limits=limits) as client:
        await one_request(client, url, "Hello", 4)  # warm-up
        start = time.perf_counter()
        results = await asyncio.gather(*(one_request(client, url, p, n) for p, n in jobs))
        wall = time.perf_counter() - start
    ttft = np.array([r["ttft"] for r in results])
    lat = np.array([r["latency"] for r in results])
    tokens = sum(r["tokens"] for r in results)
    return {
        "concurrency": concurrency,
        "output_tokens": tokens,
        "wall_s": round(wall, 3),
        "output_tokens_per_s": round(tokens / wall, 1),
        "ttft_p50_s": round(float(np.percentile(ttft, 50)), 3),
        "ttft_p95_s": round(float(np.percentile(ttft, 95)), 3),
        "latency_p50_s": round(float(np.percentile(lat, 50)), 3),
        "latency_p95_s": round(float(np.percentile(lat, 95)), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32, 64])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", action="store_true", help="save results to benchmarks/results/api_load_test.json")
    args = parser.parse_args()

    runs = []
    for c in args.concurrency:
        res = asyncio.run(run(args.url, c, args.seed))
        runs.append(res)
        print(" ".join(f"{k}={v}" for k, v in res.items()))
    if args.save:
        from benchmarks.report import save_results

        save_results("api_load_test", {"url_is_local": args.url.startswith("http://127.0.0.1"), "runs": runs})


if __name__ == "__main__":
    main()
