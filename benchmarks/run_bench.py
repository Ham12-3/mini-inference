"""Run one backend on the Stage 5 workload and save throughput and latency metrics.

    python -m benchmarks.run_bench --backend engine_continuous --rate inf
    python -m benchmarks.run_bench --backend engine_static --rate 4
    python -m benchmarks.run_bench --backend naive --rate inf --num-requests 10

Backends:
    naive              HF model, no KV cache, one request at a time (use a small subset)
    hf_static          HF generate(), static batches of up to --max-num-seqs requests
    engine_static      my engine, a new batch only when the previous one is done
    engine_continuous  my engine, continuous batching

Every backend serves the same requests (same prompts, same output lengths,
EOS ignored) in the same way: a request can only start after its arrival
time. ``--rate inf`` makes all requests arrive at t=0 (offline: measures
peak throughput); a finite rate uses Poisson arrivals (online: measures
latency under load).

Metrics, per request:
    TTFT (time to first token)  = first token time - arrival time (includes queueing)
    TPOT (time per output token) = (finish - first token) / (output tokens - 1)
    end-to-end latency           = finish - arrival
and for the run: output tokens/s and requests/s over the whole run, and peak
GPU memory (``torch.cuda.max_memory_allocated``, including preallocated caches).
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from benchmarks.report import save_results
from benchmarks.workload import Request, make_workload
from engine.loader import resolve_model_path

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_MODEL_LEN = 1024  # longest prompt (512) + longest output (512)


@dataclass
class Record:
    arrival: float  # absolute perf_counter times
    first_token: float
    finish: float
    output_tokens: int


def wait_until(t: float) -> None:
    delay = t - time.perf_counter()
    if delay > 0:
        time.sleep(delay)


# --------------------------------------------------------------------------- backends
# Each takes the workload and an absolute start time, serves every request no
# earlier than start + arrival_time, and returns one Record per request plus
# an optional timeline of (seconds since start, running batch size, queue length).


def run_engine(requests: list[Request], start: float, scheduling: str, max_num_seqs: int):
    from engine.config import EngineConfig
    from engine.engine import LLMEngine
    from engine.sequence import SamplingParams

    engine = LLMEngine(EngineConfig(model=MODEL, max_num_seqs=max_num_seqs, max_model_len=MAX_MODEL_LEN,
                                    max_num_batched_tokens=4096, scheduling=scheduling))
    # Warm-up (kernel selection, allocator), then forget it.
    engine.generate([r.prompt_ids for r in requests[:4]], SamplingParams(max_tokens=8, ignore_eos=True))
    engine.step_stats.clear()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = max(start, time.perf_counter())
    ids, i = [], 0
    while i < len(requests) or engine.has_unfinished_requests():
        now = time.perf_counter()
        while i < len(requests) and start + requests[i].arrival_time <= now:
            r = requests[i]
            ids.append(engine.add_request(r.prompt_ids, SamplingParams(max_tokens=r.output_len, ignore_eos=True),
                                          arrival_time=start + r.arrival_time))
            i += 1
        if engine.has_unfinished_requests():
            engine.step()
        else:
            wait_until(start + requests[i].arrival_time)

    records = []
    for rid in ids:
        s = engine.sequences[rid]
        records.append(Record(s.arrival_time, s.first_token_time, s.finish_time, len(s.output_token_ids)))
    timeline = [(st.time - start, st.num_running, st.num_waiting) for st in engine.step_stats]
    return start, records, timeline


class StepTimer:
    """A ``generate()`` streamer that timestamps every decode step of a batch."""

    def __init__(self) -> None:
        self.times: list[float] = []
        self._seen_prompt = False

    def put(self, value) -> None:
        if not self._seen_prompt:  # the first call carries the prompt, not a new token
            self._seen_prompt = True
            return
        self.times.append(time.perf_counter())  # put() receives CPU tensors, so the step is done

    def end(self) -> None:
        pass


def run_hf_static(requests: list[Request], start: float, max_num_seqs: int):
    from transformers import AutoTokenizer

    from baselines.hf_static import hf_static_generate, load_hf_greedy

    path = str(resolve_model_path(MODEL))
    model = load_hf_greedy(path, torch.float16, "cuda")
    pad = AutoTokenizer.from_pretrained(path).pad_token_id
    hf_static_generate(model, [r.prompt_ids for r in requests[:4]], 8, pad)  # warm-up
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = max(start, time.perf_counter())
    records: list[Record | None] = [None] * len(requests)
    timeline = []
    i = 0
    while i < len(requests):
        wait_until(start + requests[i].arrival_time)
        # Take everything that has arrived (up to the batch limit) as one static batch.
        now = time.perf_counter()
        batch = [i]
        while len(batch) < max_num_seqs and batch[-1] + 1 < len(requests) and start + requests[batch[-1] + 1].arrival_time <= now:
            batch.append(batch[-1] + 1)
        i = batch[-1] + 1
        timer = StepTimer()
        steps = max(requests[j].output_len for j in batch)  # the batch runs until its longest request is done
        timeline.append((now - start, len(batch), 0))
        hf_static_generate(model, [requests[j].prompt_ids for j in batch], steps, pad, streamer=timer)
        for j in batch:
            n = requests[j].output_len
            records[j] = Record(start + requests[j].arrival_time, timer.times[0], timer.times[n - 1], n)
        timeline.append((time.perf_counter() - start, 0, 0))
    return start, records, timeline


def run_naive(requests: list[Request], start: float, max_num_seqs: int):
    from baselines.hf_static import load_hf_greedy
    from baselines.naive import naive_generate

    model = load_hf_greedy(str(resolve_model_path(MODEL)), torch.float16, "cuda")
    naive_generate(model, torch.tensor([requests[0].prompt_ids], device="cuda"), 4)  # warm-up
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = max(start, time.perf_counter())
    records = []
    for r in requests:
        wait_until(start + r.arrival_time)
        first: list[float] = []
        out = naive_generate(model, torch.tensor([r.prompt_ids], device="cuda"), r.output_len,
                             on_token=lambda _t: first.append(time.perf_counter()) if not first else None)
        records.append(Record(start + r.arrival_time, first[0], time.perf_counter(), len(out)))
    return start, records, []


BACKENDS = {
    "naive": run_naive,
    "hf_static": run_hf_static,
    "engine_static": lambda reqs, start, n: run_engine(reqs, start, "static", n),
    "engine_continuous": lambda reqs, start, n: run_engine(reqs, start, "continuous", n),
}


# --------------------------------------------------------------------------- metrics


def summarize(records: list[Record], start: float) -> dict:
    ttft = np.array([r.first_token - r.arrival for r in records])
    tpot = np.array([(r.finish - r.first_token) / (r.output_tokens - 1) for r in records if r.output_tokens > 1])
    e2e = np.array([r.finish - r.arrival for r in records])
    duration = max(r.finish for r in records) - start
    tokens = sum(r.output_tokens for r in records)

    def pct(a: np.ndarray, q: float) -> float:
        return round(float(np.percentile(a, q)), 4)

    return {
        "num_requests": len(records),
        "output_tokens": tokens,
        "duration_s": round(duration, 3),
        "output_tokens_per_s": round(tokens / duration, 1),
        "requests_per_s": round(len(records) / duration, 3),
        "ttft_p50_s": pct(ttft, 50), "ttft_p95_s": pct(ttft, 95),
        "tpot_p50_ms": round(pct(tpot, 50) * 1000, 2), "tpot_p95_ms": round(pct(tpot, 95) * 1000, 2),
        "e2e_p50_s": pct(e2e, 50), "e2e_p95_s": pct(e2e, 95),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
    }


def downsample(timeline: list[tuple], max_points: int = 300) -> list[list[float]]:
    """Keep at most ``max_points`` evenly spaced timeline points (keeps result files small)."""
    if not timeline:
        return []
    step = max(1, math.ceil(len(timeline) / max_points))
    kept = timeline[::step] + ([timeline[-1]] if (len(timeline) - 1) % step else [])
    return [[round(t, 3), running, waiting] for t, running, waiting in kept]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    parser.add_argument("--rate", default="inf", help="requests per second, or 'inf' for all at t=0")
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    rate = None if args.rate == "inf" else float(args.rate)
    tok = AutoTokenizer.from_pretrained(resolve_model_path(MODEL))
    requests = make_workload(tok, num_requests=args.num_requests, seed=args.seed, rate=rate)

    start, records, timeline = BACKENDS[args.backend](requests, time.perf_counter(), args.max_num_seqs)
    metrics = summarize(records, start)
    print(" ".join(f"{k}={v}" for k, v in metrics.items()))

    save_results(
        f"bench_{args.backend}_rate{args.rate}",
        {
            "backend": args.backend,
            "model": MODEL,
            "dtype": "float16",
            "workload": {"num_requests": args.num_requests, "seed": args.seed, "rate": args.rate,
                         "prompt_len": [32, 512], "output_len": "70% in [16,128], 30% in [256,512]",
                         "ignore_eos": True},
            "max_num_seqs": args.max_num_seqs,
            "metrics": metrics,
            "timeline": downsample(timeline),
        },
    )


if __name__ == "__main__":
    main()
