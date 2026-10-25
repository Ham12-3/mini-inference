"""Benchmark workloads.

* ``PROMPT_POOL``: varied chat prompts, used by the batch-size sweep and the demo.
* :func:`make_workload`: the Stage 5 request set. Fixed seed, mixed prompt and
  output lengths, and optional Poisson arrival times.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

PROMPT_POOL = [
    "Explain, step by step, how a CPU cache works and why it makes programs faster.",
    "Write a Python function that merges two sorted lists into one sorted list.",
    "What causes the seasons on Earth?",
    "Summarise the plot of Romeo and Juliet in a paragraph.",
    "Give me a recipe for a simple vegetable curry.",
    "What is the difference between TCP and UDP?",
    "Write a short poem about the ocean at night.",
    "How does a hash map handle collisions?",
    "List the planets of the solar system with one fact about each.",
    "Explain what a neural network is to a ten-year-old.",
    "What are the pros and cons of working from home?",
    "Describe how vaccines train the immune system.",
    "Write a SQL query that finds the top five customers by total spend.",
    "Why is the sky blue?",
    "Compare Python and Rust for building command-line tools.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
]


def chat_prompt_ids(tokenizer, texts: Sequence[str]) -> list[list[int]]:
    """Wrap each text as a user turn in the model's chat template and tokenize it."""
    out = []
    for text in texts:
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
        out.append(tokenizer(chat).input_ids)
    return out


@dataclass(frozen=True)
class Request:
    prompt_ids: list[int]
    output_len: int  # tokens to generate (EOS is ignored so every backend does the same work)
    arrival_time: float  # seconds after the start of the run


def make_workload(
    tokenizer,
    num_requests: int = 200,
    seed: int = 0,
    prompt_len: tuple[int, int] = (32, 512),
    short_output: tuple[int, int] = (16, 128),
    long_output: tuple[int, int] = (256, 512),
    long_fraction: float = 0.3,
    rate: float | None = None,
) -> list[Request]:
    """Build a reproducible list of requests.

    Prompt lengths are uniform in ``prompt_len``. Output lengths are a mix:
    most requests are short, ``long_fraction`` of them long. That mix is what
    makes continuous batching matter: in a static batch, short requests
    finish early and their slots sit idle until the longest one is done.

    Prompts are slices of real tokenized English (the prompt pool repeated),
    so the model sees realistic tokens; their meaning does not matter for
    timing.

    ``rate`` is the mean arrival rate in requests per second. Gaps between
    arrivals are exponential (a Poisson process), the usual model for
    independent users. ``None`` means everything arrives at time 0 (an
    offline batch job, which measures peak throughput).
    """
    rng = random.Random(seed)
    corpus: list[int] = []
    text = " ".join(PROMPT_POOL)
    while len(corpus) < prompt_len[1] * 4:
        corpus += tokenizer(text).input_ids

    requests, t = [], 0.0
    for _ in range(num_requests):
        n_prompt = rng.randint(*prompt_len)
        start = rng.randint(0, len(corpus) - n_prompt)
        n_out = rng.randint(*long_output) if rng.random() < long_fraction else rng.randint(*short_output)
        if rate is not None:
            t += rng.expovariate(rate)
        requests.append(Request(corpus[start : start + n_prompt], n_out, t))
    return requests
