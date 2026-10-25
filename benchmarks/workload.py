"""Benchmark workloads.

For now: a fixed pool of varied chat prompts used by the batch-size sweep.
Stage 5 adds the mixed-length, Poisson-arrival request generator.
"""

from __future__ import annotations

from collections.abc import Sequence

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
