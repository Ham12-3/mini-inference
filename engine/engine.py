"""Generation loops.

Stage 2 has one: :func:`generate_greedy`, a single sequence with a KV cache.
The multi-request ``LLMEngine`` (scheduler + continuous batching) arrives in Stage 4.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from engine.kv_cache import SingleSequenceKVCache
from engine.model import Qwen2ForCausalLM


@torch.inference_mode()
def generate_greedy(
    model: Qwen2ForCausalLM,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    cache: SingleSequenceKVCache,
    eos_token_ids: Sequence[int] = (),
) -> list[int]:
    """Greedy-decode one sequence using a KV cache.

    1. Prefill: run the whole prompt through the model in one forward pass.
       This fills the cache with k/v for every prompt token and gives the
       logits that pick the first output token.
    2. Decode: feed back one token at a time. Each step computes k/v for that
       token only and attends over the cache.

    Returns the generated ids (without the prompt). Stops after
    ``max_new_tokens`` or right after producing a token in ``eos_token_ids``.
    """
    device = cache.k.device
    cache.reset()

    ids = torch.tensor([list(prompt_ids)], device=device)
    positions = torch.arange(ids.shape[1], device=device).unsqueeze(0)
    hidden = model(ids, positions, cache)
    cache.advance(ids.shape[1])
    # Only the last prompt position predicts the next token.
    next_id = model.compute_logits(hidden[:, -1]).argmax(dim=-1)  # [1]
    out = [int(next_id)]

    stop = set(eos_token_ids)
    while len(out) < max_new_tokens and out[-1] not in stop:
        # The new token's position is the number of tokens already cached.
        positions = torch.tensor([[cache.length]], device=device)
        hidden = model(next_id.view(1, 1), positions, cache)
        cache.advance(1)
        next_id = model.compute_logits(hidden[:, -1]).argmax(dim=-1)
        out.append(int(next_id))
    return out
