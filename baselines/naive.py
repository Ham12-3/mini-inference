"""Stage 1 baseline: generation with no KV cache.

To produce token number ``t``, we run the *whole* sequence so far (prompt plus
the ``t - 1`` tokens already generated) through the model again, then keep only
the logits of the last position. All the keys and values computed in earlier
steps are thrown away and recomputed.

That makes step ``t`` cost roughly ``prompt_len + t`` tokens of compute, so
generating ``n`` tokens costs about ``n * prompt_len + n**2 / 2`` token-passes
instead of ``prompt_len + n``. This file exists to be the slow line on the
first chart: it shows what the KV cache (Stage 2) saves us.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


@torch.inference_mode()
def naive_generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    on_token: Callable[[int], None] | None = None,
) -> list[int]:
    """Greedy-decode up to ``max_new_tokens`` tokens for one sequence, recomputing everything each step.

    Args:
        model: A Hugging Face causal LM (anything returning ``.logits``).
        input_ids: Prompt token ids, shape ``[1, prompt_len]``, on the model's device.
        max_new_tokens: How many tokens to generate.
        eos_token_id: Stop early when this token is produced. ``None`` always
            generates exactly ``max_new_tokens`` tokens (what the benchmark wants).
        on_token: Optional callback run with each new token as soon as it is
            known (the benchmark uses it to time the first token).

    Returns:
        The generated token ids (prompt not included).
    """
    ids = input_ids
    generated: list[int] = []
    for _ in range(max_new_tokens):
        # use_cache=False: the model keeps no state between calls, so every
        # call is a full forward pass over all ids.
        logits = model(input_ids=ids, use_cache=False).logits[:, -1, :]
        next_id = logits.argmax(dim=-1, keepdim=True)  # greedy: [1, 1]
        generated.append(int(next_id))
        if on_token is not None:
            on_token(generated[-1])
        if eos_token_id is not None and generated[-1] == eos_token_id:
            break
        ids = torch.cat([ids, next_id], dim=1)
    return generated
