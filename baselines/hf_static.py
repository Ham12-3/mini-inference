"""Baseline: Hugging Face ``generate()`` with static batching.

All prompts in a batch are left-padded to the same length and generated
together. The batch runs until its longest request is done; finished rows
keep being computed (as padding) until then.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def load_hf_greedy(model_path: str, dtype: torch.dtype, device: str):
    """Load the HF model set up for plain greedy decoding.

    The model's saved generation config adds ``repetition_penalty=1.1``, which
    is not greedy and costs extra work. Passing a config to ``generate()`` does
    not remove it (transformers merges the two), so we replace it.
    """
    from transformers import AutoModelForCausalLM, GenerationConfig

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, attn_implementation="sdpa")
    model.to(device).eval()
    model.generation_config = GenerationConfig(do_sample=False)
    return model


@torch.inference_mode()
def hf_static_generate(
    model,
    prompts: Sequence[Sequence[int]],
    max_new_tokens: int,
    pad_token_id: int,
    eos_token_ids: Sequence[int] | None = None,
) -> list[list[int]]:
    """Generate for a batch of prompts in one ``generate()`` call.

    With ``eos_token_ids=None`` every row generates exactly ``max_new_tokens``
    tokens (what the throughput benchmark wants). Returns each row's new
    tokens, trimmed after its first EOS.
    """
    device = model.device
    width = max(len(p) for p in prompts)
    input_ids = torch.full((len(prompts), width), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for i, p in enumerate(prompts):
        # Left padding: real tokens are right-aligned so every row's next
        # token is generated at the same column.
        input_ids[i, width - len(p) :] = torch.tensor(p, device=device)
        attention_mask[i, width - len(p) :] = 1

    kwargs = {"max_new_tokens": max_new_tokens, "pad_token_id": pad_token_id}
    if eos_token_ids is None:
        kwargs["min_new_tokens"] = max_new_tokens
    else:
        kwargs["eos_token_id"] = list(eos_token_ids)
    out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    results = []
    for row in out[:, width:].tolist():
        if eos_token_ids:
            for j, tok in enumerate(row):
                if tok in eos_token_ids:
                    row = row[: j + 1]
                    break
        results.append(row)
    return results
