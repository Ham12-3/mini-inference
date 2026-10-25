"""Generation loops.

* :func:`generate_greedy`: one sequence with a KV cache (Stage 2).
* :class:`ModelRunner`: runs one prefill or one batched decode step against a
  slot cache. Everything that serves many requests is built on it.
* :func:`generate_static`: static batching (Stage 3). All requests start
  together and the batch runs until the last one finishes.

The multi-request ``LLMEngine`` (scheduler + continuous batching) arrives in Stage 4.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from engine.kv_cache import SingleSequenceKVCache, SlotKVCache
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


class ModelRunner:
    """Runs the model against a :class:`SlotKVCache`, one forward pass per call.

    Returns float32 logits. Sampling (choosing tokens from the logits) is left
    to the caller.
    """

    def __init__(self, model: Qwen2ForCausalLM, cache: SlotKVCache) -> None:
        self.model = model
        self.cache = cache
        self.device = cache.k.device

    @torch.inference_mode()
    def prefill(self, slot: int, prompt_ids: Sequence[int]) -> torch.Tensor:
        """Process a whole prompt into ``slot``; return the logits for the next token, ``[vocab]``."""
        ctx = self.cache.prefill_context(slot)
        ids = torch.tensor([list(prompt_ids)], device=self.device)
        positions = torch.arange(ctx.start, ctx.start + ids.shape[1], device=self.device).unsqueeze(0)
        hidden = self.model(ids, positions, ctx)
        ctx.commit(ids.shape[1])
        return self.model.compute_logits(hidden[0, -1]).float()

    @torch.inference_mode()
    def decode(self, slots: Sequence[int], last_tokens: Sequence[int]) -> torch.Tensor:
        """One decode step for many sequences in ONE forward pass; returns logits ``[B, vocab]``.

        ``last_tokens[b]`` is the most recent token of the sequence in
        ``slots[b]``. Each one goes in at its own sequence's position.
        """
        ctx = self.cache.decode_context(slots)
        ids = torch.tensor(list(last_tokens), device=self.device).unsqueeze(1)  # [B, 1]
        hidden = self.model(ids, ctx.positions.unsqueeze(1), ctx)
        ctx.commit()
        return self.model.compute_logits(hidden[:, -1]).float()


def generate_static(
    runner: ModelRunner,
    prompts: Sequence[Sequence[int]],
    max_new_tokens: int,
    eos_token_ids: Sequence[int] = (),
) -> list[list[int]]:
    """Greedy static batching: prefill every prompt, then decode them together until all are done.

    Prompts are prefilled one at a time (each into its own slot). Decoding is
    batched: one forward pass per step for every unfinished sequence. A
    finished sequence drops out of the batch, but its slot stays idle until
    the whole batch is done; nothing new can join. That waiting is what
    continuous batching (Stage 4) removes.
    """
    cache = runner.cache
    if len(prompts) > cache.num_free_slots:
        raise ValueError(f"{len(prompts)} prompts but only {cache.num_free_slots} free slots")
    for p in prompts:
        if len(p) + max_new_tokens > cache.max_len:
            raise ValueError(f"prompt of {len(p)} tokens + {max_new_tokens} new exceeds max_len {cache.max_len}")

    stop = set(eos_token_ids)
    slots = [cache.allocate() for _ in prompts]
    try:
        outputs = [[int(runner.prefill(slot, p).argmax())] for slot, p in zip(slots, prompts)]

        def finished(i: int) -> bool:
            return len(outputs[i]) >= max_new_tokens or outputs[i][-1] in stop

        active = [i for i in range(len(prompts)) if not finished(i)]
        while active:
            logits = runner.decode([slots[i] for i in active], [outputs[i][-1] for i in active])
            for i, token in zip(active, logits.argmax(dim=-1).tolist()):
                outputs[i].append(token)
            active = [i for i in active if not finished(i)]
    finally:
        for slot in slots:
            cache.free(slot)
    return outputs
