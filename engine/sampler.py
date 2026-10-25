"""Turn logits into tokens, with different sampling settings for each request in the batch."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from engine.sequence import SamplingParams


class Sampler:
    """Batched per-request sampling: greedy, temperature, top-k and top-p.

    All requests in a step are handled in one set of tensor operations, each
    row with its own settings. A single seeded ``torch.Generator`` makes runs
    reproducible: the same requests in the same order give the same tokens.
    (Random draws are consumed row by row, so if the batch composition
    changes, a sampled request's tokens can change too. Greedy requests never
    depend on the batch.)
    """

    def __init__(self, seed: int, device: torch.device | str) -> None:
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

    @torch.inference_mode()
    def __call__(self, logits: torch.Tensor, params: Sequence[SamplingParams]) -> list[int]:
        """Pick one token per row of ``logits`` ([B, vocab], float32)."""
        greedy_tokens = logits.argmax(dim=-1)
        if all(p.greedy for p in params):
            return greedy_tokens.tolist()

        device = logits.device
        temps = torch.tensor([p.temperature if not p.greedy else 1.0 for p in params], device=device)
        logits = logits / temps.unsqueeze(1)

        # Sort once (most likely first); top-k and top-p are both cut-offs on this order.
        sorted_logits, sorted_ids = logits.sort(dim=-1, descending=True)
        vocab = logits.shape[-1]
        ranks = torch.arange(vocab, device=device)

        # top-k: keep ranks < k (k = 0 means keep everything).
        ks = torch.tensor([p.top_k if p.top_k > 0 else vocab for p in params], device=device)
        sorted_logits = sorted_logits.masked_fill(ranks >= ks.unsqueeze(1), float("-inf"))

        # top-p: keep the smallest prefix whose probability reaches top_p. A
        # token is dropped if the tokens ranked above it already reach top_p,
        # so the most likely token is always kept.
        probs = sorted_logits.softmax(dim=-1)
        mass_before = probs.cumsum(dim=-1) - probs
        top_ps = torch.tensor([p.top_p for p in params], device=device)
        sorted_logits = sorted_logits.masked_fill(mass_before >= top_ps.unsqueeze(1), float("-inf"))

        choice = torch.multinomial(sorted_logits.softmax(dim=-1), num_samples=1, generator=self.generator)
        sampled_tokens = sorted_ids.gather(1, choice).squeeze(1)

        is_greedy = torch.tensor([p.greedy for p in params], device=device)
        return torch.where(is_greedy, greedy_tokens, sampled_tokens).tolist()
