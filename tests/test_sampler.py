"""Per-request sampling, tested on hand-made logits."""

import torch

from engine.sampler import Sampler
from engine.sequence import SamplingParams


def test_greedy_is_argmax():
    logits = torch.randn(4, 100)
    out = Sampler(0, "cpu")(logits, [SamplingParams()] * 4)
    assert out == logits.argmax(-1).tolist()


def test_top_k_1_and_tiny_top_p_are_greedy():
    logits = torch.randn(3, 100)
    params = [SamplingParams(temperature=1.0, top_k=1), SamplingParams(temperature=1.0, top_p=1e-6),
              SamplingParams(temperature=5.0, top_k=1, top_p=0.5)]
    assert Sampler(0, "cpu")(logits, params) == logits.argmax(-1).tolist()


def test_same_seed_same_samples():
    logits = torch.randn(8, 100)
    params = [SamplingParams(temperature=1.0)] * 8
    assert Sampler(123, "cpu")(logits, params) == Sampler(123, "cpu")(logits, params)


def test_top_k_only_samples_from_top_k():
    logits = torch.randn(1, 50)
    top5 = set(logits[0].topk(5).indices.tolist())
    sampler = Sampler(0, "cpu")
    seen = {sampler(logits, [SamplingParams(temperature=2.0, top_k=5)])[0] for _ in range(300)}
    assert seen <= top5 and len(seen) > 1


def test_top_p_keeps_the_nucleus():
    # Probabilities 0.5, 0.3, 0.2 (then ~0): top_p=0.7 must keep exactly the first two.
    logits = torch.full((1, 10), -1e9)
    logits[0, :3] = torch.tensor([0.5, 0.3, 0.2]).log()
    sampler = Sampler(0, "cpu")
    seen = {sampler(logits, [SamplingParams(temperature=1.0, top_p=0.7)])[0] for _ in range(300)}
    assert seen == {0, 1}


def test_mixed_batch_keeps_greedy_rows_greedy():
    logits = torch.randn(4, 100)
    params = [SamplingParams(), SamplingParams(temperature=1.5), SamplingParams(), SamplingParams(temperature=0.7, top_p=0.9)]
    out = Sampler(0, "cpu")(logits, params)
    argmax = logits.argmax(-1).tolist()
    assert out[0] == argmax[0] and out[2] == argmax[2]
