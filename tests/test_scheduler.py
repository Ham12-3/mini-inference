"""Scheduler rules, tested without a model: tokens are faked, slots are real."""

import random

import pytest
import torch

from engine.config import EngineConfig
from engine.kv_cache import SlotKVCache
from engine.scheduler import Scheduler
from engine.sequence import FinishReason, SamplingParams, Sequence, SequenceStatus

EOS = 0
TOKEN = 7  # a fake non-EOS token


def make_scheduler(max_num_seqs=4, max_model_len=64, max_num_batched_tokens=128, scheduling="continuous"):
    cfg = EngineConfig(device="cpu", max_num_seqs=max_num_seqs, max_model_len=max_model_len,
                       max_num_batched_tokens=max_num_batched_tokens, scheduling=scheduling)
    cache = SlotKVCache(num_layers=1, num_kv_heads=1, head_dim=4, num_slots=max_num_seqs,
                        max_len=max_model_len, dtype=torch.float32, device="cpu")
    return Scheduler(cfg, cache, eos_token_ids=[EOS]), cache


def make_seq(i, prompt_len, max_tokens=8):
    return Sequence(i, [TOKEN] * prompt_len, SamplingParams(max_tokens=max_tokens), arrival_time=0.0)


def check_invariants(sched: Scheduler, cache: SlotKVCache):
    slots = [s.slot for s in sched.running]
    assert None not in slots
    assert len(slots) == len(set(slots)), f"a slot is shared: {slots}"
    assert not set(slots) & set(cache._free), "a running sequence holds a slot marked free"
    assert len(sched.running) <= sched.config.max_num_seqs


def run_step(sched: Scheduler, token=TOKEN):
    out = sched.schedule()
    for seq in out.prefills + out.decodes:
        sched.append_token(seq, token, now=0.0)
    return out


def test_slots_are_reused_and_never_shared():
    rng = random.Random(0)
    sched, cache = make_scheduler(max_num_seqs=4)
    seqs = [make_seq(i, rng.randint(3, 30), rng.randint(1, 15)) for i in range(20)]
    for s in seqs:
        sched.add(s)
    slot_history = []  # the slot each admitted sequence got
    while sched.has_work():
        out = sched.schedule()
        slot_history += [s.slot for s in out.prefills]
        check_invariants(sched, cache)
        for seq in out.prefills + out.decodes:
            sched.append_token(seq, TOKEN, now=0.0)
        check_invariants(sched, cache)
    assert all(s.status is SequenceStatus.FINISHED and s.slot is None for s in seqs)
    assert all(len(s.output_token_ids) == s.sampling.max_tokens for s in seqs)
    assert len(slot_history) == 20 and set(slot_history) == {0, 1, 2, 3}  # 20 requests through 4 slots
    assert cache.num_free_slots == 4


def test_queue_respects_max_num_seqs_and_fifo_order():
    sched, cache = make_scheduler(max_num_seqs=4)
    for i in range(10):
        sched.add(make_seq(i, 5, max_tokens=3))
    out = sched.schedule()
    assert [s.seq_id for s in out.prefills] == [0, 1, 2, 3]
    assert len(sched.running) == 4 and len(sched.waiting) == 6
    assert sched.schedule().prefills == []  # full: nobody else gets in
    admitted = [0, 1, 2, 3]
    while sched.has_work():
        admitted += [s.seq_id for s in run_step(sched).prefills]
    assert admitted == list(range(10))  # first come, first served


def test_finished_slot_is_given_to_waiting_request_next_step():
    sched, cache = make_scheduler(max_num_seqs=2)
    a, b, c = make_seq(0, 5, max_tokens=1), make_seq(1, 5, max_tokens=5), make_seq(2, 5, max_tokens=5)
    for s in (a, b, c):
        sched.add(s)
    run_step(sched)  # a and b prefill; a finishes immediately (max_tokens=1)
    assert a.is_finished and c.status is SequenceStatus.WAITING
    out = run_step(sched)
    assert out.prefills == [c] and c.slot == 0  # c took a's freed slot


def test_token_budget_limits_prefills_per_step():
    sched, cache = make_scheduler(max_num_seqs=4, max_model_len=64, max_num_batched_tokens=64)
    for i in range(3):
        sched.add(make_seq(i, 30))
    out = sched.schedule()
    assert len(out.prefills) == 2  # 30 + 30 fit in 64 tokens, the third must wait
    assert len(sched.waiting) == 1


def test_prompt_longer_than_max_model_len_is_rejected():
    sched, _ = make_scheduler(max_model_len=64)
    with pytest.raises(ValueError, match="max_model_len"):
        sched.add(make_seq(0, 64))
    with pytest.raises(ValueError, match="empty"):
        sched.add(make_seq(1, 0))


def test_finish_reasons():
    sched, cache = make_scheduler(max_num_seqs=2, max_model_len=16)
    eos_seq = make_seq(0, 4, max_tokens=10)
    long_seq = make_seq(1, 10, max_tokens=100)
    sched.add(eos_seq)
    sched.add(long_seq)
    sched.schedule()
    sched.append_token(eos_seq, EOS, now=0.0)
    assert eos_seq.finish_reason is FinishReason.EOS
    while not long_seq.is_finished:
        sched.append_token(long_seq, TOKEN, now=0.0)
    assert long_seq.finish_reason is FinishReason.MAX_MODEL_LEN
    assert long_seq.num_tokens == 16
    assert cache.num_free_slots == 2


def test_static_scheduling_waits_for_the_whole_batch():
    sched, cache = make_scheduler(max_num_seqs=2, scheduling="static")
    a, b, c = make_seq(0, 5, max_tokens=1), make_seq(1, 5, max_tokens=4), make_seq(2, 5, max_tokens=4)
    for s in (a, b, c):
        sched.add(s)
    run_step(sched)  # a and b admitted; a finishes at once, freeing a slot
    assert a.is_finished and cache.num_free_slots == 1
    while not b.is_finished:
        assert run_step(sched).prefills == []  # c must wait although a slot is free
    assert run_step(sched).prefills == [c]


def test_abort_frees_slot_or_leaves_queue():
    sched, cache = make_scheduler(max_num_seqs=1)
    running, queued = make_seq(0, 5, max_tokens=10), make_seq(1, 5, max_tokens=10)
    sched.add(running)
    sched.add(queued)
    run_step(sched)
    sched.abort(queued, now=0.0)  # still waiting: just leaves the queue
    assert queued.finish_reason is FinishReason.ABORT and not sched.waiting
    sched.abort(running, now=0.0)  # running: finishes and frees its slot
    assert running.finish_reason is FinishReason.ABORT and cache.num_free_slots == 1
    assert not sched.has_work()
