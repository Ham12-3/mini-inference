"""The scheduler decides, every step, who prefills, who decodes, and who is done.

It keeps two collections:

* ``waiting``: a FIFO queue of requests that have arrived but have no cache slot yet.
* ``running``: requests that own a slot and produce one token per step.

Each step it admits requests from the front of the queue while three limits
allow it: room in the KV cache, ``max_num_seqs`` running sequences, and the
step's token budget ``max_num_batched_tokens`` (every decode costs 1 token,
a prefill costs its prompt length). When a request finishes, its cache space
is freed immediately, so the next step can admit someone new. That
join-and-leave-every-step behaviour is continuous batching.

With the paged cache, memory can run out mid-generation; the scheduler then
preempts the newest request (see ``_make_room_for_decodes``).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

from engine.config import EngineConfig
from engine.kv_cache import PagedKVCache, SlotKVCache
from engine.sequence import FinishReason, Sequence, SequenceStatus


@dataclass
class SchedulerOutput:
    """What to run in one engine step."""

    prefills: list[Sequence] = field(default_factory=list)  # newly admitted: process the prompt
    decodes: list[Sequence] = field(default_factory=list)  # already running: one token each

    @property
    def is_empty(self) -> bool:
        return not self.prefills and not self.decodes


class Scheduler:
    def __init__(
        self, config: EngineConfig, cache: SlotKVCache | PagedKVCache, eos_token_ids: Iterable[int] = ()
    ) -> None:
        self.config = config
        self.cache = cache
        self.eos_token_ids = set(eos_token_ids)
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.num_preemptions = 0

    def add(self, seq: Sequence) -> None:
        """Queue a new request, rejecting ones that can never fit."""
        if seq.prompt_len == 0:
            raise ValueError("prompt is empty")
        if seq.prompt_len >= self.config.max_model_len:
            raise ValueError(
                f"prompt has {seq.prompt_len} tokens but max_model_len is {self.config.max_model_len}; "
                "it would leave no room to generate"
            )
        self.waiting.append(seq)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> SchedulerOutput:
        """Choose this step's prefills and decodes."""
        self._make_room_for_decodes()
        out = SchedulerOutput(decodes=list(self.running))
        static = self.config.scheduling == "static"
        if static and self.running:
            return out  # static batching: nobody joins until the whole batch is done
        budget = self.config.max_num_batched_tokens - len(out.decodes)
        while self.waiting and len(self.running) < self.config.max_num_seqs:
            seq = self.waiting[0]
            # A request preempted earlier recomputes its prompt AND the tokens
            # it already generated, so it prefills all of them.
            num_tokens = seq.num_tokens
            # The token budget protects running decodes from a long burst of
            # prefills. A static batch has no running decodes, so it fills
            # every slot at once instead.
            if num_tokens > budget and not static:
                # Strict FIFO: do not let a later, shorter prompt jump the
                # queue. The budget refills next step; with nothing running
                # it is at least max_model_len, so every valid prompt fits.
                break
            # Room for the prompt plus its first decoded token.
            if not self.cache.can_allocate(num_tokens + 1):
                break
            self.waiting.popleft()
            # With prefix caching the cache may already hold the start of the
            # prompt; then lengths[slot] > 0 and only the rest is prefilled.
            seq.slot = self.cache.allocate(num_tokens, token_ids=seq.token_ids)
            seq.num_cached_tokens = self.cache.lengths[seq.slot]
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            out.prefills.append(seq)
            budget -= num_tokens - seq.num_cached_tokens
        return out

    def _make_room_for_decodes(self) -> None:
        """Make sure every running sequence has cache space for this step's new token.

        Always true for the slot cache. The paged cache may need a new block;
        if the pool is empty we *preempt*: the most recently admitted sequence
        gives back all its blocks and returns to the front of the queue. When
        readmitted, it recomputes its KV with one prefill over its prompt plus
        the tokens it had generated, then carries on. (vLLM calls this
        preemption by recomputation; the alternative is swapping blocks out to
        CPU memory.)
        """
        for seq in list(self.running):  # oldest first
            if seq not in self.running:
                continue  # already preempted in this loop
            # Decoding writes the last sampled token at position num_tokens - 1.
            while not self.cache.ensure_capacity(seq.slot, seq.num_tokens):
                victim = self.running[-1]
                self._preempt(victim)
                if victim is seq:
                    break

    def _preempt(self, seq: Sequence) -> None:
        self.running.remove(seq)
        self.cache.free(seq.slot)
        seq.slot = None
        seq.status = SequenceStatus.WAITING
        seq.num_preemptions += 1
        self.waiting.appendleft(seq)
        self.num_preemptions += 1

    def append_token(self, seq: Sequence, token: int, now: float) -> None:
        """Record a newly sampled token and finish the sequence if it is done."""
        seq.output_token_ids.append(token)
        if seq.first_token_time is None:
            seq.first_token_time = now

        reason = None
        if token in self.eos_token_ids and not seq.sampling.ignore_eos:
            reason = FinishReason.EOS
        elif len(seq.output_token_ids) >= seq.sampling.max_tokens:
            reason = FinishReason.LENGTH
        elif seq.num_tokens >= self.config.max_model_len:
            # A sequence (prompt + output) may hold at most max_model_len
            # tokens, the size of its cache slot.
            reason = FinishReason.MAX_MODEL_LEN
        if reason is not None:
            self._finish(seq, reason, now)

    def abort(self, seq: Sequence, now: float) -> None:
        """Cancel a request wherever it is: drop it from the queue, or finish it and free its slot."""
        if seq.is_finished:
            return
        if seq in self.waiting:
            self.waiting.remove(seq)
            seq.status = SequenceStatus.FINISHED
            seq.finish_reason = FinishReason.ABORT
            seq.finish_time = now
        else:
            self._finish(seq, FinishReason.ABORT, now)

    def _finish(self, seq: Sequence, reason: FinishReason, now: float) -> None:
        seq.status = SequenceStatus.FINISHED
        seq.finish_reason = reason
        seq.finish_time = now
        self.running.remove(seq)
        # Free the slot right away: the next schedule() can hand it to a waiting request.
        self.cache.free(seq.slot)
        seq.slot = None
