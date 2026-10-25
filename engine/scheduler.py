"""The scheduler decides, every step, who prefills, who decodes, and who is done.

It keeps two collections:

* ``waiting``: a FIFO queue of requests that have arrived but have no cache slot yet.
* ``running``: requests that own a slot and produce one token per step.

Each step it admits requests from the front of the queue while three limits
allow it: a free cache slot, ``max_num_seqs`` running sequences, and the
step's token budget ``max_num_batched_tokens`` (every decode costs 1 token,
a prefill costs its prompt length). When a request finishes, its slot is
freed immediately, so the next step can admit someone new into it. That
join-and-leave-every-step behaviour is continuous batching.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

from engine.config import EngineConfig
from engine.kv_cache import SlotKVCache
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
    def __init__(self, config: EngineConfig, cache: SlotKVCache, eos_token_ids: Iterable[int] = ()) -> None:
        self.config = config
        self.cache = cache
        self.eos_token_ids = set(eos_token_ids)
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []

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
        out = SchedulerOutput(decodes=list(self.running))
        static = self.config.scheduling == "static"
        if static and self.running:
            return out  # static batching: nobody joins until the whole batch is done
        budget = self.config.max_num_batched_tokens - len(out.decodes)
        while self.waiting and self.cache.num_free_slots > 0 and len(self.running) < self.config.max_num_seqs:
            seq = self.waiting[0]
            # The token budget protects running decodes from a long burst of
            # prefills. A static batch has no running decodes, so it fills
            # every slot at once instead.
            if seq.prompt_len > budget and not static:
                # Strict FIFO: do not let a later, shorter prompt jump the
                # queue. The budget refills next step; with nothing running
                # it is at least max_model_len, so every valid prompt fits.
                break
            self.waiting.popleft()
            seq.slot = self.cache.allocate()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            out.prefills.append(seq)
            budget -= seq.prompt_len
        return out

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

    def _finish(self, seq: Sequence, reason: FinishReason, now: float) -> None:
        seq.status = SequenceStatus.FINISHED
        seq.finish_reason = reason
        seq.finish_time = now
        self.running.remove(seq)
        # Free the slot right away: the next schedule() can hand it to a waiting request.
        self.cache.free(seq.slot)
        seq.slot = None
