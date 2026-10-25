"""Requests as the engine sees them: sampling settings, state and timing."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


@dataclass
class SamplingParams:
    """How to pick tokens for one request.

    Attributes:
        temperature: 0 means greedy (always the most likely token). Higher
            values flatten the distribution and make output more varied.
        top_k: Only sample from the k most likely tokens. 0 turns it off.
        top_p: Only sample from the smallest set of tokens whose probabilities
            add up to at least top_p ("nucleus sampling"). 1.0 turns it off.
        max_tokens: Stop after this many generated tokens.
        ignore_eos: Keep going past the end-of-sequence token (benchmarks use
            this to force an exact output length).
    """

    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    max_tokens: int = 128
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0 (0 disables it)")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0


class SequenceStatus(enum.Enum):
    WAITING = "waiting"  # in the queue, no cache slot yet
    RUNNING = "running"  # owns a cache slot, decoding every step
    FINISHED = "finished"  # done, slot handed back


class FinishReason(str, enum.Enum):
    EOS = "eos"  # produced an end-of-sequence token
    LENGTH = "length"  # reached its max_tokens
    MAX_MODEL_LEN = "max_model_len"  # prompt + output filled the cache slot
    ABORT = "abort"  # cancelled by the caller (e.g. the client disconnected)


@dataclass
class Sequence:
    """One request moving through the engine.

    Timestamps are ``time.perf_counter()`` values: arrival (added to the
    queue), first token (end of prefill) and finish. Latency metrics come
    from these: TTFT = first_token - arrival, TPOT = (finish - first_token)
    / (output tokens - 1).
    """

    seq_id: int
    prompt_token_ids: list[int]
    sampling: SamplingParams
    arrival_time: float
    output_token_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    slot: int | None = None
    finish_reason: FinishReason | None = None
    first_token_time: float | None = None
    finish_time: float | None = None

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def last_token(self) -> int:
        return self.output_token_ids[-1]

    @property
    def is_finished(self) -> bool:
        return self.status is SequenceStatus.FINISHED
