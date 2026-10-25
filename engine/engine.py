"""Generation loops.

* :func:`generate_greedy`: one sequence with a KV cache (Stage 2).
* :class:`ModelRunner`: runs one prefill or one batched decode step against a
  slot cache. Everything that serves many requests is built on it.
* :func:`generate_static`: static batching (Stage 3). All requests start
  together and the batch runs until the last one finishes.

* :class:`LLMEngine`: request queue + scheduler + continuous batching (Stage 4).
"""

from __future__ import annotations

import time
from collections.abc import Sequence as SequenceOf
from dataclasses import dataclass

import torch

from engine.config import EngineConfig
from engine.cuda_graphs import GraphedDecoder
from engine.kv_cache import PagedKVCache, SingleSequenceKVCache, SlotKVCache
from engine.loader import load_eos_token_ids, load_model, resolve_model_path
from engine.model import Qwen2ForCausalLM
from engine.sampler import Sampler
from engine.scheduler import Scheduler
from engine.sequence import SamplingParams, Sequence


@torch.inference_mode()
def generate_greedy(
    model: Qwen2ForCausalLM,
    prompt_ids: SequenceOf[int],
    max_new_tokens: int,
    cache: SingleSequenceKVCache,
    eos_token_ids: SequenceOf[int] = (),
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

    def __init__(self, model: Qwen2ForCausalLM, cache: SlotKVCache | PagedKVCache) -> None:
        self.model = model
        self.cache = cache
        self.device = cache.k.device
        self.graphs: GraphedDecoder | None = None  # set by enable_cuda_graphs()

    def enable_cuda_graphs(self, max_batch: int) -> None:
        """Record the decode step as CUDA graphs; later decodes replay them when a bucket fits."""
        self.graphs = GraphedDecoder(self.model, self.cache, max_batch)
        self.graphs.capture()

    @torch.inference_mode()
    def prefill(self, slot: int, prompt_ids: SequenceOf[int]) -> torch.Tensor:
        """Process a whole prompt into ``slot``; return the logits for the next token, ``[vocab]``."""
        ctx = self.cache.prefill_context(slot)
        ids = torch.tensor([list(prompt_ids)], device=self.device)
        positions = torch.arange(ctx.start, ctx.start + ids.shape[1], device=self.device).unsqueeze(0)
        hidden = self.model(ids, positions, ctx)
        ctx.commit(ids.shape[1])
        return self.model.compute_logits(hidden[0, -1]).float()

    @torch.inference_mode()
    def prefill_packed(self, slots: SequenceOf[int], prompts: SequenceOf[SequenceOf[int]]) -> torch.Tensor:
        """Process several fresh prompts in one forward pass; returns next-token logits ``[B, vocab]``."""
        ctx = self.cache.packed_prefill_context(slots, [len(p) for p in prompts])
        ids = torch.tensor([[t for p in prompts for t in p]], device=self.device)  # [1, total tokens]
        hidden = self.model(ids, ctx.positions.unsqueeze(0), ctx)
        ctx.commit()
        return self.model.compute_logits(hidden[0, ctx.last_index]).float()

    @torch.inference_mode()
    def decode(self, slots: SequenceOf[int], last_tokens: SequenceOf[int]) -> torch.Tensor:
        """One decode step for many sequences in ONE forward pass; returns logits ``[B, vocab]``.

        ``last_tokens[b]`` is the most recent token of the sequence in
        ``slots[b]``. Each one goes in at its own sequence's position.
        """
        if self.graphs is not None:
            logits = self.graphs.decode(slots, last_tokens)
            if logits is not None:
                return logits
        ctx = self.cache.decode_context(slots)
        ids = torch.tensor(list(last_tokens), device=self.device).unsqueeze(1)  # [B, 1]
        hidden = self.model(ids, ctx.positions.unsqueeze(1), ctx)
        ctx.commit()
        return self.model.compute_logits(hidden[:, -1]).float()


def pack_by_budget(lengths: SequenceOf[int], budget: int) -> list[list[int]]:
    """Split items (by index, order kept) into groups whose total length fits ``budget``.

    ``budget <= 0`` puts every item in its own group: one-at-a-time prefill.
    An item longer than the budget still gets a group of its own.
    """
    groups: list[list[int]] = []
    total = 0
    for i, n in enumerate(lengths):
        if groups and budget > 0 and total + n <= budget:
            groups[-1].append(i)
            total += n
        else:
            groups.append([i])
            total = n
    return groups


def generate_static(
    runner: ModelRunner,
    prompts: SequenceOf[SequenceOf[int]],
    max_new_tokens: int,
    eos_token_ids: SequenceOf[int] = (),
    prefill_budget: int = 0,
) -> list[list[int]]:
    """Greedy static batching: prefill every prompt, then decode them together until all are done.

    Prompts are prefilled one at a time (each into its own slot), or packed
    into forward passes of up to ``prefill_budget`` tokens. Decoding is
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
        outputs: list[list[int]] = [[] for _ in prompts]
        for group in pack_by_budget([len(p) for p in prompts], prefill_budget):
            if len(group) == 1:
                logits = runner.prefill(slots[group[0]], prompts[group[0]]).unsqueeze(0)
            else:
                logits = runner.prefill_packed([slots[i] for i in group], [prompts[i] for i in group])
            for i, token in zip(group, logits.argmax(dim=-1).tolist()):
                outputs[i].append(token)

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


@dataclass
class RequestOutput:
    """A finished (or in-progress) request as returned to the caller."""

    request_id: int
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    text: str | None
    finish_reason: str | None
    arrival_time: float
    first_token_time: float | None
    finish_time: float | None


@dataclass
class StepStats:
    """What happened in one engine step (used for the debug log and the batch-size timeline)."""

    step: int
    time: float  # perf_counter at the end of the step
    num_prefills: int
    num_decodes: int
    num_finished: int
    num_running: int  # after the step
    num_waiting: int  # after the step


class LLMEngine:
    """Serves many requests at once with continuous batching.

    Every call to :meth:`step` does one scheduler decision, prefills the newly
    admitted requests (packed into as few forward passes as the token budget
    allows), then runs ONE batched decode for
    everything already running. Requests finish and free their slot inside a
    step, and new ones are admitted at the start of the next: the batch
    changes shape every step instead of waiting for its slowest member.
    """

    def __init__(self, config: EngineConfig) -> None:
        from transformers import AutoTokenizer

        self.config = config
        model_dir = resolve_model_path(config.model)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = load_model(str(model_dir), config.torch_dtype, config.device)
        self.eos_token_ids = load_eos_token_ids(model_dir)
        if config.kv_cache == "paged":
            self.cache = PagedKVCache.for_model(
                self.model.cfg, num_blocks=config.kv_cache_tokens // config.block_size, block_size=config.block_size,
                max_seqs=config.max_num_seqs, max_len=config.max_model_len, dtype=config.torch_dtype,
                device=config.device, enable_prefix_caching=config.enable_prefix_caching,
            )
        else:
            self.cache = SlotKVCache.for_model(
                self.model.cfg, config.max_num_seqs, config.max_model_len, config.torch_dtype, config.device
            )
        self.runner = ModelRunner(self.model, self.cache)
        if config.cuda_graphs:
            self.runner.enable_cuda_graphs(min(config.max_num_seqs, config.cuda_graph_max_batch))
        self.scheduler = Scheduler(config, self.cache, self.eos_token_ids)
        self.sampler = Sampler(config.seed, config.device)
        self.sequences: dict[int, Sequence] = {}
        self.step_stats: list[StepStats] = []
        self._next_id = 0

    def add_request(
        self,
        prompt: str | SequenceOf[int],
        sampling_params: SamplingParams | None = None,
        arrival_time: float | None = None,
    ) -> int:
        """Queue a request; returns its id. ``prompt`` is raw text or token ids."""
        ids = self.tokenizer(prompt).input_ids if isinstance(prompt, str) else list(prompt)
        seq = Sequence(
            seq_id=self._next_id,
            prompt_token_ids=ids,
            sampling=sampling_params or SamplingParams(),
            arrival_time=time.perf_counter() if arrival_time is None else arrival_time,
        )
        self.scheduler.add(seq)  # raises if the prompt can never fit
        self.sequences[seq.seq_id] = seq
        self._next_id += 1
        return seq.seq_id

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_work()

    def step(self) -> list[tuple[int, int]]:
        """Run one engine step; returns the ``(request_id, new_token)`` pairs it produced."""
        batch = self.scheduler.schedule()
        new_tokens: list[tuple[int, int]] = []

        # Prefill the newly admitted requests, packing several prompts into
        # one forward pass where the token budget allows (or one at a time
        # if batched prefill is switched off).
        # A request that was preempted earlier recomputes its prompt plus the
        # tokens it had already generated (token_ids), then continues. Tokens
        # already in the cache (a prefix-cache hit) are skipped.
        budget = self.config.max_num_batched_tokens if self.config.batched_prefill else 0
        new_tokens_of = {s.seq_id: s.token_ids[self.cache.lengths[s.slot]:] for s in batch.prefills}
        for group in pack_by_budget([len(new_tokens_of[s.seq_id]) for s in batch.prefills], budget):
            seqs = [batch.prefills[i] for i in group]
            logits = self.runner.prefill_packed([s.slot for s in seqs], [new_tokens_of[s.seq_id] for s in seqs])
            for s in seqs:
                self.cache.register_prefix(s.slot, s.token_ids)
            tokens = self.sampler(logits, [s.sampling for s in seqs])
            now = time.perf_counter()
            for seq, token in zip(seqs, tokens):
                self.scheduler.append_token(seq, token, now)
                new_tokens.append((seq.seq_id, token))

        if batch.decodes:
            logits = self.runner.decode([s.slot for s in batch.decodes], [s.last_token for s in batch.decodes])
            tokens = self.sampler(logits, [s.sampling for s in batch.decodes])
            now = time.perf_counter()
            for seq, token in zip(batch.decodes, tokens):
                self.scheduler.append_token(seq, token, now)
                new_tokens.append((seq.seq_id, token))

        stats = StepStats(
            step=len(self.step_stats),
            time=time.perf_counter(),
            num_prefills=len(batch.prefills),
            num_decodes=len(batch.decodes),
            num_finished=sum(s.is_finished for s in batch.prefills + batch.decodes),
            num_running=len(self.scheduler.running),
            num_waiting=len(self.scheduler.waiting),
        )
        self.step_stats.append(stats)
        if self.config.debug:
            print(
                f"step {stats.step:5d} | decode batch {stats.num_decodes:3d} | +{stats.num_prefills} joined "
                f"| -{stats.num_finished} finished | running {stats.num_running:3d} | waiting {stats.num_waiting:4d}"
            )
        return new_tokens

    def abort_request(self, request_id: int) -> None:
        """Cancel a request; a running one frees its cache slot immediately."""
        self.scheduler.abort(self.sequences[request_id], time.perf_counter())

    def release(self, request_id: int) -> None:
        """Forget a finished request (a long-running server calls this to keep memory flat)."""
        if not self.sequences[request_id].is_finished:
            raise ValueError("only finished requests can be released")
        del self.sequences[request_id]

    def get_output(self, request_id: int) -> RequestOutput:
        seq = self.sequences[request_id]
        return RequestOutput(
            request_id=seq.seq_id,
            prompt_token_ids=seq.prompt_token_ids,
            output_token_ids=list(seq.output_token_ids),
            text=self.tokenizer.decode(seq.output_token_ids, skip_special_tokens=True),
            finish_reason=seq.finish_reason.value if seq.finish_reason else None,
            arrival_time=seq.arrival_time,
            first_token_time=seq.first_token_time,
            finish_time=seq.finish_time,
        )

    def generate(
        self,
        prompts: SequenceOf[str | SequenceOf[int]],
        sampling_params: SamplingParams | SequenceOf[SamplingParams] | None = None,
    ) -> list[RequestOutput]:
        """Convenience loop: queue every prompt, step until all are done, return outputs in order."""
        if sampling_params is None or isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params or SamplingParams()] * len(prompts)
        ids = [self.add_request(p, sp) for p, sp in zip(prompts, sampling_params)]
        while self.has_unfinished_requests():
            self.step()
        return [self.get_output(i) for i in ids]
