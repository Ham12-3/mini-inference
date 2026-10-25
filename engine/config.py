"""Engine configuration.

Everything that controls *how* the engine runs lives here (batch limits,
cache size, dtype, device). Everything that describes *the model itself*
(hidden size, number of layers, heads, ...) is read from the model's own
``config.json`` at load time and is never hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass
class EngineConfig:
    """Settings for one engine instance.

    Attributes:
        model: Hugging Face model id or a local directory with the weights.
        dtype: "float16" (default; the T4 has no fast bfloat16), "bfloat16" or "float32".
        device: Torch device string, e.g. "cuda" or "cpu".
        max_num_seqs: Most sequences that can be running (and own a cache slot) at once.
        max_model_len: Most tokens (prompt + output) any one sequence may hold.
            This also fixes the length of each cache slot.
        max_num_batched_tokens: Token budget for one engine step. It limits how
            many prompt tokens we prefill in a single step so a burst of long
            prompts cannot stall the running decodes for too long.
        scheduling: "continuous" (default): requests join and leave the running
            batch every step. "static": a new batch is admitted only when the
            previous one has completely finished (the baseline continuous
            batching is compared against).
        kv_cache: "slot" (default): one fixed max_model_len slot per sequence.
            "paged": a pool of block_size-token blocks handed out as
            sequences grow, so the same memory holds more sequences.
        block_size: Tokens per block in the paged cache.
        enable_prefix_caching: Paged cache only. Reuse cached blocks when a new
            prompt starts with tokens already processed (e.g. a shared system
            prompt), so only the rest is prefilled.
        kv_cache_tokens: Total KV cache capacity in tokens. Defaults to
            max_num_seqs * max_model_len (what the slot cache needs). For the
            paged cache this sets the block pool size, so a paged engine with
            a larger max_num_seqs can be compared at equal memory.
        cuda_graphs: Record the decode step as CUDA graphs (one per batch-size
            and KV-length bucket) and replay them, removing most per-step
            CPU launch overhead. CUDA only.
        batched_prefill: Pack several new prompts into one prefill forward pass
            (up to max_num_batched_tokens tokens). False prefills one prompt
            per forward pass, the simpler Stage 3/4 behaviour.
        seed: Seed for the sampler's random generator, so sampled runs are reproducible.
        debug: Print the batch composition on every step.
    """

    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: str = "float16"
    device: str = "cuda"
    max_num_seqs: int = 32
    max_model_len: int = 2048
    max_num_batched_tokens: int = 4096
    scheduling: str = "continuous"
    kv_cache: str = "slot"
    block_size: int = 16
    enable_prefix_caching: bool = False
    kv_cache_tokens: int | None = None
    batched_prefill: bool = True
    cuda_graphs: bool = False
    seed: int = 0
    debug: bool = False

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {self.dtype!r}")
        if self.max_num_seqs < 1:
            raise ValueError("max_num_seqs must be at least 1")
        if self.max_model_len < 2:
            raise ValueError("max_model_len must be at least 2")
        if self.scheduling not in ("continuous", "static"):
            raise ValueError(f"scheduling must be 'continuous' or 'static', got {self.scheduling!r}")
        if self.kv_cache not in ("slot", "paged"):
            raise ValueError(f"kv_cache must be 'slot' or 'paged', got {self.kv_cache!r}")
        if self.cuda_graphs and not self.device.startswith("cuda"):
            raise ValueError("cuda_graphs needs a CUDA device")
        if self.enable_prefix_caching and self.kv_cache != "paged":
            raise ValueError("prefix caching needs kv_cache='paged'")
        if self.kv_cache_tokens is None:
            self.kv_cache_tokens = self.max_num_seqs * self.max_model_len
        if self.kv_cache_tokens < self.max_model_len:
            # Otherwise one long sequence could never fit, even alone.
            raise ValueError("kv_cache_tokens must be >= max_model_len")
        if self.max_num_batched_tokens < self.max_model_len:
            # Otherwise a prompt of max length could never be admitted.
            raise ValueError("max_num_batched_tokens must be >= max_model_len")

    @property
    def torch_dtype(self) -> torch.dtype:
        return _DTYPES[self.dtype]
