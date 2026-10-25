"""Qwen2, written from scratch in PyTorch.

The model is a stack of identical decoder layers between a token embedding
and a final norm + LM head:

    ids -> embed -> [ RMSNorm -> Attention(GQA, RoPE) -> +residual
                      RMSNorm -> SwiGLU MLP           -> +residual ] x N
        -> RMSNorm -> LM head -> logits

Two choices here differ from the Hugging Face layout (the loader maps the
weights across):

* q, k and v projections are fused into one ``qkv_proj`` matmul, and the MLP's
  gate and up projections into one ``gate_up_proj``. During decode each step
  only processes a handful of tokens, so the GPU spends a lot of its time just
  launching kernels. One big matmul instead of three small ones means fewer
  launches and better use of the GPU.
* The model does not own a KV cache. Each attention layer hands its new q, k, v
  to an :class:`AttentionContext`, which stores k/v wherever it likes and runs
  attention against everything stored so far. That way the same model works
  with no cache (parity tests), a single-sequence cache (Stage 2), a slot cache
  for batches (Stage 3) and a paged cache (Stage 6).

All sizes come from the model's ``config.json`` via :class:`Qwen2Config`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class Qwen2Config:
    """The architecture numbers we need, read from a Hugging Face ``config.json``."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    attention_bias: bool

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Qwen2Config":
        if d.get("model_type") != "qwen2":
            raise ValueError(f"expected a qwen2 config, got model_type={d.get('model_type')!r}")
        if d.get("rope_scaling"):
            raise NotImplementedError("rope_scaling is not supported")
        if d.get("use_sliding_window"):
            raise NotImplementedError("sliding-window attention is not supported")
        num_heads = d["num_attention_heads"]
        return cls(
            vocab_size=d["vocab_size"],
            hidden_size=d["hidden_size"],
            intermediate_size=d["intermediate_size"],
            num_layers=d["num_hidden_layers"],
            num_heads=num_heads,
            num_kv_heads=d.get("num_key_value_heads", num_heads),
            head_dim=d.get("head_dim") or d["hidden_size"] // num_heads,
            rms_norm_eps=d["rms_norm_eps"],
            rope_theta=d.get("rope_theta", 10000.0),
            max_position_embeddings=d["max_position_embeddings"],
            tie_word_embeddings=d.get("tie_word_embeddings", False),
            # Qwen2 always puts a bias on q/k/v (and nowhere else). Its
            # config.json does not spell this out, so default to True.
            attention_bias=d.get("attention_bias", True),
        )

    @property
    def q_size(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_kv_heads * self.head_dim


class AttentionContext(Protocol):
    """Where attention layers send their keys and values.

    ``attend`` receives the new tokens' q, k, v for one layer, after RoPE:
    q is ``[B, num_heads, T, head_dim]`` and k, v are ``[B, num_kv_heads, T, head_dim]``.
    It stores k and v (or not), runs attention over all the keys this
    sequence can see, and returns ``[B, num_heads, T, head_dim]``.
    """

    def attend(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor: ...


def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Scaled dot-product attention with grouped-query attention (GQA).

    In GQA several query heads share one key/value head (Qwen2.5-0.5B: 14
    query heads, 2 kv heads, so groups of 7). That makes the KV cache 7x
    smaller. To run it with the standard SDPA kernel we repeat each kv head
    for every query head in its group. ``repeat_interleave`` keeps the group
    order right: kv head 0 serves query heads 0..6, kv head 1 serves 7..13.

    ``attn_mask`` is a boolean mask broadcastable to ``[B, heads, Tq, Tk]``,
    True where attention is allowed.
    """
    groups = q.shape[1] // k.shape[1]
    if groups > 1:
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)


class RMSNorm(nn.Module):
    """Root-mean-square norm: scale each vector to unit RMS, then by a learned weight.

    The statistics are computed in float32 even when the model runs in
    float16; summing 896 squares in fp16 loses too much precision.
    """

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x32.to(dtype)


class RotaryEmbedding(nn.Module):
    """Rotary position embedding (RoPE) with precomputed cos/sin tables.

    RoPE encodes a token's position by rotating pairs of q/k features by an
    angle ``position * freq_i``. Because the dot product of two rotated
    vectors only depends on the *difference* of their angles, attention
    scores end up depending on relative position.

    We precompute cos and sin for every position once, in float32. A forward
    pass then just looks up rows by position. The tables are plain tensors,
    not parameters or buffers, so ``model.to(float16)`` does not round them,
    and they are built lazily on whatever device the first call uses.
    """

    def __init__(self, head_dim: int, max_positions: int, theta: float) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_positions = max_positions
        self.theta = theta
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None

    def _build(self, device: torch.device) -> None:
        # One frequency per feature pair: theta^(-2i/d) for i in [0, d/2).
        exponents = torch.arange(0, self.head_dim, 2, dtype=torch.int64).float() / self.head_dim
        inv_freq = (1.0 / (self.theta**exponents)).to(device)
        positions = torch.arange(self.max_positions, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)  # [P, d/2]
        # Qwen2 (like Llama) rotates feature i with feature i + d/2, so the
        # angle table is the half-table repeated twice.
        angles = torch.cat([freqs, freqs], dim=-1)  # [P, d]
        self._cos, self._sin = angles.cos(), angles.sin()

    def forward(self, positions: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Look up cos/sin for ``positions`` ([B, T]); returns two ``[B, 1, T, head_dim]`` tensors.

        Positions are given per token, not assumed to be ``0..T-1``. In a
        batched decode step every sequence sits at a different position, and
        this is what lets each one get its own rotation.
        """
        if self._cos is None or self._cos.device != positions.device:
            self._build(positions.device)
        cos = self._cos[positions].unsqueeze(1).to(dtype)
        sin = self._sin[positions].unsqueeze(1).to(dtype)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class Attention(nn.Module):
    def __init__(self, cfg: Qwen2Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.q_size, self.kv_size = cfg.q_size, cfg.kv_size
        self.qkv_proj = nn.Linear(cfg.hidden_size, cfg.q_size + 2 * cfg.kv_size, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(cfg.q_size, cfg.hidden_size, bias=False)

    def project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.qkv_proj(x).split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, ctx: AttentionContext
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self.project_qkv(x)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        out = ctx.attend(self.layer_idx, q, k, v)  # [B, heads, T, head_dim]
        return self.o_proj(out.transpose(1, 2).reshape(B, T, self.q_size))


class MLP(nn.Module):
    """SwiGLU feed-forward: down(silu(gate(x)) * up(x))."""

    def __init__(self, cfg: Qwen2Config) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen2Config, layer_idx: int) -> None:
        super().__init__()
        self.input_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = Attention(cfg, layer_idx)
        self.post_attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, ctx: AttentionContext
    ) -> torch.Tensor:
        # Pre-norm residual blocks: normalise the input to each block, add
        # the block's output back onto the unnormalised stream.
        x = x + self.attn(self.input_norm(x), cos, sin, ctx)
        return x + self.mlp(self.post_attn_norm(x))


class Qwen2ForCausalLM(nn.Module):
    """The full Qwen2 decoder.

    ``forward`` returns hidden states; ``compute_logits`` turns (a subset of)
    them into vocabulary logits. They are split because the LM head is big
    (vocab 151,936): during prefill we only need logits for the last prompt
    token, not for all of them.
    """

    def __init__(self, cfg: Qwen2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg, i) for i in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta)
        # With tied embeddings the LM head reuses the embedding matrix, so
        # there is no separate weight to store or load.
        self.lm_head = None if cfg.tie_word_embeddings else nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, ctx: AttentionContext) -> torch.Tensor:
        """Run the decoder.

        Args:
            input_ids: ``[B, T]`` token ids.
            positions: ``[B, T]`` absolute position of each token in its own sequence.
            ctx: Handles KV storage and attention (see :class:`AttentionContext`).

        Returns:
            Final hidden states, ``[B, T, hidden_size]``.
        """
        x = self.embed_tokens(input_ids)
        cos, sin = self.rope(positions, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin, ctx)
        return self.norm(x)

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        weight = self.embed_tokens.weight if self.lm_head is None else self.lm_head.weight
        return F.linear(hidden, weight)
