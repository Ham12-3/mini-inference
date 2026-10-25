"""Load Hugging Face Qwen2 safetensors weights into :class:`engine.model.Qwen2ForCausalLM`.

The checkpoint and our model name things differently, and our model fuses
some matrices. ``WEIGHT_MAP`` below spells out every mapping, and
``load_weights`` checks that every parameter of our model was filled
completely and that no checkpoint tensor was left unused.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

from engine.model import Qwen2Config, Qwen2ForCausalLM

# (checkpoint name pattern, our parameter name, shard)
#
# ``shard`` is None for a one-to-one copy. Otherwise it names which slice of
# a fused parameter this tensor fills: our ``qkv_proj`` is q, k and v stacked
# along the output dimension, and ``gate_up_proj`` is gate then up.
WEIGHT_MAP: list[tuple[str, str, str | None]] = [
    (r"model\.embed_tokens\.weight", "embed_tokens.weight", None),
    (r"model\.norm\.weight", "norm.weight", None),
    (r"lm_head\.weight", "lm_head.weight", None),
    (r"model\.layers\.(\d+)\.input_layernorm\.weight", r"layers.\1.input_norm.weight", None),
    (r"model\.layers\.(\d+)\.post_attention_layernorm\.weight", r"layers.\1.post_attn_norm.weight", None),
    (r"model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)", r"layers.\1.attn.qkv_proj.\2", "q"),
    (r"model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)", r"layers.\1.attn.qkv_proj.\2", "k"),
    (r"model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)", r"layers.\1.attn.qkv_proj.\2", "v"),
    (r"model\.layers\.(\d+)\.self_attn\.o_proj\.weight", r"layers.\1.attn.o_proj.weight", None),
    (r"model\.layers\.(\d+)\.mlp\.gate_proj\.weight", r"layers.\1.mlp.gate_up_proj.weight", "gate"),
    (r"model\.layers\.(\d+)\.mlp\.up_proj\.weight", r"layers.\1.mlp.gate_up_proj.weight", "up"),
    (r"model\.layers\.(\d+)\.mlp\.down_proj\.weight", r"layers.\1.mlp.down_proj.weight", None),
]

# Files needed from the Hub: weights, configs and tokenizer.
MODEL_FILE_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model"]


def resolve_model_path(model: str) -> Path:
    """Return a local directory for ``model``: as given if it is a directory, else download from the Hub (cached)."""
    if Path(model).is_dir():
        return Path(model)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, allow_patterns=MODEL_FILE_PATTERNS))


def load_config(model_dir: Path) -> Qwen2Config:
    return Qwen2Config.from_dict(json.loads((model_dir / "config.json").read_text()))


def load_eos_token_ids(model_dir: Path) -> list[int]:
    """Token ids that end generation.

    ``generation_config.json`` is preferred: for Qwen2.5-Instruct it lists both
    ``<|im_end|>`` and ``<|endoftext|>``, while ``config.json`` lists only one.
    """
    for name in ("generation_config.json", "config.json"):
        path = model_dir / name
        if path.exists():
            eos = json.loads(path.read_text()).get("eos_token_id")
            if eos is not None:
                return [eos] if isinstance(eos, int) else list(eos)
    return []


def _shard_slices(cfg: Qwen2Config) -> dict[str, slice]:
    q, kv, inter = cfg.q_size, cfg.kv_size, cfg.intermediate_size
    return {
        "q": slice(0, q),
        "k": slice(q, q + kv),
        "v": slice(q + kv, q + 2 * kv),
        "gate": slice(0, inter),
        "up": slice(inter, 2 * inter),
    }


def map_name(hf_name: str) -> tuple[str, str | None] | None:
    """Map a checkpoint tensor name to ``(our parameter name, shard)``, or None if unknown."""
    for pattern, target, shard in WEIGHT_MAP:
        m = re.fullmatch(pattern, hf_name)
        if m:
            return m.expand(target), shard
    return None


@torch.no_grad()
def load_weights(model: Qwen2ForCausalLM, model_dir: Path) -> None:
    """Copy every tensor from the checkpoint into ``model``, casting to the model's dtype and device."""
    cfg = model.cfg
    params = dict(model.named_parameters())
    slices = _shard_slices(cfg)
    expected = {
        name: ({"q", "k", "v"} if "qkv_proj" in name else {"gate", "up"} if "gate_up_proj" in name else {None})
        for name in params
    }
    filled: dict[str, set[str | None]] = {name: set() for name in params}

    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {model_dir}")
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as f:
            for hf_name in f.keys():
                mapped = map_name(hf_name)
                if mapped is None:
                    raise KeyError(f"checkpoint tensor {hf_name!r} has no mapping")
                name, shard = mapped
                if name == "lm_head.weight" and cfg.tie_word_embeddings:
                    continue  # tied: the embedding matrix already is the LM head
                dest = params[name] if shard is None else params[name][slices[shard]]
                src = f.get_tensor(hf_name)
                if src.shape != dest.shape:
                    raise ValueError(f"{hf_name}: shape {tuple(src.shape)} does not fit {name} {tuple(dest.shape)}")
                dest.copy_(src)  # casts dtype (e.g. bf16 -> fp16) and moves to the model's device
                filled[name].add(shard)

    missing = [n for n in params if filled[n] != expected[n]]
    if missing:
        raise ValueError(f"parameters not (fully) loaded from the checkpoint: {missing[:5]}{' ...' if len(missing) > 5 else ''}")


def load_model(model: str, dtype: torch.dtype = torch.float16, device: str = "cuda") -> Qwen2ForCausalLM:
    """Build our Qwen2 from a Hub id or local directory and load its weights."""
    model_dir = resolve_model_path(model)
    cfg = load_config(model_dir)
    # Build on the "meta" device first: this creates parameter shapes without
    # allocating or randomly initialising 500M numbers we would overwrite
    # anyway. ``to_empty`` then allocates real (uninitialised) memory, and
    # load_weights fills every parameter, checking that none was missed.
    with torch.device("meta"):
        net = Qwen2ForCausalLM(cfg)
    net = net.to(dtype=dtype).to_empty(device=device)
    load_weights(net, model_dir)
    return net.eval()
