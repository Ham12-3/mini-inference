"""My Qwen2 must compute the same thing as the Hugging Face implementation.

* float32: logits must match to within 1e-3 (max absolute difference), both
  for a full forward pass and for each step of prefill + cached decode.
* float16 (GPU): greedy generation must match HF ``generate()`` token for token.
"""

import pytest
import torch
from conftest import DEVICE, requires_gpu

from engine.engine import generate_greedy
from engine.kv_cache import NoCache, SingleSequenceKVCache
from engine.loader import load_eos_token_ids, load_model, map_name

FP32_TOL = 1e-3
NUM_TOKENS = 50


def hf_greedy(hf, ids, max_new_tokens, eos, min_new_tokens=0):
    """Plain greedy decoding with HF ``generate()``.

    Qwen2.5-Instruct ships a generation_config.json with
    ``repetition_penalty: 1.1``, and HF applies it even when
    ``do_sample=False``. That is not greedy decoding: it made HF pick a
    different token from its own argmax whenever the best token had already
    appeared in the text. Passing a fresh GenerationConfig to ``generate()``
    is not enough (transformers merges it with the model's saved one), so we
    replace the model's generation config itself.
    """
    from transformers import GenerationConfig

    hf.generation_config = GenerationConfig(do_sample=False, eos_token_id=eos, pad_token_id=eos[0])
    out = hf.generate(ids, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens)
    return out[0, ids.shape[1] :].tolist()


def load_hf(model_dir, dtype):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, attn_implementation="sdpa").to(DEVICE).eval()


@pytest.fixture(scope="module")
def fp32_models(model_dir):
    return load_hf(model_dir, torch.float32), load_model(str(model_dir), torch.float32, DEVICE)


def test_weight_map_covers_hf_names():
    assert map_name("model.layers.3.self_attn.k_proj.bias") == ("layers.3.attn.qkv_proj.bias", "k")
    assert map_name("model.layers.0.mlp.up_proj.weight") == ("layers.0.mlp.gate_up_proj.weight", "up")
    assert map_name("model.layers.0.self_attn.rotary_emb.inv_freq") is None


@torch.inference_mode()
def test_fp32_full_forward_logits_match(fp32_models, chat_prompt_ids):
    hf, mine = fp32_models
    for prompt in chat_prompt_ids[:3]:
        ids = torch.tensor([prompt], device=DEVICE)
        positions = torch.arange(ids.shape[1], device=DEVICE).unsqueeze(0)
        ref = hf(input_ids=ids).logits
        got = mine.compute_logits(mine(ids, positions, NoCache()))
        diff = (ref - got).abs().max().item()
        assert diff < FP32_TOL, f"max |logit diff| = {diff:.2e}"


@torch.inference_mode()
def test_fp32_cached_decode_logits_match(fp32_models, chat_prompt_ids, model_dir):
    """Prefill + 20 cached decode steps must give the same logits as HF's full forward over the same tokens."""
    hf, mine = fp32_models
    prompt = chat_prompt_ids[0]
    cache = SingleSequenceKVCache.for_model(mine.cfg, max_len=512, dtype=torch.float32, device=DEVICE)

    # Get 20 continuation tokens to feed (from HF's own greedy run).
    seq = torch.tensor([prompt], device=DEVICE)
    full = torch.tensor([prompt + hf_greedy(hf, seq, 20, load_eos_token_ids(model_dir), min_new_tokens=20)], device=DEVICE)
    ref_logits = hf(input_ids=full).logits[0]  # [len, vocab], full forward, no cache

    hidden = mine(seq, torch.arange(len(prompt), device=DEVICE).unsqueeze(0), cache)
    cache.advance(len(prompt))
    got = [mine.compute_logits(hidden[0, -1])]
    for pos in range(len(prompt), full.shape[1] - 1):
        hidden = mine(full[:, pos : pos + 1], torch.tensor([[pos]], device=DEVICE), cache)
        cache.advance(1)
        got.append(mine.compute_logits(hidden[0, -1]))

    ref = ref_logits[len(prompt) - 1 : full.shape[1] - 1]
    diff = (ref - torch.stack(got)).abs().max().item()
    assert diff < FP32_TOL, f"max |logit diff| over cached decode = {diff:.2e}"


@requires_gpu
@torch.inference_mode()
def test_fp16_greedy_matches_hf_generate(model_dir, chat_prompt_ids):
    hf = load_hf(model_dir, torch.float16)
    mine = load_model(str(model_dir), torch.float16, "cuda")
    eos = load_eos_token_ids(model_dir)
    cache = SingleSequenceKVCache.for_model(mine.cfg, max_len=1024, dtype=torch.float16, device="cuda")

    for i, prompt in enumerate(chat_prompt_ids):
        ids = torch.tensor([prompt], device="cuda")
        ref = hf_greedy(hf, ids, NUM_TOKENS, eos)
        assert len(ref) == NUM_TOKENS, f"prompt {i} hit EOS after {len(ref)} tokens; pick a longer-answer prompt"
        got = generate_greedy(mine, prompt, NUM_TOKENS, cache, eos_token_ids=eos)
        assert got == ref, f"prompt {i}: first mismatch at token {next(j for j, (a, b) in enumerate(zip(got + [-1], ref + [-1])) if a != b)}"
