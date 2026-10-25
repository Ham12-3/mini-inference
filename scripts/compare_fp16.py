"""Debug tool: find where my fp16 model and HF's fp16 model first disagree in greedy decoding.

Kept because it found a real issue: HF ``generate()`` applies the model's
``repetition_penalty: 1.1`` from generation_config.json even with
``do_sample=False``, so it is not plain greedy unless given a fresh
GenerationConfig. ``--model-generation-config`` reproduces that.

For each parity prompt it reports the first differing token and, at that
step, both models' top-2 tokens and the gap between them. A tiny gap means a
near-tie that any rounding difference can flip; a large gap means a real bug.

    python scripts/compare_fp16.py [--unfused]

``--unfused`` makes my model run q/k/v and gate/up as separate matmuls (same
weights, just not fused) to test whether fusion is what changes the rounding.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from conftest import MODEL_ID, PROMPTS  # noqa: E402

from engine.engine import generate_greedy  # noqa: E402
from engine.kv_cache import NoCache, SingleSequenceKVCache  # noqa: E402
from engine.loader import load_eos_token_ids, load_model, resolve_model_path  # noqa: E402
from engine.model import MLP, Attention  # noqa: E402


def unfuse() -> None:
    def project_qkv(self, x):
        w, b = self.qkv_proj.weight, self.qkv_proj.bias
        q, kv = self.q_size, self.kv_size
        return (
            F.linear(x, w[:q], b[:q]),
            F.linear(x, w[q : q + kv], b[q : q + kv]),
            F.linear(x, w[q + kv :], b[q + kv :]),
        )

    def mlp_forward(self, x):
        w = self.gate_up_proj.weight
        n = w.shape[0] // 2
        return self.down_proj(F.silu(F.linear(x, w[:n])) * F.linear(x, w[n:]))

    Attention.project_qkv = project_qkv
    MLP.forward = mlp_forward


def top2(logits: torch.Tensor) -> str:
    v, i = logits.float().topk(2)
    return f"top1={i[0].item()} ({v[0]:.3f})  top2={i[1].item()} ({v[1]:.3f})  gap={v[0] - v[1]:.4f}"


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unfused", action="store_true")
    parser.add_argument("--tokens", type=int, default=50)
    parser.add_argument(
        "--model-generation-config",
        action="store_true",
        help="let HF generate() use the model's generation_config.json (repetition_penalty=1.1), "
        "which reproduces the original mismatch",
    )
    args = parser.parse_args()
    if args.unfused:
        unfuse()

    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    model_dir = resolve_model_path(MODEL_ID)
    tok = AutoTokenizer.from_pretrained(model_dir)
    hf = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
    mine = load_model(str(model_dir), torch.float16, "cuda")
    eos = load_eos_token_ids(model_dir)
    cache = SingleSequenceKVCache.for_model(mine.cfg, 1024, torch.float16, "cuda")

    for n, text in enumerate(PROMPTS):
        chat = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
        prompt = tok(chat).input_ids
        ids = torch.tensor([prompt], device="cuda")
        gen_cfg = GenerationConfig(do_sample=False, max_new_tokens=args.tokens, eos_token_id=eos, pad_token_id=eos[0])
        if args.model_generation_config:
            ref = hf.generate(ids, max_new_tokens=args.tokens, do_sample=False, pad_token_id=eos[0])
        else:
            ref = hf.generate(ids, generation_config=gen_cfg)
        ref = ref[0, len(prompt) :].tolist()
        got = generate_greedy(mine, prompt, args.tokens, cache, eos_token_ids=eos)
        if got == ref:
            print(f"prompt {n}: identical ({len(ref)} tokens)")
            continue
        i = next(j for j, (a, b) in enumerate(zip(got + [-1], ref + [-1])) if a != b)
        # Re-run both models on the shared prefix (no cache) and compare the
        # logits that chose token i.
        prefix = torch.tensor([prompt + ref[:i]], device="cuda")
        hf_logits = hf(input_ids=prefix).logits[0, -1]
        my_logits = mine.compute_logits(mine(prefix, torch.arange(prefix.shape[1], device="cuda")[None], NoCache())[0, -1])
        print(f"prompt {n}: first mismatch at token {i}: hf={ref[i] if i < len(ref) else 'END'} mine={got[i] if i < len(got) else 'END'}")
        print(f"   hf   {top2(hf_logits)}")
        print(f"   mine {top2(my_logits)}")
        print(f"   max |logit diff| at that step: {(hf_logits.float() - my_logits.float()).abs().max():.4f}")


if __name__ == "__main__":
    main()
