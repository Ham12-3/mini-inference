import pytest
import torch

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Varied prompts (explanation, code, maths, a list, a story), each with a
# long enough answer that the fp16 parity test really compares 50 tokens.
PROMPTS = [
    "Explain, step by step, how a CPU cache works.",
    "Write a Python function that checks whether a string is a palindrome.",
    "What is 17 multiplied by 23? Show your working.",
    "List five interesting facts about octopuses.",
    "Write a short story about a robot who learns to paint.",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def model_dir():
    from engine.loader import resolve_model_path

    return resolve_model_path(MODEL_ID)


@pytest.fixture(scope="session")
def tokenizer(model_dir):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir)


@pytest.fixture(scope="session")
def chat_prompt_ids(tokenizer):
    """Token ids of each prompt wrapped in the model's chat template."""

    def encode(text: str) -> list[int]:
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
        return tokenizer(chat).input_ids

    return [encode(p) for p in PROMPTS]


@pytest.fixture(scope="session")
def prompts(tokenizer, chat_prompt_ids):
    """Eight prompts for engine tests: the five parity prompts plus three from the benchmark pool."""
    from benchmarks.workload import PROMPT_POOL, chat_prompt_ids as encode_chat

    return chat_prompt_ids + encode_chat(tokenizer, PROMPT_POOL[2:5])


requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
