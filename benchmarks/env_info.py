"""Collect and print environment info: GPU, library versions and the model config.

Every benchmark result stores the output of :func:`collect_env` next to its
numbers, so a result file always says what hardware and software produced it.

Run it directly to check a fresh runtime and download the model:

    python -m benchmarks.env_info --model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch

# Files we need from the Hub: weights, model config, tokenizer and generation config.
MODEL_FILE_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model"]


def collect_env() -> dict[str, Any]:
    """Return a JSON-friendly description of the hardware and software."""
    import transformers

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            gpu=props.name,
            gpu_capability=f"{props.major}.{props.minor}",
            gpu_memory_gb=round(props.total_memory / 1e9, 2),
        )
    return info


def download_model(model: str) -> Path:
    """Download (or reuse the cached copy of) a model from the Hub and return its local directory."""
    from huggingface_hub import snapshot_download

    if Path(model).is_dir():
        return Path(model)
    return Path(snapshot_download(model, allow_patterns=MODEL_FILE_PATTERNS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    print("== Environment")
    print(json.dumps(collect_env(), indent=2))

    print(f"\n== Downloading {args.model}")
    path = download_model(args.model)
    size_gb = sum(f.stat().st_size for f in path.glob("*.safetensors")) / 1e9
    print(f"local dir: {path}")
    print(f"weights:   {size_gb:.2f} GB in {len(list(path.glob('*.safetensors')))} safetensors file(s)")

    print("\n== config.json")
    print((path / "config.json").read_text())
    sys.stdout.flush()


if __name__ == "__main__":
    main()
