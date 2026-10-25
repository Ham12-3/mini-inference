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
from typing import Any

import torch

from engine.loader import resolve_model_path


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    print("== Environment")
    print(json.dumps(collect_env(), indent=2))

    print(f"\n== Downloading {args.model}")
    path = resolve_model_path(args.model)
    size_gb = sum(f.stat().st_size for f in path.glob("*.safetensors")) / 1e9
    print(f"local dir: {path}")
    print(f"weights:   {size_gb:.2f} GB in {len(list(path.glob('*.safetensors')))} safetensors file(s)")

    print("\n== config.json")
    print((path / "config.json").read_text())
    sys.stdout.flush()


if __name__ == "__main__":
    main()
