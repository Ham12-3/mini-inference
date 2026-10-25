"""Save benchmark results as JSON next to the code that produced them.

Each result file holds the measured numbers plus the environment (GPU,
library versions) so it can be read on its own later.

The result is also printed on one line between BEGIN/END markers. On Colab
that lets us copy a result out of the notebook output even though the runtime
cannot push to GitHub itself.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.env_info import collect_env

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def save_results(name: str, payload: dict[str, Any]) -> Path:
    """Write ``payload`` (plus env info and a timestamp) to ``results/<name>.json`` and print it."""
    record = {
        "name": name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "env": collect_env(),
        **payload,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"=== RESULT {name} BEGIN ===")
    print(json.dumps(record, separators=(",", ":")))
    print(f"=== RESULT {name} END ===")
    print(f"saved {path}")
    return path


def print_saved(names: list[str], packed: bool = False) -> None:
    """Print saved result files between markers, to copy results off Colab.

    ``packed`` prints zlib-compressed base64 instead of JSON: about a third of
    the size, which matters when copying by hand from notebook output.
    """
    import base64
    import zlib

    for name in names:
        record = json.loads((RESULTS_DIR / f"{name}.json").read_text())
        text = json.dumps(record, separators=(",", ":"))
        if packed:
            print(f"=== PACKED {name} BEGIN ===")
            print(base64.b64encode(zlib.compress(text.encode(), 9)).decode())
            print(f"=== PACKED {name} END ===")
        else:
            print(f"=== RESULT {name} BEGIN ===")
            print(text)
            print(f"=== RESULT {name} END ===")


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    print_saved([a for a in args if a != "--packed"], packed="--packed" in args)
