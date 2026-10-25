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


def print_saved(names: list[str]) -> None:
    """Print saved result files on one line each between markers (to copy results off Colab)."""
    for name in names:
        record = json.loads((RESULTS_DIR / f"{name}.json").read_text())
        print(f"=== RESULT {name} BEGIN ===")
        print(json.dumps(record, separators=(",", ":")))
        print(f"=== RESULT {name} END ===")


if __name__ == "__main__":
    import sys

    print_saved(sys.argv[1:])
