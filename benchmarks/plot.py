"""Make PNG charts from the JSON results in ``benchmarks/results/``.

    python -m benchmarks.plot

Each backend keeps the same colour on every chart, so a reader who learns
"blue = my engine" on chart 1 can rely on it for the rest.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from benchmarks.report import RESULTS_DIR  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"

# Fixed colour and label per backend (validated categorical palette, slots 1-4).
BACKEND_STYLE = {
    "engine": ("#2a78d6", "My engine"),
    "naive": ("#eb6834", "Naive (no KV cache)"),
    "hf": ("#1baf7a", "Hugging Face"),
    "vllm": ("#eda100", "vLLM"),
}


def _load(name: str) -> dict | None:
    path = RESULTS_DIR / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_MUTED)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _new_figure(title: str, subtitle: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    fig.suptitle(title, x=0.07, y=0.97, ha="left", fontsize=13, fontweight="bold", color=TEXT)
    fig.text(0.07, 0.905, subtitle, ha="left", fontsize=9, color=TEXT_MUTED)
    return fig, ax


def _setup_line(env: dict, extra: str = "") -> str:
    return f"{env.get('gpu', 'CPU')} · torch {env['torch']} · {extra}".rstrip(" ·")


def _save(fig: plt.Figure, name: str) -> Path:
    path = RESULTS_DIR / name
    fig.subplots_adjust(top=0.82, left=0.1, right=0.8, bottom=0.12)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def plot_single_request() -> Path | None:
    """Chart 1: single-request decode speed vs output length, naive vs KV cache."""
    series = [
        ("naive", ["single_naive", "single_naive_long"]),
        ("hf", ["single_hf_kv"]),
        ("engine", ["single_engine_kv"]),
    ]
    data = {}
    env = None
    for key, names in series:
        points: dict[int, float] = {}
        for name in names:
            res = _load(name)
            if res:
                env = env or res["env"]
                for run in res["runs"]:
                    points[run["output_len"]] = run["tokens_per_s"]
        if points:
            data[key] = sorted(points.items())
    if not data or env is None:
        return None

    fig, ax = _new_figure(
        "Single request: a KV cache keeps decode speed flat",
        _setup_line(env, "Qwen2.5-0.5B-Instruct · fp16 · greedy · 48-token prompt"),
    )
    for key, points in data.items():
        color, label = BACKEND_STYLE[key]
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6, label=label)
        # Direct label at the right end of each line.
        ax.annotate(
            f"{label}\n{ys[-1]:.0f} tok/s",
            (xs[-1], ys[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=TEXT,
        )
    ax.set_xscale("log", base=2)
    all_x = sorted({x for pts in data.values() for x, _ in pts})
    ax.set_xticks(all_x, [str(x) for x in all_x])
    ax.set_xlabel("Output length (tokens)", color=TEXT_MUTED, fontsize=10)
    ax.set_ylabel("Decode throughput (tokens/s)", color=TEXT_MUTED, fontsize=10)
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=TEXT)
    return _save(fig, "chart1_single_request.png")


def main() -> None:
    plot_single_request()


if __name__ == "__main__":
    main()
