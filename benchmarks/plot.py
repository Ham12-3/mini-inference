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
    "hf": ("#1baf7a", "HF generate()"),
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
    # Fix the layout now: direct labels are placed in screen units.
    fig.subplots_adjust(top=0.82, left=0.1, right=0.8, bottom=0.12)
    _style_axes(ax)
    fig.suptitle(title, x=0.07, y=0.97, ha="left", fontsize=13, fontweight="bold", color=TEXT)
    fig.text(0.07, 0.905, subtitle, ha="left", fontsize=9, color=TEXT_MUTED)
    return fig, ax


def _setup_line(env: dict, extra: str = "") -> str:
    return f"{env.get('gpu', 'CPU')} · torch {env['torch']} · {extra}".rstrip(" ·")


def _plot_lines(ax: plt.Axes, data: dict[str, list[tuple[float, float]]]) -> None:
    """Draw one line per backend and label each line's last point directly.

    End labels that would overlap are pushed apart vertically. Call after
    the y-limits are final.
    """
    ends = []
    for key, points in data.items():
        color, label = BACKEND_STYLE[key]
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6, label=label)
        ends.append((xs[-1], ys[-1], f"{label}\n{ys[-1]:.0f} tok/s"))
    lo, hi = ax.get_ylim()
    height_pt = ax.get_window_extent().height * 72 / ax.figure.dpi
    min_gap = 26  # points: two lines of 8.5pt text plus a little air
    placed: list[float] = []  # label centres, in points from the bottom of the axes
    for x, y, text in sorted(ends, key=lambda e: e[1]):
        y_pt = (y - lo) / (hi - lo) * height_pt
        label_pt = max([y_pt] + [p + min_gap for p in placed])
        placed.append(label_pt)
        ax.annotate(text, (x, y), xytext=(8, label_pt - y_pt), textcoords="offset points",
                    va="center", fontsize=8.5, color=TEXT, annotation_clip=False)


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
    ax.set_xscale("log", base=2)
    all_x = sorted({x for pts in data.values() for x, _ in pts})
    ax.set_xticks(all_x, [str(x) for x in all_x])
    ax.set_xlabel("Output length (tokens)", color=TEXT_MUTED, fontsize=10)
    ax.set_ylabel("Decode throughput (tokens/s)", color=TEXT_MUTED, fontsize=10)
    ax.set_ylim(0, max(y for pts in data.values() for _, y in pts) * 1.12)
    _plot_lines(ax, data)
    ax.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=TEXT)
    return _save(fig, "chart1_single_request.png")


def plot_static_batching() -> Path | None:
    """Chart 2: static batching throughput vs batch size."""
    data = {}
    env = None
    output_len = None
    for key, name in (("hf", "static_hf"), ("engine", "static_engine")):
        res = _load(name)
        if res:
            env = env or res["env"]
            output_len = res["output_len"]
            data[key] = [(r["batch_size"], r["tokens_per_s"]) for r in res["runs"]]
    if not data or env is None:
        return None

    fig, ax = _new_figure(
        "Batching: one decode step serves the whole batch",
        _setup_line(env, f"Qwen2.5-0.5B-Instruct · fp16 · {output_len} new tokens per request · static batches"),
    )
    # Reference: perfect scaling from my engine's batch-1 speed (B x single).
    if "engine" in data:
        xs = [x for x, _ in data["engine"]]
        base = data["engine"][0][1] / data["engine"][0][0]
        ax.plot(xs, [base * x for x in xs], color=TEXT_MUTED, linewidth=1, linestyle="--", label="Perfect scaling")
    all_x = sorted({x for pts in data.values() for x, _ in pts})
    ax.set_xticks(all_x, [str(x) for x in all_x])
    ax.set_xlabel("Batch size (requests decoded together)", color=TEXT_MUTED, fontsize=10)
    ax.set_ylabel("Output throughput (tokens/s)", color=TEXT_MUTED, fontsize=10)
    ax.set_ylim(bottom=0)  # leaves room for the perfect-scaling line
    _plot_lines(ax, data)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=TEXT)
    return _save(fig, "chart2_static_batching.png")


def main() -> None:
    plot_single_request()
    plot_static_batching()


if __name__ == "__main__":
    main()
