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
NL = chr(10)  # line break inside labels

# Fixed colour and label per backend (validated categorical palette, slots 1-5).
# "engine" is my engine with continuous batching wherever batching applies.
BACKEND_STYLE = {
    "engine": ("#2a78d6", "My engine"),
    "naive": ("#eb6834", "Naive (no KV cache)"),
    "hf": ("#1baf7a", "HF generate()"),
    "vllm": ("#eda100", "vLLM"),
    "engine_static": ("#e87ba4", "My engine, static batching"),
}

# Stage 5 result files: backend style key -> run_bench backend name.
BENCH_BACKENDS = [
    ("naive", "naive"),
    ("hf", "hf_static"),
    ("engine_static", "engine_static"),
    ("engine", "engine_continuous"),
    ("vllm", "vllm"),
]
BENCH_LABELS = {
    "naive": "Naive, no cache" + NL + "(10-request subset)",
    "hf": "HF generate()," + NL + "static batches",
    "engine_static": "My engine," + NL + "static batching",
    "engine": "My engine," + NL + "continuous batching",
    "vllm": "vLLM",
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


def _plot_lines(
    ax: plt.Axes,
    data: dict[str, list[tuple[float, float]]],
    fmt=lambda v: f"{v:.0f} tok/s",
    labels: dict[str, str] | None = None,
) -> None:
    """Draw one line per backend and label each line's last point directly.

    End labels that would overlap are pushed apart vertically. Call after
    the y-limits are final.
    """
    ends = []
    for key, points in data.items():
        color, label = BACKEND_STYLE[key]
        label = (labels or {}).get(key, label)
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6, label=label)
        ends.append((xs[-1], ys[-1], f"{label}\n{fmt(ys[-1])}"))
    height_pt = ax.get_window_extent().height * 72 / ax.figure.dpi
    min_gap = 26  # points: two lines of 8.5pt text plus a little air
    placed: list[float] = []  # label centres, in points from the bottom of the axes
    to_axes = ax.transData + ax.transAxes.inverted()  # data -> 0..1 of the axes (works on log scales too)
    for x, y, text in sorted(ends, key=lambda e: e[1]):
        y_pt = to_axes.transform((x, y))[1] * height_pt
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


def _bench(backend: str, rate: str) -> dict | None:
    return _load(f"bench_{backend}_rate{rate}")


def plot_serving_comparison() -> Path | None:
    """Chart 3: offline throughput and p95 end-to-end latency, every backend, same 200 requests."""
    rows = [(key, _bench(name, "inf")) for key, name in BENCH_BACKENDS]
    rows = [(key, res) for key, res in rows if res]
    if not rows:
        return None
    env = next(res["env"] for _, res in rows if "env" in res)

    fig, (ax_tp, ax_lat) = plt.subplots(1, 2, figsize=(10, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(top=0.8, left=0.2, right=0.97, bottom=0.12, wspace=0.55)
    fig.suptitle("Serving 200 mixed requests: continuous batching wins", x=0.03, y=0.97, ha="left",
                 fontsize=13, fontweight="bold", color=TEXT)
    fig.text(0.03, 0.9, _setup_line(env, "Qwen2.5-0.5B-Instruct · fp16 · all requests arrive at t=0 · "
                                         "prompts 32-512, outputs 16-512 tokens · 32 slots"),
             ha="left", fontsize=8.5, color=TEXT_MUTED)

    def bars(ax, values, title, unit, fmt):
        keys = [k for k, _ in values]
        ys = list(range(len(values)))[::-1]
        ax.barh(ys, [v for _, v in values], color=[BACKEND_STYLE[k][0] for k in keys], height=0.6)
        ax.set_yticks(ys, [BENCH_LABELS[k] for k in keys], fontsize=8.5, color=TEXT)
        ax.set_title(title, loc="left", fontsize=10, color=TEXT)
        ax.set_xlabel(unit, color=TEXT_MUTED, fontsize=9)
        _style_axes(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        top = max(v for _, v in values)
        ax.set_xlim(0, top * 1.25)
        for y, (_, v) in zip(ys, values):
            ax.text(v + top * 0.02, y, fmt(v), va="center", fontsize=8.5, color=TEXT)

    bars(ax_tp, [(k, r["metrics"]["output_tokens_per_s"]) for k, r in rows],
         "Output throughput (higher is better)", "tokens / second", lambda v: f"{v:,.0f}")
    lat = [(k, r["metrics"]["e2e_p95_s"]) for k, r in rows if k != "naive" and "e2e_p95_s" in r["metrics"]]
    bars(ax_lat, lat, "p95 end-to-end latency (lower is better)", "seconds", lambda v: f"{v:.1f} s")
    path = RESULTS_DIR / "chart3_serving_comparison.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def plot_batch_timeline() -> Path | None:
    """Chart 4: how many requests are in the running batch over time, continuous vs static."""
    runs = [(key, _bench(name, "inf")) for key, name in (("engine_static", "engine_static"), ("engine", "engine_continuous"))]
    runs = [(k, r) for k, r in runs if r and r.get("timeline")]
    if not runs:
        return None
    fig, ax = _new_figure(
        "Batch size over time: continuous batching keeps the GPU full",
        _setup_line(runs[0][1]["env"], "same 200 requests, all queued at t=0 · 32 slots"),
    )
    for key, res in runs:
        t, running = zip(*[(p[0], p[1]) for p in res["timeline"]])
        color, _ = BACKEND_STYLE[key]
        label = "Continuous" if key == "engine" else "Static"
        ax.step(t, running, where="post", color=color, linewidth=1.6, label=label)
        # Mark when the last request finished.
        done = res["metrics"]["duration_s"]
        ax.axvline(done, color=color, linewidth=1, linestyle=":")
        ax.text(done - 0.8, 34.5, f"{label} done: {done:.0f} s", color=TEXT, fontsize=8.5, ha="right")
    ax.set_xlabel("Time since start (seconds)", color=TEXT_MUTED, fontsize=10)
    ax.set_ylabel("Requests in the running batch", color=TEXT_MUTED, fontsize=10)
    ax.set_ylim(0, 37)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=TEXT, bbox_to_anchor=(1.0, 1.0))
    return _save(fig, "chart4_batch_timeline.png")


def plot_latency_vs_load() -> Path | None:
    """Chart 5: p95 time to first token as the request rate grows, static vs continuous."""
    rates = ["2", "4", "inf"]
    data = {}
    env = None
    for key, name in (("engine_static", "engine_static"), ("engine", "engine_continuous"), ("hf", "hf_static")):
        pts = [(i, r["metrics"]["ttft_p95_s"]) for i, rate in enumerate(rates) if (r := _bench(name, rate))]
        if pts:
            data[key] = pts
            env = env or _bench(name, "inf") or _bench(name, "2")
    if not data or env is None:
        return None
    fig, ax = _new_figure(
        "Time to first token under load: continuous stays under 0.1 s",
        _setup_line(env["env"], "200 requests, Poisson arrivals · p95 TTFT (includes queueing)"),
    )
    ax.set_yscale("log")
    ax.set_ylim(0.02, 400)
    _plot_lines(ax, data, fmt=lambda v: f"{v:.2f} s" if v < 1 else f"{v:.0f} s",
                labels={"engine": "My engine, continuous", "engine_static": "My engine, static",
                        "hf": "HF generate(), static"})
    ax.set_xticks(range(len(rates)), ["2 req/s", "4 req/s", "all at once"])
    ax.set_xlabel("Request arrival rate", color=TEXT_MUTED, fontsize=10)
    ax.set_ylabel("p95 time to first token (s, log scale)", color=TEXT_MUTED, fontsize=10)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=TEXT)
    return _save(fig, "chart5_latency_vs_load.png")


def main() -> None:
    plot_single_request()
    plot_static_batching()
    plot_serving_comparison()
    plot_batch_timeline()
    plot_latency_vs_load()


if __name__ == "__main__":
    main()
