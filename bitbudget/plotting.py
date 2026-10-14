"""Figures: quality-vs-size Pareto frontier and the per-layer bit map."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

# Fixed order/colors so figures from different runs are comparable.
METHOD_STYLE = {
    "fp16": ("#6b7280", "*"),
    "rtn": ("#9ca3af", "o"),
    "mlx-awq": ("#60a5fa", "s"),
    "mlx-mixed": ("#a78bfa", "D"),
    "mlx-dynamic": ("#f59e0b", "^"),
    "bitbudget-noawq": ("#34d399", "v"),
    "bitbudget-uniform": ("#0ea5e9", "P"),
    "bitbudget-output": ("#a3a3a3", "x"),
    "bitbudget": ("#dc2626", "X"),
}


def load_results(path: str | Path) -> list[dict]:
    """Read a results JSONL file; later records with the same ``tag`` are merged
    into earlier ones (e.g. an energy measurement added after evaluation)."""
    merged: dict[str, dict] = {}
    order = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        tag = r.get("tag") or f"row{len(order)}"
        if tag not in merged:
            merged[tag] = {}
            order.append(tag)
        merged[tag].update(r)
    return [merged[t] for t in order]


def plot_pareto(results: list[dict], out: str | Path, metric: str = "ppl",
                x: str = "bpw", title: str | None = None,
                xlabel: str | None = None) -> Path:
    groups = defaultdict(list)
    for r in results:
        if metric in r and r.get(metric) is not None and x in r:
            groups[r.get("method", "?")].append(r)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for method, rows in groups.items():
        rows = sorted(rows, key=lambda r: r[x])
        color, marker = METHOD_STYLE.get(method, ("#111827", "o"))
        xs = [r[x] for r in rows]
        ys = [r[metric] for r in rows]
        if method == "fp16":
            ax.axhline(ys[0], color=color, ls="--", lw=1, label=f"fp16 ({ys[0]:.3g})")
            continue
        ax.plot(xs, ys, marker=marker, color=color, lw=1.6 if method == "bitbudget" else 1.0,
                ms=7, label=method)
    ax.set_xlabel(xlabel or ("bits per weight (whole model, mlx-lm measure)" if x == "bpw" else x))
    ax.set_ylabel({"ppl": "perplexity (lower is better)",
                   "kl": "KL to fp16 (nats/token)"}.get(metric, metric))
    ys_all = [r[metric] for rows in groups.values() for r in rows if r[metric] > 0]
    if metric == "ppl" and ys_all and max(ys_all) / min(ys_all) > 5:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_bitmap(plan: dict[str, int], out: str | Path, title: str | None = None) -> Path:
    """Heatmap: decoder block on the x-axis, projection on the y-axis."""
    pat = re.compile(r"\.(\d+)\.(.+)$")
    cells = {}
    rows = []
    n_blocks = 0
    for path, b in plan.items():
        m = pat.search(path)
        if not m:
            continue
        i, name = int(m.group(1)), m.group(2).split(".")[-1]
        cells[(name, i)] = b
        n_blocks = max(n_blocks, i + 1)
        if name not in rows:
            rows.append(name)
    order = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    rows = [r for r in order if r in rows] + [r for r in rows if r not in order]
    levels = [2, 3, 4, 5, 6, 8]
    grid = [[cells.get((r, i), float("nan")) for i in range(n_blocks)] for r in rows]
    cmap = ListedColormap(["#7f1d1d", "#dc2626", "#f59e0b", "#fde68a", "#93c5fd", "#1d4ed8"])
    norm = BoundaryNorm([1.5, 2.5, 3.5, 4.5, 5.5, 7, 9], cmap.N)
    fig, ax = plt.subplots(figsize=(max(6, n_blocks * 0.32), 3.2), dpi=150)
    im = ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    ax.set_xticks(range(n_blocks), [str(i) for i in range(n_blocks)], fontsize=7)
    ax.set_xlabel("decoder block")
    cb = fig.colorbar(im, ax=ax, ticks=levels, fraction=0.03, pad=0.02)
    cb.set_label("bits")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
