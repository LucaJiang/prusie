"""Max |Δalpha| against IBSS iterations from the published benchmark records.

Reads docs/assets/benchmark/benchmark_records.jsonl.gz and writes
docs/assets/benchmark/v030_alpha_vs_iterations.{svg,pdf}.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from v030_plots import (
    CASE_POINT,
    FLOOR,
    HIGHLIGHT,
    INK2,
    PANELS,
    TITLES,
    ZERO_ROW,
    density_scatter,
    error_axis,
    grid,
    note,
    save,
    sci,
    style,
)

MAX_ITER = 100


def load(path):
    rows = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            record = json.loads(line)
            if record["kind"] != "primary":
                continue
            iterations = max(np.median(record["prusie"]["niter"]), np.median(record["R"]["niter"]))
            rows.append(
                {
                    "panel": record["panel"],
                    "iter_max": float(iterations),
                    "alpha_error": float(record["max_abs_alpha_error"]),
                    "both_at_limit": bool(record["both_nonconverged_at_limit"]),
                }
            )
    return rows


def plot(assets_dir, output_dir):
    """Render the figure from `assets_dir`/benchmark_records.jsonl.gz into `output_dir`."""
    rows = load(Path(assets_dir) / "benchmark_records.jsonl.gz")
    style("prusie-v030-alpha-vs-iterations")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    fig, axes = plt.subplots(
        1, 2, figsize=(8.0, 3.3), layout="constrained", gridspec_kw={"width_ratios": [1, 1.5]}
    )
    for ax, panel in zip(axes, PANELS, strict=True):
        selected = [row for row in rows if row["panel"] == panel]
        x = np.array([row["iter_max"] for row in selected])
        e = np.array([row["alpha_error"] for row in selected])
        limit = np.array([row["both_at_limit"] for row in selected])
        zero = e == 0
        y = np.where(zero, ZERO_ROW * 10 ** rng.uniform(-0.25, 0.25, e.size), np.maximum(e, FLOOR))
        x_shown = x + rng.uniform(-0.3, 0.3, x.size)
        if panel == "GWAS":
            ax.scatter(x_shown[~limit], y[~limit], s=9, color=CASE_POINT, alpha=0.55, linewidths=0)
        else:
            density_scatter(ax, np.maximum(x_shown[~limit], 0.7), y[~limit], CASE_POINT, light=0.8)
        if limit.any():
            ax.scatter(
                MAX_ITER + rng.uniform(-1.2, 1.2, limit.sum()),
                y[limit],
                s=12,
                marker="D",
                facecolor="white",
                edgecolor=HIGHLIGHT,
                linewidth=0.7,
                zorder=5,
            )
        median = float(np.median(e))
        ax.axhline(median, color=INK2, lw=0.7, ls=(0, (1, 1.5)), zorder=3)
        note(ax, f"median {sci(median)}\nexact 0: {int(zero.sum()):,}", x=0.06, y=0.26)
        error_axis(ax)
        ax.set_xscale("linear")
        ax.set_xlim(0, MAX_ITER + 4)
        ax.set_xticks((1, 20, 40, 60, 80, 100))
        ax.set_title(f"{TITLES[panel]}")
        ax.set_xlabel("IBSS iterations (larger of the two programs)")
        grid(ax, "x")
    axes[0].set_ylabel("Max |Δα|")
    handles = [
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=4,
            mfc=CASE_POINT,
            mec="none",
            label="At least one program converged",
        ),
        Line2D(
            [],
            [],
            ls="",
            marker="D",
            ms=4,
            mfc="white",
            mec=HIGHLIGHT,
            mew=0.9,
            label="Neither program converged (100 iterations)",
        ),
        Line2D([], [], color=INK2, lw=0.7, ls=(0, (1, 1.5)), label="Panel median"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, columnspacing=1.8)
    save(fig, output, "v030_alpha_vs_iterations")


def main():
    docs_assets = Path(__file__).resolve().parents[3] / "docs" / "assets" / "benchmark"
    plot(docs_assets, docs_assets)


if __name__ == "__main__":
    main()
