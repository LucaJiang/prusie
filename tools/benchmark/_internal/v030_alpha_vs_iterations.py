"""Max |Δalpha| vs iterations figure from the published benchmark records.

Reads docs/assets/benchmark/benchmark_records.jsonl.gz and writes
docs/assets/benchmark/v030_alpha_vs_iterations.{svg,pdf}.
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter

COLORS = {"converged": "#0072B2", "nonconverged": "#D55E00"}
PANELS = ("GWAS", "eQTL")
MAX_ITER = 100
ALPHA_GATE = 1e-5


def median(values):
    values = sorted(values)
    n = len(values)
    if n == 0:
        return math.nan
    if n % 2:
        return float(values[n // 2])
    return 0.5 * (values[n // 2 - 1] + values[n // 2])


def load(path):
    rows = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            record = json.loads(line)
            if record["kind"] != "primary":
                continue
            prusie_iters = median(record["prusie"]["niter"])
            r_iters = median(record["R"]["niter"])
            rows.append(
                {
                    "panel": record["panel"],
                    "iter_prusie": prusie_iters,
                    "iter_r": r_iters,
                    "iter_max": max(prusie_iters, r_iters),
                    "alpha_error": float(record["max_abs_alpha_error"]),
                    "both_at_limit": bool(record["both_nonconverged_at_limit"]),
                }
            )
    return rows


def panel_counts(rows):
    at_limit = sum(1 for row in rows if row["both_at_limit"])
    exceedance = sum(1 for row in rows if row["alpha_error"] > ALPHA_GATE)
    return len(rows), at_limit, exceedance


def plot(rows, output):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "prusie-v030-alpha-vs-iterations",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for ax, panel in zip(axes, PANELS, strict=True):
        panel_rows = [row for row in rows if row["panel"] == panel]
        converged = [row for row in panel_rows if not row["both_at_limit"]]
        at_limit = [row for row in panel_rows if row["both_at_limit"]]

        maximum = max((row["alpha_error"] for row in panel_rows), default=0.0)
        ax.scatter(
            [row["iter_max"] for row in converged],
            [row["alpha_error"] for row in converged],
            s=9,
            alpha=0.35,
            rasterized=True,
            color=COLORS["converged"],
            label="At least one backend converged",
        )
        if at_limit:
            ax.scatter(
                [row["iter_max"] for row in at_limit],
                [row["alpha_error"] for row in at_limit],
                s=22,
                marker="D",
                alpha=0.85,
                color=COLORS["nonconverged"],
                label=f"Both at iteration limit (n={len(at_limit)})",
            )

        ax.axhline(
            ALPHA_GATE,
            color="#444",
            linestyle="--",
            linewidth=0.8,
            label=f"Alpha gate: {ALPHA_GATE:g}",
        )
        total, _, exceedance = panel_counts(panel_rows)
        ax.set(
            title=f"{panel} (n={total:,}; exceedances={exceedance})",
            xlabel="Median IBSS iterations per case (max of prusie / R)",
            ylabel="Maximum absolute alpha difference",
            yscale="symlog",
        )
        ax.set_yscale("symlog", linthresh=1e-14)
        upper = max(3 * maximum, 1e-13, 3 * ALPHA_GATE)
        ax.set_ylim(0, upper)
        ax.yaxis.set_major_locator(FixedLocator((0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4)))
        ax.yaxis.set_major_formatter(
            FuncFormatter(
                lambda value, _: "0" if value == 0 else f"$10^{{{int(np.log10(value))}}}$"
            )
        )
        ax.set_xlim(0, MAX_ITER + 2)
        ax.grid(alpha=0.15)
        ax.legend(frameon=False, fontsize=8, loc="upper left")

    for ext in ("svg", "pdf"):
        metadata = {"Date": None} if ext == "svg" else {"CreationDate": None, "ModDate": None}
        fig.savefig(
            output / f"v030_alpha_vs_iterations.{ext}", bbox_inches="tight", metadata=metadata
        )
    plt.close(fig)


def main():
    docs_assets = Path(__file__).resolve().parents[3] / "docs" / "assets" / "benchmark"
    rows = load(docs_assets / "benchmark_records.jsonl.gz")
    plot(rows, docs_assets)


if __name__ == "__main__":
    main()
