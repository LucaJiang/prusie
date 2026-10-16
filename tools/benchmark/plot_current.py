#!/usr/bin/env python3
"""Plot fitting runtime and memory for genomic loci and matrix sizes.

Uses the same runtime/memory definitions and scientific figure style as
tools/generate_results.py. Each input is a data-safe summary containing records
with n_variants and per-backend runtime_seconds, peak_memory_mib, sampled_memory,
niter and convergence. No genotype, variant identity or association input is used.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"prusie": "#0072B2", "susieR": "#D55E00"}


def save(fig, output, name):
    for ext in ("svg", "png"):
        metadata = {"Date": None} if ext == "svg" else {"Software": "prusie benchmark tools"}
        path = output / (name + "." + ext)
        fig.savefig(path, bbox_inches="tight", dpi=180, metadata=metadata)
        if ext == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real", type=Path, required=True)
    p.add_argument("--scaling", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    real = json.loads(args.real.read_text())
    scaling = json.loads(args.scaling.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "prusie-complete-call-20261012",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    for key, ylabel, filename in [
        ("runtime", "Median fitting runtime (ms)", "runtime_by_variants"),
        ("memory", "Peak resident memory (MiB)", "memory_by_variants"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), layout="constrained")
        for ax, group, title in zip(axes, ("gwas", "eqtl"), ("GWAS", "eQTL"), strict=True):
            rows = [
                r
                for r in real["records"]
                if r["group"] == group
                and r["timing_eligible"]
                and (key != "memory" or r["memory_eligible"])
            ]
            for backend, marker in [("prusie", "o"), ("susieR", "^")]:
                y = [
                    r["backends"][backend]["runtime_seconds"]["median"] * 1000
                    if key == "runtime"
                    else r["backends"][backend]["peak_memory_mib"]
                    for r in rows
                ]
                ax.scatter(
                    [r["n_variants"] for r in rows],
                    y,
                    color=COLORS[backend],
                    marker=marker,
                    s=14,
                    alpha=0.65,
                    linewidths=0,
                    label=backend,
                )
            ax.set(
                xlabel="Variants in analysis",
                ylabel=ylabel,
                title=title,
                xscale="log",
                yscale="log",
            )
            ax.grid(alpha=0.15)
            ax.legend(frameon=False, fontsize=9)
        save(fig, args.output, filename)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), layout="constrained")
    for backend in COLORS:
        rows = sorted(
            [r for r in scaling["records"] if r["timing_eligible"]], key=lambda r: r["n_variants"]
        )
        x = [r["n_variants"] for r in rows]
        b = [r["backends"][backend] for r in rows]
        med = np.array([z["runtime_seconds"]["median"] for z in b])
        low = np.array([z["runtime_seconds"]["q1"] for z in b])
        high = np.array([z["runtime_seconds"]["q3"] for z in b])
        axes[0, 0].errorbar(
            x,
            med,
            yerr=[med - low, high - med],
            marker="o",
            color=COLORS[backend],
            label=backend,
            capsize=3,
        )
        axes[1, 0].plot(
            x, [z["niter"] for z in b], marker="o", color=COLORS[backend], label=backend
        )
        memory = sorted(
            [r for r in scaling["records"] if r["memory_eligible"]], key=lambda r: r["n_variants"]
        )
        xm = [r["n_variants"] for r in memory]
        bm = [r["backends"][backend] for r in memory]
        axes[0, 1].plot(
            xm,
            [z["peak_memory_mib"] / 1024 for z in bm],
            marker="o",
            color=COLORS[backend],
            label=backend,
        )
        for key, linestyle, label in [
            ("baseline_rss_bytes", "--", "prefit baseline"),
            ("sampled_incremental_peak_bytes", "-", "sampled increment"),
        ]:
            valid = [
                (r["n_variants"], r["backends"][backend]["sampled_memory"])
                for r in memory
                if r["backends"][backend]["sampled_memory"].get("status") == "ok"
            ]
            axes[1, 1].plot(
                [p for p, s in valid],
                [s[key] / 2**30 for p, s in valid],
                marker="o",
                linestyle=linestyle,
                color=COLORS[backend],
                label=f"{backend}: {label}",
            )
    for ax, title, ylabel in zip(
        axes.flat,
        [
            "A  Fitting runtime: median and IQR",
            "B  Peak resident memory",
            "C  Completed IBSS sweeps",
            "D  Sampled resident memory",
        ],
        ["Seconds", "GiB", "Iterations", "GiB"],
        strict=True,
    ):
        ax.set(title=title, xlabel="Variables (p)", ylabel=ylabel, xscale="log")
        ax.grid(alpha=0.15)
        ax.legend(frameon=False, fontsize=8)
    axes[0, 0].set_yscale("log")
    axes[0, 1].set_yscale("log")
    # A sampled increment can legitimately be zero when an allocation spike is
    # missed or existing resident pages suffice. Keep that lower bound visible.
    axes[1, 1].set_yscale("symlog", linthresh=0.001)
    axes[1, 1].set_ylim(bottom=0)
    save(fig, args.output, "controlled_scaling")


if __name__ == "__main__":
    main()
