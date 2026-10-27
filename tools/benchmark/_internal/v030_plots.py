"""Deterministic figures from exported 0.3.0 records; no fitting or R required."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, ScalarFormatter
from v030_records import finite

COLORS = {"prusie": "#0072B2", "susieR": "#D55E00"}
PANELS = ("GWAS", "eQTL")


def save(fig, output, name):
    for ext in ("svg", "pdf"):
        metadata = {"Date": None} if ext == "svg" else {"CreationDate": None, "ModDate": None}
        fig.savefig(output / (name + "." + ext), bbox_inches="tight", metadata=metadata)
    plt.close(fig)


def ecdf(ax, values, **kwargs):
    values = sorted(v for v in values if finite(v) and v > 0)
    if values:
        ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", **kwargs)


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "prusie-v030-frozen-real-panels",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def finish(axes):
    for ax in np.asarray(axes).flat:
        ax.grid(alpha=0.15)


def plot(summary, m1, records, output):
    style()
    primary = [r for r in records if r["kind"] == "primary"]
    for key, ylabel, filename in (
        ("runtime", "Median fit time (s)", "v030_runtime_by_variants"),
        ("memory", "Fresh-process peak RSS (MiB)", "v030_memory_by_variants"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
        for ax, panel in zip(axes, PANELS, strict=True):
            rows = [
                r
                for r in primary
                if r["panel"] == panel and r["timing_pair" if key == "runtime" else "memory_pair"]
            ]
            for backend, marker in (("prusie", "o"), ("susieR", "^")):
                y = [
                    r[backend]["runtime"]["median"]
                    if key == "runtime"
                    else r[backend]["peak_rss_bytes"] / 2**20
                    for r in rows
                ]
                ax.scatter(
                    [r["p"] for r in rows],
                    y,
                    s=7 if panel == "eQTL" else 15,
                    alpha=0.3 if panel == "eQTL" else 0.65,
                    color=COLORS[backend],
                    marker=marker,
                    linewidths=0,
                    label=backend,
                )
            ax.set(
                title=f"{panel} (paired n={len(rows):,})",
                xlabel="Variants (p)",
                ylabel=ylabel,
                xscale="log",
                yscale="log",
            )
            ax.legend(frameon=False)
        finish(axes)
        save(fig, output, filename)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.8), layout="constrained")
    for column, panel in enumerate(PANELS):
        rows = [r for r in primary if r["panel"] == panel]
        for ax, field, label in (
            (axes[0, column], "speedup", "R / prusie median fit time"),
            (axes[1, column], "memory_ratio", "R / prusie fresh-process peak RSS"),
        ):
            values = [r[field] for r in rows if finite(r[field]) and r[field] > 0]
            ecdf(ax, values, color=COLORS["prusie"], linewidth=1.5)
            ax.axvline(1, color="#777777", linestyle="--", linewidth=0.8)
            ax.set(
                title=f"{panel} (paired n={len(values):,})",
                xlabel=label,
                ylabel="Fraction of cases",
                xscale="log",
                ylim=(0, 1.02),
            )
            if field == "memory_ratio":
                ax.xaxis.set_major_locator(FixedLocator((1, 2, 3, 4, 6, 8)))
                ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.xaxis.set_minor_formatter(NullFormatter())
    finish(axes)
    save(fig, output, "v030_ratio_distributions")

    lookup = {(r["panel"], r["case_id"], r["kind"]): r for r in records}
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.8), layout="constrained")
    for column, panel in enumerate(PANELS):
        base = [r for r in primary if r["panel"] == panel and (panel == "GWAS" or r["p"] >= 1024)]
        all_ratios = []
        for count, color in ((2, "#009E73"), (4, "#0072B2"), (8, "#CC79A7")):
            paired = [(r, lookup[(panel, r["case_id"], f"threads{count}")]) for r in base]
            ratios = [
                a["prusie"]["runtime"]["median"] / b["prusie"]["runtime"]["median"]
                for a, b in paired
                if a["prusie"]["time_complete"] and b["prusie"]["time_complete"]
            ]
            all_ratios.append(ratios)
            ecdf(axes[0, column], ratios, color=color, label=f"{count} threads (n={len(ratios):,})")
        axes[0, column].axvline(1, color="#777777", linestyle="--", linewidth=0.8)
        axes[0, column].set(
            title=panel,
            xlabel="Median t1 / median tk fit time",
            ylabel="Fraction of cases",
            xscale="log",
            ylim=(0, 1.02),
        )
        axes[0, column].legend(frameon=False, fontsize=8)
        valid = [
            (k, np.quantile(v, (0.25, 0.5, 0.75)))
            for k, v in zip((2, 4, 8), all_ratios, strict=True)
            if v
        ]
        if valid:
            x = [1] + [k for k, _ in valid]
            q = np.array([[1, 1, 1]] + [list(v) for _, v in valid])
            axes[1, column].plot(x, q[:, 1], "o-", color=COLORS["prusie"])
            axes[1, column].fill_between(x, q[:, 0], q[:, 2], alpha=0.15, color=COLORS["prusie"])
        axes[1, column].axhline(1, color="#777777", linestyle="--", linewidth=0.8)
        axes[1, column].set(
            xlabel="Requested native threads",
            ylabel="Across-case speed-up: median and IQR",
            xticks=(1, 2, 4, 8),
        )
    finish(axes)
    save(fig, output, "v030_thread_scaling")

    pairs = [
        (lookup[(r["panel"], r["case_id"], "primary")], r)
        for r in records
        if r["kind"] == "overwrite"
    ]
    pairs = [
        (a, b)
        for a, b in pairs
        if a["prusie"]["memory_complete"] and b["prusie"]["memory_complete"]
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    for position, (label, color) in enumerate(
        (("overwrite_ld=False", COLORS["prusie"]), ("overwrite_ld=True", "#009E73"))
    ):
        axes[0].scatter(
            [a["p"] for a, _ in pairs],
            [pair[position]["prusie"]["peak_rss_bytes"] / 2**20 for pair in pairs],
            s=14,
            alpha=0.6,
            color=color,
            label=label,
        )
    axes[0].set(
        xlabel="Variants (p)",
        ylabel="Fresh-process peak RSS (MiB)",
        xscale="log",
        yscale="log",
        title=f"GWAS (paired n={len(pairs)})",
    )
    axes[0].legend(frameon=False, fontsize=8)
    ecdf(
        axes[1],
        [a["prusie"]["peak_rss_bytes"] / b["prusie"]["peak_rss_bytes"] for a, b in pairs],
        color="#009E73",
    )
    axes[1].axvline(1, color="#777777", linestyle="--", linewidth=0.8)
    axes[1].set(xlabel="Peak RSS: False / True", ylabel="Fraction of GWAS loci", ylim=(0, 1.02))
    finish(axes)
    save(fig, output, "v030_overwrite_memory")

    windows = sorted((r for r in records if r["kind"].startswith("window")), key=lambda r: r["p"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    for backend in COLORS:
        timed = [r for r in windows if r[backend] and r[backend]["time_complete"]]
        if timed:
            med = np.array([r[backend]["runtime"]["median"] for r in timed])
            low = np.array([r[backend]["runtime"]["q1"] for r in timed])
            high = np.array([r[backend]["runtime"]["q3"] for r in timed])
            axes[0].errorbar(
                [r["p"] for r in timed],
                med,
                yerr=(med - low, high - med),
                marker="o",
                color=COLORS[backend],
                label=backend,
                capsize=3,
            )
            axes[2].plot(
                [r["p"] for r in timed],
                [np.median(r[backend]["niter"]) for r in timed],
                "o-",
                color=COLORS[backend],
                label=backend,
            )
        memory = [r for r in windows if r[backend] and r[backend]["memory_complete"]]
        axes[1].plot(
            [r["p"] for r in memory],
            [r[backend]["peak_rss_bytes"] / 2**30 for r in memory],
            "o-",
            color=COLORS[backend],
            label=backend,
        )
    for ax, ylabel in zip(
        axes,
        (
            "Median fit time and repeat IQR (s)",
            "Fresh-process peak RSS (GiB)",
            "Completed IBSS sweeps",
        ),
        strict=True,
    ):
        ax.set(xlabel="Leading variants in one fixed T2D locus", ylabel=ylabel, xscale="log")
        ax.legend(frameon=False)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    finish(axes)
    save(fig, output, "v030_controlled_scaling")

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.8), layout="constrained")
    for column, panel in enumerate(PANELS):
        for index, (field, label) in enumerate(
            (
                ("max_abs_alpha_error", "Maximum absolute alpha difference"),
                ("max_abs_pip_error", "Maximum absolute PIP difference"),
            )
        ):
            rows = [r for r in primary if r["panel"] == panel and finite(r[field])]
            axes[index, column].scatter(
                [r["p"] for r in rows],
                [r[field] for r in rows],
                s=9,
                alpha=0.4,
                color=COLORS["prusie"],
            )
            axes[index, column].set(
                title=f"{panel} (available n={len(rows):,})",
                xlabel="Variants (p)",
                ylabel=label,
                xscale="log",
                yscale="symlog",
            )
            axes[index, column].set_yscale("symlog", linthresh=1e-14)
            maximum = max((r[field] for r in rows), default=0)
            if index == 0:
                axes[index, column].axhline(
                    1e-5, color="#D55E00", linestyle="--", linewidth=0.8, label="Alpha gate: 1e-5"
                )
                axes[index, column].legend(frameon=False, fontsize=8)
                maximum = max(maximum, 1e-5)
            axes[index, column].set_ylim(0, max(3 * maximum, 1e-13))
            axes[index, column].yaxis.set_major_locator(
                FixedLocator((0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-5))
            )
            axes[index, column].yaxis.set_major_formatter(
                FuncFormatter(
                    lambda value, _: "0" if value == 0 else f"$10^{{{int(np.log10(value))}}}$"
                )
            )
    finish(axes)
    save(fig, output, "v030_numerical_agreement")

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), layout="constrained")
    for column, (panel, title) in enumerate(
        zip(("panel", "fresh_panel", "largep"), ("Primary", "Confirmation", "Large-p"), strict=True)
    ):
        ax = axes[0, column]
        c1 = {
            r["case_id"]: r
            for r in m1["records"]
            if r["comparison"] == "C1_V030_vs_BASE" and r["panel"] == panel
        }
        c2 = {
            r["case_id"]: r
            for r in m1["records"]
            if r["comparison"] == "C2_V030_vs_MAIN" and r["panel"] == panel
        }
        assert c1.keys() == c2.keys()
        assert all(
            c1[k]["candidate_retained_seconds"] == c2[k]["candidate_retained_seconds"] for k in c1
        )
        totals = [
            sum(sum(r["parent_retained_seconds"]) for r in c2.values()),
            sum(sum(r["parent_retained_seconds"]) for r in c1.values()),
            sum(sum(r["candidate_retained_seconds"]) for r in c1.values()),
        ]
        normalized = np.array(totals) / totals[0]
        ax.plot((0, 1, 2), normalized, "o-", color=COLORS["prusie"])
        for i, value in enumerate(normalized):
            ax.annotate(
                f"{value:.3g}",
                (i, value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        ax.set(
            xticks=(0, 1, 2),
            xticklabels=("MAIN", "base", "0.3.0"),
            title=f"{title} (n={len(c1)})",
            ylabel="Total retained fit time / MAIN",
            ylim=(0, max(normalized) * 1.2),
        )
        increments = [
            np.median([np.median(r["parent_retained_incremental_bytes"]) for r in c2.values()]),
            np.median([np.median(r["parent_retained_incremental_bytes"]) for r in c1.values()]),
            np.median([np.median(r["candidate_retained_incremental_bytes"]) for r in c1.values()]),
        ]
        axes[1, column].plot((0, 1, 2), np.array(increments) / 2**20, "o-", color="#009E73")
        axes[1, column].set(
            xticks=(0, 1, 2),
            xticklabels=("MAIN", "base", "0.3.0"),
            ylabel="Median v10 RSS increment (MiB)",
            ylim=(0, max(increments) / 2**20 * 1.1),
        )
    finish(axes)
    save(fig, output, "v030_cumulative_chain")
