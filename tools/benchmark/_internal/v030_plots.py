"""Deterministic benchmark figures from the published 0.3.0 records; no fitting or R."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib import patheffects, ticker
from matplotlib.lines import Line2D

# Okabe-Ito blue/vermillion, shared with the manuscript figures.
COLORS = {"prusie": "#0072B2", "susieR": "#D55E00"}
DARK = {"prusie": "#004a75", "susieR": "#8f3d00"}
LABELS = {"prusie": "prusie 0.3.0", "susieR": "susieR 0.16.6"}
KEYS = {"prusie": "prusie", "susieR": "R"}
HIGHLIGHT = "#CC79A7"
# Analysis-type identity where both panels share one axis; validated as a pair.
PANEL_COLORS = {"GWAS": "#4a3aa7", "eQTL": "#199e70"}
CASE_POINT = "#4d5d6c"
INK = "#1f1f1e"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"
PANELS = ("GWAS", "eQTL")
TITLES = {"GWAS": "GWAS: 200 T2D loci", "eQTL": "eQTL: 14,157 genes"}
BINS = {"GWAS": (7, 5), "eQTL": (14, 20)}
XTICKS = {"GWAS": ((2000, 5000, 10000, 20000), (1400, 22000)), "eQTL": ((10, 100, 1000), (5, 8500))}
WINDOWS = (500, 1000, 2000, 4000, 10000, 19221)
GIB = 2.0**30
HALO = [patheffects.Stroke(linewidth=3.0, foreground="white"), patheffects.Normal()]
TEXT_HALO = [patheffects.Stroke(linewidth=2.5, foreground="white"), patheffects.Normal()]
WIDE = (8.0, 3.3)


def style(salt):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": salt,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save(fig, output, name):
    for ext in ("svg", "pdf"):
        metadata = {"Date": None} if ext == "svg" else {"CreationDate": None, "ModDate": None}
        fig.savefig(output / f"{name}.{ext}", bbox_inches="tight", dpi=300, metadata=metadata)
    plt.close(fig)


def read_jsonl(path):
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def primary(records, panel):
    return [r for r in records if r["kind"] == "primary" and r["panel"] == panel]


def plain_log(axis, ticks):
    axis.set_major_locator(ticker.FixedLocator(ticks))
    axis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.10g}"))
    axis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10)))
    axis.set_minor_formatter(ticker.NullFormatter())


def grid(ax, axis="y"):
    ax.grid(axis=axis, which="major", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def note(ax, text, x=0.03, y=0.97, **kwargs):
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        path_effects=TEXT_HALO,
        zorder=7,
        **kwargs,
    )


def density_scatter(ax, x, y, colour, size=2.0, light=0.82):
    """Points coloured by local log-space density so the dense eQTL cloud keeps its shape."""
    lx, ly = np.log10(x), np.log10(y)
    h, xe, ye = np.histogram2d(lx, ly, bins=(70, 55))
    ix = np.clip(np.digitize(lx, xe) - 1, 0, h.shape[0] - 1)
    iy = np.clip(np.digitize(ly, ye) - 1, 0, h.shape[1] - 1)
    d = np.log1p(h[ix, iy]) / np.log1p(h.max())
    order = np.argsort(d, kind="stable")
    base = np.array(mcolors.to_rgb(colour))
    rgb = base + (1 - base) * (light * (1 - d[order]))[:, None]
    ax.scatter(x[order], y[order], s=size, c=rgb, linewidths=0, rasterized=True, zorder=2)


def binned_median(p, y, n_bins, min_count):
    order = np.argsort(p)
    p, y = p[order], y[order]
    edges = np.geomspace(p.min(), p.max() * (1 + 1e-12), n_bins + 1)
    index = np.digitize(p, edges) - 1
    keep = [i for i in range(n_bins) if (index == i).sum() >= min_count]
    return (
        np.array([np.median(p[index == i]) for i in keep]),
        np.array([np.median(y[index == i]) for i in keep]),
    )


def median_line(ax, x, y, backend):
    ax.plot(
        x,
        y,
        color=DARK[backend],
        lw=1.5,
        marker="o",
        ms=4,
        mfc="white",
        mec=DARK[backend],
        mew=1.1,
        path_effects=HALO,
        zorder=5,
    )


def backend_handles(extra=()):
    handles = [
        Line2D(
            [],
            [],
            color=DARK[b],
            marker="o",
            mfc="white",
            mec=DARK[b],
            mew=1.1,
            lw=1.5,
            ms=4,
            label=LABELS[b],
        )
        for b in ("prusie", "susieR")
    ]
    return handles + list(extra)


def scatter_panel(ax, panel, p, values):
    for backend in ("susieR", "prusie"):
        if panel == "GWAS":
            ax.scatter(
                p, values[backend], s=8, color=COLORS[backend], alpha=0.45, linewidths=0, zorder=2
            )
        else:
            density_scatter(ax, p, values[backend], COLORS[backend])
    for backend in ("susieR", "prusie"):
        median_line(ax, *binned_median(p, values[backend], *BINS[panel]), backend)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks, limits = XTICKS[panel]
    plain_log(ax.xaxis, ticks)
    ax.set_xlim(*limits)
    ax.set_title(TITLES[panel])
    ax.set_xlabel("Variants in region, p")
    grid(ax)


def by_variants(records, output, metric, filename):
    fig, axes = plt.subplots(1, 2, figsize=WIDE, layout="constrained")
    for ax, panel in zip(axes, PANELS, strict=True):
        rows = primary(records, panel)
        p = np.array([r["p"] for r in rows], dtype=float)
        if metric == "memory":
            values = {
                b: np.array([r[KEYS[b]]["peak_rss_diagnostic_bytes"] for r in rows]) / GIB
                for b in KEYS
            }
            ratio = float(np.median(values["susieR"] / values["prusie"]))
            note(ax, f"Median paired susieR/prusie peak ratio {ratio:.1f}×")
            line_p = np.geomspace(p.min(), p.max(), 200)
            ax.plot(line_p, 8 * line_p**2 / GIB, color=INK2, lw=0.9, ls=(0, (4, 2)), zorder=3)
        else:
            field = "times" if metric == "complete" else "fitting_times"
            values = {b: np.array([np.median(r[KEYS[b]][field]) for r in rows]) for b in KEYS}
            total = {b: sum(sum(r[KEYS[b]][field]) for r in rows) for b in KEYS}
            kind = "runtime" if metric == "complete" else "fitting time"
            note(ax, f"Cumulative susieR/prusie {kind} {total['susieR'] / total['prusie']:.1f}×")
        scatter_panel(ax, panel, p, values)
    if metric == "memory":
        plain_log(axes[0].yaxis, (0.1, 0.2, 0.5, 1, 2, 5, 10, 20))
        axes[0].set_ylim(0.1, 40)
        plain_log(axes[1].yaxis, (0.05, 0.1, 0.2, 0.5, 1, 2, 5))
        axes[1].set_ylim(0.05, 5)
        axes[0].set_ylabel("Peak memory (GiB)")
        extra = [
            Line2D(
                [], [], color=INK2, lw=0.9, ls=(0, (4, 2)), label="One dense LD matrix, 8p² bytes"
            )
        ]
    else:
        plain_log(axes[0].yaxis, (0.01, 0.1, 1, 10, 100))
        axes[0].set_ylim(0.01, 800)
        plain_log(axes[1].yaxis, (0.001, 0.01, 0.1, 1, 10, 100))
        axes[1].set_ylim(2e-4, 400)
        axes[0].set_ylabel(
            "Complete-call time (s)" if metric == "complete" else "Fitting-stage time (s)"
        )
        extra = []
    fig.legend(
        handles=backend_handles(extra),
        loc="outside upper center",
        ncol=3,
        handlelength=2.4,
        columnspacing=2.2,
    )
    save(fig, output, filename)


def ratio_distributions(records, output):
    fields = (
        ("times", "Complete-call time"),
        ("fitting_times", "Fitting-stage time"),
        ("memory", "Peak memory"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.9), layout="constrained")
    for ax, (field, title) in zip(axes, fields, strict=True):
        for panel, colour in PANEL_COLORS.items():
            rows = primary(records, panel)
            if field == "memory":
                ratio = np.array(
                    [
                        r["R"]["peak_rss_diagnostic_bytes"]
                        / r["prusie"]["peak_rss_diagnostic_bytes"]
                        for r in rows
                    ]
                )
            else:
                ratio = np.array(
                    [np.median(r["R"][field]) / np.median(r["prusie"][field]) for r in rows]
                )
            ratio = np.sort(ratio)
            ax.step(
                ratio,
                np.arange(1, ratio.size + 1) / ratio.size,
                where="post",
                color=colour,
                lw=1.6,
                label=f"{panel} (n={ratio.size:,})",
            )
            med = float(np.median(ratio))
            ax.plot([med], [0.5], marker="o", ms=4.5, mfc="white", mec=colour, mew=1.2, zorder=5)
            ax.annotate(
                f"{med:.1f}×",
                (med, 0.5),
                xytext=(6 if panel == "GWAS" else -6, 0),
                textcoords="offset points",
                ha="left" if panel == "GWAS" else "right",
                va="center",
                fontsize=8,
                color=INK,
                path_effects=TEXT_HALO,
            )
        ax.axhline(0.5, color=GRID, lw=0.6, zorder=0)
        ax.set_xscale("log")
        if field == "memory":
            plain_log(ax.xaxis, (1, 2, 5, 10))
            ax.set_xlim(1, 10)
        else:
            plain_log(ax.xaxis, (5, 10, 20, 50, 100, 200))
            ax.set_xlim(5, 250)
        ax.set_ylim(0, 1.02)
        ax.set_title(title)
        ax.set_xlabel("susieR / prusie, per case")
        grid(ax, "x")
    axes[0].set_ylabel("Fraction of cases")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, handlelength=2.0)
    save(fig, output, "v030_ratio_distributions")


def overwrite_memory(overwrite, output):
    p = np.array([r["p"] for r in overwrite], dtype=float)
    default = np.array([r["measurements"]["default"]["peak_rss_bytes"] for r in overwrite], float)
    inplace = np.array([r["measurements"]["overwrite"]["peak_rss_bytes"] for r in overwrite], float)
    diff = (inplace - default) / 2**10
    same = diff == 0
    fig, (left, right) = plt.subplots(
        1, 2, figsize=WIDE, layout="constrained", gridspec_kw={"width_ratios": [1.25, 1]}
    )
    left.scatter(
        p,
        default / GIB,
        s=12,
        color=COLORS["prusie"],
        alpha=0.8,
        linewidths=0,
        zorder=3,
        label="overwrite_ld=False",
    )
    left.scatter(
        p,
        inplace / GIB,
        s=30,
        facecolors="none",
        edgecolors=COLORS["susieR"],
        linewidths=0.8,
        zorder=4,
        label="overwrite_ld=True",
    )
    left.set_xscale("log")
    left.set_yscale("log")
    plain_log(left.yaxis, (0.2, 0.5, 1, 2, 5))
    left.set_ylabel("prusie peak memory (GiB)")
    left.set_title(f"Fresh-process peak, {len(p)} GWAS loci")
    left.legend(loc="upper left", handletextpad=0.3, borderaxespad=0.2)
    right.axhline(0, color=INK2, lw=0.8, zorder=2)
    right.scatter(p[same], diff[same], s=10, color=INK2, alpha=0.45, linewidths=0, zorder=3)
    right.scatter(p[~same], diff[~same], s=30, color=HIGHLIGHT, linewidths=0, zorder=4)
    for x, d in zip(p[~same], diff[~same], strict=True):
        right.annotate(
            f"+{d:.0f} KiB",
            (x, d),
            xytext=(-7, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=8,
        )
    right.set_ylim(-4, 16)
    right.set_yticks((0, 4, 8, 12, 16))
    right.set_ylabel("Peak difference, True − False (KiB)")
    right.set_title("Paired difference")
    note(right, f"{int(same.sum())} of {same.size} loci: identical peak")
    for ax in (left, right):
        ax.set_xscale("log")
        plain_log(ax.xaxis, XTICKS["GWAS"][0])
        ax.set_xlim(*XTICKS["GWAS"][1])
        ax.set_xlabel("Variants in region, p")
        grid(ax)
    save(fig, output, "v030_overwrite_memory")


def controlled_scaling(records, replay, output):
    windows = {r["p"]: r for r in records if r["kind"].startswith("window")}
    assert tuple(sorted(windows)) == WINDOWS
    p = np.array(WINDOWS, dtype=float)

    def summary(backend, field):
        rows = []
        for size in WINDOWS:
            if backend == "susieR" and size == replay["p"]:
                key = "complete" if field == "times" else "fitting"
                rows.append((replay[f"{key}_median"], replay[f"{key}_q1"], replay[f"{key}_q3"]))
            else:
                values = windows[size][KEYS[backend]][field]
                rows.append(
                    (np.median(values), np.quantile(values, 0.25), np.quantile(values, 0.75))
                )
        return np.array(rows).T

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.8), layout="constrained")
    labels = ("500", "1k", "2k", "4k", "10k", "19.2k")
    for ax, field, title in (
        (axes[0, 0], "times", "Complete call"),
        (axes[0, 1], "fitting_times", "Fitting stage"),
    ):
        for backend in ("susieR", "prusie"):
            med, q1, q3 = summary(backend, field)
            ax.fill_between(p, q1, q3, color=COLORS[backend], alpha=0.25, lw=0, zorder=2)
            median_line(ax, p, med, backend)
        ax.set_yscale("log")
        plain_log(ax.yaxis, (0.01, 0.1, 1, 10, 100))
        ax.set_ylim(0.005, 1000)
        ax.set_ylabel("Time per call (s)")
        ax.set_title(title)
    memory = axes[1, 0]
    for backend in ("susieR", "prusie"):
        peaks = np.array([windows[s][KEYS[backend]]["peak_rss_diagnostic_bytes"] for s in WINDOWS])
        median_line(memory, p, peaks / GIB, backend)
    memory.plot(p, 8 * p**2 / GIB, color=INK2, lw=0.9, ls=(0, (4, 2)), zorder=3)
    memory.set_yscale("log")
    plain_log(memory.yaxis, (0.002, 0.01, 0.1, 1, 10))
    memory.set_ylim(0.0015, 40)
    memory.set_ylabel("Peak memory (GiB)")
    memory.set_title("Backend-process peak memory")
    iters = np.array([int(np.median(windows[s]["prusie"]["niter"])) for s in WINDOWS])
    assert np.array_equal(iters, [int(np.median(windows[s]["R"]["niter"])) for s in WINDOWS])
    it_ax = axes[1, 1]
    it_ax.plot(p, iters, color=INK2, lw=1.3, marker="o", ms=4, mfc="white", mec=INK2, mew=1.1)
    for x, n in zip(p, iters, strict=True):
        it_ax.annotate(
            str(n),
            (x, n),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    it_ax.set_ylim(0, 34)
    it_ax.set_ylabel("Completed IBSS iterations")
    it_ax.set_title("Iterations (identical for both)")
    for ax in axes.flat:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(ticker.FixedLocator(WINDOWS))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(labels))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.set_xlim(400, 25000)
        ax.set_xlabel("Leading variants of one T2D locus, p")
        grid(ax)
    extra = [Line2D([], [], color=INK2, lw=0.9, ls=(0, (4, 2)), label="One dense LD matrix")]
    fig.legend(
        handles=backend_handles(extra),
        loc="outside upper center",
        ncol=3,
        handlelength=2.4,
        columnspacing=2.2,
    )
    save(fig, output, "v030_controlled_scaling")


def sci(value):
    mantissa, exponent = f"{value:.2g}".split("e")
    return f"{mantissa}×10$^{{{int(exponent)}}}$"


ZERO_ROW = 10**-18.6
FLOOR = 1e-17


def error_axis(ax):
    ax.set_yscale("log")
    ax.set_ylim(10**-19.2, 3e-3)
    ticks = (ZERO_ROW, FLOOR, 1e-14, 1e-11, 1e-8, 1e-5)
    labels = ("0", "≤10$^{-17}$", "10$^{-14}$", "10$^{-11}$", "10$^{-8}$", "10$^{-5}$")
    ax.yaxis.set_major_locator(ticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(labels))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.axhspan(10**-19.2, 10**-17.8, color="#f3f2ef", lw=0, zorder=0)
    for y in ticks[1:]:
        ax.axhline(y, color=GRID, lw=0.6, zorder=0)


def numerical_agreement(records, output):
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.6), layout="constrained", sharex="col")
    rng = np.random.default_rng(7)
    for column, panel in enumerate(PANELS):
        rows = primary(records, panel)
        p = np.array([r["p"] for r in rows], dtype=float)
        limit = np.array([r["both_nonconverged_at_limit"] for r in rows])
        jitter = 10 ** rng.uniform(-0.25, 0.25, p.size)
        for row, (field, label) in enumerate(
            (("max_abs_alpha_error", "Max |Δα|"), ("max_abs_pip_error", "Max |ΔPIP|"))
        ):
            ax = axes[row, column]
            e = np.array([r[field] for r in rows], dtype=float)
            zero = e == 0
            shown = np.where(zero, ZERO_ROW * jitter, np.maximum(e, FLOOR))
            if panel == "GWAS":
                ax.scatter(
                    p[~limit],
                    shown[~limit],
                    s=8,
                    color=CASE_POINT,
                    alpha=0.55,
                    linewidths=0,
                    zorder=2,
                )
            else:
                density_scatter(ax, p[~limit], shown[~limit], CASE_POINT, light=0.8)
            if limit.any():
                ax.scatter(
                    p[limit],
                    shown[limit],
                    s=10,
                    marker="D",
                    facecolor="white",
                    edgecolor=HIGHLIGHT,
                    linewidth=0.7,
                    zorder=5,
                )
            median = float(np.median(e))
            ax.axhline(median, color=INK2, lw=0.7, ls=(0, (1, 1.5)), zorder=3)
            floored = int(((e > 0) & (e < FLOOR)).sum())
            text = f"median {sci(median)}\nexact 0: {int(zero.sum()):,}"
            if floored:
                text += f"; below 10$^{{-17}}$: {floored:,}"
            note(ax, text, y=0.26)
            error_axis(ax)
            ax.set_xscale("log")
            if column == 0:
                ax.set_ylabel(label)
        axes[0, column].set_title(TITLES[panel])
        ticks, limits = XTICKS[panel]
        plain_log(axes[1, column].xaxis, ticks)
        axes[1, column].set_xlim(*limits)
        axes[1, column].set_xlabel("Variants in region, p")
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
    save(fig, output, "v030_numerical_agreement")


def thread_scaling(threads, output):
    fig, ax = plt.subplots(figsize=(5.2, 3.0), layout="constrained")
    rng = np.random.default_rng(11)
    for position, count in enumerate((2, 4, 8)):
        rows = [r for r in threads["records"] if r["comparison"] == f"C4_t{count}_vs_t1"]
        ratio = np.array([r["ratio"] for r in rows])
        for offset, (data_type, colour) in zip((-0.14, 0.14), PANEL_COLORS.items(), strict=True):
            sel = np.array([r["data_type"] == data_type for r in rows])
            x = position + offset + rng.uniform(-0.07, 0.07, sel.sum())
            ax.scatter(
                x,
                ratio[sel],
                s=12,
                color=colour,
                alpha=0.7,
                linewidths=0,
                zorder=3,
                label=f"{data_type} (n={sel.sum()})" if position == 0 else None,
            )
        med = float(np.median(ratio))
        ax.hlines(med, position - 0.32, position + 0.32, color=INK, lw=1.6, zorder=4)
        ax.text(
            position,
            0.43,
            f"median {med:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            path_effects=TEXT_HALO,
        )
    ax.axhline(1, color=INK2, lw=0.8, ls=(0, (4, 2)), zorder=2)
    ax.text(-0.47, 0.99, "no change", va="top", ha="left", fontsize=7.5, color=INK2)
    ax.set_xticks((0, 1, 2), ("2 threads", "4 threads", "8 threads"))
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0.4, 1.15)
    ax.set_ylabel("Time relative to 1 thread")
    ax.set_title(f"prusie n_threads, {len({r['case_id'] for r in threads['records']})} cases")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.97), ncol=2, handletextpad=0.2)
    grid(ax)
    save(fig, output, "thread_scaling")


def plot(assets_dir, output_dir):
    """Render every benchmark figure from the published records in `assets_dir`."""
    assets = Path(assets_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(assets / "benchmark_records.jsonl.gz")
    replay = json.loads((assets / "window10000_retime.json").read_text())
    overwrite = read_jsonl(assets / "benchmark_overwrite_records.jsonl.gz")
    threads = json.loads((assets / "thread_scaling.json").read_text())
    style("prusie-v030-benchmark")
    by_variants(records, output, "complete", "v030_runtime_by_variants")
    by_variants(records, output, "fitting", "benchmark_fitting_by_variants")
    by_variants(records, output, "memory", "v030_memory_by_variants")
    ratio_distributions(records, output)
    overwrite_memory(overwrite, output)
    controlled_scaling(records, replay, output)
    numerical_agreement(records, output)
    thread_scaling(threads, output)
