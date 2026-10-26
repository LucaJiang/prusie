#!/usr/bin/env python3
"Regenerate fine-mapping tables, records and vector figures from retained data."

from __future__ import annotations

import csv
import io
import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/benchmark/_internal"))
from render_tables import render  # noqa: E402 - Local benchmark tools.
from summary import FIELDS, paired, summarize  # noqa: E402 - Local tool path must be set first.


def text_json(value):
    return json.dumps(value, indent=2, allow_nan=False) + "\n"


def table(headers, rows):
    # A vertical bar in a scientific label is text, not another table column.
    def cell(value):
        return str(value).replace("|", "&#124;")

    return (
        "\n| "
        + " | ".join(map(cell, headers))
        + " |\n| "
        + " | ".join(["---"] * len(headers))
        + " |\n"
        + "".join("| " + " | ".join(map(cell, row)) + " |\n" for row in rows)
    )


def f(x):
    return f"{x:.3g}"


def main():
    current = ROOT / "docs/assets/benchmark/v030_summary.json"
    if current.exists():
        import argparse

        from v030_tables import render as render_v030

        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--pages", help="Comma-separated repository-relative pages; omit for all blocks"
        )
        parser.add_argument("--check", action="store_true")
        args = parser.parse_args()
        changed = render_v030(
            ROOT, pages=args.pages.split(",") if args.pages else None, check=args.check
        )
        print("Current 0.3.0 blocks:", len(changed), "stale" if args.check else "updated")
        return
    if len(sys.argv) > 1:
        raise SystemExit(
            "Page-scoped/check regeneration requires the current 0.3.0 retained records"
        )
    evidence = json.loads((ROOT / "docs/assets/benchmark_records.json").read_text())
    cases = [paired(c) for c in evidence["cases"] if c["stage"] == "C"]
    assert len(cases) == len(evidence["cases"]), "Canonical records must contain fine-mapping only"
    groups = summarize(cases)
    real = ["GWAS", "eQTL"]
    outputs = {}

    def out(name, value):
        outputs[ROOT / name] = value.encode() if isinstance(value, str) else value

    def fragment(file, key, value, prefix="generated"):
        path = ROOT / file
        s = outputs.get(path, path.read_bytes()).decode()
        pattern = (
            rf"<!-- {prefix}:{key}"
            rf":start -->.*?<!-- {prefix}:{key}"
            rf":end -->"
        )
        assert re.search(pattern, s, re.S), f"Missing marker {file}:{key}"
        s = re.sub(
            pattern,
            lambda m: (
                f"<!-- {prefix}:{key}:start -->\n{value.rstrip()}\n<!-- {prefix}:{key}:end -->"
            ),
            s,
            flags=re.S,
        )
        out(file, s)

    summary = dict(
        schema_version=1,
        operators=dict(
            runtime_speedup=(
                "exp(mean(log(median_repeat_time_susieR / median_repeat_time_prusie)))"
            ),
            absolute_runtime=("Across-case median of within-case retained-repeat medians, seconds"),
            peak_memory=(
                "Across-case median of independent full-process peak resident set size, MiB"
            ),
            memory_ratio="Median of per-case peak_susieR / peak_prusie",
            memory_reduction="Median of per-case 1 - peak_prusie / peak_susieR",
            error=(
                "Maximum absolute difference over all valid case/array "
                "elements; zero-based index retained"
            ),
        ),
        provenance=evidence["provenance"],
        groups=groups,
    )
    out("docs/assets/r_comparison_summary.json", text_json(summary))
    out(
        "docs/assets/r_comparison_numerical.json",
        text_json(
            [
                dict(
                    case_id=c["case_id"],
                    stage="C",
                    panel=c["panel"],
                    group=c["group"],
                    n_variants=c["n_variants"],
                    **c["numerical"],
                )
                for c in cases
            ]
        ),
    )
    out(
        "docs/assets/r_comparison_protocol.json",
        text_json(dict(provenance=evidence["provenance"], environments=evidence["environments"])),
    )
    repeats = [
        dict(
            case_id=c["case_id"],
            stage="C",
            panel=c["panel"],
            group=c["group"],
            backend=k,
            repeat=i + 1,
            time_seconds=t,
        )
        for c in cases
        for k, b in c["backends"].items()
        for i, t in enumerate(b["repeat_seconds"])
    ]
    out("docs/assets/r_comparison_repeats.json", text_json(repeats))
    buf = io.StringIO()
    columns = [
        "case_id",
        "panel",
        "group",
        "n_variants",
        "max_iter",
        "timing_eligible",
        "prusie_niter",
        "susieR_niter",
        "prusie_converged",
        "susieR_converged",
        "prusie_seconds",
        "susieR_seconds",
        "speedup",
        "prusie_peak_mib",
        "susieR_peak_mib",
        "memory_ratio",
        "memory_reduction",
        "max_abs_pip_error",
    ]
    writer = csv.DictWriter(buf, columns, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for c in cases:
        row = {
            k: c[k]
            for k in (
                "case_id",
                "panel",
                "group",
                "n_variants",
                "max_iter",
                "timing_eligible",
                "speedup",
                "memory_ratio",
                "memory_reduction",
            )
        }
        for k in ("prusie", "susieR"):
            b = c["backends"][k]
            row.update(
                {
                    k + "_niter": b["niter"],
                    k + "_converged": b["converged"],
                    k + "_seconds": b.get("runtime_seconds", {}).get("median"),
                    k + "_peak_mib": b["peak_memory_mib"],
                }
            )
        row["max_abs_pip_error"] = (
            c["numerical"].get("errors", {}).get("pip", {}).get("max_abs_error")
        )
        writer.writerow(row)
    out("docs/assets/r_comparison_cases.tsv", buf.getvalue())
    earlier = table(
        ["Analysis", "Speed-up (geometric mean)", "Peak RSS ratio (median)"],
        [
            [
                g,
                f"{groups[g]['runtime_speedup_geometric_mean']:.2g}×",
                f"{groups[g]['memory_ratio_distribution']['median']:.2g}×",
            ]
            for g in real
        ],
    )
    fragment("docs/performance.md", "earlier", earlier)
    render(ROOT, fragment, table)
    panel = evidence["provenance"]["real_panel"]
    panel_text = (
        f"The data cover {panel['regions']}"
        f" genomic windows: 100 analyses of one "
        f"FinnGen R10 inflammatory "
        f"bowel disease GWAS (K11_IBD_STRICT; "
        f"n = 412181) and 100 "
        f"BLUEPRINT monocyte eQTL analyses (QTD000021; "
        f"n = 191), "
        f"representing {panel['distinct_genes']}"
        f" distinct genes. The interval metadata "
        f"contain {panel['overlapping_interval_pairs']}"
        f" overlapping window pairs. Analyses "
        f"share study participants "
        f"and reference samples; the 200 trait "
        f"analyses are not 200 "
        f"independent loci."
    )
    fragment("docs/performance.md", "panel", panel_text)
    maxima = table(
        [
            "Analysis",
            "Analyses",
            "Variants per analysis",
            "Maximum absolute PIP difference",
            "Trace: case / variant index (zero-based)",
        ],
        [
            [
                g,
                groups[g]["valid_numerical_analyses"],
                "–".join(map(str, groups[g]["variant_count_range"])),
                f(groups[g]["errors"]["pip"]["max_abs_error"]),
                (
                    f"{groups[g]['errors']['pip']['case_id']}"
                    f" / {groups[g]['errors']['pip']['index_0based'][0]}"
                ),
            ]
            for g in real
        ],
    )
    maxima += "\n" + " ".join(
        (
            f"Across the evaluated {g}"
            f" analyses, the maximum absolute difference "
            f"in variant PIPs "
            f"relative to susieR 0.16.6 was {groups[g]['errors']['pip']['max_abs_error']:.2e}"
            f"."
        )
        for g in real
    )
    fragment("docs/numerical_accuracy.md", "maxima", maxima)
    units = {
        "alpha": "Probability",
        "mu": "Standardized effect",
        "mu2": "Squared standardized effect",
        "lbf_variable": "Natural-log BF",
        "lbf": "Natural-log BF",
        "V": "Squared effect",
        "sigma2": "Phenotype variance",
        "elbo": "Natural-log objective",
        "KL": "Natural-log divergence",
    }
    errors = table(
        [
            "Quantity",
            "Scale / units",
            "GWAS maximum absolute difference",
            "eQTL maximum absolute difference",
        ],
        [
            [k, units[k], *[f(groups[g]["errors"][k]["max_abs_error"]) for g in real]]
            for k in FIELDS
            if k != "pip"
        ],
    )
    errors += (
        "\nMaxima are computed elementwise across every valid "
        "analysis, including iteration-limit outputs. The "
        "machine summary includes the case, array index, both "
        "values at the maximum and reference magnitude for each "
        "field. ELBO traces are compared only with matching "
        "shapes; per-case records retain any shape mismatch.\n"
    )
    fragment("docs/numerical_accuracy.md", "errors", errors)
    csrows = []
    discrepancies = []
    for g in real:
        rs = [c for c in cases if c["group"] == g]
        cr = [d for c in rs for d in c["numerical"].get("credible_set_differences", [])]
        mismatch = [
            (c["case_id"], d["component_id_0based"])
            for c in rs
            for d in c["numerical"].get("credible_set_differences", [])
            if not d.get("members_match", False)
        ]
        discrepancies += mismatch
        csrows.append(
            [
                g,
                sum(len(c["numerical"]["credible_sets"]["prusie"]) for c in rs),
                sum(len(c["numerical"]["credible_sets"]["susieR"]) for c in rs),
                len(mismatch),
                f(max((d.get("posterior_mass_abs_error", 0.0) for d in cr), default=0.0)),
                f(
                    max(
                        (max(d.get("purity_abs_error", {}).values(), default=0.0) for d in cr),
                        default=0.0,
                    )
                ),
            ]
        )
    cs = table(
        [
            "Analysis",
            "prusie sets",
            "susieR sets",
            "Component/member differences",
            "Maximum mass difference",
            "Maximum purity difference",
        ],
        csrows,
    )
    cs += (
        "\nSets are matched by original component ID and "
        "compared as member sets; returned ordering does not "
        "affect the comparison. All actual masses, purity "
        "statistics and zero-based members are retained.\n"
    )
    if discrepancies:
        cs += (
            "\nComponent/member discrepancies: "
            + ", ".join(f"{c}, component {k}" for c, k in discrepancies)
            + ".\n"
        )
    fragment("docs/numerical_accuracy.md", "cs", cs)
    limits = [
        c
        for c in cases
        if c["panel"] == "real" and c["case_id"] in groups[c["group"]]["iteration_limit_cases"]
    ]
    limit = "; ".join(
        (
            f"{c['case_id']}"
            f": prusie {c['backends']['prusie']['niter']}"
            f" iterations, susieR {c['backends']['susieR']['niter']}"
            f" iterations, both unconverged"
        )
        for c in limits
    )
    fragment(
        "docs/numerical_accuracy.md",
        "limits",
        limit
        + (
            ". Both calls completed and their outputs enter the "
            "numerical maxima and timing summaries."
        ),
    )
    # Deterministic scientific vector figures.
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 10,
            "svg.hashsalt": "prusie-standalone-1",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    colors = ["#0072B2", "#D55E00"]

    def save(fig, name):
        for ext in ("svg", "pdf"):
            b = io.BytesIO()
            md = {"Date": None} if ext == "svg" else {"CreationDate": None, "ModDate": None}
            fig.savefig(b, format=ext, bbox_inches="tight", metadata=md)
            out(f"docs/assets/{name}.{ext}", b.getvalue())
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3), layout="constrained")
    for ax, g, color in zip(axes, real, colors, strict=True):
        rows = [c for c in cases if c["group"] == g and c["numerical"]["outputs_valid"]]
        ax.scatter(
            [c["n_variants"] for c in rows],
            [c["numerical"]["errors"]["pip"]["max_abs_error"] for c in rows],
            s=15,
            color=color,
            alpha=0.65,
        )
        ax.set(xlabel="Variants in analysis", ylabel="Maximum absolute PIP difference", title=g)
        ax.set_yscale("symlog", linthresh=1e-14)
        ax.set_xscale("log")
        ax.grid(alpha=0.15)
    save(fig, "pip_accuracy")
    toy = ROOT / "docs/assets/toy_result.json"
    if toy.exists():
        t = json.loads(toy.read_text())
        ref = np.load(ROOT / "src/prusie/data/toy/reference.npz")
        inp = np.load(ROOT / "src/prusie/data/toy/inputs.npz")
        fig, axes = plt.subplots(2, 1, figsize=(8.6, 4.6), sharex=True, layout="constrained")
        x = np.arange(1, 501)
        axes[0].scatter(x, inp["z"], s=8, color=colors[0])
        axes[0].set(ylabel="Signed z")
        axes[1].scatter(x, t["pip"], s=12, color=colors[0], label="prusie")
        axes[1].plot(x, ref["pip"], color=colors[1], lw=0.7, alpha=0.7, label="susieR")
        axes[1].set(xlabel="Synthetic variant index (one-based)", ylabel="PIP", ylim=(-0.03, 1.05))
        axes[1].legend(frameon=False)
        for ax in axes:
            for pos in [100, 350]:
                ax.axvline(pos, color="#777777", ls="--", lw=0.7)
        save(fig, "toy_fit")
        info = (
            f"The independent 500-variant toy example "
            f"fit converged in "
            f"{t['niter']}"
            f" iterations and returned {len(t['cs'])}"
            f" credible sets. Its maximum absolute "
            f"PIP difference from "
            f"the separately generated susieR 0.16.6 "
            f"reference was {t['comparison']['errors']['pip']['max_abs_error']:.3g}"
            f"."
        )
        for page in ["docs/example.md", "docs/numerical_accuracy.md"]:
            fragment(page, "toy", info)
        cs = table(
            ["Original component", "Synthetic variants", "Returned posterior mass", "Minimum |r|"],
            [
                [
                    c["component_id_0based"],
                    ", ".join(f"syn{j + 1:04d}" for j in c["members_0based"]),
                    f"{c['coverage']:.6f}",
                    f"{c['purity']['min_abs_corr']:.6f}",
                ]
                for c in t["cs"]
            ],
        )
        fragment("docs/example.md", "toy-cs", cs)
    for path, payload in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    print(
        "Generated",
        len(outputs),
        "outputs;",
        len(cases),
        "fine-mapping cases;",
        len(repeats),
        "retained repeats",
    )


if __name__ == "__main__":
    main()
