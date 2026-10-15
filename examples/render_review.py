"""Render the retained review fits; only explicit --refresh-sensitivity refits.

From the repository root, with the analysis extra installed:
    python examples/render_review.py --output-dir output/review --render-source-sha SHA \
        --sensitivity-dir docs/review/post_analysis_fixes/examples
This reads the saved alternative fits as well as the original baseline,
refinement posterior and LD diagnostics. Only an explicit
``--refresh-sensitivity`` recomputes alternative fits; it cannot be combined
with ``--sensitivity-dir``.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from post_analysis import make_figures

import prusie
from prusie.diagnostics import resolve_convergence
from prusie.report import _plain, load_analysis, save_analysis, write_report

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / "docs/review/post_analysis/examples"
NUMERIC_SOURCE = "e45914e512e04feef37cf7310bc0b309582d6058"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, value):
    path.write_text(
        json.dumps(_plain(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def render(case, output, render_source, refresh, sensitivity_dir=None):
    if refresh and sensitivity_dir is not None:
        raise ValueError("--refresh-sensitivity cannot be combined with --sensitivity-dir")
    old = HISTORICAL / case
    fits = load_analysis(old / "fits")
    fit, refined = fits["baseline"], fits["refined"]
    output.mkdir(parents=True, exist_ok=True)
    e = prusie.load_example() if case == "stable" else prusie.load_convergence_example()
    inputs, params = e["inputs"], dict(e["parameters"])
    if case == "stable":
        R, z = inputs["R"], inputs["z"]
    else:
        diagonal = np.sqrt(np.diag(inputs["XtX"]))
        R = inputs["XtX"] / diagonal[:, None] / diagonal[None, :]
        beta = inputs["Xty"] / np.diag(inputs["XtX"])
        se = np.sqrt(
            (inputs["yty"] - beta * inputs["Xty"]) / (inputs["n"] - 2) / np.diag(inputs["XtX"])
        )
        z = beta / se
    control, resolution = resolve_convergence(fit)
    if refresh:
        if case == "stable":
            alternatives = prusie.run_sensitivity(
                inputs,
                [
                    {"name": "larger_L", "changes": {"L": params["L"] + 1}},
                    {
                        "name": "fixed_variance",
                        "changes": {
                            "scaled_prior_variance": 0.05,
                            "estimate_prior_variance": False,
                        },
                    },
                    {"name": "coverage90", "report": {"coverage": 0.9}},
                ],
                baseline=fit,
                baseline_settings=params,
            )["fits"]
            alternatives = {key: value for key, value in alternatives.items() if key != "baseline"}
        else:
            alternatives = {
                "larger_L": prusie.susie_suff_stat(
                    **inputs, **(params | dict(L=12, convergence=control))
                )
            }
        save_analysis(alternatives, output / "sensitivity_fits")
        dump(
            output / "sensitivity_source.json",
            dict(
                source_sha=render_source,
                version=prusie.__version__,
                recomputed_fits=["larger_L", "fixed_variance"]
                if case == "stable"
                else ["larger_L"],
                report_only=["coverage90"] if case == "stable" else [],
                requested_convergence=control.options(),
                convergence_resolution=resolution,
            ),
        )
    saved = output if sensitivity_dir is None else sensitivity_dir / case
    alternatives = load_analysis(saved / "sensitivity_fits")
    sensitivity_source = json.loads((saved / "sensitivity_source.json").read_text())
    compared = (
        {"baseline": fit} | alternatives
        if case == "stable"
        else {"baseline": fit, "refined": refined} | alternatives
    )
    comparison = prusie.compare_fits(compared, baseline="baseline", R=R)
    previous = json.loads((old / "comparison.json").read_text())
    previous_pip = {row["variant_id"]: row["pips"] for row in previous["variants"]}
    max_pip_difference = max(
        abs(row["pips"][setting] - previous_pip[row["variant_id"]][setting])
        for row in comparison["variants"]
        for setting in comparison["settings"]
    )
    # Record observed agreement; do not infer it merely from the unchanged inputs.
    dump(
        output / "sensitivity_validation.json",
        dict(
            old_numeric_source=NUMERIC_SOURCE,
            new_numeric_source=sensitivity_source["source_sha"],
            maximum_absolute_pip_difference=max_pip_difference,
            variants=len(comparison["variants"]),
            settings=comparison["settings"],
        ),
    )
    provenance = json.loads((old / "provenance.json").read_text())
    provenance.update(
        numeric_source_sha=NUMERIC_SOURCE,
        render_source_sha=render_source,
        render_version=prusie.__version__,
        sensitivity_source=sensitivity_source,
        elapsed_seconds_scope="Original numeric analysis; not this rendering pass",
        reused=[
            "baseline and refined posterior/trace from original saved fits",
            "original LD diagnostics",
        ],
        recomputed=[
            "posterior summary from saved fit",
            "signal/component comparison",
            "figures",
            "HTML",
            "compact analysis summary",
            "evidence index",
        ],
        source_files={
            (
                str(path.relative_to(ROOT))
                if path.is_relative_to(ROOT)
                else "saved_sensitivity/" + case + "/" + str(path.relative_to(saved))
            ): sha(path)
            for path in [
                old / "fits/analysis.json",
                old / "fits/arrays.npz",
                old / "ld_diagnostics.json",
                saved / "sensitivity_fits/analysis.json",
                saved / "sensitivity_fits/arrays.npz",
                saved / "sensitivity_source.json",
            ]
        },
    )
    summary = prusie.summarize(
        refined,
        R,
        z=z,
        input_info={
            "ld_source": provenance["reference_sources"],
            "in_sample_ld": True,
            "genome_build": "not_assessed",
        },
    )
    ld = json.loads((old / "ld_diagnostics.json").read_text())
    figures = make_figures(fit, refined, summary, R, ld, comparison)
    receipt = write_report(
        summary,
        output,
        figures=figures,
        ld=ld,
        comparison=comparison,
        provenance=provenance,
        annotations={
            "example_kind": "synthetic toy example"
            if case == "stable"
            else (
                "official convergence sufficient-statistics example; no "
                "biological mechanism inferred"
            )
        },
        title=f"prusie {case}: fine-mapping results",
    )
    if not (output / "figure_index.json").exists():
        dump(
            output / "figure_index.json",
            [
                dict(title=label, png=f"figure_{i}.png", svg=f"figure_{i}.svg")
                for i, label in enumerate(figures, 1)
            ],
        )
    dump(output / "plot_data.json", {label: parts[2] for label, parts in figures.items()})
    for name, value in [
        ("posterior_summary", summary),
        ("comparison", comparison),
        ("ld_diagnostics", ld),
        ("provenance", provenance),
    ]:
        dump(output / f"{name}.json", value)
    # Lossless exported structures are identical after removing JSON whitespace.
    for path in [output / "analysis.json", output / "evidence.json"]:
        dump(path, json.loads(path.read_text()))
    import matplotlib.pyplot as plt

    plt.close("all")
    return dict(
        case=case,
        figures=len(figures),
        maximum_sensitivity_pip_difference=max_pip_difference,
        evidence_sha256=receipt.get("evidence_sha256", None),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render-source-sha", required=True)
    parser.add_argument("--refresh-sensitivity", action="store_true")
    parser.add_argument(
        "--sensitivity-dir",
        type=Path,
        help=(
            "Read saved case/sensitivity_fits from this directory; no fits or inputs are changed."
        ),
    )
    parser.add_argument("--case", choices=["stable", "problem", "both"], default="both")
    args = parser.parse_args()
    if len(args.render_source_sha) != 40 or any(
        c not in "0123456789abcdef" for c in args.render_source_sha
    ):
        parser.error("--render-source-sha must be the full source commit ID")
    if args.refresh_sensitivity and args.sensitivity_dir is not None:
        parser.error("--refresh-sensitivity cannot be combined with --sensitivity-dir")
    for case in ["stable", "problem"] if args.case == "both" else [args.case]:
        print(
            json.dumps(
                render(
                    case,
                    args.output_dir / case,
                    args.render_source_sha,
                    args.refresh_sensitivity,
                    args.sensitivity_dir,
                )
            )
        )


if __name__ == "__main__":
    main()
