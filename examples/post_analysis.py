"""Run the official susieR FinemappingConvergence toy example, entirely offline.

python examples/post_analysis.py --output-dir output/post-analysis --case official
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

import prusie
from prusie.report import load_analysis, save_analysis, write_report
from prusie.viz import (
    plot_convergence,
    plot_ld_diagnostics,
    plot_locus,
    plot_prior_influence,
    plot_refinement,
    plot_sensitivity,
    plot_signal,
)


def checksum(a):
    a = np.asarray(a)
    return hashlib.sha256(a.tobytes(order="C")).hexdigest()


def source_identity():
    """Identify example, refinement and rendering source separately from the native core."""
    import prusie._native

    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    sources = [
        Path(__file__),
        Path(prusie.__file__).with_name("refinement.py"),
        Path(prusie.__file__).with_name("report.py"),
        Path(prusie.__file__).parent / "viz/_locus.py",
    ]
    return dict(
        checkout_head=revision.stdout.strip() if revision.returncode == 0 else "not_assessed",
        source_files_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        native_sha256=hashlib.sha256(Path(prusie._native.__file__).read_bytes()).hexdigest(),
        note=(
            "Source hashes identify the example, refinement tracking and presentation; "
            "this record is not a final wheel freeze."
        ),
    )


def make_figures(fit, refined, summary, R, ld, comparison):
    "Screen-sized views, with every retained CS and separate variance details."
    matched = prusie.compare_fits({"before": fit, "after": refined}, baseline="before", R=R)
    display_names = {"baseline": "Baseline", "refined": "After refinement", "larger_L": "L = 12"}
    figures = {
        "locus overview": plot_locus(summary, R=R, layout="compact"),
        "convergence before and after": plot_convergence(fit, after=refined, layout="compact"),
        "refinement": plot_refinement(fit, refined, comparison=matched, layout="compact"),
        "LD diagnostics": plot_ld_diagnostics(ld, layout="compact"),
        "sensitivity main signals": plot_sensitivity(
            comparison, candidates_only=True, layout="compact", display_names=display_names
        ),
    }
    components = sorted(summary["credible_sets"], key=lambda cs: int(cs["component_id"]))
    for cs in components:
        component = cs["component_id"]
        detail = next(row for row in summary["components"] if row["component_id"] == component)
        candidates = list(dict.fromkeys(cs["members"] + detail["ranking"][:10]))
        figures[f"signal detail component {component}"] = plot_signal(
            summary, component, R=R, max_variants=12, layout="compact", ld_measure="r2"
        )
        figures[f"prior influence component {component} (CS and top 10)"] = plot_prior_influence(
            summary, component, variant_ids=candidates, layout="compact"
        )
    if not components:
        figures["signal detail"] = plot_signal(summary, R=R)
        figures["prior influence"] = plot_prior_influence(summary)
    figures["variance details before and after"] = plot_convergence(
        fit, after=refined, layout="variance"
    )
    return figures


def export_records(output, records):
    """Readable full-precision records alongside lossless analysis persistence."""
    for name, value in records.items():
        (output / f"{name}.json").write_text(
            json.dumps(value, default=lambda item: item.tolist(), allow_nan=False, indent=2) + "\n"
        )


def write_source_assets(output):
    """Keep fixture attribution bytes and supply full terms from repository notices."""
    import shutil
    from importlib.resources import files

    output.mkdir(parents=True, exist_ok=True)
    for name in ("metadata.json", "LICENSE.txt", "LICENSE-BSD-3-Clause.txt"):
        shutil.copyfile(files("prusie").joinpath("data/convergence", name), output / name)
    (output / "README.txt").write_text(
        "LICENSE.txt and LICENSE-BSD-3-Clause.txt preserve the two identical attribution "
        "stubs from the frozen fixture. BSD-3-Clause.txt supplies the complete license terms.\n"
    )
    notices = Path(__file__).resolve().parents[1] / "THIRD_PARTY_NOTICES.md"
    terms = notices.read_text().split("### susieR\n", 1)[1].split("### R Brent_fmin", 1)[0]
    full = terms[terms.index("Copyright (c)") :].strip() + "\n"
    (output / "BSD-3-Clause.txt").write_text(full)
    return dict(
        full_terms="BSD-3-Clause.txt",
        full_terms_source="THIRD_PARTY_NOTICES.md / susieR",
        attribution="LICENSE.txt",
        fixture_license_stub="LICENSE-BSD-3-Clause.txt",
    )


def execute(case, output):
    started = time.perf_counter()
    e = prusie.load_convergence_example()
    inputs = e["inputs"]
    params = e["parameters"]
    fit = prusie.susie_suff_stat(**inputs, **params, track_fit=True)
    refined = prusie.refine_fit(fit, prusie.RefinementConfig(inner_tol=0.001), track_fit=True)
    diagonal = np.sqrt(np.diag(inputs["XtX"]))
    R = inputs["XtX"] / diagonal[:, None] / diagonal[None, :]
    # Marginal t statistics on the supplied response scale, with n - 2 df.
    beta = inputs["Xty"] / np.diag(inputs["XtX"])
    se = np.sqrt(
        (inputs["yty"] - beta * inputs["Xty"]) / (inputs["n"] - 2) / np.diag(inputs["XtX"])
    )
    z = beta / se
    n = inputs["n"]
    alternate = prusie.susie_suff_stat(**inputs, **dict(params, L=12))
    comparison = prusie.compare_fits(
        {"baseline": fit, "refined": refined, "larger_L": alternate}, R=R
    )
    source = e["metadata"]["source_url"]
    annotations = {
        "example_kind": e["metadata"]["description"],
        "association_source": (
            "Two-sided normal approximation applied to marginal t statistics derived "
            "from XtX, Xty and yty (df = n - 2); these are not external GWAS z scores."
        ),
        "ld_diagnostic_scope": (
            "In-sample signed LD r is normalized from the same XtX. Diagnostics use "
            "the t-derived statistics with a normal approximation; the near-singular "
            "matrix requires the recorded regularization. No flags does not validate "
            "an external reference; paired susieR flip-flag agreement was not tested."
        ),
        "report_notes": [
            e["metadata"]["description"],
            "Component IDs are original zero-based fit rows. Variant IDs are zero-based "
            "positional indices; genomic positions, alleles and genome build are unavailable.",
            "PIP, alpha, signed r and displayed r² are unitless. Effect estimates use "
            "the scale of the supplied sufficient statistics. CS count is not a causal count.",
        ],
    }
    summary = prusie.summarize(
        refined,
        R,
        z=z,
        input_info={"ld_source": source, "in_sample_ld": True, "genome_build": "not_assessed"},
    )
    ld = prusie.diagnose_ld(z, R, n, variant_ids=fit.variant_ids)
    figures = make_figures(fit, refined, summary, R, ld, comparison)
    license_files = write_source_assets(output)
    provenance = dict(
        data_checksums={k: checksum(v) for k, v in inputs.items()},
        variant_harmonization=(
            "Input order preserved; allele labels unavailable in this toy fixture."
        ),
        parameters=params,
        tracking=dict(baseline=True, refinement=True, candidate_trajectory="original IBSS sweeps"),
        initial_state=("cold start; refinement stage B uses stage A posterior moments"),
        reference_sources=[source],
        software_versions={
            "prusie": prusie.__version__,
            "numpy": np.__version__,
            "native": fit.backend_version,
        },
        random_seed=e["metadata"].get("seed", "frozen upstream fixture; no new random generation"),
        elapsed_seconds=time.perf_counter() - started,
        fixture=e["metadata"],
        license_files=license_files,
        source_identity=source_identity(),
        skipped=[
            "gene track: no physical coordinate metadata",
        ],
    )
    receipt = write_report(
        summary,
        output,
        figures=figures,
        ld=ld,
        comparison=comparison,
        provenance=provenance,
        annotations=annotations,
        title="Toy example: susieR FinemappingConvergence",
    )
    save_analysis({"baseline": fit, "refined": refined}, output / "fits")
    restored = load_analysis(output / "fits")
    assert restored["refined"].alpha.tobytes() == refined.alpha.tobytes()
    save_analysis(
        {"baseline": fit, "refined": refined, "larger_L": alternate}, output / "sensitivity_fits"
    )
    gain = prusie.evaluate_objective(refined)["elbo"] - prusie.evaluate_objective(fit)["elbo"]
    assert gain > 17
    np.testing.assert_allclose(refined.pip, e["reference"]["latest_refined_pip"], atol=1e-5, rtol=0)
    receipt["recomputed_objective_gain"] = gain
    export_records(
        output,
        dict(
            provenance=provenance,
            comparison=comparison,
            posterior_summary=summary,
            ld_diagnostics=ld,
        ),
    )
    receipt.update(
        case=case,
        ibss_converged=refined.converged,
        refinement_status=refined.refinement_status,
        fit_module=prusie.__file__,
        elapsed_seconds=time.perf_counter() - started,
    )
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    import matplotlib.pyplot as plt

    plt.close("all")
    return receipt


def render_saved(case, source, output):
    """Render a saved summary, diagnostics and posterior without fitting again."""
    import shutil

    record = load_analysis(source)
    fits = load_analysis(source / "fits")
    example = prusie.load_convergence_example()
    inputs = example["inputs"]
    diagonal = np.sqrt(np.diag(inputs["XtX"]))
    R = inputs["XtX"] / diagonal[:, None] / diagonal[None, :]
    figures = make_figures(
        fits["baseline"],
        fits["refined"],
        record["summary"],
        R,
        record["ld"],
        record["comparison"],
    )
    if source.resolve() != output.resolve():
        shutil.copytree(source, output, dirs_exist_ok=True)
    license_files = write_source_assets(output)
    provenance = dict(record["provenance"])
    provenance["license_files"] = license_files
    provenance.update(
        render_version=prusie.__version__,
        render_source_identity=source_identity(),
        rendering_scope="Saved numerical fits/diagnostics reused; no refitting",
    )
    receipt = write_report(
        record["summary"],
        output,
        figures=figures,
        **{key: record[key] for key in ("ld", "comparison", "annotations")},
        provenance=provenance,
        title="Toy example: susieR FinemappingConvergence",
    )
    export_records(
        output,
        dict(
            provenance=provenance,
            comparison=record["comparison"],
            posterior_summary=record["summary"],
            ld_diagnostics=record["ld"],
        ),
    )
    import matplotlib.pyplot as plt

    plt.close("all")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", choices=("official",), default="official")
    parser.add_argument(
        "--saved-dir",
        type=Path,
        help="Render saved case directories, using their recorded summaries and fits.",
    )
    args = parser.parse_args()
    if args.saved_dir is None:
        receipt = execute(args.case, args.output_dir / args.case)
    else:
        receipt = render_saved(args.case, args.saved_dir / args.case, args.output_dir / args.case)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
