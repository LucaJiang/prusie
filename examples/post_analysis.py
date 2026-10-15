"""Run stable and local-optimum toy examples, entirely offline.

python examples/post_analysis.py --output-dir output/post-analysis --case both
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import prusie
from prusie.plots import (
    plot_convergence,
    plot_ld_diagnostics,
    plot_locus,
    plot_prior_influence,
    plot_refinement,
    plot_sensitivity,
    plot_signal,
)
from prusie.report import load_analysis, save_analysis, write_report


def checksum(a):
    a = np.asarray(a)
    return hashlib.sha256(a.tobytes(order="C")).hexdigest()


def make_figures(fit, refined, summary, R, ld, comparison):
    "Screen-sized views, with every retained CS and separate variance details."
    matched = prusie.compare_fits({"before": fit, "after": refined}, baseline="before", R=R)
    display_names = {
        "baseline": "Baseline",
        "refined": "After refinement",
        "coverage90": "CS coverage\n90%",
    }
    for name, settings in comparison.get("fit_settings", {}).items():
        if name == "larger_L" and "L" in settings:
            display_names[name] = f"L = {settings['L']}"
    display_names["fixed_variance"] = "Scaled prior variance\n0.05 (fixed)"
    display_names = {
        name: label
        for name, label in display_names.items()
        if name in comparison.get("settings", [])
    }
    figures = {
        "locus overview": plot_locus(summary, R=R, layout="compact"),
        "convergence before and after": plot_convergence(fit, after=refined, layout="compact"),
        "refinement": plot_refinement(fit, refined, comparison=matched, layout="compact"),
        "LD diagnostics": plot_ld_diagnostics(ld, layout="compact"),
        "sensitivity main signals": plot_sensitivity(
            comparison, candidates_only=True, layout="compact", display_names=display_names
        ),
    }
    components = sorted(
        summary["credible_sets"],
        key=lambda cs: (
            -max(row["pip"] for row in summary["variants"] if row["variant_id"] in cs["members"]),
            cs["component_id"],
        ),
    )
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


def interpret_saved_report(output, model):
    "Save optional AI output separately, citing the already written evidence."
    from prusie.ai import CodexCLI

    bundle = json.loads((output / "evidence.json").read_text(encoding="utf-8"))
    sources = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in ("evidence.json", "analysis_summary.json", "summary.txt")
    }
    try:
        interpretation = CodexCLI(
            output / "ai", model=model, provider="openai", max_attempts=1
        ).interpret(bundle, consent=True)
    except Exception as exc:
        # Optional service failures must not discard the completed analysis.
        interpretation = {
            "status": "failed",
            "reason": str(exc),
            "error_type": type(exc).__name__,
            "transmitted": None,
            "model_requested": model,
        }
    interpretation["evidence_sha256"] = bundle["sha256"]
    interpretation["source_files_sha256"] = sources
    (output / "ai_interpretation.json").write_text(
        json.dumps(interpretation, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return interpretation


def execute(case, output, interpret=False, model=None):
    if interpret and case != "stable":
        raise ValueError("Optional interpretation requires the synthetic stable example")
    if interpret and (not isinstance(model, str) or not model.strip()):
        raise ValueError("Optional interpretation requires an explicit model")
    started = time.perf_counter()
    if case == "stable":
        e = prusie.load_example()
        inputs = e["inputs"]
        params = dict(e["parameters"])
        fit = prusie.susie_rss(**inputs, **params, track_fit=True, retain_data=True)
        R = inputs["R"]
        z = inputs["z"]
        n = inputs["n"]
        refined = prusie.refine_fit(fit, prusie.RefinementConfig(candidate_budget=4))
        sensitivity = prusie.run_sensitivity(
            inputs,
            [
                {"name": "larger_L", "changes": {"L": params["L"] + 1}},
                {
                    "name": "fixed_variance",
                    "changes": {"scaled_prior_variance": 0.05, "estimate_prior_variance": False},
                },
                {"name": "coverage90", "report": {"coverage": 0.9}},
            ],
            baseline=fit,
            baseline_settings=params,
        )
        comparison = sensitivity["comparison"]
        source = "prusie packaged synthetic toy example; frozen generator seed 20261007"
        annotations = {
            "example_kind": ("synthetic toy example; no biological mechanism is inferred")
        }
    else:
        e = prusie.load_convergence_example()
        inputs = e["inputs"]
        params = e["parameters"]
        fit = prusie.susie_suff_stat(**inputs, **params, track_fit=True)
        refined = prusie.refine_fit(fit, prusie.RefinementConfig(inner_tol=0.001))
        diagonal = np.sqrt(np.diag(inputs["XtX"]))
        R = inputs["XtX"] / diagonal[:, None] / diagonal[None, :]
        # Derived signed marginal t statistics from the supplied centered sums.
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
            "example_kind": (
                "official convergence toy sufficient statistics; local-optimum solver example"
            )
        }
    summary = prusie.summarize(
        refined,
        R,
        z=z,
        input_info={"ld_source": source, "in_sample_ld": True, "genome_build": "not_assessed"},
    )
    ld = prusie.diagnose_ld(z, R, n, variant_ids=fit.variant_ids)
    figures = make_figures(fit, refined, summary, R, ld, comparison)
    provenance = dict(
        data_checksums={k: checksum(v) for k, v in inputs.items()},
        variant_harmonization=(
            "Input order preserved; allele labels unavailable in this toy fixture."
        ),
        parameters=params,
        initial_state=("cold start; refinement stage B uses stage A posterior moments"),
        reference_sources=[source],
        software_versions={
            "prusie": prusie.__version__,
            "numpy": np.__version__,
            "native": fit.backend_version,
        },
        random_seed=e["metadata"].get("seed", "frozen upstream fixture; no new random generation"),
        elapsed_seconds=time.perf_counter() - started,
        skipped=[
            "AI explanation is separate from the statistical report",
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
        title="prusie " + case + " toy example",
    )
    save_analysis({"baseline": fit, "refined": refined}, output / "fits")
    restored = load_analysis(output / "fits")
    assert restored["refined"].alpha.tobytes() == refined.alpha.tobytes()
    if case == "problem":
        gain = prusie.evaluate_objective(refined)["elbo"] - prusie.evaluate_objective(fit)["elbo"]
        assert gain > 17
        np.testing.assert_allclose(
            refined.pip, e["reference"]["latest_refined_pip"], atol=1e-5, rtol=0
        )
        receipt["objective_gain"] = gain
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
    if interpret:
        interpret_saved_report(output, model)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", choices=("stable", "problem", "both"), default="both")
    parser.add_argument(
        "--interpret",
        action="store_true",
        help=(
            "Consent to send the synthetic stable evidence bundle "
            "to the official Codex CLI. No LD/genotypes are sent."
        ),
    )
    parser.add_argument(
        "--model",
        help=("Explicit provider model identifier for --interpret; no default is assumed."),
    )
    args = parser.parse_args()
    if args.interpret and args.case != "stable":
        parser.error("--interpret requires --case stable (synthetic evidence only)")
    if args.interpret and not args.model:
        parser.error("--interpret requires --model with an available provider model identifier")
    for case in ("stable", "problem") if args.case == "both" else (args.case,):
        receipt = execute(case, args.output_dir / case, interpret=args.interpret, model=args.model)
        print(json.dumps(receipt))


if __name__ == "__main__":
    main()
