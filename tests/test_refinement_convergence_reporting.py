"""fit.converged keeps susieR's outer-refinement meaning; reports separate the two levels."""

from pathlib import Path

import pytest

import prusie
from prusie.report import analysis_summary, write_report


@pytest.fixture(scope="module")
def stopped_then_refined():
    example = prusie.load_example()
    parameters = dict(example["parameters"], max_iter=2)
    fit = prusie.susie_rss(**example["inputs"], **parameters, track_fit="compact", retain_data=True)
    return example, fit, prusie.refine_fit(fit)


def test_iteration_limit_fit_refines_to_outer_convergence(stopped_then_refined):
    _, fit, refined = stopped_then_refined
    assert fit.converged is False
    assert fit.convergence["stop_reason"] == "iteration_limit"
    assert refined.refinement["status"] in ("completed", "no_improvement")
    assert refined.converged is True
    assert refined.convergence["refinement_converged"] is True
    assert refined.convergence["inner_ibss_converged"] is False
    assert refined.convergence["stop_reason"] == "iteration_limit"
    assert fit.converged is False


def test_assessment_and_comparison_keep_both_levels(stopped_then_refined):
    _, fit, refined = stopped_then_refined
    assessment = prusie.assess_convergence(refined)
    assert assessment["inner_ibss_converged"] is False
    assert assessment["refinement_converged"] is True
    assert assessment["status"] != "converged"
    row = prusie.compare_fits({"before": fit, "after": refined})["fit_status"][1]
    assert row["converged"] is True and row["inner_ibss_converged"] is False


def test_reports_flag_inner_nonconvergence_and_explain_converged(stopped_then_refined, tmp_path):
    example, _, refined = stopped_then_refined
    summary = prusie.summarize(refined, example["inputs"]["R"])
    compact = analysis_summary(summary)
    assert "ibss_not_converged" in {flag["code"] for flag in compact["flags"]}
    note = compact["fit"]["convergence_note"]
    assert "fit.converged is True" in note and "iteration_limit" in note
    assert note in prusie.summary_text(summary)
    report = write_report(summary, tmp_path)
    page = Path(report["html"]).read_text(encoding="utf-8")
    assert "fit.converged is True" in page
    assert 'data-flag="ibss_not_converged"' in page


def test_unrefined_fit_has_no_convergence_note():
    example = prusie.load_example()
    fit = prusie.susie_rss(**example["inputs"], **example["parameters"], track_fit="compact")
    compact = analysis_summary(prusie.summarize(fit, example["inputs"]["R"]))
    assert compact["fit"]["convergence_note"] is None
