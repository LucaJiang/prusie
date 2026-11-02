"""Actionable report flags, additive tables and older-record compatibility."""

import copy
import csv
import json
import platform
from pathlib import Path

import numpy as np
import pytest

from prusie.report import analysis_summary, load_analysis, summary_text, write_report


def recorded_summary():
    return dict(
        variants=[dict(variant_id="a", index=0, pip=0.9, rank=1)],
        components=[dict(component_id=3, active=True)],
        credible_sets=[dict(component_id=3, members=["a"], representative="a")],
        convergence=dict(
            stop_reason="criterion_satisfied", niter=4, tol=0.001, elbo_decrease_count=0
        ),
        trace=[],
    )


def recorded_comparison():
    return dict(
        schema_version="prusie.comparison.v2",
        settings=["baseline", "L_half"],
        baseline="baseline",
        signal_relations=[],
        fit_status=[
            dict(
                setting=setting,
                converged=True,
                inner_ibss_converged=True,
                refinement_status="not_requested",
                niter=4,
                stop_reason="criterion_satisfied",
                cs_count=1,
            )
            for setting in ("baseline", "L_half")
        ],
        cs_stability=[
            dict(
                baseline_component=3,
                representative="a",
                setting="L_half",
                relation="unmatched",
                matched_components=[],
                jaccard=None,
                same_representative=None,
                sum_abs_delta_pip=0.1,
                missing_members=[],
            )
        ],
        cs_stability_summary=[
            dict(baseline_component=3, representative="a", recovered_in=0, assessed_in=1)
        ],
    )


def recorded_refinement():
    return dict(
        status="completed",
        stop_reason="no_improvement_above_absolute_tol",
        total_fits=2,
        before=dict(elbo=-20.0),
        after=dict(elbo=-19.0),
        accepted_inner_ibss_converged=False,
        history=[
            dict(
                round=1,
                component=3,
                accepted=True,
                objective=-19.0,
                candidate_cs=[["b"]],
                candidate_cs_index=[7],
            )
        ],
        competing_candidates=[
            dict(
                round=1,
                component=3,
                objective=-19.0,
                gap_to_round_best=0.0,
                same_cs_as_incumbent=False,
                same_cs_as_accepted=True,
            ),
            dict(
                round=1,
                component=7,
                objective=-19.5,
                gap_to_round_best=0.5,
                same_cs_as_incumbent=True,
                same_cs_as_accepted=False,
            ),
        ],
        signal_comparison=dict(
            schema_version="prusie.comparison.v2",
            settings=["before", "after"],
            baseline="before",
            components=dict(
                before=[dict(component_id=3, cs_retained=True, cs=["a"], top="a")],
                after=[dict(component_id=7, cs_retained=True, cs=["b"], top="b")],
            ),
            signal_relations=[
                dict(
                    setting_a="before",
                    setting_b="after",
                    component_a=3,
                    component_b=None,
                    status="unmatched",
                ),
                dict(
                    setting_a="before",
                    setting_b="after",
                    component_a=None,
                    component_b=7,
                    status="unmatched",
                ),
            ],
        ),
    )


def flag_map(compact):
    return {row["code"]: row for row in compact["flags"]}


def test_every_flag_has_actionable_message_and_evidence(tmp_path):
    summary = recorded_summary()
    summary["convergence"].update(
        stop_reason="iteration_limit", inner_ibss_converged=False, elbo_decrease_count=1
    )
    summary["refinement"] = recorded_refinement()
    comparison = recorded_comparison()
    comparison["fit_status"][1].update(converged=False, inner_ibss_converged=False)
    ld = dict(variants=[dict(variant_id="a", flag=True, status="assessed")])
    original = copy.deepcopy((summary, comparison, ld))
    compact = analysis_summary(summary, comparison=comparison, ld=ld)
    expected = {
        "ibss_not_converged",
        "elbo_decreased",
        "refinement_changed_cs",
        "refinement_accepted_inner_not_converged",
        "competing_refinement_solution",
        "sensitivity_cs_not_recovered",
        "sensitivity_fit_not_converged",
        "ld_flagged",
    }
    assert set(flag_map(compact)) == expected
    assert all(
        set(flag) == {"code", "severity", "message", "evidence_pointer"}
        and flag["severity"] in {"attention", "info"}
        for flag in compact["flags"]
    )
    assert compact["fit"]["assessment"]["schema_version"] == "prusie.convergence_assessment.v1"
    text = summary_text(summary, comparison=comparison, ld=ld)
    for flag in compact["flags"]:
        assert ("Check: " + flag["message"] in text) == (flag["severity"] == "attention")
    receipt = write_report(summary, tmp_path, comparison=comparison, ld=ld)
    page = Path(receipt["html"]).read_text(encoding="utf-8")
    assert page.index("</header>") < page.index('id="first-checks"') < page.index("<nav")
    for flag in compact["flags"]:
        assert f'data-flag="{flag["code"]}"' in page
    assert (summary, comparison, ld) == original
    assert load_analysis(tmp_path)["summary"] == summary


def test_old_records_leave_new_checks_unevaluated(tmp_path):
    summary = recorded_summary()
    summary["refinement"] = dict(status="completed", history=[], total_fits=0)
    comparison = dict(settings=["baseline", "L_half"], baseline="baseline", signals=[])
    compact = analysis_summary(summary, comparison=comparison)
    checks = {row["code"]: row["status"] for row in compact["flag_checks"]}
    assert compact["flags"] == []
    for code in (
        "refinement_changed_cs",
        "refinement_accepted_inner_not_converged",
        "competing_refinement_solution",
        "sensitivity_cs_not_recovered",
        "sensitivity_fit_not_converged",
    ):
        assert checks[code] == "not evaluated"
    receipt = write_report(summary, tmp_path, comparison=comparison)
    page = Path(receipt["html"]).read_text(encoding="utf-8")
    assert "No flags were raised by the checks with recorded evidence." in page
    assert "Not evaluated: competing candidates were not recorded." in page
    assert "not evaluated" in page


@pytest.mark.parametrize("gap,raised", [(0.0, False), (0.49, False), (0.5, True), (1.0, True)])
def test_competing_candidate_threshold_is_inclusive_and_display_only(gap, raised, tmp_path):
    summary = recorded_summary()
    summary["refinement"] = recorded_refinement()
    before = copy.deepcopy(summary)
    compact = analysis_summary(summary, competing_elbo_gap=gap)
    assert ("competing_refinement_solution" in flag_map(compact)) == raised
    assert compact["refinement"]["competing_elbo_gap"] == gap
    receipt = write_report(summary, tmp_path, competing_elbo_gap=gap)
    saved = json.loads(Path(receipt["analysis_summary"]).read_text(encoding="utf-8"))
    assert saved["flags"] == compact["flags"]
    assert Path(receipt["text_summary"]).read_text(encoding="utf-8") == (
        summary_text(summary, competing_elbo_gap=gap) + "\n"
    )
    assert summary == before


@pytest.mark.parametrize("gap", [-1, float("inf"), float("nan"), True, "1"])
def test_invalid_competing_threshold_rejected_before_writing(gap, tmp_path):
    with pytest.raises(ValueError, match="competing_elbo_gap"):
        analysis_summary(recorded_summary(), competing_elbo_gap=gap)
    with pytest.raises(ValueError, match="competing_elbo_gap"):
        write_report(recorded_summary(), tmp_path / "report", competing_elbo_gap=gap)
    assert not (tmp_path / "report").exists()


def test_cs_change_ignores_cs_order_and_uses_member_identity():
    summary = recorded_summary()
    summary["refinement"] = recorded_refinement()
    comparison = summary["refinement"]["signal_comparison"]
    comparison["components"] = dict(
        before=[
            dict(component_id=3, cs_retained=True, cs=["a", "b"]),
            dict(component_id=7, cs_retained=True, cs=["c"]),
        ],
        after=[
            dict(component_id=3, cs_retained=True, cs=["c"]),
            dict(component_id=7, cs_retained=True, cs=["b", "a"]),
        ],
    )
    assert "refinement_changed_cs" not in flag_map(analysis_summary(summary))
    comparison["components"]["after"][1]["cs"] = ["a", "d"]
    assert "refinement_changed_cs" in flag_map(analysis_summary(summary))


def test_new_tables_export_full_precision_and_resolvable_evidence(tmp_path):
    summary = recorded_summary()
    summary["refinement"] = recorded_refinement()
    comparison = recorded_comparison()
    receipt = write_report(summary, tmp_path, comparison=comparison)
    filenames = {Path(name).name for name in receipt["tables"]}
    assert {
        "convergence_assessment.csv",
        "refinement_credible_sets.csv",
        "competing_candidates.csv",
        "fit_status.csv",
        "cs_stability.csv",
        "cs_stability_summary.csv",
    } <= filenames
    with (tmp_path / "tables/cs_stability.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["sum_abs_delta_pip"] == "0.1"
    with (tmp_path / "tables/refinement_attempts.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert json.loads(rows[0]["candidate_cs"]) == [["b"]]
    evidence = json.loads(Path(receipt["evidence"]).read_text(encoding="utf-8"))["data"]
    for flag in analysis_summary(summary, comparison=comparison)["flags"]:
        value = evidence
        for part in flag["evidence_pointer"].split("/")[1:]:
            value = value[int(part)] if isinstance(value, list) else value[part]
        assert value is not None
    page = Path(receipt["html"]).read_text(encoding="utf-8")
    assert "Refinement credible-set membership" in page
    assert "Sensitivity fit status" in page and "Baseline credible-set recovery" in page


def test_software_versions_auto_and_caller_values_win(tmp_path):
    import prusie

    receipt = write_report(recorded_summary(), tmp_path / "auto")
    automatic = load_analysis(Path(receipt["html"]).parent)["provenance"]["software_versions"]
    assert automatic == dict(
        prusie=prusie.__version__,
        numpy=np.__version__,
        python=platform.python_version(),
        platform=platform.platform(),
        source="auto",
    )
    supplied = dict(prusie="recorded-version", source="caller")
    write_report(
        recorded_summary(), tmp_path / "supplied", provenance=dict(software_versions=supplied)
    )
    assert load_analysis(tmp_path / "supplied")["provenance"]["software_versions"] == supplied


def test_figure_section_override_and_validation(tmp_path):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(2, 1))
    axis.plot([0, 1], [0, 1])
    try:
        receipt = write_report(
            recorded_summary(),
            tmp_path / "valid",
            figures={"Posterior paths": figure},
            figure_sections={"Posterior paths": "fit"},
        )
        index = json.loads(Path(receipt["figure_index"]).read_text(encoding="utf-8"))
        assert index[0]["section"] == "fit"
        for mapping in ({"Posterior paths": "unknown"}, {"absent label": "fit"}):
            with pytest.raises(ValueError, match="Unknown figure"):
                write_report(
                    recorded_summary(),
                    tmp_path / "invalid",
                    figures={"Posterior paths": figure},
                    figure_sections=mapping,
                )
        assert not (tmp_path / "invalid").exists()
    finally:
        plt.close(figure)


def test_real_toy_report_with_refinement_sensitivity_and_ld(tmp_path):
    from prusie import diagnose_ld, load_example, refine_fit, run_sensitivity, summarize, susie_rss

    example = load_example()
    fit = susie_rss(**example["inputs"], **example["parameters"], retain_data=True, track_fit=True)
    refined = refine_fit(fit, track_fit=True)
    sensitivity = run_sensitivity(
        example["inputs"],
        [{"name": "coverage_0.90", "report": {"coverage": 0.90}}],
        baseline=fit,
        baseline_settings=example["parameters"],
    )
    summary = summarize(refined)
    ld = diagnose_ld(**example["inputs"])
    receipt = write_report(summary, tmp_path, comparison=sensitivity["comparison"], ld=ld)
    compact = json.loads(Path(receipt["analysis_summary"]).read_text(encoding="utf-8"))
    assert compact["fit"]["assessment"]["inner_ibss_converged"] == fit.converged
    assert compact["refinement"]["credible_set_changes"] is not None
    assert len(compact["sensitivity"]["fit_status"]) == 2
    assert compact["sensitivity"]["cs_stability_summary"]
    checks = {row["code"]: row["status"] for row in compact["flag_checks"]}
    assert checks["sensitivity_cs_not_recovered"] != "not evaluated"
    assert checks["sensitivity_fit_not_converged"] != "not evaluated"
    assert (checks["ld_flagged"] != "not evaluated") == (compact["ld"]["assessed"] > 0)
    assert (tmp_path / "tables/competing_candidates.csv").exists()
