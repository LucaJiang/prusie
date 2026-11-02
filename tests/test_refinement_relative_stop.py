"""The susieR-compatible relative stop is kept, recorded in nats and flagged."""

import pytest

import prusie
from prusie import RefinementConfig, refine_fit, susie_suff_stat
from prusie.report import analysis_summary


@pytest.fixture(scope="module")
def local_optimum():
    example = prusie.datasets.load_convergence_example()
    fit = susie_suff_stat(**example["inputs"], **example["parameters"], retain_data=True)
    return fit, refine_fit(fit)


def no_relative(fit):
    tol = fit.params["tol"]
    return RefinementConfig(
        max_rounds=100,
        absolute_tol=tol,
        relative_tol=None,
        max_iter=fit.params["max_iter"],
        inner_tol=tol,
    )


def test_default_stops_at_relative_threshold_and_records_it(local_optimum):
    fit, refined = local_optimum
    record = refined.refinement
    assert record["stop_reason"] == "relative_improvement_threshold"
    stop = record["relative_stop"]
    assert stop["relative_tol"] == fit.params["tol"]
    assert stop["objective"] == record["best_objective"]
    assert stop["threshold"] == fit.params["tol"] * abs(stop["objective"])
    assert 0 < stop["last_accepted_gain"] < stop["threshold"]
    assert stop["last_accepted_gain"] == record["best_objective"] - record["before"]["elbo"]


def test_relative_none_searches_on_without_relative_record(local_optimum):
    fit, refined = local_optimum
    further = refine_fit(fit, no_relative(fit))
    record = further.refinement
    assert record["stop_reason"] == "no_improvement_above_absolute_tol"
    assert record["relative_stop"] is None
    assert len({row["round"] for row in record["history"]}) > len(
        {row["round"] for row in refined.refinement["history"]}
    )
    assert record["best_objective"] >= refined.refinement["best_objective"]


def test_report_notes_the_relative_stop(local_optimum):
    fit, refined = local_optimum
    compact = analysis_summary(prusie.summarize(refined))
    flags = {flag["code"]: flag for flag in compact["flags"]}
    flag = flags["refinement_stopped_by_relative_threshold"]
    assert flag["severity"] == "info"
    assert "relative_tol=None" in flag["message"] and "nats" in flag["message"]
    assert "Note: " + flag["message"] in prusie.summary_text(prusie.summarize(refined))
    unrefined = analysis_summary(prusie.summarize(fit))
    checks = {row["code"]: row["status"] for row in unrefined["flag_checks"]}
    assert checks["refinement_stopped_by_relative_threshold"] == "skipped"
