"""Post-hoc assessment uses recorded sweep diagnostics without changing a fit."""

import copy
import math

import numpy as np
import pytest

from prusie import summarize, susie_rss
from prusie.datasets import load_example
from prusie.diagnostics import assess_convergence


def recorded(deltas=(8.0, 4.0, 2.0), **convergence):
    """A saved tail of sweeps with two active components and one inactive one."""
    return dict(
        convergence=dict(
            stop_reason="iteration_limit", niter=20, tol=0.125, ibss_converged=False, **convergence
        ),
        variants=[dict(index=i, variant_id=name) for i, name in enumerate(("a", "b", "c"))],
        trace=[
            dict(
                iteration=i + 18,
                elbo=-100.0 + i,
                delta_elbo=delta,
                max_abs_delta_pip=0.0002,
                active=[True, True, False],
                top_variant=[0, 2, 1],
                entropy=[0.2, 0.4, 1.0],
                cycle_lag=None,
                events=[],
            )
            for i, delta in enumerate(deltas)
        ],
    )


def test_slow_monotone_extrapolation_and_contract():
    result = assess_convergence(recorded())
    assert set(result) == {
        "schema_version",
        "status",
        "stop_reason",
        "niter",
        "tol",
        "inner_ibss_converged",
        "refinement_converged",
        "trace_available",
        "tail",
        "elbo_tail",
        "inference_stability",
        "moving_components",
        "elbo_decrease_count",
        "nonfinite_elbo_count",
        "message",
    }
    assert result["schema_version"] == "prusie.convergence_assessment.v1"
    assert result["status"] == "slow_monotone"
    assert result["stop_reason"] == "iteration_limit"
    assert result["niter"] == 20 and result["tol"] == 0.125
    assert result["inner_ibss_converged"] is False
    assert result["refinement_converged"] is None
    assert result["trace_available"] and result["tail"] == 3
    assert result["elbo_tail"] == dict(
        ratio=0.5, extrapolated_remaining_gain=2.0, estimated_extra_sweeps=4
    )
    assert result["inference_stability"] == dict(
        top_variants_unchanged=True,
        max_abs_delta_pip_tail=0.0002,
        stable=True,
        pip_stability=0.001,
    )
    assert result["moving_components"] == []
    assert result["elbo_decrease_count"] == result["nonfinite_elbo_count"] == 0
    assert "iterations" in result["message"]


@pytest.mark.parametrize(
    "deltas,status",
    [
        ((1.0, -0.1, 0.01), "oscillating"),
        ((1.0, -0.1, 0.0, 0.01), "oscillating"),
        ((1.0, -0.1, -0.01), "unclassified"),
        ((1.0, 2.0, 3.0), "unclassified"),
        ((1.0, 1.0, 1.0), "unclassified"),
        ((1.0, 0.0, 0.0), "unclassified"),
        ((1.0, None, 0.1), "unclassified"),
        ((1.0, float("nan"), 0.1), "unclassified"),
        ((1.0,), "unclassified"),
    ],
)
def test_signed_elbo_tail_patterns(deltas, status):
    result = assess_convergence(recorded(deltas))
    assert result["status"] == status
    assert result["elbo_tail"]["ratio"] is None


@pytest.mark.parametrize(
    "event,status",
    [
        ("numerical_failure", "numerical_failure"),
        ("elbo_decreased", "elbo_decreasing"),
        ("cycle_detected", "oscillating"),
    ],
)
def test_recorded_events_override_geometric_tail(event, status):
    summary = recorded()
    summary["trace"][-1]["events"] = [event]
    assert assess_convergence(summary)["status"] == status


def test_status_precedence_and_tail_event_counts():
    summary = recorded((4.0, 2.0, 1.0, 0.5))
    summary["trace"][0]["events"] = ["numerical_failure", "nonfinite_elbo"]
    summary["trace"][1]["events"] = ["elbo_decreased", "cycle_detected"]
    summary["trace"][1]["elbo"] = float("nan")
    result = assess_convergence(summary)
    assert result["status"] == "numerical_failure"
    assert result["elbo_decrease_count"] == 1
    assert result["nonfinite_elbo_count"] == 2
    assert assess_convergence(summary, tail=3)["status"] == "elbo_decreasing"
    final = assess_convergence(summary, tail=2)
    assert final["status"] == "slow_monotone"
    assert final["elbo_decrease_count"] == final["nonfinite_elbo_count"] == 0
    summary["convergence"]["ibss_converged"] = True
    assert assess_convergence(summary)["status"] == "converged"


@pytest.mark.parametrize("lag", [2, 5])
def test_recorded_cycle_lag_is_respected(lag):
    summary = recorded()
    summary["trace"][-1]["cycle_lag"] = lag
    assert assess_convergence(summary)["status"] == "oscillating"


def test_lag_one_fixed_point_is_not_oscillation():
    summary = recorded()
    summary["trace"][-1]["cycle_lag"] = 1
    summary["trace"][-1]["events"] = ["fixed_point"]
    assert assess_convergence(summary)["status"] == "slow_monotone"


@pytest.mark.parametrize("with_trace", [False, True])
@pytest.mark.parametrize("criterion", ["criterion_satisfied", "averaged_cycle"])
def test_successful_stop_precedes_trace_requirement(with_trace, criterion):
    summary = recorded()
    if criterion == "criterion_satisfied":
        summary["convergence"]["stop_reason"] = "criterion_satisfied"
    else:
        summary["trace"][-1]["events"] = ["alpha_pip_cycle_averaged"]
        summary["convergence"]["last"] = copy.deepcopy(summary["trace"][-1])
    if not with_trace:
        summary["trace"] = []
    result = assess_convergence(summary)
    assert result["status"] == "converged"
    assert result["trace_available"] is with_trace
    assert result["tail"] == (3 if with_trace else 0)


def test_no_trace_and_legacy_missing_diagnostics():
    summary = recorded()
    summary["trace"] = []
    result = assess_convergence(summary)
    assert result["status"] == "not_assessable"
    assert result["niter"] == 20 and result["stop_reason"] == "iteration_limit"
    assert "track_fit=True" in result["message"]
    assert result["inference_stability"]["stable"] is None
    assert result["moving_components"] == []
    missing = assess_convergence({})
    assert missing["status"] == "not_assessable"
    assert missing["niter"] is missing["tol"] is missing["inner_ibss_converged"] is None


def test_recorded_inner_and_outer_flags_are_separate():
    summary = recorded(inner_ibss_converged=False, refinement_converged=True)
    summary["converged"] = True
    result = assess_convergence(summary)
    assert result["status"] != "converged"
    assert result["inner_ibss_converged"] is False
    assert result["refinement_converged"] is True
    summary["convergence"]["inner_ibss_converged"] = True
    summary["convergence"]["refinement_converged"] = False
    assert assess_convergence(summary)["status"] == "converged"


def test_median_ratio_tolerance_and_missing_tolerance():
    summary = recorded((10.0, 8.0, 4.0, 1.0))
    summary["convergence"]["tol"] = 2.0
    result = assess_convergence(summary)
    assert result["elbo_tail"] == dict(
        ratio=0.5, extrapolated_remaining_gain=1.0, estimated_extra_sweeps=0
    )
    summary["convergence"]["tol"] = None
    assert assess_convergence(summary)["elbo_tail"]["estimated_extra_sweeps"] is None


def test_first_sweep_missing_deltas_are_not_treated_as_failed_checks():
    summary = recorded((None, 2.0, 1.0))
    for i, row in enumerate(summary["trace"]):
        row["iteration"] = i + 1
    summary["trace"][0]["max_abs_delta_pip"] = None
    result = assess_convergence(summary)
    assert result["status"] == "slow_monotone"
    assert result["inference_stability"]["stable"] is True


@pytest.mark.parametrize(
    "null_index,last_internal,expected", [(1, 2, "b"), (1, 1, None), (3, 3, None)]
)
def test_internal_top_variant_indices_map_with_null_column(null_index, last_internal, expected):
    summary = recorded(null_index=null_index)
    summary["trace"][-1]["top_variant"][0] = last_internal
    result = assess_convergence(summary)
    component = next(row for row in result["moving_components"] if row["component_id"] == 0)
    assert component["last_top_variant"] == expected
    assert component["top_variant_changes"] == 1


def test_old_summary_with_appended_null_and_no_mapping_still_avoids_false_id():
    summary = recorded()
    summary["trace"][-1]["top_variant"][0] = 3
    assert assess_convergence(summary)["moving_components"][0]["last_top_variant"] is None


def test_moving_components_keep_original_ids_and_exclude_inactive_components():
    summary = recorded()
    summary["trace"][1]["top_variant"] = [1, 0, 2]
    summary["trace"][-1]["top_variant"] = [1, 2, 0]
    summary["trace"][-1]["entropy"] = [0.1, 0.3, 0.5]
    result = assess_convergence(summary)
    assert [row["component_id"] for row in result["moving_components"]] == [1, 0]
    assert [row["top_variant_changes"] for row in result["moving_components"]] == [2, 1]
    assert result["moving_components"][0]["entropy_change"] == pytest.approx(-0.1)
    assert result["inference_stability"]["top_variants_unchanged"] is False
    assert result["inference_stability"]["stable"] is False
    summary["trace"][0]["active"][1] = False
    assert [row["component_id"] for row in assess_convergence(summary)["moving_components"]] == [0]


def test_entropy_movement_does_not_require_a_top_variant_change():
    summary = recorded()
    summary["trace"][-1]["entropy"][1] = 0.41
    result = assess_convergence(summary)
    assert len(result["moving_components"]) == 1
    assert result["moving_components"][0]["top_variant_changes"] == 0
    assert result["moving_components"][0]["last_top_variant"] == "c"
    assert result["inference_stability"]["stable"] is True


def test_stability_threshold_is_strict_and_uses_recorded_distance_entering_tail():
    summary = recorded()
    summary["trace"][-2]["max_abs_delta_pip"] = 0.001
    result = assess_convergence(summary, tail=2)
    assert result["inference_stability"]["max_abs_delta_pip_tail"] == 0.001
    assert result["inference_stability"]["stable"] is False
    assert assess_convergence(summary, pip_stability=0.01)["inference_stability"]["stable"] is True


@pytest.mark.parametrize("missing", ["max_abs_delta_pip", "top_variant", "active"])
def test_incomplete_stability_metrics_are_not_reported_as_stable(missing):
    summary = recorded()
    del summary["trace"][1][missing]
    result = assess_convergence(summary)
    assert result["inference_stability"]["stable"] is None


def test_missing_activity_flag_leaves_component_stability_unevaluated():
    summary = recorded()
    summary["trace"][1]["active"][0] = None
    assert assess_convergence(summary)["inference_stability"]["stable"] is None


def test_one_saved_sweep_keeps_recorded_distance_without_claiming_component_stability():
    summary = recorded((1.0,))
    result = assess_convergence(summary)
    assert result["tail"] == 1
    assert result["inference_stability"]["max_abs_delta_pip_tail"] == 0.0002
    assert result["inference_stability"]["top_variants_unchanged"] is None
    assert result["inference_stability"]["stable"] is None


def test_summary_variant_rows_are_mapped_by_input_index():
    summary = recorded(null_index=1)
    summary["variants"].reverse()
    summary["trace"][-1]["top_variant"][0] = 2
    assert assess_convergence(summary)["moving_components"][0]["last_top_variant"] == "b"


def test_missing_top_variant_does_not_hide_recorded_entropy_movement():
    summary = recorded()
    del summary["trace"][1]["top_variant"]
    summary["trace"][-1]["entropy"][0] = 0.3
    result = assess_convergence(summary)
    assert result["inference_stability"]["stable"] is None
    assert result["moving_components"][0]["entropy_change"] == pytest.approx(0.1)
    assert result["moving_components"][0]["top_variant_changes"] is None


@pytest.mark.parametrize("tail", [True, False, 1, 0, -1, 2.0, "5", None])
def test_tail_validation(tail):
    with pytest.raises(ValueError, match="tail"):
        assess_convergence({}, tail=tail)


@pytest.mark.parametrize("threshold", [0.0, 1.0, -0.1, math.nan, math.inf, True, "0.1", None])
def test_pip_stability_validation(threshold):
    with pytest.raises(ValueError, match="pip_stability"):
        assess_convergence({}, pip_stability=threshold)


def test_rejects_nonfit_input():
    with pytest.raises(TypeError, match="SusieResult"):
        assess_convergence([])


def test_tracked_packaged_toy_at_iteration_limit_and_input_immutability(monkeypatch):
    from prusie import _native

    example = load_example()
    parameters = dict(example["parameters"], max_iter=2, tol=1e-14, track_fit=True)
    with pytest.warns(RuntimeWarning, match="iteration_limit"):
        fit = susie_rss(**example["inputs"], **parameters)
    before = copy.deepcopy(fit.convergence), copy.deepcopy(fit.trace), fit.pip.copy()
    saved = summarize(fit)
    monkeypatch.setattr(_native, "fit", lambda *args, **kwargs: pytest.fail("assessment refitted"))
    direct = assess_convergence(fit)
    restored = assess_convergence(saved)
    assert direct == restored
    assert direct["stop_reason"] == "iteration_limit"
    assert direct["inner_ibss_converged"] is False
    assert direct["niter"] == direct["tail"] == 2
    assert direct["status"] != "converged"
    assert fit.convergence == before[0] and fit.trace == before[1]
    np.testing.assert_array_equal(fit.pip, before[2])
    assert fit.converged is False


def test_fit_without_trace_and_legacy_convergence():
    fit = susie_rss([5.0, 0.0], np.eye(2), n=100, L=1)
    assert assess_convergence(fit)["status"] == "converged"
    fit.convergence = {}
    result = assess_convergence(fit)
    assert result["status"] == "converged"
    assert result["niter"] == fit.niter
    assert result["inner_ibss_converged"] is fit.converged


def test_real_null_fit_and_summary_preserve_index_mapping():
    fit = susie_rss(
        [5.0, 0.0], np.eye(2), n=100, L=1, variant_ids=["a", "b"], null_weight=0.2, track_fit=True
    )
    assert fit.convergence["null_index"] == 2
    assert summarize(fit)["convergence"]["null_index"] == 2
    assert assess_convergence(fit) == assess_convergence(summarize(fit))
