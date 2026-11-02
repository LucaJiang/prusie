"""Refinement adds candidate records without changing the search or posterior."""

import copy

import numpy as np
import pytest

from prusie import (
    RefinementConfig,
    load_convergence_example,
    load_example,
    refine_fit,
    susie_rss,
    susie_suff_stat,
)
from prusie.refinement import _cs_identity, _cs_members


@pytest.fixture(scope="module")
def refined_toy():
    example = load_example()
    fit = susie_rss(**example["inputs"], **example["parameters"], retain_data=True)
    return fit, refine_fit(fit, track_fit=True)


def test_toy_candidate_records_describe_original_stage_b(refined_toy):
    baseline, refined = refined_toy
    record = refined.refinement
    evaluated = [row for row in record["history"] if row["candidate_cs"] is not None]
    assert evaluated
    assert len(record["competing_candidates"]) == len(evaluated)
    for row in evaluated:
        assert len(row["candidate_cs"]) == len(row["candidate_cs_index"])
        assert all(isinstance(member, str) for cs in row["candidate_cs"] for member in cs)
        assert row["objective"] == row["stages"]["B"]["objective"]
        competing = next(
            c
            for c in record["competing_candidates"]
            if (c["round"], c["component"]) == (row["round"], row["component"])
        )
        round_best = max(r["objective"] for r in evaluated if r["round"] == row["round"])
        assert competing["objective"] == row["objective"]
        assert competing["gap_to_round_best"] == round_best - row["objective"]
        same_round = [r for r in evaluated if r["round"] == row["round"]]
        retained = (
            round_best if any(r["accepted"] for r in same_round) else row["incumbent_objective"]
        )
        assert competing["gap_to_retained"] == retained - row["objective"]
        assert isinstance(competing["same_cs_as_incumbent"], bool)
        assert isinstance(competing["same_cs_as_accepted"], bool)
    assert baseline.refinement["status"] == "not_executed"
    accepted = [row for row in evaluated if row["accepted"]]
    if accepted:
        assert (
            record["accepted_inner_ibss_converged"] == accepted[-1]["stages"]["B"]["ibss_converged"]
        )
    else:
        assert record["accepted_inner_ibss_converged"] is None


def test_cs_equality_ignores_order_but_preserves_member_identity():
    assert _cs_identity([["a", "b"], ["c"]]) == _cs_identity([["c"], ["b", "a"]])
    assert _cs_identity([["a", "b"], ["c"]]) != _cs_identity([["a", "c"], ["b"]])


def test_competing_records_ignore_candidate_cs_order(monkeypatch):
    import prusie.refinement as module

    example = load_convergence_example()
    baseline = susie_suff_stat(**example["inputs"], **example["parameters"])
    original_run = module._run
    stage_b = []

    def permute_candidate(*args, **kwargs):
        result = original_run(*args, **kwargs)
        if kwargs.get("reset_variances"):
            result.sets = dict(result.sets)
            result.sets["cs"] = result.sets["cs"][::-1]
            result.sets["cs_index"] = result.sets["cs_index"][::-1]
            stage_b.append(result)
        return result

    monkeypatch.setattr(module, "_run", permute_candidate)
    refined = refine_fit(baseline, RefinementConfig(max_rounds=1, absolute_tol=1e20))
    incumbent = _cs_identity(_cs_members(baseline))
    for row, candidate, competing in zip(
        refined.refinement["history"],
        stage_b,
        refined.refinement["competing_candidates"],
        strict=True,
    ):
        assert row["candidate_cs"] == _cs_members(candidate)
        assert row["candidate_cs_index"] == list(candidate.sets["cs_index"])
        assert competing["same_cs_as_incumbent"] == (
            _cs_identity(_cs_members(candidate)) == incumbent
        )
        assert competing["same_cs_as_accepted"] == competing["same_cs_as_incumbent"]
    assert refined.refinement["accepted_inner_ibss_converged"] is None


def test_accepted_inner_status_and_relative_denominator_are_recorded():
    example = load_convergence_example()
    baseline = susie_suff_stat(**example["inputs"], **example["parameters"], track_fit=True)
    original = copy.deepcopy(baseline.as_dict())
    result = refine_fit(baseline, track_fit=True)
    accepted = [row for row in result.refinement["history"] if row["accepted"]]
    assert accepted
    last = accepted[-1]
    assert (
        result.refinement["accepted_inner_ibss_converged"] == last["stages"]["B"]["ibss_converged"]
    )
    assert (
        result.refinement["accepted_inner_ibss_converged"]
        == result.convergence["inner_ibss_converged"]
    )
    assert result.converged == result.convergence["refinement_converged"]
    for row in result.refinement["history"]:
        assert row["relative_improvement"] == row["improvement"] / max(1, abs(row["objective"]))
    for field in ("alpha", "mu", "mu2", "V", "pip", "elbo", "KL", "XtXr"):
        assert getattr(baseline, field).tobytes() == original[field].tobytes()
    assert baseline.refinement == original["refinement"]


def test_no_support_and_no_attempts_have_unevaluated_records():
    fit = susie_rss(z=[6.0, 0.0], R=np.eye(2), n=200, L=1, retain_data=True)
    budget = refine_fit(fit, RefinementConfig(candidate_budget=0))
    assert budget.refinement["competing_candidates"] == []
    assert budget.refinement["accepted_inner_ibss_converged"] is None
    fit.sets = dict(cs=[np.array([0, 1])], cs_index=np.array([0]), purity={})
    no_support = refine_fit(fit)
    row = no_support.refinement["history"][0]
    assert row["reason"] == "no_prior_support"
    assert row["candidate_cs"] is None and row["candidate_cs_index"] is None
    assert no_support.refinement["competing_candidates"] == []
    assert no_support.refinement["total_fits"] == 0


def test_member_ids_map_internal_null_column():
    fit = susie_rss(
        z=[6.0, 0.0, 3.0], R=np.eye(3), n=200, L=2, null_weight=0.1, variant_ids=["a", "b", "c"]
    )
    # Internal null positions are handled even for a nonterminal saved null column.
    fit.null_index = 1
    fit.sets = dict(cs=[np.array([2, 3]), np.array([0])], cs_index=np.array([1, 0]))
    assert _cs_members(fit) == [["b", "c"], ["a"]]


@pytest.mark.parametrize("failure", ["invalid_state", "nonfinite_objective", "no_return"])
def test_failed_stage_b_preserves_available_cs_without_entering_selection(monkeypatch, failure):
    import prusie.refinement as module

    original_run = module._run

    def fail_candidate(*args, **kwargs):
        if kwargs.get("reset_variances") and failure == "no_return":
            raise RuntimeError("stage B failed before returning")
        result = original_run(*args, **kwargs)
        if kwargs.get("reset_variances"):
            if failure == "invalid_state":
                result.mu[0, 0] = np.nan
            else:
                result.elbo[-1] = np.nan
        return result

    fit = susie_rss(z=[6.0, 0.0], R=np.eye(2), n=200, L=1, retain_data=True)
    monkeypatch.setattr(module, "_run", fail_candidate)
    result = refine_fit(fit)
    assert result.refinement_status == "all_candidates_failed"
    assert result.refinement["accepted_inner_ibss_converged"] is None
    row = result.refinement["history"][0]
    if failure == "no_return":
        assert row["candidate_cs"] is None and row["candidate_cs_index"] is None
        assert result.refinement["competing_candidates"] == []
    else:
        assert row["candidate_cs"] is not None and row["candidate_cs_index"] is not None
        competing = result.refinement["competing_candidates"][0]
        assert competing["objective"] is None and competing["gap_to_round_best"] is None
        assert competing["gap_to_retained"] is None
    for name in ("alpha", "mu", "mu2", "V", "pip", "elbo", "KL", "XtXr"):
        assert getattr(fit, name).tobytes() == getattr(result, name).tobytes()
