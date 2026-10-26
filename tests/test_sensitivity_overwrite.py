"""Sensitivity refits preserve inputs independently of baseline LD ownership."""

import copy

import numpy as np
import pytest

from prusie import ConvergenceConfig, continue_fit, susie_rss
from prusie.comparison import run_sensitivity


def rss_inputs():
    return dict(
        z=np.array([6.0, 1.5, 0.1]),
        # Force native symmetry repair while remaining within the comparison
        # helper's existing correlation-matrix tolerance.
        R=np.array([[1.0, 0.2, 0.1], [0.200000000001, 1.0, 0.15], [0.1, 0.15, 1.0]]),
        n=200,
        variant_ids=["a", "b", "c"],
    )


def fit_settings():
    return dict(
        L=1,
        n_threads=2,
        max_iter=10,
        tol=1e-8,
        check_input=False,
        estimate_prior_variance=False,
        scaled_prior_variance=0.1,
    )


def assert_unchanged(data, before):
    assert set(data) == set(before)
    for key in data:
        np.testing.assert_array_equal(data[key], before[key])


def assert_same_fit(actual, expected):
    for name in ("alpha", "mu", "mu2", "V", "pip", "elbo"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    assert actual.params["n_threads"] == expected.params["n_threads"] == 2
    assert actual.params["convergence"] == expected.params["convergence"]


@pytest.mark.parametrize("change", [{"L": 2}, {"refine": True}])
@pytest.mark.parametrize("explicit_false", [False, True])
def test_overwrite_baseline_refits_match_safe_direct_calls(change, explicit_false):
    data, settings = rss_inputs(), fit_settings()
    before = copy.deepcopy(data)
    consumed = copy.deepcopy(data)
    baseline = susie_rss(**consumed, **settings, overwrite_ld=True)
    assert not np.array_equal(consumed["R"], data["R"])
    baseline_alpha = baseline.alpha.copy()
    out = run_sensitivity(
        data,
        [{"name": "changed", "changes": change}],
        baseline=baseline,
        baseline_settings={"overwrite_ld": False} if explicit_false else None,
    )
    expected = susie_rss(**copy.deepcopy(data), **dict(settings, **change))
    actual = out["fits"]["changed"]
    assert_same_fit(actual, expected)
    assert actual.params["overwrite_ld"] is False
    assert "overwrite_ld" not in out["baseline_settings"]
    if "refine" in change:
        assert actual.refinement_status == expected.refinement_status != "not_executed"
    assert out["fits"]["baseline"] is baseline
    assert baseline.params["overwrite_ld"] is True
    np.testing.assert_array_equal(baseline.alpha, baseline_alpha)
    assert_unchanged(data, before)


@pytest.mark.parametrize("provided", [False, True])
def test_safe_continuation_preserves_explicit_convergence_threads_and_inputs(provided):
    data, settings = rss_inputs(), fit_settings()
    settings.update(convergence=ConvergenceConfig("joint", min_iter=2), retain_data=True)
    before = copy.deepcopy(data)
    direct = susie_rss(**copy.deepcopy(data), **settings)
    out = run_sensitivity(
        dict(data, overwrite_ld=False),
        [{"name": "continued", "continuation": {"max_iter": 4}}],
        baseline=direct if provided else None,
        baseline_settings={"overwrite_ld": False}
        if provided
        else dict(settings, overwrite_ld=False),
    )
    assert_same_fit(out["fits"]["continued"], continue_fit(direct, max_iter=4))
    assert out["fits"]["baseline"].params["overwrite_ld"] is False
    assert_unchanged(data, before)


def test_overwrite_baseline_continuation_rejects_before_any_refit():
    data = rss_inputs()
    before = copy.deepcopy(data)
    baseline = susie_rss(**copy.deepcopy(data), **fit_settings(), overwrite_ld=True)
    calls = []
    with pytest.raises(ValueError, match=r"Continuation needs retained data.*retain_data=True"):
        run_sensitivity(
            data,
            [
                {"name": "otherwise_valid", "changes": {"L": 2}},
                {"name": "continued", "continuation": {"max_iter": 4}},
            ],
            baseline=baseline,
            fit_function=lambda **kwargs: calls.append(kwargs),
        )
    assert calls == []
    assert_unchanged(data, before)


@pytest.mark.parametrize("location", ["inputs", "settings", "alternative"])
@pytest.mark.parametrize("overwrite", [True, np.bool_(True), 1, "yes", None])
def test_explicit_destructive_or_invalid_policy_rejects_entire_plan(location, overwrite):
    data, settings, alternative = rss_inputs(), fit_settings(), rss_inputs()
    {"inputs": data, "settings": settings, "alternative": alternative}[location]["overwrite_ld"] = (
        overwrite
    )
    before, alternative_before = copy.deepcopy(data), copy.deepcopy(alternative)
    calls = []
    with pytest.raises(ValueError, match=r"overwrite_ld"):
        run_sensitivity(
            data,
            [
                {"name": "otherwise_valid", "changes": {"L": 2}},
                {"name": "alternative", "inputs": alternative},
            ],
            baseline_settings=settings,
            fit_function=lambda **kwargs: calls.append(kwargs),
        )
    assert calls == []
    assert_unchanged(data, before)
    assert_unchanged(alternative, alternative_before)


def test_explicit_false_alternative_preserves_shared_input_mapping():
    data, settings = rss_inputs(), fit_settings()
    alternative = dict(data, overwrite_ld=np.bool_(False))
    before = copy.deepcopy(alternative)
    out = run_sensitivity(
        data,
        [{"name": "alternative", "inputs": alternative}],
        baseline_settings=settings,
    )
    assert_same_fit(out["fits"]["alternative"], susie_rss(**copy.deepcopy(data), **settings))
    assert out["fits"]["alternative"].params["overwrite_ld"] is False
    assert_unchanged(alternative, before)
