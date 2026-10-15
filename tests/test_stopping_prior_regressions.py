"""Public regressions for signed ELBO stopping and the current-sigma2 guard."""

import warnings
from pathlib import Path

import numpy as np
import pytest

from prusie import ConvergenceConfig, susie_rss, susie_suff_stat


def cycle_case(**kwargs):
    # Frozen from the complete reported seed-312 generator, including its
    # unused p-by-p normal draw. Provenance accompanies the fixture.
    with np.load(Path(__file__).parent / "data" / "cycle_seed312.npz") as data:
        args = dict(z=data["z"], R=data["R"], n=5000, L=5, check_prior=False, max_iter=100)
    args.update(kwargs)
    return susie_rss(**args)


def assert_same_fit(a, b):
    for field in ("alpha", "mu", "mu2", "V", "pip", "elbo", "KL", "XtXr", "lbf", "lbf_variable"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    assert (a.niter, a.converged, a.sigma2) == (b.niter, b.converged, b.sigma2)
    assert a.convergence["stop_reason"] == b.convergence["stop_reason"]
    assert [x.tolist() for x in a.sets["cs"]] == [x.tolist() for x in b.sets["cs"]]


@pytest.mark.parametrize("tracking", [False, "compact", "dense"])
def test_seed312_default_elbo_cycles_are_diagnostic_only(tracking):
    default = cycle_case(track_fit=tracking)
    explicit = cycle_case(track_fit=tracking, convergence=ConvergenceConfig("elbo"))
    lag1 = cycle_case(track_fit=tracking, convergence=ConvergenceConfig(cycle_lag=1))
    assert default.converged
    assert default.convergence["stop_reason"] == "criterion_satisfied"
    assert_same_fit(default, explicit)
    assert_same_fit(default, lag1)
    assert_same_fit(default, cycle_case())
    if tracking:
        events = [row for row in default.trace if "cycle_detected" in row["events"]]
        assert events, "This input must exercise the previously premature exit"
        assert any(row["delta_elbo"] > default.params["tol"] for row in events)


def test_cycle_diagnostics_respect_minimum_and_iteration_limit():
    with pytest.warns(RuntimeWarning, match="iteration_limit"):
        f = cycle_case(max_iter=30, convergence=ConvergenceConfig(min_iter=60), track_fit=True)
    assert f.niter == 30 and not f.converged
    assert f.convergence["stop_reason"] == "iteration_limit"
    assert any("cycle_detected" in row["events"] for row in f.trace)
    g = cycle_case(max_iter=100, convergence=ConvergenceConfig(min_iter=60))
    assert g.niter >= 60 and g.converged


@pytest.mark.parametrize("criterion", ["alpha_pip", "joint"])
def test_explicit_posterior_modes_never_stop_before_minimum(criterion):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f = cycle_case(max_iter=30, convergence=ConvergenceConfig(criterion, min_iter=60))
    assert f.niter == 30 and not f.converged
    assert f.convergence["stop_reason"] == "iteration_limit"


def guard_case(*, standardize, estimate_sigma, method):
    return dict(
        XtX=9 * np.eye(2),
        Xty=[3.0, 0.0],
        yty=100000.0,
        n=10,
        L=1,
        scaled_prior_variance=0.0045,
        residual_variance=1.0,
        estimate_prior_variance=method != "fixed",
        estimate_prior_method="optim" if method == "fixed" else method,
        estimate_residual_variance=estimate_sigma,
        standardize=standardize,
        check_prior=True,
        max_iter=5,
    )


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("method", ["fixed", "EM"])
def test_prior_guard_changes_decision_after_sigma2_update(standardize, method):
    args = guard_case(standardize=standardize, estimate_sigma=True, method=method)
    # The first sweep is admissible under the initial variance. On the next
    # sweep sigma2 is much larger and the 100*max(z^2) bound is much smaller.
    with pytest.warns(RuntimeWarning, match="iteration_limit"):
        first = susie_suff_stat(**dict(args, max_iter=1))
    assert first.sigma2 > 1000.0
    assert np.max(first.V) <= 100.0
    with pytest.raises(ValueError, match=r"prior variance is unreasonably large"):
        susie_suff_stat(**args)
    fixed = susie_suff_stat(**dict(args, estimate_residual_variance=False))
    assert fixed.sigma2 == 1.0
    unchecked = susie_suff_stat(**dict(args, check_prior=False, max_iter=2))
    assert np.max(unchecked.V) > 100.0 / first.sigma2


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("estimate_sigma", [False, True])
@pytest.mark.parametrize("method", ["optim", "EM", "simple", "fixed"])
def test_prior_guard_safe_case_and_zero_information(standardize, estimate_sigma, method):
    args = dict(
        XtX=np.diag([99.0, 99.0, 0.0]),
        Xty=[30.0, 2.0, 0.0],
        yty=99.0,
        n=100,
        L=2,
        standardize=standardize,
        estimate_residual_variance=estimate_sigma,
        estimate_prior_variance=method != "fixed",
        estimate_prior_method="optim" if method == "fixed" else method,
    )
    guarded = susie_suff_stat(**args, check_prior=True)
    unchecked = susie_suff_stat(**args, check_prior=False)
    assert_same_fit(guarded, unchecked)
    assert np.isfinite(guarded.pip).all()


def test_first_finite_objective_is_not_a_nonfinite_fallback():
    with pytest.warns(RuntimeWarning, match="iteration_limit"):
        f = susie_rss(z=[2.0, 0.0], R=np.eye(2), n=100, L=1, max_iter=1)
    assert f.convergence["effective_criterion"] == "elbo"
    assert f.convergence["fallback_iterations"] == []
    assert f.convergence["nonfinite_elbo_count"] == 0


def test_infinite_objective_reports_numerical_failure_before_minimum():
    with pytest.warns(RuntimeWarning, match="numerical_failure"):
        f = susie_suff_stat(
            [[1.0]],
            [0.0],
            1e308,
            10,
            L=1,
            scaled_prior_variance=0.0,
            residual_variance=1e308,
            estimate_prior_variance=False,
            estimate_residual_variance=False,
            standardize=False,
            convergence=ConvergenceConfig(min_iter=50),
        )
    assert f.niter == 1 and not f.converged
    assert f.convergence["stop_reason"] == "numerical_failure"
    assert not f.convergence["elbo_valid"]
