"""Retained post-analysis is explicit about its existing Gaussian replay scope."""

import numpy as np
import pytest

import prusie
from prusie import _native

X = np.array(
    [
        [0, 1, 0],
        [1, 0, 0],
        [2, 1, 0],
        [0, 2, 1],
        [1, 1, 2],
        [2, 0, 1],
        [0, 0, 2],
        [2, 2, 1],
    ],
    dtype=float,
)
Y = np.array([0.2, 0.8, 1.9, -0.1, 0.7, 2.1, -0.2, 1.6])
R = np.eye(3)
Z = np.array([2.1, 0.5, 0.1])


def call_fitter(kind, **extra):
    options = dict(L=2, coverage=None, max_iter=20, check_prior=False)
    if kind == "mixture":
        options.update(prior_variance_grid=[0.01, 0.1, 0.5], mixture_weights=[0.2, 0.4, 0.4])
    elif kind == "NIG":
        options["estimate_residual_method"] = "NIG"
    elif kind == "slot":
        options["slot_prior"] = prusie.slot_prior_betabinom()
    elif kind == "pip":
        options["convergence_method"] = "pip"
    elif kind == "greedy":
        options["L_greedy"] = 1
    elif kind == "median":
        options["median_abs_corr"] = 0.5
    elif kind == "extension":
        options["cs_extension_corr"] = 0.9
    options.update(extra)
    if kind in {"individual", "CSC"}:
        options.pop("check_prior", None)
        predictors = X
        if kind == "CSC":
            predictors = pytest.importorskip("scipy.sparse").csc_matrix(X)
        return prusie.susie(predictors, Y, **options)
    if kind == "trend":
        options.pop("check_prior")
        return prusie.susie_trendfilter(Y, use_mad=False, **options)
    if kind == "RSS_X":
        return prusie.susie_rss(Z, X=X, n=8, **options)
    if kind == "finite":
        return prusie.susie_rss(Z, R, 8, R_finite=20, **options)
    if kind == "mismatch":
        return prusie.susie_rss(Z, R, 8, R_mismatch="eb", **options)
    if kind == "multipanel":
        return prusie.susie_rss(Z, [R, R], 8, **options)
    if kind == "lambda":
        return prusie.susie_rss_lambda(Z, R, 8, lambda_=0.2, **options)
    return prusie.susie_ss(7 * R, np.sqrt(7) * Z, 7, 8, **options)


UNSUPPORTED = [
    "individual",
    "CSC",
    "trend",
    "RSS_X",
    "finite",
    "mismatch",
    "multipanel",
    "lambda",
    "mixture",
    "NIG",
    "slot",
    "pip",
    "greedy",
    "median",
    "extension",
]


@pytest.fixture
def forbid_scientific_native_calls(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("unsupported explicit retained analysis must fail before fitting")

    for name in ("fit", "fit_design_dense", "fit_design_csc", "fit_design_trend", "ser_update"):
        monkeypatch.setattr(_native, name, fail)


UNSUPPORTED_REQUESTS = {
    "retain_data": {"retain_data": True},
    "refinement_config": {"refine": prusie.RefinementConfig()},
}


@pytest.mark.parametrize(
    "kind,flag",
    [
        (kind, flag)
        for kind in UNSUPPORTED
        for flag in UNSUPPORTED_REQUESTS
        if not (kind == "lambda" and flag == "retain_data")
    ]
    + [("multipanel", "refine")],
)
def test_explicit_unsupported_request_fails_before_native(
    kind, flag, forbid_scientific_native_calls
):
    request = UNSUPPORTED_REQUESTS.get(flag, {"refine": True})
    with pytest.raises(ValueError, match="Retained continuation/refinement is unavailable"):
        call_fitter(kind, **request)


@pytest.mark.parametrize(
    "kind,flag",
    [
        (kind, flag)
        for kind in UNSUPPORTED
        if kind != "multipanel"
        for flag in UNSUPPORTED_REQUESTS
        if not (kind == "lambda" and flag == "retain_data")
    ],
)
def test_init_only_prepares_before_enforcing_retained_analysis(
    kind, flag, forbid_scientific_native_calls
):
    inputs = call_fitter(kind, init_only=True, **UNSUPPORTED_REQUESTS[flag])
    assert isinstance(inputs, prusie.SusieInputs)
    assert inputs._request.prepared is None
    assert inputs._request.signature is None
    with pytest.raises(ValueError, match="Retained continuation/refinement is unavailable"):
        prusie.susie_workhorse(inputs)


@pytest.mark.filterwarnings("ignore:IBSS did not converge:RuntimeWarning")
@pytest.mark.parametrize("kind", UNSUPPORTED)
def test_default_advanced_fit_and_tracking_do_not_expose_unsafe_snapshot(kind):
    fit = call_fitter(kind, track_fit="compact")
    assert isinstance(fit, prusie.SusieResult)
    assert fit._prepared is None
    assert fit.data_signature is None
    assert fit.convergence["retained_analysis_available"] is False
    assert fit.convergence["retained_analysis_limitations"]
    assert np.isfinite(fit.alpha).all()
    np.testing.assert_allclose(fit.alpha.sum(axis=1), 1, atol=1e-10, rtol=0)
    with pytest.raises(ValueError, match="retained data"):
        prusie.evaluate_objective(fit)
    with pytest.raises(ValueError, match="retained data"):
        prusie.continue_fit(fit, max_iter=1)


@pytest.mark.filterwarnings("ignore:IBSS did not converge:RuntimeWarning")
@pytest.mark.parametrize("kind", ["mixture", "NIG", "slot", "pip", "finite", "lambda"])
def test_advanced_warm_start_does_not_enable_legacy_retained_analysis(kind):
    initial = call_fitter(kind)
    fit = call_fitter(kind, model_init=initial)
    assert fit._prepared is None
    assert fit.data_signature is None
    assert fit.convergence["retained_analysis_limitations"]
    np.testing.assert_allclose(fit.alpha.sum(axis=1), 1, atol=1e-10, rtol=0)


@pytest.mark.filterwarnings("ignore:IBSS did not converge:RuntimeWarning")
@pytest.mark.parametrize("route", ["SS", "R_RSS", "X_known_var_y"])
def test_static_materialized_gaussian_retains_existing_helpers(route):
    options = dict(L=2, retain_data=True, coverage=None, check_prior=False)
    if route == "SS":
        fit = prusie.susie_ss(7 * R, np.sqrt(7) * Z, 7, 8, **options)
    elif route == "R_RSS":
        fit = prusie.susie_rss(Z, R, 8, **options)
    else:
        fit = prusie.susie_rss(bhat=0.2 * Z, shat=0.2, X=X, n=8, var_y=1, **options)
    assert fit._prepared is not None
    assert fit.data_signature == fit._prepared.signature
    assert fit.convergence["retained_analysis_limitations"] == []
    assert np.isfinite(prusie.evaluate_objective(fit)["elbo"])
    continued = prusie.continue_fit(fit, max_iter=1)
    assert continued._prepared is fit._prepared


@pytest.mark.filterwarnings("ignore:IBSS did not converge:RuntimeWarning")
def test_requested_pip_rule_keeps_legacy_configuration_separate():
    legacy = prusie.ConvergenceConfig(criterion="joint", alpha_tol=0.1, pip_tol=0.2)
    fit = call_fitter("pip", convergence=legacy, tol=0.002)
    record = fit.convergence
    assert record["requested_convergence_method"] == "pip"
    assert record["requested_criterion"] == "pip"
    assert record["requested_config"]["criterion"] == "pip"
    assert record["requested_config"]["alpha_tol"] == 0.002
    assert record["requested_config"]["pip_tol"] == 0.002
    assert record["alpha_tol"] == 0.002
    assert record["pip_tol"] == 0.002
    assert record["legacy_requested_config"] == legacy.options()
    assert fit.params["convergence"] == legacy.options()


@pytest.mark.filterwarnings("ignore:IBSS did not converge:RuntimeWarning")
def test_dynamic_rule_override_preserves_callers_requested_method():
    fit = call_fitter("finite", convergence_method="elbo")
    assert fit.convergence["requested_convergence_method"] == "elbo"
    assert fit.convergence["requested_criterion"] == "pip"
    assert fit.convergence["requested_config"]["criterion"] == "pip"
    assert fit.convergence["legacy_requested_config"]["criterion"] == "elbo"


@pytest.mark.filterwarnings("ignore:IBSS did not converge:RuntimeWarning")
@pytest.mark.parametrize("kind", [kind for kind in UNSUPPORTED if kind != "multipanel"])
def test_refine_true_refits_models_outside_the_retained_scope(kind):
    fit = call_fitter(kind, refine=True, coverage=0.5, min_abs_corr=0.0)
    assert isinstance(fit, prusie.SusieResult)
    assert fit._prepared is None
    assert np.isfinite(fit.alpha).all()
    if fit.refinement_status == "not_executed":
        assert not fit.sets["cs"]
    else:
        assert fit.refinement["method"] == "block_coordinate_ascent"
        assert fit.refinement["total_fits"] % 2 == 0
