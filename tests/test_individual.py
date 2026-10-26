"""Complete component-alpha checks against frozen susieR feature fixtures."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

import prusie

REFERENCE = Path(__file__).parent / "fixtures/susier_parity_0166/reference.json"
FIXTURE = json.loads(REFERENCE.read_text())
CASES = [
    name
    for name, case in FIXTURE["cases"].items()
    if case["api"] in ("susie", "susie_ss") and case["status"] == "ok"
]
SLOT_CASES = [
    name for name in CASES if (FIXTURE["cases"][name].get("special") or "").startswith("slot")
]


@pytest.mark.parametrize("name", SLOT_CASES)
@pytest.mark.parametrize("sparse_design", [False, True])
@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("null_weight", [0.0, 0.1])
def test_slot_prior_predictions_are_activity_weighted_posterior_means(
    name, sparse_design, standardize, null_weight
):
    case = FIXTURE["cases"][name]
    data = FIXTURE["datasets"][case["data"]]
    X, y = np.asarray(data["X"]), np.asarray(data["y"])
    special = case["special"]
    prior = (
        prusie.slot_prior_betabinom(a_beta=1, b_beta=2)
        if special == "slot_betabinom"
        else prusie.slot_prior_poisson(
            3, nu=8, update_schedule="batch" if special.endswith("batch") else "sequential"
        )
    )
    options = dict(
        L=3,
        max_iter=100,
        tol=0.001,
        estimate_residual_variance=False,
        residual_variance=1,
        n_purity="auto",
    )
    options.update(case["options"] or {})
    options.update(slot_prior=prior, standardize=standardize, null_weight=null_weight)
    fit = prusie.susie(sparse.csc_matrix(X) if sparse_design else X, y, **options)
    assert np.any(fit.slot_weights < 0.99), "Exercise an uncertain component's activity"
    centered = (X - X.mean(axis=0)) / fit.X_column_scale_factors[: X.shape[1]]
    # Average each component's conditional prediction over its activity event.
    contributions = centered @ (fit.alpha * fit.mu)[:, : X.shape[1]].T
    prediction = y.mean() + contributions @ fit.slot_weights
    np.testing.assert_allclose(fit.fitted, prediction, atol=1e-12, rtol=0)
    np.testing.assert_allclose(X @ fit.coef + fit.intercept, prediction, atol=1e-12, rtol=0)
    np.testing.assert_allclose(prusie.predict(fit, X), fit.fitted, atol=1e-12, rtol=0)
    full_coef = prusie.coef(fit)
    np.testing.assert_allclose(full_coef[1 : 1 + X.shape[1]], fit.coef, atol=1e-12, rtol=0)
    assert len(full_coef) == 1 + X.shape[1] + bool(null_weight)
    np.testing.assert_allclose(
        fit.intercept, y.mean() - X.mean(axis=0) @ fit.coef, atol=1e-12, rtol=0
    )
    if standardize and not null_weight:
        # Preserve the original component-alpha reference; the historical R
        # intercept export also omitted activity and remains unchanged on disk.
        np.testing.assert_allclose(fit.alpha, case["reference"]["alpha"], atol=1e-5, rtol=0)


@pytest.mark.parametrize("name", CASES)
def test_final_complete_alpha_matches_frozen_R(name):
    case = FIXTURE["cases"][name]
    data = FIXTURE["datasets"][case["data"]]
    options = dict(
        L=3,
        max_iter=100,
        tol=0.001,
        estimate_residual_variance=False,
        residual_variance=1,
        n_purity="auto",
    )
    options.update(case["options"] or {})
    if "na.rm" in options:
        options["na_rm"] = options.pop("na.rm")
    special = case["special"]
    if special == "slot_betabinom":
        options["slot_prior"] = prusie.slot_prior_betabinom(a_beta=1, b_beta=2)
    elif special in ("slot_poisson_sequential", "slot_poisson_batch"):
        options["slot_prior"] = prusie.slot_prior_poisson(
            3, nu=8, update_schedule="batch" if special.endswith("batch") else "sequential"
        )
    if case["api"] == "susie":
        X, y = np.asarray(data["X"]), np.asarray(data["y"])
        if special == "sparse":
            X = sparse.csc_matrix(X)
        if special == "missing_y":
            y = y.copy()
            y[[2, 18]] = np.nan
        fit = prusie.susie(X, y, **options)
    else:
        fit = prusie.susie_ss(data["XtX"], data["Xty"], data["yty"], data["n"], **options)
    expected = np.asarray(case["reference"]["alpha"])
    assert fit.alpha.shape == expected.shape
    assert np.isfinite(fit.alpha).all()
    assert ((fit.alpha >= 0) & (fit.alpha <= 1)).all()
    np.testing.assert_allclose(fit.alpha.sum(axis=1), 1, atol=1e-10, rtol=0)
    np.testing.assert_allclose(fit.alpha, expected, atol=1e-5, rtol=0)


def test_sparse_fitter_does_not_densify_predictors(monkeypatch):
    data = FIXTURE["datasets"]["N2finemapping"]
    X = sparse.csc_matrix(data["X"])
    original = sparse.csc_matrix.toarray

    def reject_predictor_densification(self, *args, **kwargs):
        if self.shape == X.shape:
            raise AssertionError("The full sparse predictor matrix was densified")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(sparse.csc_matrix, "toarray", reject_predictor_densification)
    fit = prusie.susie(
        X,
        data["y"],
        L=3,
        max_iter=100,
        tol=0.001,
        residual_variance=1,
        estimate_residual_variance=False,
    )
    assert fit.alpha.shape == (3, X.shape[1])


def test_init_only_and_sufficient_stat_helpers():
    data = FIXTURE["datasets"]["N2finemapping"]
    inputs = prusie.susie(
        data["X"],
        data["y"],
        L=3,
        init_only=True,
        residual_variance=1,
        estimate_residual_variance=False,
    )
    assert isinstance(inputs, prusie.SusieInputs)
    assert inputs["data"]["X"].shape == np.asarray(data["X"]).shape
    fit = prusie.susie_workhorse(inputs)
    assert fit.alpha.shape[0] == 3
    summary = prusie.compute_suff_stat(data["X"], data["y"])
    reference = FIXTURE["auxiliary"]["compute_suff_stat"]
    for name in ("XtX", "Xty", "yty", "X_colmeans", "y_mean", "n"):
        np.testing.assert_allclose(summary[name], reference[name], atol=1e-10, rtol=1e-12)
    regression = prusie.univariate_regression(data["X"], data["y"])
    for name in ("betahat", "sebetahat"):
        np.testing.assert_allclose(
            regression[name],
            FIXTURE["auxiliary"]["univariate_regression"][name],
            atol=1e-12,
            rtol=1e-12,
        )
