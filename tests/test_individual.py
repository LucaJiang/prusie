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
