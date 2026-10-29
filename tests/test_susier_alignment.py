"""Gaussian preprocessing and refinement against independent pinned-R results."""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from prusie import _native, susie_rss, susie_suff_stat

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/pinned_susier_checks.json").read_text(encoding="utf-8")
)


def numeric(value):
    if isinstance(value, list):
        return [numeric(item) for item in value]
    if isinstance(value, str) and value in ("NaN", "Inf", "-Inf"):
        return float(value)
    return value


@pytest.mark.parametrize("case", FIXTURE["boundaries"], ids=lambda case: case["id"])
def test_pinned_r_valid_gaussian_boundaries(case):
    arguments = dict(
        L=1,
        estimate_residual_variance=False,
        estimate_prior_variance=False,
        residual_variance=1,
        max_iter=100,
    )
    if case["api"] == "rss":
        arguments.update(z=[3.0, 0.0], R=np.eye(2), n=100)
        fit_function = susie_rss
    else:
        arguments.update(XtX=99 * np.eye(2), Xty=[30.0, 0.0], yty=99, n=100)
        fit_function = susie_suff_stat
    arguments.update({key: numeric(value) for key, value in case["args"].items()})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fit = fit_function(**arguments)
    assert np.isfinite(fit.pip).all() and ((fit.pip >= 0) & (fit.pip <= 1)).all()
    assert np.max(np.abs(fit.pip - case["pip"])) < 1e-5


@pytest.mark.parametrize("case", FIXTURE["refinement"], ids=lambda case: case["id"])
def test_pinned_r_default_refinement_and_original_scale(case):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        arguments = dict(case["args"])
        if case.get("initial_state"):
            arguments["s_init"] = case["initial_state"]
        fit = susie_rss(**arguments)
    reference = case["reference"]
    assert np.max(np.abs(fit.pip - np.atleast_1d(reference["pip"]))) < 1e-5
    assert fit.converged == reference["converged"]
    if case["args"].get("refine"):
        assert fit.refinement["config"]["max_rounds"] == 100
        assert fit.refinement["config"]["max_iter"] == case["args"].get("max_iter", 50)
        assert fit.refinement["config"]["absolute_tol"] == case["args"].get("tol", 0.0001)


@pytest.mark.parametrize("entry", ["z", "Xty"])
def test_missing_summary_is_zeroed_without_mutating_readonly_input(entry):
    values = np.array([np.nan, 0.0])
    values.flags.writeable = False
    before = values.tobytes()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if entry == "z":
            fit = susie_rss(values, np.eye(2), 100, L=1, check_prior=False)
        else:
            fit = susie_suff_stat(
                99 * np.eye(2), values, 99, 100, L=1, estimate_residual_variance=False
            )
    assert values.tobytes() == before and not values.flags.writeable
    assert np.isfinite(fit.pip).all()
    assert any("Xty are replaced" in str(item.message) for item in caught) == (entry == "Xty")


def test_default_uses_implicit_scaling_with_materialized_native_oracle(
    monkeypatch, record_property
):
    assert _native.MATRIX_OPERATOR_VERSION == 1
    original = _native.fit
    settings_seen = []

    def capture(matrix, response, yty, n, settings):
        settings_seen.append(dict(settings))
        return original(matrix, response, yty, n, settings)

    matrix = np.array([[1.0, 0.2], [0.2, 1.0]])
    z = np.array([3.0, 0.0])
    before = [array.tobytes() for array in (matrix, z)]
    monkeypatch.setattr(_native, "fit", capture)
    fit = susie_rss(z, matrix, 100, L=1, variant_ids=["second", "first"])
    assert len(settings_seen) == 1
    settings = settings_seen[0]
    assert settings["matrix_global_scale"] == 99
    assert np.array_equal(settings["matrix_scales"], [1.0, 1.0])
    working = np.asfortranarray(99 * matrix)
    response = np.ascontiguousarray(np.sqrt(99) * np.sqrt(99 / (z**2 + 98)) * z)
    oracle_settings = {
        key: value for key, value in settings.items() if not key.startswith("matrix_")
    }
    expected = original(working, response, 99, 100, oracle_settings)
    assert fit.alpha.shape == expected["alpha"].shape == (1, 2)
    assert fit.variant_ids.tolist() == ["second", "first"]
    for alpha in (fit.alpha, expected["alpha"]):
        assert np.isfinite(alpha).all() and ((alpha >= 0) & (alpha <= 1)).all()
        np.testing.assert_allclose(alpha.sum(axis=1), 1, atol=1e-10, rtol=0)
    np.testing.assert_allclose(fit.alpha, expected["alpha"], atol=1e-5, rtol=0)
    for field in ("mu", "mu2", "V", "sigma2", "elbo", "lbf", "KL"):
        record_property(
            field + "_max_abs",
            float(np.max(np.abs(np.asarray(getattr(fit, field)) - np.asarray(expected[field])))),
        )
    record_property("niter_values", [fit.niter, expected["niter"]])
    record_property("converged_values", [fit.converged, expected["converged"]])
    assert [array.tobytes() for array in (matrix, z)] == before


@pytest.mark.parametrize("scale", [1e-100, 1.0, 1e100])
def test_projection_check_has_no_absolute_response_floor(scale):
    with pytest.warns(RuntimeWarning, match="nonzero eigenspace"):
        fit = susie_suff_stat(
            [[99.0, 0.0], [0.0, 0.0]],
            [scale, scale * 1e-6],
            99,
            100,
            L=1,
            check_input=True,
            check_prior=False,
            estimate_prior_variance=False,
            estimate_residual_variance=False,
        )
    assert fit.qc["projection_squared_fraction"] > np.finfo(float).eps


def test_disabled_estimation_does_not_validate_unused_method():
    options = dict(L=1, estimate_prior_variance=False)
    first = susie_rss([3.0, 0.0], np.eye(2), 100, **options)
    second = susie_rss([3.0, 0.0], np.eye(2), 100, estimate_prior_method="unused", **options)
    np.testing.assert_array_equal(first.pip, second.pip)


@pytest.mark.parametrize("shat", [[np.inf, 0.1], np.inf])
def test_positive_infinite_standard_errors_follow_zero_z_route(shat):
    options = dict(L=1, check_prior=False, estimate_prior_variance=False, max_iter=100)
    beta = np.array([0.3, 0.2])
    fit = susie_rss(bhat=beta, shat=shat, R=np.eye(2), n=100, **options)
    expected = susie_rss(z=beta / shat, R=np.eye(2), n=100, **options)
    np.testing.assert_array_equal(fit.pip, expected.pip)
    if np.ndim(shat):
        np.testing.assert_allclose(
            fit.pip, [0.13612544949868066, 0.8638745505013193], atol=1e-5, rtol=0
        )


def test_infinite_objective_raises_like_pinned_r():
    with pytest.raises(ValueError, match="produced an infinite ELBO"):
        susie_suff_stat(
            np.eye(2),
            [0.0, 0.0],
            1e308,
            1.7e308,
            L=1,
            standardize=False,
            scaled_prior_variance=0,
            estimate_prior_variance=False,
            estimate_residual_variance=False,
            residual_variance=1,
        )


def test_warm_start_accepts_new_summaries_without_retained_data():
    arguments = dict(L=1, estimate_prior_variance=False, residual_variance=1, max_iter=100)
    original = susie_rss([3.0, 0.0], np.eye(2), 100, **arguments)
    assert original.data_signature is None
    warm = susie_rss([2.0, 0.0], np.eye(2), 100, s_init=original, **arguments)
    cold = susie_rss([2.0, 0.0], np.eye(2), 100, **arguments)
    np.testing.assert_allclose(warm.pip, cold.pip, atol=1e-5, rtol=0)
