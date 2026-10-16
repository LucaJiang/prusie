"""Analytic and native checks for the deliberately restricted NumPy oracle."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp

import prusie

_path = Path(__file__).with_name("reference_numpy.py")
_spec = spec_from_file_location("restricted_numpy_susie", _path)
ref = module_from_spec(_spec)
_spec.loader.exec_module(ref)


def inputs():
    return dict(
        XtX=np.array([[99.0, 29.7, 0.0], [29.7, 99.0, 9.9], [0.0, 9.9, 99.0]]),
        Xty=np.array([45.0, 20.0, -25.0]),
        yty=99.0,
        n=100,
        L=2,
        max_iter=100,
        estimate_residual_variance=False,
    )


@pytest.mark.parametrize("estimate_sigma", [False, True])
@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("method", ["optim", "EM", "simple", "fixed"])
def test_restricted_reference_native_probability_agreement(estimate_sigma, standardize, method):
    kw = dict(
        inputs(),
        estimate_residual_variance=estimate_sigma,
        standardize=standardize,
        prior_weights=[1.0, 2.0, 3.0],
        estimate_prior_variance=method != "fixed",
    )
    method = "optim" if method == "fixed" else method
    a = ref.susie_ss_ref(**kw, method=method)
    b = prusie.susie_suff_stat(**kw, estimate_prior_method=method)
    np.testing.assert_allclose(a["pip"], b.pip, atol=1e-5, rtol=0)
    np.testing.assert_allclose(a["alpha"].sum(1), 1.0, atol=1e-12, rtol=0)
    assert np.isfinite(a["elbo"]).all()


@pytest.mark.parametrize("v", [0.0, 0.2])
def test_one_effect_closed_form(v):
    # One sweep of a fixed-V SER has a direct normal-mixture posterior.
    q = np.diag([20.0, 50.0])
    u = np.array([4.0, -5.0])
    sigma = 1.2
    pi = np.array([0.25, 0.75])
    s = sigma / np.diag(q)
    bh = u / np.diag(q)
    lb = -0.5 * np.log1p(v / s) + 0.5 * bh**2 * v / (s * (v + s))
    expected_alpha = np.exp(lb + np.log(pi) - logsumexp(lb + np.log(pi)))
    variance = v * s / (v + s)
    expected_mu = variance / s * bh
    f = ref.susie_ss_ref(
        q,
        u,
        99.0,
        100,
        L=1,
        scaled_prior_variance=v,
        residual_variance=sigma,
        estimate_residual_variance=False,
        estimate_prior_variance=False,
        standardize=False,
        max_iter=1,
        prior_weights=pi,
    )
    if v:
        np.testing.assert_allclose(f["alpha"][0], expected_alpha, atol=1e-15, rtol=0)
        np.testing.assert_allclose(f["mu"][0], expected_mu, atol=1e-15, rtol=0)
        np.testing.assert_allclose(f["mu2"][0], variance + expected_mu**2, atol=1e-15, rtol=0)
    else:
        np.testing.assert_array_equal(f["alpha"][0], pi)
        np.testing.assert_array_equal(f["pip"], [0.0, 0.0])
        np.testing.assert_array_equal(f["mu2"], [[0.0, 0.0]])
    assert not f["converged"] and f["niter"] == 1


@pytest.mark.parametrize(
    "override",
    [
        {"XtX": np.diag([99.0, 99.0, 0.0])},
        {"XtX": np.diag([99.0, 99.0, -1.0])},
        {"XtX": np.diag([99.0, 99.0, 1e-20])},
        {"XtX": np.array([[99.0, 200.0, 0.0], [200.0, 99.0, 0.0], [0.0, 0.0, 99.0]])},
        {"Xty": [1.0, np.nan, 2.0]},
        {"Xty": [1.0, np.inf, 2.0]},
        {"Xty": [1.0, 1e200, 2.0]},
        {"yty": np.inf},
        {"n": 2},
        {"residual_variance": 0.0},
        {"residual_variance": 1e-100},
        {"prior_weights": [0.0, 1.0, 1.0]},
        {"prior_weights": [-1.0, 1.0, 1.0]},
        {"prior_weights": [1e-200, 1.0, 1.0]},
        {"prior_weights": [1.0, np.nan, 1.0]},
        {"prior_weights": [1.0, 2.0]},
        {"method": "unsupported"},
        {"L": 0},
        {"max_iter": 0},
    ],
)
def test_reference_rejects_unsupported_boundaries(override):
    with pytest.raises(ValueError):
        ref.susie_ss_ref(**dict(inputs(), **override))


def test_unsupported_api_is_not_silently_ignored():
    with pytest.raises(TypeError):
        ref.susie_ss_ref(**inputs(), check_prior=True)
    with pytest.raises(TypeError):
        ref.susie_ss_ref(**inputs(), refine=True)
    with pytest.raises(ValueError):
        ref._lbf(np.ones(2), np.array([1e-20, 1.0]), 0.2)


def test_exact_weights_are_scale_invariant():
    kw = inputs()
    a = ref.susie_ss_ref(**kw, prior_weights=[1.0, 2.0, 3.0])
    b = ref.susie_ss_ref(**kw, prior_weights=[2.0, 4.0, 6.0])
    for key in ("pip", "alpha", "mu", "mu2", "V", "elbo"):
        np.testing.assert_array_equal(a[key], b[key])


def test_reference_cs_coverage_purity_and_deduplication():
    alpha = np.array([[0.6, 0.35, 0.05], [0.7, 0.26, 0.04], [0.01, 0.01, 0.98]])
    corr = np.array([[1.0, 0.8, 0.0], [0.8, 1.0, 0.1], [0.0, 0.1, 1.0]])
    cs = ref.credible_sets_ref(alpha, np.ones(3), corr, coverage=0.9, min_abs_corr=0.5)
    assert [x[:2] for x in cs] == [(0, [0, 1]), (2, [2])]
    assert cs[0][2] == pytest.approx(0.95) and cs[0][3] == 0.8
    assert cs[1][2] == 0.98 and cs[1][3] == 1.0
