"""Hand-calculated and invariant tests independent of R and real-data fixtures."""
import os
import subprocess
import sys

import numpy as np
import pytest

from prusie import _native, susie_rss, susie_suff_stat
from prusie.posterior import credible_sets, marginal_pip


def test_fixed_single_effect_against_closed_form():
    z = np.array([4., -1., 0.])
    n, V = 100., 0.2
    adjusted = z * np.sqrt((n - 1) / (z * z + n - 2))
    Xty = np.sqrt(n - 1) * adjusted
    fit = susie_rss(z, np.eye(3), n, L=1, estimate_prior_variance=False,
                    scaled_prior_variance=V)
    postvar = 1 / (1 / V + n - 1)
    mu = postvar * Xty
    lbf = -.5 * np.log1p(V * (n - 1)) + .5 * Xty * Xty * postvar
    w = np.exp(lbf - max(lbf))
    np.testing.assert_allclose(fit.alpha[0], w / sum(w), atol=1e-13)
    np.testing.assert_allclose(fit.mu[0], mu, atol=1e-13)
    np.testing.assert_allclose(fit.mu2[0], postvar + mu * mu, atol=1e-13)
    np.testing.assert_allclose(fit.lbf_variable[0], lbf, atol=1e-13)
    assert fit.niter == 2 and fit.converged
    assert fit.backend_version.startswith("prusie-rust/")


@pytest.mark.parametrize('capability', [0, 1])
def test_rss_suff_stat_same_transformation(monkeypatch, record_property, capability):
    if capability and getattr(_native, 'MATRIX_OPERATOR_VERSION', 0) != 1:
        pytest.skip('This native build has no actual matrix operator')
    monkeypatch.setattr(_native, 'MATRIX_OPERATOR_VERSION', capability, raising=False)
    z = np.array([3., 2., -1.])
    R = np.array([[1., .2, -.1], [.2, 1., 0.], [-.1, 0., 1.]])
    n = 191.
    adjusted = z * np.sqrt((n - 1) / (z * z + n - 2))
    before = (z.copy(), R.copy())
    a = susie_rss(z, R, n, L=2, estimate_prior_variance=False)
    b = susie_suff_stat((n - 1) * R, np.sqrt(n - 1) * adjusted, n - 1, n,
                       L=2, estimate_prior_variance=False, estimate_residual_variance=False)
    np.testing.assert_array_equal(z, before[0])
    np.testing.assert_array_equal(R, before[1])
    np.testing.assert_allclose(a.pip, b.pip, atol=1e-5, rtol=0)
    for fit in [a, b]:
        assert np.isfinite(fit.pip).all() and ((fit.pip >= 0) & (fit.pip <= 1)).all()
        assert np.isfinite(fit.alpha).all() and ((fit.alpha >= 0) & (fit.alpha <= 1)).all()
    for name in ("alpha", "mu", "mu2", "V", "elbo", "lbf_variable"):
        left, right = getattr(a, name), getattr(b, name)
        if not capability:
            np.testing.assert_array_equal(left, right)
        else:
            record_property(name + '_max_abs', float(np.max(np.abs(left - right))))
    record_property('iterations', [a.niter, b.niter])


def test_original_scale_equivariance():
    beta, se, n = np.array([.2, -.1, .05]), np.array([.05, .04, .02]), 191.
    kwargs = dict(R=np.eye(3), n=n, L=1, estimate_prior_variance=False)
    a = susie_rss(bhat=beta, shat=se, var_y=2., **kwargs)
    b = susie_rss(bhat=beta * 3, shat=se * 3, var_y=18., **kwargs)
    np.testing.assert_allclose(a.alpha, b.alpha, atol=1e-13)
    np.testing.assert_allclose(b.coef, a.coef * 3, atol=1e-13)
    np.testing.assert_allclose(b.sigma2, a.sigma2 * 9, atol=1e-13)


def test_sequential_ibss_changes_later_effect():
    args = dict(z=[8., 3., 1.], R=np.eye(3), n=100., L=2,
                estimate_prior_variance=False, max_iter=1)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        fit = susie_rss(**args)
    # A simultaneous update would produce identical component rows.
    assert not np.allclose(fit.alpha[0], fit.alpha[1])
    assert fit.niter == 1 and not fit.converged


def test_null_and_zero_variance_are_explicit():
    fit = susie_rss([0., 0.], np.eye(2), 100, L=2,
                    null_weight=.1, scaled_prior_variance=0,
                    estimate_prior_variance=False, variant_ids=["v1", "v2"])
    assert fit.alpha.shape == (2, 3)
    assert fit.null_index == 2
    np.testing.assert_array_equal(fit.pip, [0., 0.])
    assert fit.sets["cs"] == []
    np.testing.assert_array_equal(fit.mu, np.zeros((2, 3)))
    np.testing.assert_array_equal(fit.mu2, np.zeros((2, 3)))


def test_singular_correlation_allowed():
    fit = susie_rss([3., 3.], np.ones((2, 2)), 100, L=1,
                    estimate_prior_variance=False, check_input=True)
    np.testing.assert_allclose(fit.alpha, [[.5, .5]], atol=1e-12)
    np.testing.assert_array_equal(fit.sets["cs"][0], [0, 1])


def test_posterior_dedup_prior_filter_and_tie_order():
    alpha = np.array([[.5, .5, 0.], [.5, .5, 0.], [.01, .01, .98]])
    result = credible_sets(alpha, np.array([.2, .2, 0.]), np.ones((3, 3)))
    np.testing.assert_array_equal(result["cs_index"], [0])
    np.testing.assert_array_equal(result["cs"][0], [0, 1])
    np.testing.assert_allclose(marginal_pip(alpha, [1., 0., 0.]), [.5, .5, 0.])


@pytest.mark.parametrize("kwargs, match", [
    ({"z": [np.nan, 0.]}, "finite"),
    ({"R": [[1., .4], [.2, 1.]]}, "symmetric"),
    ({"R": [[1., 2.], [2., 1.]]}, "correlation"),
    ({"n": 1}, "n must"),
    ({"prior_weights": [0., 0.]}, "positive sum"),
    ({"prior_weights": [-1., 2.]}, "nonnegative"),
    ({"variant_ids": ["x", "x"]}, "distinct"),
    ({"L": 0}, "positive integer"),
    ({"bhat": [1., 2.], "shat": [1., 1.]}, "either"),
])
def test_invalid_inputs_error(kwargs, match):
    args = dict(z=[1., 0.], R=np.eye(2), n=100)
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        susie_rss(**args)


@pytest.mark.parametrize("option", ["refine", "track_fit", "verbose"])
def test_unsupported_options_are_not_ignored(option):
    with pytest.raises(NotImplementedError, match=option):
        susie_rss([1., 0.], np.eye(2), 100, **{option: True})


def test_no_r_executable_required():
    environment = os.environ.copy()
    environment["PATH"] = ""
    code = "import shutil; assert shutil.which('R') is None; from prusie import susie_rss; f=susie_rss([3.,0.],[[1.,0.],[0.,1.]],100,L=1); assert f.converged"
    subprocess.run([sys.executable, "-c", code], env=environment, check=True)
