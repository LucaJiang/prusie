"""Analytical RSS reconstruction, independent of real phenotype metadata."""
import warnings
import numpy as np
import pytest
import prusie


@pytest.mark.parametrize('layout', ['C', 'F', 'strided', 'readonly'])
@pytest.mark.parametrize('standardize,null_weight,estimate_residual', [
    (True, 0., False), (False, 0., False), (True, .1, False),
    (True, 0., True), (False, .1, True),
])
@pytest.mark.parametrize('asymmetry', [0., 1e-14])
def test_known_variance_matches_explicit_sufficient_statistics(
        layout, standardize, null_weight, estimate_residual, asymmetry):
    # A fixed, positive definite analytical correlation matrix; no simulated data.
    R = np.array([[1., .4, .2, 0.], [.4, 1., .1, .2],
                  [.2, .1, 1., .3], [0., .2, .3, 1.]])
    R[0, 1] += asymmetry
    if layout == 'F':
        R = np.asfortranarray(R)
    elif layout == 'strided':
        backing = np.zeros((4, 8)); backing[:, ::2] = R; R = backing[:, ::2]
    elif layout == 'readonly':
        R.flags.writeable = False
    beta = np.array([.3, -.1, .02, .04])
    se = np.array([.07, .1, .03, .05])
    original = [x.copy() for x in (R, beta, se)]
    n, variance = 101., 3.5
    z = beta / se
    adjustment = (n - 1) / (z * z + n - 2)
    sd = np.sqrt(variance * adjustment / (se * se))
    raw = R * sd[:, None] * sd[None, :]
    crossproduct = (raw + raw.T) / 2
    crossresponse = (np.sqrt(adjustment) * z) * np.sqrt(adjustment) * variance / se
    options = dict(L=2, max_iter=100, tol=.001, standardize=standardize,
                   null_weight=null_weight, prior_weights=[1., 0., 2., 1.],
                   estimate_residual_variance=estimate_residual,
                   scaled_prior_variance=.2, check_prior=False,
                   variant_ids=['a', 'b', 'c', 'd'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        expected = prusie.susie_suff_stat(crossproduct, crossresponse,
                                           (n - 1) * variance, n, **options)
        actual = prusie.susie_rss(R=R, n=n, bhat=beta, shat=se,
                                   var_y=variance, **options)
    for field in ['alpha', 'mu', 'mu2', 'lbf_variable', 'lbf', 'V', 'sigma2',
                  'pip', 'elbo', 'KL', 'XtXr', 'X_column_scale_factors']:
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    assert actual.niter == expected.niter
    assert actual.converged == expected.converged
    assert actual.sets == expected.sets
    for array, before in zip((R, beta, se), original):
        np.testing.assert_array_equal(array, before)


@pytest.mark.parametrize('shat', [.1, np.array([.1, .1])])
def test_scalar_and_vector_standard_errors_known_variance(shat):
    actual = prusie.susie_rss(R=np.eye(2), n=100, bhat=[.5, 0.],
                               shat=shat, var_y=2., L=1)
    explicit = prusie.susie_rss(R=np.eye(2), n=100, bhat=[.5, 0.],
                                 shat=np.array([.1, .1]), var_y=2., L=1)
    np.testing.assert_array_equal(actual.pip, explicit.pip)
