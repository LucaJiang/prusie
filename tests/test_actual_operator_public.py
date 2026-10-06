"""Independent D check of actual native operator on existing analytical cases."""
import warnings
import numpy as np
import pytest
from prusie import _native, susie_rss, susie_suff_stat

@pytest.mark.parametrize('kind', ['known_n', 'missing_n', 'effects', 'no_information'])
@pytest.mark.parametrize('layout', ['C', 'F', 'strided'])
@pytest.mark.parametrize('null_weight', [0., .2])
@pytest.mark.parametrize('standardize', [False, True])
def test_actual_operator_public_boundary(monkeypatch, kind, layout, null_weight, standardize):
    r = np.array([[1. + 7e-9, -.2, -0.], [-.2, 1., .125],
                  [0., .125 + 1e-14, 1. - 7e-9]])
    options = dict(L=2, null_weight=null_weight, standardize=standardize,
                   prior_weights=[.7, 0., .3], coverage=None if null_weight else .95,
                   check_input=kind in {'known_n', 'no_information'})
    if kind == 'no_information':
        matrix = r * 99.
        matrix[-1, :] = matrix[:, -1] = 0.
        function = susie_suff_stat
        args = dict(Xty=[12., 2., 0.], yty=99., n=100., estimate_residual_variance=False)
        matrix_key = 'XtX'
    else:
        matrix = r
        function = susie_rss
        args = dict(n=None if kind == 'missing_n' else 100.)
        args.update(dict(bhat=[.3, .1, -.2], shat=[.1, .2, .3], var_y=1.4)
                    if kind == 'effects' else dict(z=[3., 1., -.2]))
        matrix_key = 'R'
    if layout == 'F':
        matrix = np.asfortranarray(matrix)
    elif layout == 'strided':
        backing = np.empty((6, 6))
        backing[::2, ::2] = matrix
        matrix = backing[::2, ::2]
    matrix.flags.writeable = False
    before = matrix.view(np.uint64).copy()
    args[matrix_key] = matrix

    original = _native.fit
    observed = []
    def capture(matrix, response, yty, n, settings):
        observed.append('matrix_global_scale' in settings)
        return original(matrix, response, yty, n, settings)
    monkeypatch.setattr(_native, 'fit', capture)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        monkeypatch.setattr(_native, 'MATRIX_OPERATOR_VERSION', 0)
        expected = function(**args, **options)
        monkeypatch.setattr(_native, 'MATRIX_OPERATOR_VERSION', 1)
        actual = function(**args, **options)
    assert observed == [False, True]
    assert np.all(np.isfinite(actual.pip))
    assert np.all((actual.pip >= 0) & (actual.pip <= 1))
    np.testing.assert_allclose(actual.pip, expected.pip, atol=1e-5, rtol=0)
    np.testing.assert_array_equal(actual.variant_ids, expected.variant_ids)
    assert actual.alpha.shape == expected.alpha.shape
    assert actual.params == expected.params
    np.testing.assert_array_equal(matrix.view(np.uint64), before)
