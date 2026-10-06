"""API wiring proof using exact materialization, not the new native operator."""
import warnings
import numpy as np
import pytest
from prusie import _native, susie_rss, susie_suff_stat


def assert_results_equal(a, b):
    for name in ['alpha', 'mu', 'mu2', 'pip', 'V', 'elbo', 'XtXr', 'KL',
                 'lbf', 'lbf_variable', 'sigma2', 'niter', 'converged',
                 'X_column_scale_factors', 'intercept']:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    assert a.params == b.params
    assert a.qc == b.qc
    assert a.warnings == b.warnings
    for name in ['cs_index', 'coverage', 'actual_coverage']:
        np.testing.assert_array_equal(a.sets[name], b.sets[name])
    assert len(a.sets['cs']) == len(b.sets['cs'])
    for x, y in zip(a.sets['cs'], b.sets['cs']):
        np.testing.assert_array_equal(x, y)
    for name in a.sets['purity']:
        np.testing.assert_array_equal(a.sets['purity'][name], b.sets['purity'][name])


@pytest.mark.parametrize('kind', ['known_n', 'missing_n', 'effects', 'no_information'])
@pytest.mark.parametrize('layout', ['C', 'F', 'strided'])
@pytest.mark.parametrize('null_weight', [0., .2])
@pytest.mark.parametrize('standardize', [False, True])
def test_capability_dispatch_exact_materialization(monkeypatch, kind, layout, null_weight, standardize):
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
    expected = {}

    def capture(crossproduct, crossresponse, yty, n, native_options):
        expected['matrix'] = crossproduct.copy()
        expected['response'] = crossresponse.copy()
        expected['options'] = dict(native_options)
        return original(crossproduct, crossresponse, yty, n, native_options)

    monkeypatch.setattr(_native, 'MATRIX_OPERATOR_VERSION', 0, raising=False)
    monkeypatch.setattr(_native, 'fit', capture)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        baseline = function(**args, **options)

    def adapter(raw_matrix, crossresponse, yty, n, native_options):
        native_options = dict(native_options)
        global_scale = native_options.pop('matrix_global_scale')
        inverse = np.asarray(native_options.pop('matrix_inverse_scales'))
        scales = np.asarray(native_options.pop('matrix_scales'))
        prepared = np.array(raw_matrix, dtype=np.float64, order='F', copy=True)
        np.multiply(prepared, global_scale, out=prepared)
        np.multiply(prepared, inverse[None, :], out=prepared)
        np.divide(prepared, scales[:, None], out=prepared)
        np.testing.assert_array_equal(prepared.view(np.uint64), expected['matrix'].view(np.uint64))
        np.testing.assert_array_equal(crossresponse, expected['response'])
        assert native_options == expected['options']
        return original(prepared, crossresponse, yty, n, native_options)

    monkeypatch.setattr(_native, 'MATRIX_OPERATOR_VERSION', 1)
    monkeypatch.setattr(_native, 'fit', adapter)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        actual = function(**args, **options)
    assert_results_equal(baseline, actual)
    np.testing.assert_array_equal(matrix.view(np.uint64), before)


@pytest.mark.parametrize('capability', [0, 2])
def test_unknown_capability_retains_materialized_contract(monkeypatch, capability):
    original = _native.fit
    called = []

    def capture(matrix, response, yty, n, options):
        assert not any(key.startswith('matrix_') for key in options)
        assert matrix.flags.f_contiguous
        called.append(True)
        return original(matrix, response, yty, n, options)

    monkeypatch.setattr(_native, 'MATRIX_OPERATOR_VERSION', capability, raising=False)
    monkeypatch.setattr(_native, 'fit', capture)
    susie_rss([2., 0.], np.eye(2), 100, L=1)
    assert called


def test_capability_keeps_public_invalid_correlation_rejection(monkeypatch):
    monkeypatch.setattr(_native, 'MATRIX_OPERATOR_VERSION', 1, raising=False)
    for matrix in [np.eye(4)[::-1, :], np.full((4, 4), np.nan)]:
        with pytest.raises(ValueError):
            susie_rss(np.arange(4, dtype=float), matrix, 100, L=1)


def test_nonfinite_factors_keep_original_materialization(monkeypatch):
    # Standardization can produce infinite scales while the prepared matrix is
    # finite zero. Such inputs must retain the old API behavior rather than
    # violating the optional operator's finite-positive factor contract.
    original = _native.fit
    calls = []

    def capture(matrix, response, yty, n, options):
        assert not any(key.startswith('matrix_') for key in options)
        assert np.isfinite(matrix).all()
        calls.append(True)
        return original(matrix, response, yty, n, options)

    monkeypatch.setattr(_native, 'MATRIX_OPERATOR_VERSION', 1, raising=False)
    monkeypatch.setattr(_native, 'fit', capture)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        result = susie_suff_stat(np.eye(2) * np.finfo(float).max, [0., 0.],
            np.finfo(float).eps, 1. + np.finfo(float).eps, L=1, max_iter=1,
            estimate_prior_variance=False, estimate_residual_variance=False)
    assert calls
    assert np.isfinite(result.pip).all()
