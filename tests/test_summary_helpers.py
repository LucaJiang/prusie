"""Analytical tests for complete, immutable native API/CS helpers."""
import numpy as np
import pytest

from prusie import _native, susie_rss, susie_suff_stat
from prusie.api import _matrix


def layouts(a):
    backing = np.empty((2 * len(a), 2 * len(a)))
    backing[::2, ::2] = a
    return [a.copy(), np.asfortranarray(a), backing[::2, ::2], a[::-1, ::-1]]


@pytest.mark.parametrize('size', [1, 31, 32, 33, 65, 130])
@pytest.mark.parametrize('defect', ['none', 'nan', 'inf', 'asymmetric', 'negative_diagonal'])
def test_complete_matrix_validation(size, defect):
    a = np.eye(size)
    if defect == 'nan':
        a[-1, 0] = np.nan
    elif defect == 'inf':
        a[0, -1] = np.inf
    elif defect == 'asymmetric':
        if size == 1:
            return
        a[0, -1] = .1
    elif defect == 'negative_diagonal':
        a[-1, -1] = -1
    for value in layouts(a):
        before = value.copy()
        if defect == 'none':
            np.testing.assert_array_equal(_matrix(value, 'XtX', size), value)
        else:
            with pytest.raises(ValueError):
                _matrix(value, 'XtX', size)
        np.testing.assert_array_equal(value, before)


@pytest.mark.parametrize('scale', [0., 1e-12, 1., 1e8, 1e200])
def test_symmetry_tolerance_matches_full_numpy(scale):
    tolerance = 1e-12 + 1e-12 * abs(scale)
    for delta in [0., tolerance * .999, tolerance, tolerance * 1.001, 2 * tolerance]:
        a = np.array([[1., scale], [scale + delta, 1.]])
        expected = np.allclose(a, a.T, rtol=1e-12, atol=1e-12)
        for value in layouts(a):
            if expected:
                _native.validate_matrix(value, 'XtX')
            else:
                with pytest.raises(ValueError, match='symmetric'):
                    _native.validate_matrix(value, 'XtX')


@pytest.mark.parametrize('scaled', [False, True])
def test_cs_all_pairs_and_float_order(scaled):
    a = np.array([[4., -.8, 1.4, 0.], [-.8, 1., -.3, 0.],
                  [1.4, -.3, 9., 0.], [0., 0., 0., 0.]])
    members = np.array([0, 1, 2, 3], dtype=np.int64)
    inv = np.array([.5, 1., 1. / 3, 0.]) if scaled else None
    for value in layouts(a):
        before = value.copy()
        sub = value[np.ix_(members, members)]
        if inv is not None:
            sub = sub * inv[:, None] * inv[None, :]
        expected = np.abs(sub[np.triu_indices(len(members), 1)])
        actual = _native.cs_pairwise_abs(value, members, inv)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(value, before)


def test_invalid_native_cs_indices_raise():
    for members in [[-1, 0], [0, 2]]:
        with pytest.raises(ValueError):
            _native.cs_pairwise_abs(np.eye(2), np.array(members, dtype=np.int64))


@pytest.mark.parametrize('last_pair', [.49, .5, .51, np.nan])
def test_complete_cs_rejection_includes_last_pair(last_pair):
    a = np.ones((4, 4))
    a[2, 3] = a[3, 2] = last_pair
    indices = np.arange(4, dtype=np.int64)
    values = _native.cs_pairwise_abs(a, indices, min_abs_corr=.5)
    if not last_pair >= .5:
        assert values is None
    else:
        np.testing.assert_array_equal(values, np.abs(a[np.triu_indices(4, 1)]))


def test_public_readonly_inputs_and_partial_zero_prior():
    a = np.array([[1., .2, -.1], [.2, 1., .3], [-.1, .3, 1.]])
    z = np.array([4., 1., 0.])
    a.flags.writeable = False
    z.flags.writeable = False
    fit = susie_rss(z, a, 100, L=2, prior_weights=[1., 0., 2.], null_weight=.2)
    writable = susie_rss(z.copy(), a.copy(), 100, L=2,
                         prior_weights=[1., 0., 2.], null_weight=.2)
    # SER's existing prior floor is independent of layout/readonly handling.
    np.testing.assert_array_equal(fit.pip, writable.pip)
    assert fit.params['prior_weights'][1] == 0
    assert np.isfinite(fit.pip).all()
    np.testing.assert_array_equal(a, [[1., .2, -.1], [.2, 1., .3], [-.1, .3, 1.]])


@pytest.mark.parametrize('diagonal', [1., 1. + 1e-8, 1. - 1e-8, 1. + 1.001e-8])
@pytest.mark.parametrize('offdiag', [0., 1. + 1e-8, -1. - 1e-8, 1. + 1.001e-8])
def test_correlation_validation_exact_boundaries(diagonal, offdiag):
    a = np.array([[diagonal, offdiag], [offdiag, 1.]])
    valid = (np.allclose(a.diagonal(), 1., atol=1e-8, rtol=0)
             and not np.any(np.abs(a) > 1. + 1e-8))
    for value in layouts(a):
        if valid:
            np.testing.assert_array_equal(_matrix(value, 'R', 2, correlation=True), value)
        else:
            with pytest.raises(ValueError, match='signed correlation'):
                _matrix(value, 'R', 2, correlation=True)


@pytest.mark.parametrize('n', [2.5, 100., 1e7])
@pytest.mark.parametrize('options', [{}, {'null_weight': .2}, {'standardize': False},
                                    {'check_input': True}, {'prior_weights': [1., 0., 2.]}])
@pytest.mark.parametrize('capability', [0, 1])
def test_lazy_rss_matches_materialized_sufficient_statistics(n, options, monkeypatch, record_property, capability):
    if capability and getattr(_native, 'MATRIX_OPERATOR_VERSION', 0) != 1:
        pytest.skip('This native build has no actual matrix operator')
    monkeypatch.setattr(_native, 'MATRIX_OPERATOR_VERSION', capability, raising=False)
    r = np.array([[1. + 7e-9, .2, -.1], [.2, 1., .3], [-.1, .3, 1. - 7e-9]])
    z = np.array([3., -1., .2])
    adj = (n - 1) / (z * z + n - 2)
    x_ty = np.sqrt(n - 1) * (np.sqrt(adj) * z)
    kwargs = dict(L=2, estimate_residual_variance=False, check_prior=False,
                  max_iter=100, tol=.001, **options)
    crossproduct = (n - 1) * r
    before = [x.copy() for x in (r, z, crossproduct, x_ty)]
    materialized = susie_suff_stat(crossproduct, x_ty, n - 1, n, **kwargs)
    lazy = susie_rss(z, r, n, **kwargs)
    for actual, expected in zip((r, z, crossproduct, x_ty), before):
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_allclose(materialized.pip, lazy.pip, atol=1e-5, rtol=0)
    for fit in [materialized, lazy]:
        assert np.isfinite(fit.pip).all() and ((fit.pip >= 0) & (fit.pip <= 1)).all()
        assert np.isfinite(fit.alpha).all() and ((fit.alpha >= 0) & (fit.alpha <= 1)).all()
    for field in ['alpha', 'mu', 'mu2', 'lbf_variable', 'lbf', 'V', 'pip', 'elbo',
                  'X_column_scale_factors', 'KL', 'XtXr', 'niter', 'converged']:
        left, right = getattr(materialized, field), getattr(lazy, field)
        if not capability:
            np.testing.assert_array_equal(left, right)
        elif np.shape(left) == np.shape(right) and field != 'converged':
            record_property(field + '_max_abs', float(np.max(np.abs(np.asarray(left) - np.asarray(right)))))
        else:
            record_property(field + '_values', [np.asarray(left).tolist(), np.asarray(right).tolist()])
    for field in ['cs_index', 'coverage', 'actual_coverage']:
        if not capability:
            np.testing.assert_array_equal(materialized.sets[field], lazy.sets[field])
        else:
            record_property('sets_' + field, [materialized.sets[field].tolist(), lazy.sets[field].tolist()])
    for field in materialized.sets['purity']:
        if not capability:
            np.testing.assert_array_equal(materialized.sets['purity'][field], lazy.sets['purity'][field])
        else:
            record_property('purity_' + field, [materialized.sets['purity'][field].tolist(), lazy.sets['purity'][field].tolist()])


def test_scaled_xtx_symmetry_is_rechecked_in_rss():
    r = np.array([[1., 9e-13], [0., 1.]])
    _matrix(r, 'R', 2, correlation=True)
    with pytest.raises(ValueError, match='symmetric'):
        susie_rss([2., 1.], r, 100, L=1)


def test_scaled_xtx_overflow_is_rejected_in_rss():
    r = np.eye(2) * (1. + 1e-9)
    with pytest.raises(ValueError, match='finite'):
        susie_rss([2., 1.], r, np.finfo(float).max, L=1, check_prior=False)


@pytest.mark.parametrize('value', [np.inf, -np.inf, np.nan])
def test_equal_nonfinite_pairs_are_rejected(value):
    r = np.eye(40)
    r[1, 9] = r[9, 1] = value
    with pytest.raises(ValueError, match='finite'):
        _matrix(r, 'R', len(r), correlation=True)


def test_signed_zero_pair_is_valid_and_unmodified():
    r = np.eye(40)
    r[1, 9] = -0.
    r[9, 1] = 0.
    before = r.view(np.uint64).copy()
    _matrix(r, 'R', len(r), correlation=True)
    np.testing.assert_array_equal(r.view(np.uint64), before)


@pytest.mark.parametrize('scale', [1., 99., 412180., np.finfo(float).max])
@pytest.mark.parametrize('defect', ['none', 'close_symmetry', 'bad_symmetry',
                                   'range', 'nonfinite', 'diagonal'])
def test_combined_rss_checks_match_two_complete_checks(scale, defect):
    a = np.eye(40)
    if defect == 'close_symmetry':
        a[1, 9] = 9e-13
    elif defect == 'bad_symmetry':
        a[1, 9] = 3e-12
    elif defect == 'range':
        a[1, 9] = a[9, 1] = 1. + 1.001e-8
    elif defect == 'nonfinite':
        a[1, 9] = a[9, 1] = np.nan
    elif defect == 'diagonal':
        a[-1, -1] = 1. + 1e-9
    for matrix in layouts(a):
        with np.errstate(over='ignore', invalid='ignore'):
            scaled = matrix * scale
            valid = (np.isfinite(matrix).all()
                     and np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-12)
                     and np.allclose(matrix.diagonal(), 1., atol=1e-8, rtol=0)
                     and not np.any(np.abs(matrix) > 1. + 1e-8)
                     and np.isfinite(scaled).all()
                     and np.allclose(scaled, scaled.T, atol=1e-12, rtol=1e-12)
                     and not np.any(scaled.diagonal() < 0))
        if valid:
            _native.validate_rss_matrix(matrix, scale)
        else:
            with pytest.raises(ValueError):
                _native.validate_rss_matrix(matrix, scale)


def test_public_suff_stat_does_not_expose_matrix_validation_bypass():
    with pytest.raises(TypeError):
        susie_suff_stat(np.eye(2), [1., 0.], 99., 100, _rss_matrix_validated=True)
    with pytest.raises(ValueError, match='finite'):
        susie_suff_stat([[1., np.nan], [np.nan, 1.]], [1., 0.], 99., 100)


@pytest.mark.parametrize('n', [None, 2.5, 100.])
@pytest.mark.parametrize('standardize', [False, True])
@pytest.mark.parametrize('diagonal', [1., 1. + 7e-9])
def test_native_input_preserves_bits_and_ownership_for_unit_scales(monkeypatch, n, standardize, diagonal):
    monkeypatch.setattr(_native, "MATRIX_OPERATOR_VERSION", 0, raising=False)
    r = np.array([[diagonal, -0., -.25], [0., 1., .125], [-.25, .125, 1.]])
    materialized = r if n is None else (n - 1) * r
    scales = (np.sqrt(materialized.diagonal() / (n - 1))
              if standardize and n is not None else np.ones(3))
    expected = materialized.T * (1. / scales)[None, :] / scales[:, None]

    class Captured(Exception):
        pass

    for matrix in layouts(r):
        # Reverse layouts reorder the mathematical input; obtain its independent
        # materialized reference in that order before entering the public call.
        raw = matrix if n is None else (n - 1) * matrix
        scale = (np.sqrt(raw.diagonal() / (n - 1))
                 if standardize and n is not None else np.ones(3))
        expected = raw.T * (1. / scale)[None, :] / scale[:, None]
        before = matrix.view(np.uint64).copy()

        def capture(crossproduct, *args):
            assert crossproduct.flags.f_contiguous
            assert not np.shares_memory(crossproduct, matrix)
            np.testing.assert_array_equal(crossproduct.view(np.uint64), expected.view(np.uint64))
            raise Captured

        monkeypatch.setattr(_native, 'fit', capture)
        with pytest.raises(Captured):
            susie_rss([2., 1., .1], matrix, n, L=1, standardize=standardize)
        np.testing.assert_array_equal(matrix.view(np.uint64), before)


@pytest.mark.parametrize('size', [1, 7, 8, 9, 31, 33])
@pytest.mark.parametrize('multiplier', [1., 99., 1e100])
def test_contiguous_preparation_simd_exact_bits(size, multiplier):
    if not _native.preparation_simd_supported():
        pytest.skip('Runtime AVX512F is unavailable; public NumPy fallback is tested separately')
    a = np.arange(size * size, dtype=float).reshape(size, size) - 5
    a[0, 0] = -0.
    scales = np.geomspace(1e-50, 1e50, size)
    inverse = 1. / scales
    a.flags.writeable = False
    before = a.view(np.uint64).copy()
    expected = (a.T * multiplier) * inverse[None, :] / scales[:, None]
    actual = _native.prepare_crossproduct(a, inverse, scales, multiplier)
    assert actual.flags.f_contiguous
    assert not np.shares_memory(actual, a)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(a.view(np.uint64), before)


def test_contiguous_preparation_layout_fallback():
    a = np.arange(81, dtype=float).reshape(9, 9)
    scales = np.ones(9)
    assert _native.prepare_crossproduct(np.asfortranarray(a), scales, scales) is None
    storage = np.empty((18, 18))
    storage[::2, ::2] = a
    assert _native.prepare_crossproduct(storage[::2, ::2], scales, scales) is None


def test_public_numpy_fallback_matches_dispatch(monkeypatch):
    monkeypatch.setattr(_native, "MATRIX_OPERATOR_VERSION", 0, raising=False)
    r = np.array([[1. + 7e-9, -.2, .3], [-.2, 1., -.1], [.3, -.1, 1. - 7e-9]])
    kwargs = dict(z=[3., 1., -.2], R=r, n=100, L=2, null_weight=.2)
    dispatched = susie_rss(**kwargs)
    calls = []
    def numpy_fallback(*args):
        calls.append(True)
        return None
    monkeypatch.setattr(_native, 'prepare_crossproduct', numpy_fallback)
    fallback = susie_rss(**kwargs)
    assert calls, 'Materialized fallback must actually invoke preparation'
    for name in ['alpha', 'mu', 'mu2', 'pip', 'V', 'elbo', 'XtXr', 'niter', 'converged']:
        np.testing.assert_array_equal(getattr(dispatched, name), getattr(fallback, name))


def test_contiguous_preparation_invalid_shape():
    with pytest.raises(ValueError, match='dimensions'):
        _native.prepare_crossproduct(np.eye(3), np.ones(2), np.ones(3))


@pytest.mark.parametrize('scale', [1., -0., -99., np.finfo(float).max / 2,
    np.nextafter(np.finfo(float).max / 2, np.inf), np.finfo(float).max, np.inf, np.nan])
@pytest.mark.parametrize('pair', [0., -0., 1. + 1e-8, 1. + 1.001e-8, np.inf, np.nan])
def test_conservative_scale_bound_keeps_complete_numpy_gate(scale, pair):
    a = np.eye(33)
    a[0, -1] = a[-1, 0] = pair
    for matrix in layouts(a):
        with np.errstate(over='ignore', invalid='ignore'):
            scaled = matrix * scale
            valid = (np.isfinite(matrix).all()
                     and np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-12)
                     and np.allclose(matrix.diagonal(), 1., atol=1e-8, rtol=0)
                     and not np.any(np.abs(matrix) > 1. + 1e-8)
                     and np.isfinite(scaled).all()
                     and np.allclose(scaled, scaled.T, atol=1e-12, rtol=1e-12)
                     and not np.any(scaled.diagonal() < 0))
        before = matrix.view(np.uint64).copy()
        if valid:
            _native.validate_rss_matrix(matrix, scale)
        else:
            with pytest.raises(ValueError):
                _native.validate_rss_matrix(matrix, scale)
        np.testing.assert_array_equal(matrix.view(np.uint64), before)


@pytest.mark.parametrize('size', [4, 5, 7, 8, 31, 32, 33, 35])
def test_transposed_proof_reads_every_entry_and_preserves_bits(size):
    row = np.arange(size, dtype=float)
    matrix = (row[:, None] + row[None, :]) / (4 * size)
    np.fill_diagonal(matrix, 1.)
    for view in layouts(matrix):
        before = view.view(np.uint64).copy()
        _native.validate_rss_matrix(view, 99., True)
        _native.validate_rss_matrix(view, 99., False)
        np.testing.assert_array_equal(view.view(np.uint64), before)
    # Each location must be read, including lower-triangle and tail entries.
    for i in range(size):
        for j in range(size):
            original = matrix[i, j]
            matrix[i, j] = np.nan
            with pytest.raises(ValueError, match='finite'):
                _native.validate_rss_matrix(matrix, 99., True)
            matrix[i, j] = original


@pytest.mark.parametrize('delta', [0., 9e-13, 3e-12])
def test_transpose_proof_falls_back_for_near_symmetry(delta):
    matrix = np.eye(35)
    matrix[1, 33] = delta
    for scale in [1., 99., np.finfo(float).max, -0., np.inf]:
        for view in layouts(matrix):
            outcomes = []
            for use_simd in [False, True]:
                try:
                    _native.validate_rss_matrix(view, scale, use_simd)
                    outcomes.append('valid')
                except ValueError as error:
                    outcomes.append(str(error))
            assert outcomes[0] == outcomes[1]


"""Negative strides must preserve logical diagonal and symmetry semantics."""
import numpy as np
import pytest
from prusie import _native


@pytest.mark.parametrize('size', [4, 5, 32, 33])
@pytest.mark.parametrize('layout', ['row', 'column', 'both', 'Frow', 'Fcolumn'])
def test_single_axis_reverse_validation_matches_scalar(size, layout):
    matrix = np.eye(size)
    if layout.startswith('F'):
        matrix = np.asfortranarray(matrix)
    if layout in {'row', 'Frow'}:
        matrix = matrix[::-1, :]
    elif layout in {'column', 'Fcolumn'}:
        matrix = matrix[:, ::-1]
    else:
        matrix = matrix[::-1, ::-1]
    before = matrix.view(np.uint64).copy()
    result = []
    for use_simd in [False, True]:
        try:
            _native.validate_rss_matrix(matrix, 99., use_simd)
            result.append('valid')
        except ValueError as error:
            result.append(str(error))
    assert result[0] == result[1]
    np.testing.assert_array_equal(matrix.view(np.uint64), before)
