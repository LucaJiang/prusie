"""Valid byte-layout inputs are normalized before typed native views exist."""
import numpy as np
import pytest
from prusie import _native, susie_rss, susie_suff_stat
from prusie.posterior import credible_sets
from prusie._array_layout import native_array


def unusual(values, kind):
    values = np.asarray(values)
    offset = 1 if kind == 'unaligned' else 0
    stride = 9 if kind in {'byte', 'reverse'} else values.dtype.itemsize
    strides = (stride,) if values.ndim == 1 else (values.shape[1] * stride, stride)
    backing = np.zeros(values.size * max(stride, 8) + 16, dtype=np.uint8)
    view = np.ndarray(values.shape, dtype=values.dtype, buffer=backing, offset=offset, strides=strides)
    view[...] = values
    if kind == 'reverse':
        view = view[::-1] if values.ndim == 1 else view[::-1, ::-1]
    view.flags.writeable = False
    return view


@pytest.mark.parametrize('size', [3, 4])
@pytest.mark.parametrize('kind', ['byte', 'unaligned', 'reverse'])
@pytest.mark.parametrize('interface', ['rss', 'suff_stat'])
def test_public_byte_layout_matches_aligned_copy(size, kind, interface):
    r = unusual(np.eye(size), kind)
    z = unusual(np.arange(size, dtype=float) + .1, kind)
    before = [value.tobytes() for value in (r, z)]
    kwargs = dict(L=2, max_iter=100, tol=.001, estimate_residual_variance=False,
                  prior_weights=np.arange(size) % 2, null_weight=.1)
    if interface == 'rss':
        actual = susie_rss(z, r, 100, **kwargs)
        expected = susie_rss(z.copy(), r.copy(), 100, **kwargs)
    else:
        actual = susie_suff_stat(r, z, 99., 100, **kwargs)
        expected = susie_suff_stat(r.copy(), z.copy(), 99., 100, **kwargs)
    for field in ['pip', 'alpha', 'mu', 'mu2', 'V', 'elbo', 'XtXr']:
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    assert [value.tobytes() for value in (r, z)] == before
    assert not r.flags.writeable and not z.flags.writeable


@pytest.mark.parametrize('kind', ['byte', 'unaligned', 'reverse'])
def test_complete_cs_byte_layout(kind):
    r = unusual(np.full((4, 4), .8) + np.eye(4) * .2, kind)
    alpha = np.full((2, 4), .25)
    actual = credible_sets(alpha, np.ones(2), r)
    expected = credible_sets(alpha, np.ones(2), r.copy())
    for key in ['cs_index', 'coverage', 'actual_coverage']:
        np.testing.assert_array_equal(actual[key], expected[key])
    for key in actual['purity']:
        np.testing.assert_array_equal(actual['purity'][key], expected['purity'][key])


@pytest.mark.parametrize('kind', ['byte', 'unaligned', 'reverse'])
@pytest.mark.parametrize('boundary', ['matrix', 'rss', 'prepare_matrix', 'prepare_inverse',
                                     'prepare_scales', 'cs_matrix', 'cs_inverse', 'cs_members',
                                     'finite', 'fit_matrix', 'fit_vector'])
def test_direct_native_rejects_unsupported_geometry(kind, boundary):
    matrix = unusual(np.eye(3), kind)
    vector = unusual(np.ones(3), kind)
    members = unusual(np.arange(3, dtype=np.int64), kind)
    good = np.ones(3)
    options = dict(l=1, prior_variance=.2, residual_variance=1.,
                   prior_weights=[1/3]*3, estimate_prior_variance=False,
                   estimate_prior_method='optim', estimate_residual_variance=False,
                   check_null_threshold=0., max_iter=100, tol=.001,
                   check_prior=False, prior_tol=1e-9, null_index=None)
    calls = {
        'matrix': lambda: _native.validate_matrix(matrix, 'XtX'),
        'rss': lambda: _native.validate_rss_matrix(matrix, 99.),
        'prepare_matrix': lambda: _native.prepare_crossproduct(matrix, good, good),
        'prepare_inverse': lambda: _native.prepare_crossproduct(np.eye(3), vector, good),
        'prepare_scales': lambda: _native.prepare_crossproduct(np.eye(3), good, vector),
        'cs_matrix': lambda: _native.cs_pairwise_abs(matrix, np.arange(3,dtype=np.int64)),
        'cs_inverse': lambda: _native.cs_pairwise_abs(np.eye(3), np.arange(3,dtype=np.int64), vector),
        'cs_members': lambda: _native.cs_pairwise_abs(np.eye(3), members),
        'finite': lambda: _native.complete_finite_scan(vector),
        'fit_matrix': lambda: _native.fit(matrix, good, 99., 100., options),
        'fit_vector': lambda: _native.fit(np.eye(3), vector, 99., 100., options),
    }
    with pytest.raises(ValueError, match='aligned|byte strides'):
        calls[boundary]()


@pytest.mark.parametrize('size', [0, 1])
@pytest.mark.parametrize('stride', [9, np.iinfo(np.intp).min])
def test_degenerate_geometry_checked_before_view(size, stride):
    buffer = np.zeros(16, dtype=np.uint8)
    vector = np.ndarray((size,), dtype=float, buffer=buffer, strides=(stride,))
    if size:
        vector[0] = 1.
        normalized = native_array(vector)
        assert normalized.ctypes.data % 8 == 0 and normalized.strides == (8,)
        np.testing.assert_array_equal(normalized, [1.])
    with pytest.raises(ValueError, match='aligned|byte strides'):
        _native.complete_finite_scan(vector)
    matrix = np.ndarray((size, size), dtype=float, buffer=buffer, strides=(stride, stride))
    with pytest.raises(ValueError, match='aligned|byte strides'):
        _native.validate_rss_matrix(matrix, 99.)
    if size:
        result = susie_rss(vector, matrix, 100., L=1)
        assert np.isfinite(result.pip).all()


def test_disabled_and_singleton_cs_do_not_convert_unused_correlation():
    class Unused:
        def __array__(self, *args, **kwargs):
            raise AssertionError('Unused correlation must not be converted')
    assert credible_sets(np.ones((1, 1)), [1.], Unused(), coverage=None)['cs'] == []
    assert len(credible_sets(np.ones((1, 1)), [1.], Unused())['cs']) == 1
