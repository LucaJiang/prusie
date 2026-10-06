"""Supplemental analytical acceptance-bound cases requested by independent D."""
import numpy as np
import pytest
from prusie import _native


BOUND = np.finfo(float).max / 2


@pytest.mark.parametrize('scale', [np.nextafter(BOUND, 0), BOUND,
    np.nextafter(BOUND, np.inf), np.finfo(float).max, -BOUND, np.inf, np.nan])
@pytest.mark.parametrize('pair', [(0., -0.), (0., 9e-13),
    (1., np.nextafter(1., np.inf)), (1. + 5e-13, 1. + 6e-13)])
def test_scaled_symmetry_orientations_and_overflow_fallback(scale, pair):
    for transpose in (False, True):
        matrix = np.eye(33)
        matrix[0, -1], matrix[-1, 0] = pair
        if transpose:
            matrix = matrix.T
        matrix.flags.writeable = False
        before = matrix.view(np.uint64).copy()
        with np.errstate(over='ignore', invalid='ignore'):
            scaled = matrix * scale
            accepted = (np.isfinite(matrix).all()
                and np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-12)
                and np.allclose(matrix.diagonal(), 1., atol=1e-8, rtol=0)
                and np.max(np.abs(matrix)) <= 1. + 1e-8
                and np.isfinite(scaled).all()
                and np.allclose(scaled, scaled.T, atol=1e-12, rtol=1e-12)
                and np.all(scaled.diagonal() >= 0))
        if accepted:
            _native.validate_rss_matrix(matrix, scale)
        else:
            with pytest.raises(ValueError):
                _native.validate_rss_matrix(matrix, scale)
        np.testing.assert_array_equal(matrix.view(np.uint64), before)
