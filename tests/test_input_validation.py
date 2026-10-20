"""Complete validation across floating-point encodings, scaling and threads."""

import numpy as np
import pytest

from prusie import _native, susie_rss, susie_suff_stat

BOUND = np.finfo(float).max / 2


@pytest.mark.parametrize(
    "scale",
    [
        486018.0,
        np.nextafter(BOUND, 0),
        BOUND,
        np.nextafter(BOUND, np.inf),
        np.finfo(float).max,
        -BOUND,
        np.inf,
        np.nan,
    ],
)
@pytest.mark.parametrize(
    "pair",
    [(0.0, -0.0), (0.0, 9e-13), (1.0, np.nextafter(1.0, np.inf)), (1.0 + 5e-13, 1.0 + 6e-13)],
)
@pytest.mark.parametrize("use_simd", [False, True])
def test_scaled_symmetry_decision_and_overflow_safety(scale, pair, use_simd):
    for transpose in (False, True):
        matrix = np.eye(33)
        matrix[0, -1], matrix[-1, 0] = pair
        if transpose:
            matrix = matrix.T
        matrix.flags.writeable = False
        before = matrix.view(np.uint64).copy()
        with np.errstate(over="ignore", invalid="ignore"):
            scaled = matrix * scale
            accepted = (
                np.isfinite(matrix).all()
                and np.isfinite(scaled).all()
                and np.all(scaled.diagonal() >= 0)
            )
        if accepted:
            assert _native.validate_rss_matrix(matrix, scale, use_simd) == np.array_equal(
                scaled, scaled.T
            )
        else:
            with pytest.raises(ValueError):
                _native.validate_rss_matrix(matrix, scale, use_simd)
        np.testing.assert_array_equal(matrix.view(np.uint64), before)


@pytest.mark.parametrize("layout", ["C", "F", "strided", "reversed"])
@pytest.mark.parametrize("asymmetry", [1e-15, 9e-13])
@pytest.mark.parametrize("use_simd", [False, True])
def test_large_n_rss_repairs_working_xtx_without_mutating_input(layout, asymmetry, use_simd):
    matrix = np.eye(33)
    matrix[0, -1] = asymmetry
    if layout == "F":
        matrix = matrix.T
    elif layout == "strided":
        backing = np.zeros((66, 66))
        backing[::2, ::2] = matrix
        matrix = backing[::2, ::2]
    elif layout == "reversed":
        matrix = matrix[::-1, ::-1]
    matrix.flags.writeable = False
    before = matrix.tobytes()
    scale = 486018.0
    assert np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-12)
    assert not np.allclose(matrix * scale, matrix.T * scale, atol=1e-12, rtol=1e-12)
    assert _native.validate_rss_matrix(matrix, scale, use_simd) is False
    z = np.linspace(-1.0, 3.0, len(matrix))
    z.flags.writeable = False
    z_before = z.tobytes()
    with pytest.warns(RuntimeWarning, match="XtX is not symmetric"):
        fit = susie_rss(z=z, R=matrix, n=scale + 1, L=2, estimate_residual_variance=False)
    assert any("XtX is not symmetric" in message for message in fit.warnings)
    assert np.isfinite(fit.pip).all() and np.all((fit.pip >= 0) & (fit.pip <= 1))
    assert matrix.tobytes() == before and not matrix.flags.writeable
    assert z.tobytes() == z_before and not z.flags.writeable


@pytest.mark.parametrize("use_simd", [False, True])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize(
    "pair,diagonal,message",
    [
        ((np.nan, 0.0), 1.0, "finite"),
        ((np.inf, np.inf), 1.0, "finite"),
        ((0.0, -np.inf), 1.0, "finite"),
        ((0.0, 0.0), -1.0, "nonnegative"),
        ((0.0, 0.0), np.nan, "finite"),
        ((0.0, 0.0), np.inf, "finite"),
    ],
)
def test_rss_retains_complete_raw_rejection(use_simd, transpose, pair, diagonal, message):
    matrix = np.eye(65)
    matrix[0, -1], matrix[-1, 0] = pair
    matrix[-1, -1] = diagonal
    if transpose:
        matrix = matrix.T
    matrix.flags.writeable = False
    before = matrix.tobytes()
    with pytest.raises(ValueError, match=message):
        _native.validate_rss_matrix(matrix, 486018.0, use_simd)
    with pytest.raises(ValueError, match=message):
        susie_rss(z=np.zeros(len(matrix)), R=matrix, n=486019.0)
    assert matrix.tobytes() == before and not matrix.flags.writeable


@pytest.mark.parametrize("transpose", [False, True])
def test_direct_xtx_warns_and_repairs_internal_copy(transpose):
    matrix = np.eye(3) * 486018.0
    matrix[0, -1] = 1e-15 * 486018.0
    if transpose:
        matrix = matrix.T
    matrix.flags.writeable = False
    before = matrix.tobytes()
    with pytest.warns(RuntimeWarning, match="XtX is not symmetric"):
        fit = susie_suff_stat(matrix, np.zeros(3), 486018.0, 486019.0)
    assert np.isfinite(fit.pip).all()
    assert matrix.tobytes() == before and not matrix.flags.writeable


def test_public_rss_rejects_scaled_diagonal_overflow():
    matrix = np.eye(3) * (1.0 + 5e-9)
    matrix.flags.writeable = False
    before = matrix.tobytes()
    with pytest.raises(ValueError, match="XtX must contain only finite values"):
        susie_rss(z=np.zeros(3), R=matrix, n=np.finfo(float).max)
    assert matrix.tobytes() == before and not matrix.flags.writeable


@pytest.mark.parametrize(
    "bits",
    [
        0,
        1,
        0x8000000000000000,
        0x8000000000000001,
        0x7FEFFFFFFFFFFFFF,
        0xFFEFFFFFFFFFFFFF,
        0x7FF0000000000000,
        0xFFF0000000000000,
        0x7FF0000000000001,
        0xFFF0000000000001,
        0x7FF8000000000000,
        0xFFF8000000001234,
    ],
)
@pytest.mark.parametrize("position", [0, 7, 128])
def test_complete_finite_predicates_preserve_encoding(bits, position):
    storage = np.zeros(129, dtype=np.uint64)
    storage[position] = bits
    values = storage.view(np.float64)
    values.flags.writeable = False
    before = storage.copy()
    expected = bool(np.isfinite(values).all())
    assert _native.complete_finite_scan(values, False) == expected
    assert _native.complete_finite_scan(values, True) == expected
    np.testing.assert_array_equal(storage, before)


@pytest.mark.parametrize("threads", ["1", "2", "4", "8", "invalid", "0"])
def test_validation_environment_keeps_complete_decisions(monkeypatch, threads):
    monkeypatch.setenv("PRUSIE_NUM_THREADS", threads)
    matrix = np.eye(1025)
    _native.validate_rss_matrix(matrix, 99.0)
    matrix[0, -1] = 1e-15
    matrix.flags.writeable = False
    before = matrix.tobytes()
    _native.validate_rss_matrix(matrix, 486018.0)
    assert matrix.tobytes() == before
    matrix.flags.writeable = True
    matrix[0, -1] = np.nan
    with pytest.raises(ValueError, match=r"finite"):
        _native.validate_rss_matrix(matrix, 99.0)
