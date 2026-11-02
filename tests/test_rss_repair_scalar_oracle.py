"""Native RSS validation and repair match a scalar Python reference bit for bit."""

import warnings

import numpy as np
import pytest

from prusie import _native
from prusie._validation import _symmetrize_matrix


def scalar_reference(a, scale):
    """Blocked scan with the native error priority, written as plain Python."""
    n = len(a)
    flags = [True] * 7
    out = np.empty((n, n), dtype=np.float64)
    with np.errstate(all="ignore"):
        for ii in range(0, n, 32):
            for jj in range(ii, n, 32):
                for i in range(ii, min(ii + 32, n)):
                    for j in range(max(jj, i), min(jj + 32, n)):
                        x, y = a[i, j], a[j, i]
                        sx, sy = x * scale, y * scale
                        value = (sx + sy) / 2.0
                        flags[0] &= np.isfinite(x) and np.isfinite(y)
                        flags[2] &= np.isfinite(sx) and np.isfinite(sy)
                        flags[4] &= sx == sy
                        flags[5] &= np.isfinite(value)
                        if i == j:
                            flags[1] &= x >= 0
                            flags[3] &= sx >= 0
                            flags[6] &= value >= 0
                        out[i, j] = out[j, i] = value
    for valid, message in zip(
        flags[:4],
        [
            "R must contain only finite values",
            "R diagonal must be nonnegative",
            "XtX must contain only finite values",
            "XtX diagonal must be nonnegative",
        ],
        strict=False,
    ):
        if not valid:
            raise ValueError(message)
    if flags[4]:
        return True, None, None
    error = (
        "XtX must contain only finite values"
        if not flags[5]
        else "XtX diagonal must be nonnegative"
        if not flags[6]
        else None
    )
    return False, out, error


def outcome(fn, a, scale):
    try:
        symmetric, repair, error = fn(a, scale)
    except ValueError as exc:
        return "error", str(exc)
    return "ok", symmetric, error, None if repair is None else repair.view(np.uint64)


def assert_outcome(a, b):
    assert a[:3] == b[:3]
    if a[0] == "ok":
        if a[3] is None:
            assert b[3] is None
        else:
            np.testing.assert_array_equal(a[3], b[3])


def layout(a, order):
    if order in ("C", "F"):
        return np.array(a, dtype=np.float64, order=order)
    if order == "negative":
        return np.ascontiguousarray(a[::-1, ::-1])[::-1, ::-1]
    b = np.empty((2 * len(a), 2 * len(a)))
    b[::2, ::2] = a
    return b[::2, ::2]


# Keep every size/layout interaction: C/F use 4×4 SIMD pairs from n=4,
# while strided/negative inputs use the scalar 32-wide traversal. All scales
# also cross a 32-wide boundary in every layout; repeating them at every size
# does not select another arithmetic or validation path.
@pytest.mark.parametrize(
    "n,order,scale",
    [
        pytest.param(n, order, 1.0, id=f"{order}-n{n}-unit")
        for n in [3, 4, 5, 7, 8, 9, 16, 17, 31, 32, 33, 63, 64, 65]
        for order in ["C", "F", "strided", "negative"]
    ]
    + [
        pytest.param(33, order, scale, id=f"{order}-n33-{label}")
        for label, scale in [
            ("zero", 0.0),
            ("negative-zero", -0.0),
            ("negative", -1.0),
            ("ordinary", 99.0),
            ("maximum", np.finfo(float).max),
            ("minimum-normal", np.finfo(float).tiny),
            ("subnormal", np.nextafter(0.0, 1.0)),
            ("infinity", np.inf),
            ("nan", np.nan),
        ]
        for order in ["C", "F", "strided", "negative"]
    ],
)
def test_flags_repair_bits_and_input_invariance_match_reference(n, order, scale):
    k = np.arange(n * n).reshape(n, n)
    a = layout(((17 * k + k // n * 11) % 67) / 128.0 - 0.25, order)
    np.fill_diagonal(a, 1.0)
    a[0, 1], a[1, 0] = -0.0, np.nextafter(0.0, 1.0)
    original = a.copy().view(np.uint64)
    expected = outcome(scalar_reference, a, scale)
    actual = outcome(_native.validate_and_repair_rss, a, scale)
    assert_outcome(actual, expected)
    np.testing.assert_array_equal(a.view(np.uint64), original)


# NaN checks every original location in both physical orders. Other encodings
# retain diagonal, scalar off-diagonal and both sides of a complete SIMD tile.
# Positions 4=(0,4) and 36=(4,0) exercise the native 4×4 pair loads, unlike the
# diagonal-block and ninth-row/column scalar tails in the original locations.
@pytest.mark.parametrize(
    "position,value,order",
    [
        pytest.param(position, np.nan, order, id=f"{order}-p{position}-nan")
        for position in [0, 1, 4, 8, 9, 10, 36, 40, 44, 72, 79, 80]
        for order in ["C", "F"]
    ]
    + [
        pytest.param(position, value, order, id=f"{order}-p{position}-{label}")
        for position in [0, 1, 4, 36, 80]
        for label, value in [
            ("positive-infinity", np.inf),
            ("negative-infinity", -np.inf),
            ("negative", -1.0),
            ("negative-zero", -0.0),
            ("maximum", np.finfo(float).max),
        ]
        for order in ["C", "F"]
    ],
)
def test_each_position_error_priority(position, value, order):
    a = np.array(np.eye(9), order=order)
    a.flat[position] = value
    actual = outcome(_native.validate_and_repair_rss, a, -2.0)
    assert_outcome(actual, outcome(scalar_reference, a, -2.0))


def test_symmetric_general_repair_overflow_is_suppressed():
    a = np.full((5, 5), 2.0)
    assert _native.validate_and_repair_rss(a, np.finfo(float).max / 2.0) == (True, None, None)


def test_warning_precedes_deferred_repair_error_and_early_errors_do_not_warn():
    a = np.eye(5)
    a[0, 1], a[1, 0] = 1.0, 0.9
    symmetric, repair, error = _native.validate_and_repair_rss(a, np.finfo(float).max)
    assert not symmetric and error == "XtX must contain only finite values"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="XtX must contain only finite values"):
            _symmetrize_matrix(
                a, "XtX", np.finfo(float).max, [], _prepared=repair, _repair_error=error
            )
    assert len(caught) == 1 and "symmetric" in str(caught[0].message)
    a[0, 0] = -1.0
    a[2, 3] = np.nan
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="R must contain only finite values"):
            _native.validate_and_repair_rss(a, 99.0)
    assert not caught
