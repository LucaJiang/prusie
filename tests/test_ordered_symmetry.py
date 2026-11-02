"""The tiled preparation keeps every required ordered float64 result."""

import numpy as np
import pytest

from prusie import _native
from prusie._validation import _symmetrize_matrix


# Layout and tile boundaries interact in indexing; scaling is elementwise.
# Keep all size/layout pairs and both scale branches for every layout at n=33.
@pytest.mark.parametrize(
    "p,scale,layout",
    [
        pytest.param(p, 1.0, layout, id=f"{layout}-n{p}-unit")
        for p in [1, 2, 31, 32, 33, 65]
        for layout in ["C", "F", "reversed", "stride"]
    ]
    + [
        pytest.param(33, scale, layout, id=f"{layout}-n33-{label}")
        for label, scale in [
            ("ordinary", 799.0),
            ("large-n", 486018.0),
            ("small", 1e-200),
            ("large", 1e200),
        ]
        for layout in ["C", "F", "reversed", "stride"]
    ],
)
def test_average_matches_numpy_bits_for_tiles_scales_and_layouts(p, scale, layout):
    values = np.arange(p * p, dtype=float).reshape(p, p) / max(p * p, 1)
    values = np.nextafter(values - 0.5, np.inf)
    if layout == "F":
        values = np.asfortranarray(values)
    elif layout == "reversed":
        values = values[::-1, ::-1]
    elif layout == "stride":
        storage = np.zeros((2 * p, 2 * p))
        storage[::2, ::2] = values
        values = storage[::2, ::2]
    before = values.copy()
    working = values if scale == 1 else values * scale
    expected = np.ascontiguousarray((working + working.T) / 2)
    actual = _native.symmetrize_matrix(values, scale)
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(values, before)


def test_signed_zero_subnormal_and_overflow_preserve_expression():
    tiny = np.nextafter(0.0, 1.0)
    values = np.array([[tiny, -0.0, 1e308], [0.0, -tiny, -1e308], [1e308, 1e308, 1.0]])
    with np.errstate(over="ignore", invalid="ignore"):
        for scale in [1.0, 2.0, 1e-200]:
            working = values if scale == 1 else values * scale
            expected = (working + working.T) / 2
            actual = _native.symmetrize_matrix(values, scale)
            np.testing.assert_array_equal(actual, expected)
            finite = np.isfinite(expected)
            np.testing.assert_array_equal(
                actual[finite].view(np.uint64), expected[finite].view(np.uint64)
            )
    with (
        pytest.warns(RuntimeWarning, match="not symmetric"),
        pytest.raises(ValueError, match="finite"),
    ):
        _symmetrize_matrix(values, "XtX", 2.0)
