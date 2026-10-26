"""Output validation must preserve the original repair's errors and float bits."""

import numpy as np
import pytest

from prusie import _native
from prusie._validation import _symmetrize_matrix


def reference_repair(values, name, scale):
    with np.errstate(over="ignore", invalid="ignore"):
        working = values if scale == 1 else values * scale
        repaired = (working + working.T) / 2
    _native.validate_matrix(repaired, name)
    return np.ascontiguousarray(repaired)


@pytest.mark.parametrize("p", [1, 2, 31, 32, 33, 65])
@pytest.mark.parametrize("layout", ["C", "F", "reverse", "stride"])
@pytest.mark.parametrize("scale", [1.0, 486018.0, 1e-200])
def test_checked_repair_preserves_finite_bits_and_input(p, layout, scale):
    values = np.arange(p * p, dtype=float).reshape(p, p) / max(1, p * p)
    np.fill_diagonal(values, 1.0)
    if p > 1:
        values[0, 1] = -0.0
        values[1, 0] = 0.0
    if layout == "F":
        values = np.asfortranarray(values)
    elif layout == "reverse":
        values = values[::-1, ::-1]
    elif layout == "stride":
        backing = np.zeros((p * 2, p * 2))
        backing[::2, ::2] = values
        values = backing[::2, ::2]
    before = values.copy()
    actual = _native.symmetrize_matrix(values, scale, "Gram")
    expected = reference_repair(values, "Gram", scale)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(values.view(np.uint64), before.view(np.uint64))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 1e308])
@pytest.mark.parametrize("position", [(0, 0), (0, 64), (64, 0), (64, 64)])
@pytest.mark.parametrize("name", ["XtX", "Gram"])
def test_error_message_and_priority_match_reference(bad, position, name):
    values = np.eye(65)
    values[0, 0] = -1.0
    values[position] = bad
    scale = 2.0
    with pytest.raises(ValueError) as expected:
        reference_repair(values, name, scale)
    with (
        pytest.warns(RuntimeWarning, match="not symmetric"),
        pytest.raises(ValueError) as actual,
    ):
        _symmetrize_matrix(values, name, scale)
    assert str(actual.value) == str(expected.value)


def test_negative_diagonal_remains_rejected():
    values = np.array([[-1.0, 0.2], [0.3, 1.0]])
    with (
        pytest.warns(RuntimeWarning, match="not symmetric"),
        pytest.raises(ValueError, match=r"^XtX diagonal must be nonnegative$"),
    ):
        _symmetrize_matrix(values, "XtX")
