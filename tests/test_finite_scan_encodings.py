"""Finiteness classifications include NaN payloads and remain read only."""
import numpy as np
import pytest
from prusie import _native


@pytest.mark.parametrize('bits', [0, 1, 0x8000000000000000, 0x8000000000000001,
    0x7fefffffffffffff, 0xffefffffffffffff, 0x7ff0000000000000,
    0xfff0000000000000, 0x7ff0000000000001, 0xfff0000000000001,
    0x7ff8000000000000, 0xfff8000000001234])
@pytest.mark.parametrize('position', [0, 7, 128])
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
