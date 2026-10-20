"""Array layout normalization before arrays are passed to the native extension."""

from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

_Scalar = TypeVar("_Scalar", bound=np.generic)


def native_array(array: NDArray[_Scalar]) -> NDArray[_Scalar]:
    """Return ``array``, or a C-ordered copy if its layout cannot be viewed natively.

    The values are unchanged. A copy is made for misaligned data, for strides
    that are not whole multiples of the element size, and for the minimum
    representable stride.
    """
    item = array.dtype.itemsize
    # Alignment flags do not cover every degenerate stride, so check directly.
    if array.ctypes.data % array.dtype.alignment or any(
        step % item or step == np.iinfo(np.intp).min for step in array.strides
    ):
        return np.array(array, dtype=array.dtype, order="C", copy=True)
    return array
