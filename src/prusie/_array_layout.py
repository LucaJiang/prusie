"""Normalize NumPy layouts before crossing the typed native boundary."""
import numpy as np


def native_array(array):
    """Preserve values; copy layouts the Rust NumPy view cannot represent."""
    item = array.dtype.itemsize
    # NumPy alignment flags alone are insufficient for degenerate dimensions.
    if (array.ctypes.data % array.dtype.alignment
            or any(step % item or step == np.iinfo(np.intp).min for step in array.strides)):
        return np.array(array, dtype=array.dtype, order="C", copy=True)
    return array
