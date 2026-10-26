"""Eligibility and cleanup for one private RSS preparation executor."""

import sys

import numpy as np

from . import _native


def _eligible_preparation(matrix, count):
    """Inspect normalized preparation inputs, independent of process hooks.

    Public RSS validation has resolved the integer count, response scale and
    matrix shape before this check. Other fitting options do not change this
    exact raw/scaled symmetry proof and ordered matrix repair.
    """
    return (
        count > 1
        and type(matrix) is np.ndarray
        and matrix.ndim == 2
        and matrix.shape[0] >= 1024
        and matrix.shape[0] == matrix.shape[1]
        and matrix.dtype == np.dtype(np.float64)
        and matrix.dtype.metadata is None
        and matrix.dtype.isnative
        and matrix.flags.c_contiguous
        and matrix.flags.aligned
        and matrix.strides == (matrix.shape[0] * 8, 8)
    )


def create_executor(matrix, count):
    """Return an inert private owner for eligible, validated preparation."""
    if not _eligible_preparation(matrix, count):
        return None
    # Architecture/FP checks and worker construction happen only under the
    # native helper's checked read borrow. Neither executes Python callbacks.
    return _native._new_fit_executor(count)


def close_executor(executor):
    """Join every worker while preserving an already-active primary exception."""
    primary = sys.exc_info()[0] is not None
    try:
        executor.close()
    except BaseException:
        if not primary:
            raise
