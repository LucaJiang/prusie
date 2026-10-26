"""Shared validation, input normalization and warning helpers."""

from __future__ import annotations

import operator
import warnings as warning_module

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _native
from ._array_layout import native_array


def _vector(value: ArrayLike, name: str, size: int | None = None) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if size is not None and array.size != size:
        raise ValueError(f"{name} must have length {size}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(native_array(array))


def _summary_vector(value, name, collected=None):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if np.any(np.isinf(array)):
        raise ValueError(f"{name} must not contain infinite values")
    missing = np.isnan(array)
    if np.any(missing):
        array = array.copy()
        array[missing] = 0.0
        label = "z-scores" if name == "z" else name
        _warn(f"NA values in {label} are replaced with 0.", collected)
    return np.ascontiguousarray(native_array(array))


def _validated_matrix(
    value: ArrayLike,
    name: str,
    size: int,
    *,
    correlation=False,
    scale=1.0,
    rss_scale=None,
) -> tuple[NDArray[np.float64], bool]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size})")
    array = native_array(array)
    # Finite, symmetry and diagonal checks run natively without p × p temporaries.
    if rss_scale is None:
        symmetric = _native.validate_matrix(array, name, correlation, scale)
    else:
        symmetric = _native.validate_rss_matrix(array, rss_scale, proof_threads=1)
    # native_array has already ensured alignment and element-compatible strides;
    # any copy made here is C-ordered, matching the public input orientation.
    return np.ascontiguousarray(array), symmetric


def _symmetrize_matrix(
    array, name, scale=1.0, collected=None, *, _prepared=None, _repair_error=None
):
    if collected is None:
        collected = []
    _warn(f"{name} is not symmetric; using ({name} + t({name}))/2.", collected)
    if _prepared is not None:
        if _repair_error is not None:
            raise ValueError(_repair_error)
        return _prepared
    repaired = _native.symmetrize_matrix(array, scale, name)
    return np.ascontiguousarray(repaired)


def _matrix(value, name, size, *, correlation=False):
    array, symmetric = _validated_matrix(value, name, size, correlation=correlation)
    return array if symmetric else _symmetrize_matrix(array, name)


def _scalar(value: float, name: str, *, lower: float = 0.0, strict: bool = False) -> float:
    if np.ndim(value) != 0:
        raise ValueError(f"{name} must be a scalar")
    value = float(value)
    if not np.isfinite(value) or (value <= lower if strict else value < lower):
        op = ">" if strict else ">="
        raise ValueError(f"{name} must be finite and {op} {lower}")
    return value


def _integer(value: int, name: str) -> int:
    if (
        isinstance(value, bool)
        or not np.isscalar(value)
        or not np.isfinite(value)
        or int(value) != value
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _thread_count(value: int) -> int:
    """Resolve a per-call integer request without changing process-wide state."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("n_threads must be an integer between 1 and 32")
    try:
        count = operator.index(value)
    except TypeError:
        raise ValueError("n_threads must be an integer between 1 and 32") from None
    if not 1 <= count <= 32:
        raise ValueError("n_threads must be an integer between 1 and 32")
    return count


def _metadata(variant_ids, variant_metadata, m):
    if variant_ids is None and m > 0:
        # Positional labels '0', ..., 'm-1', with the same minimal Unicode
        # width as an explicitly supplied list of decimal strings.
        ids = np.arange(m).astype(f"<U{len(str(m - 1))}")
    else:
        ids = np.array(
            [str(i) for i in range(m)] if variant_ids is None else variant_ids,
            dtype=str,
            copy=True,
        )
        if ids.shape != (m,) or len(set(ids.tolist())) != m or np.any(ids == ""):
            raise ValueError("variant_ids must contain m distinct nonempty IDs in input order")
    metadata = (
        {}
        if variant_metadata is None
        else {key: np.asarray(value).copy() for key, value in variant_metadata.items()}
    )
    if any(value.shape != (m,) for value in metadata.values()):
        raise ValueError("Each variant_metadata column must have length m")
    return ids, metadata


def _warn(message: str, collected: list[str]) -> None:
    collected.append(message)
    warning_module.warn(message, RuntimeWarning, stacklevel=3)
