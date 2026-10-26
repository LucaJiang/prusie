"""Strict, opt-in caller-workspace contract for ordinary z/R/n RSS only."""

from __future__ import annotations

import numpy as np

from ._validation import (
    _integer,
    _metadata,
    _scalar,
    _thread_count,
    _vector,
)

_SCALARS = frozenset(
    (
        bool,
        int,
        float,
        str,
        np.bool_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.longdouble,
        np.str_,
    )
)


_VECTORS = frozenset(("z", "scaled_prior_variance", "prior_weights", "variant_ids"))


_NONE = frozenset(
    [
        "X",
        "maf",
        "control",
        "bhat",
        "shat",
        "var_y",
        "s_init",
        "model_init",
        "prior_variance_grid",
        "mixture_weights",
        "alpha0",
        "beta0",
        "L_greedy",
        "slot_prior",
        "median_abs_corr",
        "cs_extension_corr",
        "convergence",
    ]
)


_FALSE = frozenset(("refine", "retain_data", "track_fit", "init_only", "check_input"))


def _span(array):
    pointer = array.__array_interface__["data"][0]
    if not array.size:
        return pointer, pointer
    low = high = 0
    for size, stride in zip(array.shape, array.strides, strict=False):
        offset = (size - 1) * stride
        low += min(0, offset)
        high += max(0, offset)
    return pointer + low, pointer + high + array.itemsize


def _layout(matrix):
    if type(matrix) is not np.ndarray:
        raise ValueError("overwrite_ld requires a plain NumPy ndarray R")
    if (
        matrix.dtype != np.dtype(np.float64)
        or matrix.dtype.metadata is not None
        or not matrix.dtype.isnative
        or matrix.ndim != 2
        or not matrix.shape[0]
        or matrix.shape[0] != matrix.shape[1]
        or not matrix.flags.c_contiguous
        or not matrix.flags.aligned
        or not matrix.flags.writeable
        or matrix.strides != (matrix.shape[0] * 8, 8)
        or matrix.size > np.iinfo(np.intp).max // 8
    ):
        raise ValueError("overwrite_ld requires square writable aligned native float64 C-order R")
    owner = matrix
    seen = set()
    while owner.base is not None:
        if id(owner) in seen or type(owner.base) is not np.ndarray:
            raise ValueError("overwrite_ld requires NumPy-owned heap storage for R")
        seen.add(id(owner))
        owner = owner.base
    if not owner.flags.owndata:
        raise ValueError("overwrite_ld requires NumPy-owned heap storage for R")
    low, high = _span(matrix)
    root_low, root_high = _span(owner)
    if not root_low <= low < high <= root_high:
        raise ValueError("overwrite_ld R extends outside its NumPy owner")
    return (matrix.dtype, matrix.shape, matrix.strides, low, high, id(owner))


def prepare(arguments):
    """Retain all arguments and reject hooks/overlaps before any coercion."""
    matrix = arguments["R"]
    layout = _layout(matrix)
    low, high = layout[3:5]
    arrays = []
    seen = set()

    def walk(value, name):
        if value is None or type(value) in _SCALARS:
            return
        if type(value) is np.ndarray:
            other_low, other_high = _span(value)
            if other_low < high and low < other_high:
                raise ValueError(f"overwrite_ld R overlaps argument {name}")
            if value.dtype.kind not in "biufUS" or value.dtype.metadata is not None:
                raise ValueError(f"overwrite_ld requires plain nonobject values for {name}")
            if value.ndim != 1:
                raise ValueError(f"overwrite_ld requires one-dimensional arrays for {name}")
            arrays.append(value)
            return
        if type(value) in (list, tuple, dict):
            if id(value) in seen:
                raise ValueError("overwrite_ld does not accept recursive containers")
            seen.add(id(value))
            if type(value) is dict:
                for key, item in value.items():
                    if type(key) is not str:
                        raise ValueError("overwrite_ld metadata keys must be plain strings")
                    walk(item, name)
            else:
                for item in value:
                    walk(item, name)
            seen.remove(id(value))
            return
        raise ValueError(f"overwrite_ld does not accept custom objects for {name}")

    for name, value in arguments.items():
        if name != "R":
            walk(value, name)
    for name in _NONE:
        if arguments[name] is not None:
            raise ValueError(f"overwrite_ld does not support {name}")
    for name in _FALSE:
        if type(arguments[name]) not in (bool, np.bool_) or bool(arguments[name]):
            raise ValueError(f"overwrite_ld requires {name}=False")
    for name in ("maf_thresh", "z_ld_weight", "null_weight"):
        if type(arguments[name]) not in _SCALARS or arguments[name] != 0:
            raise ValueError(f"overwrite_ld requires {name}=0")
    for name, expected in (
        ("R_mismatch", "none"),
        ("estimate_residual_method", "MoM"),
        ("unmappable_effects", "none"),
    ):
        if type(arguments[name]) is not str or arguments[name] != expected:
            raise ValueError(f"overwrite_ld requires {name}={expected!r}")
    if arguments["R_finite"] is not None and arguments["R_finite"] is not False:
        raise ValueError("overwrite_ld requires R_finite=None or False")
    if arguments["convergence_method"] is not None and arguments["convergence_method"] != "elbo":
        raise ValueError("overwrite_ld supports only elbo convergence")
    if type(arguments["z_method"]) not in (str, np.str_) or arguments["z_method"] not in (
        "wald",
        "score",
    ):
        raise ValueError("overwrite_ld requires plain wald or score z_method")
    if arguments["z"] is None or arguments["n"] is None:
        raise ValueError("overwrite_ld requires z and n")
    for name, value in arguments.items():
        if name == "R" or value is None:
            continue
        if name == "variant_metadata":
            if type(value) is not dict:
                raise ValueError("overwrite_ld metadata must be a plain dictionary")
            vectors = value.values()
        elif name in _VECTORS:
            vectors = (value,) if type(value) in (list, tuple, np.ndarray) else ()
        else:
            if type(value) not in _SCALARS:
                raise ValueError(f"overwrite_ld requires a plain scalar for {name}")
            continue
        for vector in vectors:
            if type(vector) not in (list, tuple, np.ndarray):
                raise ValueError("overwrite_ld requires plain vector metadata")
            if any(type(item) not in _SCALARS for item in vector):
                raise ValueError(f"overwrite_ld requires a flat plain vector for {name}")
    strings = {
        "z_method",
        "R_mismatch",
        "estimate_residual_method",
        "convergence_method",
        "unmappable_effects",
        "estimate_prior_method",
    }
    booleans = _FALSE | {
        "estimate_residual_variance",
        "check_prior",
        "estimate_prior_variance",
        "standardize",
        "verbose",
        "overwrite_ld",
    }
    for name, value in arguments.items():
        if (
            name == "R"
            or value is None
            or name in _VECTORS
            or name in ("variant_metadata", "R_finite")
        ):
            continue
        if name in booleans:
            if type(value) not in (bool, np.bool_):
                raise ValueError(f"overwrite_ld requires a bool for {name}")
        elif name in strings or (name == "n_purity" and value == "auto"):
            if type(value) is not str and not (name == "z_method" and type(value) is np.str_):
                raise ValueError(f"overwrite_ld requires a plain string for {name}")
        elif type(value) in (bool, np.bool_, str, np.str_):
            raise ValueError(f"overwrite_ld requires a numeric scalar for {name}")
    for name in ("z", "scaled_prior_variance", "prior_weights"):
        value = arguments[name]
        if type(value) in (str, np.str_):
            raise ValueError(f"overwrite_ld requires numeric values for {name}")
        if type(value) in (list, tuple, np.ndarray) and any(
            type(item) in (str, np.str_) for item in value
        ):
            raise ValueError(f"overwrite_ld requires numeric values for {name}")
    return arguments, layout, arrays


def recheck(state):
    """Recheck the original storage/aliases after permitted ordinary conversions."""
    arguments, layout, arrays = state
    if _layout(arguments["R"]) != layout:
        raise ValueError("overwrite_ld R storage changed before repair")
    low, high = layout[3:5]
    for array in arrays:
        other_low, other_high = _span(array)
        if other_low < high and low < other_high:
            raise ValueError("overwrite_ld R overlaps another input before repair")


def _overwrite_prepare(arguments):
    from ._overwrite import prepare

    state = prepare(arguments)
    _thread_count(arguments["n_threads"])
    L = _integer(arguments["L"], "L")
    _integer(arguments["max_iter"], "max_iter")
    # Reject inexpensive matrix-independent errors before taking workspace.
    # These checks use the ordinary rules without normalizing weights twice.
    _scalar(arguments["n"], "n", lower=1.0, strict=True)
    for name in ("r_tol", "residual_variance_lowerbound"):
        _scalar(arguments[name], name)
    _scalar(1e-4 if arguments["tol"] is None else arguments["tol"], "tol")
    for name in ("greedy_lbf_cutoff", "check_null_threshold"):
        _scalar(arguments[name], name, lower=-np.inf)
    prior_tol = float(arguments["prior_tol"])
    if np.isnan(prior_tol) or prior_tol < 0:
        raise ValueError("prior_tol must be a nonnegative scalar")
    lower = arguments["residual_variance_lowerbound"]
    upper = arguments["residual_variance_upperbound"]
    if upper is not None and (np.isnan(upper) or upper < lower):
        raise ValueError("residual_variance_upperbound must be at least the lower bound")
    if arguments["residual_variance"] is not None:
        _scalar(arguments["residual_variance"], "residual_variance", strict=True)
    for name in ("coverage", "min_abs_corr"):
        value = arguments[name]
        if value is not None and (not np.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"{name} must be in [0, 1]")
    value = arguments["scaled_prior_variance"]
    if np.ndim(value) == 0:
        _scalar(value, "scaled_prior_variance")
    elif np.any(_vector(value, "scaled_prior_variance", L) < 0):
        raise ValueError("scaled_prior_variance must be nonnegative")
    weights = arguments["prior_weights"]
    if weights is not None:
        weights = _vector(weights, "prior_weights", arguments["R"].shape[0])
        if np.any(weights < 0) or not np.isfinite(weights.sum()) or weights.sum() <= 0:
            raise ValueError("prior_weights must be nonnegative with a finite positive sum")
    if arguments["estimate_prior_variance"] and arguments["estimate_prior_method"] not in (
        "optim",
        "EM",
        "simple",
        "fixed_mixture",
    ):
        raise ValueError("estimate_prior_method must be 'optim', 'EM', or 'simple'")
    value = arguments["n_purity"]
    if value is not None and value != "auto" and not np.isfinite(value):
        raise ValueError("n_purity must be a finite number or None")
    _metadata(arguments["variant_ids"], arguments["variant_metadata"], arguments["R"].shape[0])
    return state


def _overwrite_recheck(state):
    from ._overwrite import recheck

    recheck(state)
