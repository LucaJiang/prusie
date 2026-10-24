"""Conservative admission and cleanup for one private RSS execution owner."""

import io
import sys
from types import BuiltinMethodType

import numpy as np


# This partition is deliberately narrower than the public fitting API. Other
# supported types/options retain their parent preparation and core thread count.
_ARGUMENTS = frozenset("""z R n X z_method maf maf_thresh R_finite R_mismatch
control bhat shat var_y z_ld_weight estimate_residual_variance prior_variance
check_prior L scaled_prior_variance residual_variance estimate_prior_variance
estimate_prior_method check_null_threshold prior_tol r_tol prior_weights
null_weight standardize max_iter n_threads overwrite_ld s_init model_init
estimate_residual_method residual_variance_lowerbound residual_variance_upperbound
prior_variance_grid mixture_weights convergence_method alpha0 beta0 L_greedy
greedy_lbf_cutoff slot_prior median_abs_corr cs_extension_corr unmappable_effects
init_only coverage min_abs_corr tol verbose track_fit check_input refine
convergence retain_data n_purity variant_ids variant_metadata""".split())
_NONE = frozenset("""X maf control bhat shat var_y s_init model_init
prior_weights prior_variance_grid mixture_weights alpha0 beta0 L_greedy
slot_prior median_abs_corr cs_extension_corr convergence variant_metadata""".split())
_FALSE = frozenset("""overwrite_ld init_only verbose track_fit check_input
refine retain_data""".split())
_ZERO = frozenset(("z_ld_weight", "null_weight", "maf_thresh"))
_BOOL = frozenset("""estimate_residual_variance check_prior
estimate_prior_variance standardize""".split())
_NUMERIC = frozenset("""prior_variance L scaled_prior_variance
check_null_threshold prior_tol r_tol max_iter residual_variance_lowerbound
greedy_lbf_cutoff""".split())
_NULLABLE_NUMERIC = frozenset("""residual_variance residual_variance_upperbound
coverage min_abs_corr tol""".split())
_CHOICES = {
    "R_finite": (None,),
    "R_mismatch": ("none",),
    "estimate_residual_method": ("MoM",),
    "unmappable_effects": ("none",),
    "convergence_method": (None, "elbo"),
    "z_method": ("wald", "score"),
}


def _ordinary_hooks(api):
    # The existing strict predicate distinguishes standard list recording from
    # displayed warnings, and rejects logging/custom display/dispatch functions.
    if not api._ordinary_overwrite_warning_sink():
        return False
    implementation = api.warning_module._showwarnmsg_impl
    recording = (type(implementation) is BuiltinMethodType
                 and type(implementation.__self__) is list
                 and implementation.__name__ == "append")
    if not recording and (sys.stderr is not sys.__stderr__
                          or type(sys.stderr) is not io.TextIOWrapper):
        return False
    if sys.gettrace() is not None or sys.getprofile() is not None:
        return False
    if np.geterrcall() is not None:
        return False
    return all(value in ("ignore", "warn", "raise") for value in np.geterr().values())


def _allowed(arguments):
    """Inspect exact types without coercion or adding a validation exception."""
    if not _ARGUMENTS.issubset(arguments):
        return False
    if any(arguments[name] is not None for name in _NONE):
        return False
    if any(arguments[name] is not False for name in _FALSE):
        return False
    if any(type(arguments[name]) not in (int, float) or arguments[name] != 0
           for name in _ZERO):
        return False
    if any(type(arguments[name]) is not bool for name in _BOOL):
        return False
    if any(type(arguments[name]) not in (int, float) for name in _NUMERIC):
        return False
    if any(arguments[name] is not None and type(arguments[name]) not in (int, float)
           for name in _NULLABLE_NUMERIC):
        return False
    for name, choices in _CHOICES.items():
        value = arguments[name]
        if value is not None and type(value) is not str:
            return False
        if value not in choices:
            return False
    if type(arguments["estimate_prior_method"]) is not str:
        return False
    purity = arguments["n_purity"]
    if not (purity is None or type(purity) in (int, float)
            or (type(purity) is str and purity == "auto")):
        return False
    ids = arguments["variant_ids"]
    if ids is not None:
        if type(ids) in (list, tuple):
            if not all(type(value) is str for value in ids):
                return False
        elif not (type(ids) is np.ndarray and ids.ndim == 1
                  and ids.dtype.kind in "US" and ids.dtype.metadata is None):
            return False
    return True


def create_executor(arguments):
    """Return an inert private owner, or keep the complete parent route."""
    from . import api

    count = arguments["n_threads"]
    matrix = arguments["R"]
    if type(count) is not int or not 2 <= count <= 32:
        return None
    if not (type(matrix) is np.ndarray and matrix.ndim == 2
            and matrix.shape[0] >= 1024 and matrix.shape[0] == matrix.shape[1]
            and matrix.dtype is api._FUSED_RSS_DTYPE and matrix.flags.c_contiguous
            and matrix.flags.aligned and matrix.strides == (matrix.shape[0] * 8, 8)):
        return None
    if not _allowed(arguments) or not _ordinary_hooks(api):
        return None
    # The binding factory itself is inert. Architecture/FP checks and worker
    # construction happen only under the native helper's checked read borrow.
    return api._native._new_fit_executor(count)


def close_executor(executor):
    """Join every worker while preserving an already-active primary exception."""
    primary = sys.exc_info()[0] is not None
    try:
        executor.close()
    except BaseException:
        if not primary:
            raise
