"""Public SuSiE fitting functions for summary and sufficient statistics.

``susie_rss`` converts marginal association statistics and signed LD into
working sufficient statistics; ``susie_suff_stat`` validates them, runs the
native IBSS solver and assembles a :class:`~prusie.result.SusieResult`.
"""

from __future__ import annotations

import math
import warnings as warning_module
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _native
from ._array_layout import native_array
from .control import RSSControl, SlotPrior, susie_rss_control
from .diagnostics import (
    ConvergenceConfig,
    convergence_summary,
    nonconvergence_message,
    resolve_convergence,
    tracking_level,
)
from .individual import SusieInputs
from .posterior import credible_sets, marginal_pip
from .refinement import RefinementConfig, initial_options, make_prepared, refine_fit
from .result import SusieResult


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
        if name == "Xty":
            _warn("NA values in Xty are replaced with 0.", collected)
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
        symmetric = _native.validate_rss_matrix(array, rss_scale)
    # native_array has already ensured alignment and element-compatible strides;
    # any copy made here is C-ordered, matching the public input orientation.
    return np.ascontiguousarray(array), symmetric


def _symmetrize_matrix(array, name, scale=1.0, collected=None):
    if collected is None:
        collected = []
    _warn(f"{name} is not symmetric; using ({name} + t({name}))/2.", collected)
    with np.errstate(over="ignore", invalid="ignore"):
        working = array if scale == 1.0 else array * scale
        repaired = (working + working.T) / 2
    _native.validate_matrix(repaired, name)
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


def _legacy_analysis_limitations(
    options: Mapping[str, Any],
    *,
    design: Any = None,
    dynamic: Any = None,
    L_greedy: int | None = None,
    median_abs_corr: float | None = None,
    cs_extension_corr: float | None = None,
) -> tuple[str, ...]:
    """Describe models/controls the existing retained-analysis engine cannot replay."""
    conditions = (
        (design is not None, "matrix-free design"),
        (dynamic is not None, "dynamic RSS model"),
        (options.get("prior_variance_grid") is not None, "mixture prior"),
        (options.get("estimate_residual_method") == "NIG", "NIG residual model"),
        (options.get("slot_prior") is not None, "slot-activity prior"),
        (options.get("criterion") == "pip", "convergence_method='pip'"),
        (L_greedy is not None, "L_greedy"),
        (median_abs_corr is not None, "median_abs_corr"),
        (cs_extension_corr is not None, "cs_extension_corr"),
    )
    return tuple(reason for active, reason in conditions if active)


def _require_legacy_analysis(
    retain_data: bool,
    refine: bool | RefinementConfig,
    limitations: tuple[str, ...],
) -> None:
    """Reject explicit unsupported post-analysis requests before scientific fitting."""
    if limitations and (retain_data or refine is not False):
        raise ValueError(
            "Retained continuation/refinement is unavailable for "
            + ", ".join(limitations)
            + ". It supports static materialized Gaussian models with fixed L, "
            "ELBO or ConvergenceConfig stopping, and default median_abs_corr/"
            "cs_extension_corr. Use retain_data=False; for single-panel models, "
            "refine=True (not a RefinementConfig) refits them as susieR does."
        )


@dataclass(frozen=True)
class _FitRequest:
    """Validated native inputs and metadata used by every fitting entry point."""

    crossproduct: NDArray[np.float64]
    crossresponse: NDArray[np.float64]
    yty: float
    n: float
    options: dict[str, Any]
    XtX: NDArray[np.float64]
    diagonal: NDArray[np.float64]
    scales: NDArray[np.float64]
    means: NDArray[np.float64]
    y_mean_value: float
    ids: NDArray[np.str_]
    metadata: dict[str, NDArray[Any]]
    coverage: float | None
    min_abs_corr: float | None
    null_index: int | None
    n_purity: int | str | None
    prior_tol: float
    collected: list[str]
    qc: dict[str, Any]
    params: dict[str, Any]
    control: Any
    resolution: dict[str, Any]
    prepared: Any
    signature: str | None
    refine: bool | RefinementConfig
    _rss_matrix_scale: float
    design: Any = None
    dynamic: Any = None
    deferred_model_init: SusieInputs | None = None
    analysis_limitations: tuple[str, ...] = ()
    requested_convergence_method: str | None = None


# Both public fitters share input preparation and result assembly.
def susie_suff_stat(
    XtX: ArrayLike,
    Xty: ArrayLike,
    yty: float,
    n: float,
    *,
    X_colmeans: ArrayLike | None = None,
    y_mean: float | None = None,
    L: int = 10,
    scaled_prior_variance: float | ArrayLike = 0.2,
    residual_variance: float | None = None,
    estimate_residual_variance: bool = True,
    estimate_prior_variance: bool = True,
    estimate_prior_method: str = "optim",
    check_null_threshold: float = 0.0,
    prior_tol: float = 1e-9,
    r_tol: float = 1e-8,
    prior_weights: ArrayLike | None = None,
    null_weight: float = 0.0,
    standardize: bool = True,
    max_iter: int = 100,
    s_init: SusieResult | Mapping[str, Any] | None = None,
    model_init: SusieResult | Mapping[str, Any] | None = None,
    estimate_residual_method: str = "MoM",
    residual_variance_lowerbound: float = 0.0,
    residual_variance_upperbound: float | None = np.inf,
    prior_variance_grid: ArrayLike | None = None,
    mixture_weights: ArrayLike | None = None,
    convergence_method: str | None = None,
    alpha0: float | None = None,
    beta0: float | None = None,
    L_greedy: int | None = None,
    greedy_lbf_cutoff: float = 0.1,
    slot_prior: SlotPrior | Mapping[str, Any] | None = None,
    median_abs_corr: float | None = None,
    cs_extension_corr: float | None = None,
    unmappable_effects: str = "none",
    init_only: bool = False,
    coverage: float | None = 0.95,
    min_abs_corr: float | None = 0.5,
    tol: float | None = 1e-4,
    verbose: bool = False,
    track_fit: bool | str = False,
    check_input: bool = False,
    refine: bool | RefinementConfig = False,
    convergence: ConvergenceConfig | None = None,
    retain_data: bool = False,
    check_prior: bool = False,
    n_purity: int | str | None = None,
    variant_ids: Sequence[str] | None = None,
    variant_metadata: Mapping[str, ArrayLike] | None = None,
    maf: ArrayLike | None = None,
    maf_thresh: float = 0.0,
) -> SusieResult | SusieInputs:
    """Fit SuSiE to centered sufficient statistics.

    Parameters
    ----------
    XtX : array_like, shape (p, p)
        Centered predictor crossproduct XᵀX for p variants, converted to float64.
        Must be finite. Asymmetric input is warned about and averaged internally.
    Xty : array_like, shape (p,)
        Centered predictor-response crossproduct Xᵀy, in the order of XtX.
    yty : float
        Centered response sum of squares yᵀy; must be positive.
    n : float
        Sample size; must exceed one. It is never inferred from the dimension
        of XtX.
    X_colmeans : array_like or float, optional
        Predictor means, used only to compute ``intercept``.
    y_mean : float, optional
        Response mean, used only to compute ``intercept``. Without both means
        the returned intercept is NaN.
    L : int, default 10
        Maximum number of single effects. The number fitted is min(L, q), where
        q counts the variants plus the optional null column.
    scaled_prior_variance : float or array_like, default 0.2
        Initial prior effect variance V (σ₀²) divided by yᵀy/(n - 1).
        A nonnegative scalar or a length-L nonnegative vector; values above
        one are allowed, including when ``standardize=True``.
    residual_variance : float, optional
        Initial residual variance σ², or its fixed value when it is not
        estimated. Defaults to yᵀy/(n - 1).
    estimate_residual_variance : bool, default True
        Update σ² after each sweep that does not stop the iteration.
    estimate_prior_variance : bool, default True
        Estimate the prior variance V_ℓ of each single effect.
    estimate_prior_method : {'optim', 'EM', 'simple'}, default 'optim'
        'optim' maximizes the single-effect regression (SER) marginal
        likelihood over log V; 'EM' sets V_ℓ to the posterior second moment
        after each SER update; 'simple' only compares the current V_ℓ with
        V_ℓ = 0.
    check_null_threshold : float, default 0
        With 'optim' or 'simple', V_ℓ is set to zero unless its log marginal
        likelihood exceeds that of V_ℓ = 0 by more than this value.
    prior_tol : float, default 1e-9
        After the final sweep, including at the iteration limit, components
        with V_ℓ < prior_tol have V_ℓ, moments, Bayes factors and KL set to
        zero and alpha reset to the prior weights. PIPs use components with
        V_ℓ > prior_tol; credible sets use a fixed activity threshold of 1e-9.
    r_tol : float, default 1e-8
        Eigenvalue tolerance for ``check_input``.
    prior_weights : array_like, shape (p,), optional
        Nonnegative prior inclusion weights π with a finite positive sum.
        Normalized after null-column augmentation. Defaults to uniform.
    null_weight : float, default 0
        Prior weight of an optional last null column with zero crossproducts.
        Must lie in [0, 1); 0 omits the column.
    standardize : bool, default True
        Scale predictors to unit sample variance, d_j = (XᵀX_jj/(n - 1))^½.
    max_iter : int, default 100
        Maximum number of iterative Bayesian stepwise selection (IBSS) sweeps.
    s_init : SusieResult or mapping, optional
        Working-coordinate posterior state; may initialize a new data set.
        A mapping needs ``alpha``, ``mu`` and ``mu2``. Supplied variant IDs,
        allele metadata, working scales and null layout must agree, but a
        retained-data signature is not required. ``V`` and ``sigma2`` are
        optional. An initial state with more components increases effective L.
        Existing components are
        used and any further components start at the prior; fitted products
        are recomputed from the state.
    coverage : float or None, default 0.95
        Requested posterior mass of each credible set (CS). None disables CS
        construction; PIPs are still returned.
    min_abs_corr : float or None, default 0.5
        Minimum absolute pairwise correlation (purity) for a CS to be
        retained. None disables CS construction.
    tol : float, default 1e-4
        ELBO tolerance; the default rule converges when 0 ≤ ΔELBO < tol.
    verbose : bool, default False
        Log convergence and credible-set counts through the prusie logger.
    track_fit : bool or {'compact', 'dense'}, default False
        Record per-sweep convergence metrics. True or 'compact' stores exact
        distances and a thresholded alpha display; 'dense' also stores full
        posterior arrays and is intended for small problems.
    check_input : bool, default False
        Eigendecompose XtX, raise if its smallest eigenvalue is below -r_tol,
        and warn if Xty lies outside the span of the eigenvectors whose
        eigenvalues exceed machine epsilon.
    refine : bool or RefinementConfig, default False
        Run the credible-set exclusion search; True uses its default
        configuration. For static materialized Gaussian models with fixed L,
        ELBO or ConvergenceConfig stopping and default median_abs_corr and
        cs_extension_corr, this is :func:`refine_fit`. For matrix-free designs,
        mixture/NIG/slot models, dynamic RSS, L_greedy, convergence_method='pip'
        or nondefault CS controls, ``refine=True`` refits the model as susieR
        does (block coordinate ascent over credible sets) and records the search
        in ``refinement``; a RefinementConfig is not available for these models.
    convergence : ConvergenceConfig, optional
        Stopping rule (ELBO, alpha/PIP or joint). Defaults to the ELBO rule,
        or to the controls recorded in ``s_init`` when it is a fit.
    retain_data : bool, default False
        Retain read-only inputs for continuation, refinement and objective
        evaluation within the same supported scope as ``refine``;
        unsupported explicit requests raise ValueError before fitting.
        Tracking, initialization and refinement imply retention only within
        that scope. Other models remain fit-capable with the default flags
        and do not retain a snapshot for these helpers. No dense Gram is formed to
        retain a matrix-free design. Without retention, ``data_signature`` is
        None. Initialization-only calls prepare inputs without this rejection.
    check_prior : bool, default False
        Raise an error if an estimated V_ℓ exceeds 100 max_j z_j², where z_j is
        the marginal z-score at the current σ².
    n_purity : int, optional
        Accepted for compatibility with susieR. With matrix input, purity
        always uses all pairs of CS members.
    variant_ids : sequence of str, optional
        Distinct nonempty variant identifiers in input order. Defaults to
        positional labels '0', ..., 'p-1'.
    variant_metadata : mapping of str to array_like, optional
        Length-p columns, for example chromosome or alleles, copied into the
        result.
    model_init : SusieResult or mapping, optional
        Initial working-coordinate posterior; s_init is a backwards-compatible alias.
    estimate_residual_method : {'MoM', 'MLE', 'NIG'}, default 'MoM'
        Residual-variance estimator or conjugate Normal-Inverse-Gamma model.
    residual_variance_lowerbound, residual_variance_upperbound : float
        Residual variance bounds. None for the upper bound means unbounded.
    prior_variance_grid, mixture_weights : array_like, optional
        Positive prior variances and normalized weights for a fixed Gaussian mixture.
    alpha0, beta0 : float, optional
        NIG hyperparameters, defaulting to 1/sqrt(n).
    L_greedy : int, optional
        Grow the component count by this increment until an empty effect appears.
    greedy_lbf_cutoff : float, default 0.1
        Log-Bayes-factor cutoff defining an empty effect for greedy growth.
    slot_prior : SlotPrior or mapping, optional
        Beta-Binomial or Gamma-Poisson effect-activity prior.
    convergence_method : {'elbo', 'pip'}, optional
        susieR stopping rule; pip checks component alpha and aggregate PIP changes
        and detects cycles. Use convergence for the existing diagnostic controls.
    median_abs_corr, cs_extension_corr : float, optional
        Alternative CS purity threshold and correlation threshold for proxy extension.
    init_only : bool, default False
        Return validated data and parameters without fitting.
    unmappable_effects : {'none'}, default 'none'
        SuSiE-inf, SuSiE-ash and Mr.ASH are outside the supported model set.
    maf, maf_thresh : optional
        Retain variants whose supplied minor-allele frequency exceeds maf_thresh.

    Returns
    -------
    SusieResult or SusieInputs
        Variational posterior (alpha, mu, mu2), Bayes factors, prior and
        residual variances, ELBO trace, PIPs, credible sets and run metadata.

    Raises
    ------
    ValueError
        Invalid shapes, nonfinite values, out-of-range parameters,
        inconsistent initialization or a native numerical failure.
    TypeError
        ``s_init``, ``refine`` or ``convergence`` has an unsupported type.

    Warns
    -----
    RuntimeWarning
        Nonconvergence and numerical diagnostics. Messages are also stored in
        ``SusieResult.warnings``.

    Notes
    -----
    The model is the Gaussian sum of single effects regression
    y = Xb + ε, ε ~ N(0, σ²I), b = ∑_ℓ b_ℓ, in which each b_ℓ has one nonzero
    entry with prior N(0, V_ℓ) at a location drawn from π. IBSS updates the L
    single effects in sequence on the working statistics D⁻¹XᵀXD⁻¹ and
    D⁻¹Xᵀy, where D = diag(d). The Gaussian updates follow susieR 0.16.6.
    As in the dense numeric susieR path, exact off-diagonal inequality triggers
    a warning and internal replacement by (XtX + XtXᵀ)/2. With check_input=False,
    NaNs in Xty are replaced with zero with a warning; check_input=True rejects
    them. Nonfinite XtX, infinite Xty and averaging overflow raise an error.
    Caller arrays are never modified.
    ``mu`` and ``mu2`` are on the working predictor scale; ``coef`` returns
    posterior mean effects on the input scale. CS purity uses correlations
    derived from the complete XtX submatrix of each set.

    References
    ----------
    Wang G, Sarkar A, Carbonetto P, Stephens M (2020). A simple new approach to
    variable selection in regression, with application to genetic fine mapping.
    J R Stat Soc B 82:1273-1300. Reference implementation: susieR 0.16.6,
    R/sufficient_stats_methods.R.

    Examples
    --------
    >>> fit = susie_suff_stat([[99., 0.], [0., 99.]], [30., 0.], 99., 100,
    ...                       L=1, estimate_residual_variance=False)
    >>> fit.alpha.shape
    (1, 2)
    """
    return _fit(_prepare_suff_stat(**locals()))


def _prepare_suff_stat(
    XtX: ArrayLike,
    Xty: ArrayLike,
    yty: float,
    n: float,
    *,
    X_colmeans: ArrayLike | None = None,
    y_mean: float | None = None,
    L: int = 10,
    scaled_prior_variance: float | ArrayLike = 0.2,
    residual_variance: float | None = None,
    estimate_residual_variance: bool = True,
    estimate_prior_variance: bool = True,
    estimate_prior_method: str = "optim",
    check_null_threshold: float = 0.0,
    prior_tol: float = 1e-9,
    r_tol: float = 1e-8,
    prior_weights: ArrayLike | None = None,
    null_weight: float = 0.0,
    standardize: bool = True,
    max_iter: int = 100,
    s_init: SusieResult | Mapping[str, Any] | None = None,
    model_init: SusieResult | Mapping[str, Any] | None = None,
    estimate_residual_method: str = "MoM",
    residual_variance_lowerbound: float = 0.0,
    residual_variance_upperbound: float | None = np.inf,
    prior_variance_grid: ArrayLike | None = None,
    mixture_weights: ArrayLike | None = None,
    convergence_method: str | None = None,
    alpha0: float | None = None,
    beta0: float | None = None,
    L_greedy: int | None = None,
    greedy_lbf_cutoff: float = 0.1,
    slot_prior: SlotPrior | Mapping[str, Any] | None = None,
    median_abs_corr: float | None = None,
    cs_extension_corr: float | None = None,
    unmappable_effects: str = "none",
    init_only: bool = False,
    coverage: float | None = 0.95,
    min_abs_corr: float | None = 0.5,
    tol: float | None = 1e-4,
    verbose: bool = False,
    track_fit: bool | str = False,
    check_input: bool = False,
    refine: bool | RefinementConfig = False,
    convergence: ConvergenceConfig | None = None,
    retain_data: bool = False,
    check_prior: bool = False,
    n_purity: int | str | None = None,
    variant_ids: Sequence[str] | None = None,
    variant_metadata: Mapping[str, ArrayLike] | None = None,
    maf: ArrayLike | None = None,
    maf_thresh: float = 0.0,
    _rss_matrix_scale=1.0,
    _rss_matrix_validated=False,
    _design=None,
    _dynamic=None,
    _summary_missing_policy: str | None = None,
) -> _FitRequest:
    """Validate a sufficient-statistics request for the shared fitting path."""
    collected = []
    if model_init is not None and s_init is not None:
        raise ValueError("Specify only model_init or its s_init alias")
    s_init = model_init if model_init is not None else s_init
    if unmappable_effects != "none":
        raise ValueError("SuSiE-inf, SuSiE-ash and Mr.ASH are outside this implementation")
    if estimate_residual_method not in ("MoM", "MLE", "NIG"):
        raise ValueError("estimate_residual_method must be 'MoM', 'MLE', or 'NIG'")
    if convergence_method not in (None, "elbo", "pip"):
        raise ValueError("convergence_method must be 'elbo' or 'pip'")
    lower = _scalar(residual_variance_lowerbound, "residual_variance_lowerbound")
    upper = np.inf if residual_variance_upperbound is None else float(residual_variance_upperbound)
    if np.isnan(upper) or upper < lower:
        raise ValueError("residual_variance_upperbound must be at least the lower bound")
    if prior_variance_grid is not None:
        grid = _vector(prior_variance_grid, "prior_variance_grid")
        mix = (
            np.full(len(grid), 1 / len(grid))
            if mixture_weights is None
            else _vector(mixture_weights, "mixture_weights", len(grid))
        )
        if np.any(grid <= 0) or np.any(mix < 0) or abs(float(mix.sum()) - 1) >= 1e-8:
            raise ValueError("prior_variance_grid must be positive and mixture_weights normalized")
        prior_variance_grid, mixture_weights = grid, mix
        estimate_prior_variance, estimate_prior_method = False, "fixed_mixture"
    elif mixture_weights is not None:
        raise ValueError("mixture_weights requires prior_variance_grid")
    if slot_prior is not None:
        if not isinstance(slot_prior, Mapping):
            raise TypeError("slot_prior must be a SlotPrior or mapping")
        slot_prior = dict(slot_prior)
    if L_greedy is not None:
        L_greedy = _integer(L_greedy, "L_greedy")
    greedy_lbf_cutoff = _scalar(greedy_lbf_cutoff, "greedy_lbf_cutoff", lower=-np.inf)
    n = _scalar(n, "n", lower=1.0, strict=True)
    for name, value in (
        ("standardize", standardize),
        ("estimate_prior_variance", estimate_prior_variance),
        ("estimate_residual_variance", estimate_residual_variance),
        ("check_input", check_input),
        ("check_prior", check_prior),
        ("retain_data", retain_data),
        ("verbose", verbose),
    ):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be a boolean")
    yty = _scalar(yty, "yty")
    Xty = _vector(Xty, "Xty") if check_input else _summary_vector(Xty, "Xty", collected)
    m = Xty.size
    if _design is not None:
        XtX = None
    elif _rss_matrix_validated:
        # Only susie_rss sets this flag, after validating every raw and scaled
        # matrix entry; direct susie_suff_stat calls always validate XtX here.
        assert isinstance(XtX, np.ndarray) and XtX.shape == (m, m) and XtX.dtype == np.float64
    else:
        XtX, symmetric = _validated_matrix(XtX, "XtX", m, scale=_rss_matrix_scale)
        if not symmetric:
            XtX = _symmetrize_matrix(XtX, "XtX", _rss_matrix_scale, collected)
            _rss_matrix_scale = 1.0
    original_m = m
    ids, metadata = _metadata(variant_ids, variant_metadata, m)
    if maf is not None:
        frequencies = _vector(maf, "maf", m)
        threshold = _scalar(maf_thresh, "maf_thresh")
        if np.any((frequencies < 0) | (frequencies > 1)) or threshold > 1:
            raise ValueError("maf and maf_thresh must lie in [0, 1]")
        selected = np.flatnonzero(frequencies > threshold)
        if not selected.size:
            raise ValueError("MAF filtering removed every predictor")
        if _design is not None:
            _design = replace(
                _design,
                X=_design.X[:, selected],
                means=_design.means[selected],
                scales=_design.scales[selected],
                diagonal=_design.diagonal[selected],
            )
        else:
            XtX = XtX[np.ix_(selected, selected)]
        Xty, ids = Xty[selected], ids[selected]
        metadata = {key: value[selected] for key, value in metadata.items()}
        if prior_weights is not None:
            prior_weights = np.asarray(prior_weights)[selected]
        if X_colmeans is not None and np.ndim(X_colmeans) != 0:
            X_colmeans = np.asarray(X_colmeans)[selected]
        m = len(selected)
    elif maf_thresh != 0:
        raise ValueError("maf_thresh requires maf")
    L, max_iter = _integer(L, "L"), _integer(max_iter, "max_iter")
    tol = _scalar(
        (1e-6 if estimate_residual_method == "NIG" else 1e-4) if tol is None else tol, "tol"
    )
    if np.ndim(prior_tol) != 0 or np.isnan(float(prior_tol)) or float(prior_tol) < 0:
        raise ValueError("prior_tol must be a nonnegative scalar")
    prior_tol = float(prior_tol)
    r_tol = _scalar(r_tol, "r_tol")
    check_null_threshold = _scalar(check_null_threshold, "check_null_threshold", lower=-np.inf)
    if np.ndim(scaled_prior_variance) == 0:
        scaled_prior_variance = _scalar(scaled_prior_variance, "scaled_prior_variance")
    else:
        scaled_prior_variance = _vector(scaled_prior_variance, "scaled_prior_variance", L)
        if np.any(scaled_prior_variance < 0):
            raise ValueError("scaled_prior_variance must be nonnegative")
        scaled_prior_variance = scaled_prior_variance.tolist()
    if yty == 0 and (np.any(Xty != 0) or residual_variance is None or estimate_residual_variance):
        raise ValueError("yty=0 requires Xty=0 and a fixed positive residual_variance")
    response_variance = yty / (n - 1) if _design is None else _design.response_variance
    if estimate_residual_method == "NIG":
        alpha0 = _scalar(1 / np.sqrt(n) if alpha0 is None else alpha0, "alpha0", strict=True)
        beta0 = _scalar(1 / np.sqrt(n) if beta0 is None else beta0, "beta0", strict=True)
        estimate_residual_variance = True
        if estimate_prior_variance:
            estimate_prior_method = "EM"
    sigma2 = (
        response_variance
        if residual_variance is None
        else _scalar(residual_variance, "residual_variance", strict=True)
    )
    if estimate_prior_variance and estimate_prior_method not in (
        "optim",
        "EM",
        "simple",
        "fixed_mixture",
    ):
        raise ValueError("estimate_prior_method must be 'optim', 'EM', or 'simple'")
    for name, value in (
        ("coverage", coverage),
        ("min_abs_corr", min_abs_corr),
        ("median_abs_corr", median_abs_corr),
        ("cs_extension_corr", cs_extension_corr),
    ):
        if value is not None and (not np.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"{name} must be in [0, 1]")
    if n_purity is not None and n_purity != "auto":
        if (
            isinstance(n_purity, (bool, np.bool_))
            or not np.isscalar(n_purity)
            or not np.isfinite(n_purity)
        ):
            raise ValueError("n_purity must be a finite number or None")
        n_purity = int(np.floor(n_purity))
    qc: dict[str, Any] = {
        "matrix_symmetric": True,
        "psd_checked": bool(check_input),
        "input_columns": m,
        "input_missing_policy": _summary_missing_policy or ("reject" if check_input else "zero"),
        "original_input_columns": original_m,
        "variant_ids_are_positional": variant_ids is None,
    }
    if check_input and _design is None:
        eigenvalues, eigenvectors = np.linalg.eigh(
            XtX if _rss_matrix_scale == 1.0 else XtX * _rss_matrix_scale
        )
        qc["minimum_eigenvalue"] = float(eigenvalues[0])
        if eigenvalues[0] < 0 and eigenvalues[0] <= -r_tol:
            raise ValueError("XtX is not positive semidefinite")
        low = eigenvectors[:, eigenvalues <= np.finfo(float).eps]
        magnitude = float(np.max(np.abs(Xty)))
        fraction = 0.0
        if magnitude > 0 and low.shape[1]:
            normalized = Xty / magnitude
            coefficients = low.T @ normalized
            fraction = float(np.sum(coefficients**2) / np.sum(normalized**2))
        qc["projection_squared_fraction"] = fraction
        if fraction > np.finfo(float).eps:
            _warn("Xty does not lie in the nonzero eigenspace of XtX", collected)
    weights = (
        np.full(m, 1.0 / m)
        if prior_weights is None
        else _vector(prior_weights, "prior_weights", m).copy()
    )
    if np.any(weights < 0) or not np.isfinite(weights.sum()) or weights.sum() <= 0:
        raise ValueError("prior_weights must be nonnegative with a finite positive sum")
    null_weight = _scalar(0.0 if null_weight is None else null_weight, "null_weight")
    if null_weight >= 1:
        raise ValueError("null_weight must be < 1")
    null_index = m if null_weight > 0 else None
    if null_index is not None:
        # As in susieR: augment the raw weights, then normalize the whole vector.
        weights = np.r_[weights * (1 - null_weight), null_weight]
        if _design is None:
            XtX = np.pad(XtX, ((0, 1), (0, 1)))
        Xty = np.r_[Xty, 0.0]
    # susieR 0.16.6 normalizes default and supplied weights alike.
    weights /= math.fsum(weights)
    p = len(Xty)
    effective_L = min(L, p)
    means = np.full(p, np.nan) if X_colmeans is None else np.asarray(X_colmeans, dtype=float)
    if means.ndim == 0:
        means = np.full(p, means)
    elif null_index is not None and means.shape == (m,):
        means = np.r_[means, 0.0]
    if means.shape != (p,):
        raise ValueError("X_colmeans must be scalar or have one value per predictor")
    y_mean_value = np.nan if y_mean is None else float(y_mean)
    diagonal = XtX.diagonal().copy() if _design is None else _design.diagonal.copy()
    if _design is not None and null_index is not None:
        diagonal = np.r_[diagonal, 0.0]
    if _rss_matrix_scale != 1.0:
        diagonal *= _rss_matrix_scale
    scales = np.sqrt(diagonal / (n - 1)) if standardize else np.ones(p)
    if _design is not None:
        scales = _design.scales.copy()
        if null_index is not None:
            scales = np.r_[scales, 1.0]
    scales[scales == 0] = 1.0
    # Working crossproduct D⁻¹ XtX D⁻¹ in native column-major layout.
    inverse_scales = 1.0 / scales
    # Version 1 accepts finite positive factors; keep the materialized path for
    # unsupported native contracts and extreme derived scales (e.g. sqrt overflow).
    matrix_operator = (
        _design is None
        and getattr(_native, "MATRIX_OPERATOR_VERSION", 0) == 1
        and math.isfinite(_rss_matrix_scale)
        and _rss_matrix_scale > 0
        and np.isfinite(scales).all()
        and (scales > 0).all()
        and np.isfinite(inverse_scales).all()
        and (inverse_scales > 0).all()
    )
    if _design is not None:
        crossproduct = None
    elif matrix_operator:
        # The native operator applies the scalar and diagonal scales itself and
        # falls back to full materialization; pass the unscaled matrix.
        crossproduct = XtX.T
    else:
        crossproduct = _native.prepare_crossproduct(XtX, inverse_scales, scales, _rss_matrix_scale)
        if crossproduct is None:
            # NumPy route when runtime SIMD support or contiguous input is absent.
            crossproduct = np.empty((p, p), dtype=np.float64, order="F")
            if _rss_matrix_scale == 1.0:
                np.multiply(XtX.T, inverse_scales[None, :], out=crossproduct)
            else:
                # Apply the RSS scalar in place in the single p × p allocation,
                np.multiply(XtX.T, _rss_matrix_scale, out=crossproduct)
                np.multiply(crossproduct, inverse_scales[None, :], out=crossproduct)
            np.divide(crossproduct, scales[:, None], out=crossproduct)
    crossresponse = np.ascontiguousarray(Xty / scales)
    options = dict(
        l=effective_L,
        prior_variance=float(np.asarray(scaled_prior_variance).flat[0])
        * (1.0 if estimate_residual_method == "NIG" else response_variance),
        residual_variance=sigma2,
        prior_weights=weights,
        estimate_prior_variance=bool(estimate_prior_variance),
        estimate_prior_method=estimate_prior_method,
        estimate_residual_variance=bool(estimate_residual_variance),
        check_null_threshold=check_null_threshold,
        max_iter=max_iter,
        tol=tol,
        check_prior=bool(check_prior),
        prior_tol=prior_tol,
        null_index=null_index,
    )
    if isinstance(scaled_prior_variance, list):
        options["component_prior_variance"] = np.asarray(scaled_prior_variance[:effective_L]) * (
            1.0 if estimate_residual_method == "NIG" else response_variance
        )
    options.update(
        estimate_residual_method=estimate_residual_method,
        residual_variance_lowerbound=lower,
        residual_variance_upperbound=upper,
    )
    if prior_variance_grid is not None:
        options.update(prior_variance_grid=prior_variance_grid, mixture_weights=mixture_weights)
    if estimate_residual_method == "NIG":
        options.update(alpha0=alpha0, beta0=beta0)
    if slot_prior is not None:
        options["slot_prior"] = slot_prior
    if matrix_operator:
        options.update(
            matrix_global_scale=float(_rss_matrix_scale),
            matrix_inverse_scales=inverse_scales,
            matrix_scales=scales,
        )
    control, resolution = resolve_convergence(
        s_init if hasattr(s_init, "params") else None, convergence
    )
    level = tracking_level(track_fit)
    options.update(control.options(), track_level=level)
    if convergence_method is not None:
        options["criterion"] = convergence_method
        if convergence_method == "pip":
            options.update(alpha_tol=tol, pip_tol=tol)
    if slot_prior is not None or (estimate_residual_method == "NIG" and effective_L > 1):
        options.update(criterion="pip", alpha_tol=tol, pip_tol=tol)
    # Only an owned snapshot can supply a stable identity for later reuse.
    # Ordinary fits do not retain caller buffers or scan an unused hash.
    signature = None
    prepared = None
    analysis_limitations = _legacy_analysis_limitations(
        options,
        design=_design,
        dynamic=_dynamic,
        L_greedy=L_greedy,
        median_abs_corr=median_abs_corr,
        cs_extension_corr=cs_extension_corr,
    )
    if not analysis_limitations and (
        retain_data or level or refine is not False or s_init is not None
    ):
        prepared = make_prepared(
            crossproduct,
            crossresponse,
            yty,
            n,
            options,
            XtX,
            diagonal,
            scales,
            means,
            y_mean_value,
            _rss_matrix_scale,
            ids,
            metadata,
        )
        signature = prepared.signature
    if s_init is not None:
        initial_alpha = (
            s_init.alpha
            if hasattr(s_init, "alpha")
            else s_init.get("alpha")
            if isinstance(s_init, Mapping)
            else None
        )
        initial_shape = np.shape(initial_alpha)
        if len(initial_shape) == 2 and initial_shape[0] > effective_L:
            effective_L = initial_shape[0]
            options["l"] = effective_L
            if prepared is not None:
                prepared = replace(
                    prepared, options=MappingProxyType({**prepared.options, "l": effective_L})
                )
            _warn(f"Requested L={L} is smaller than model_init; using L={effective_L}.", collected)
        options.update(
            initial_options(
                _model_init_state(s_init),
                ids=ids,
                metadata=metadata,
                scales=scales,
                null_index=null_index,
                signature=signature,
                L=effective_L,
                weights=weights,
                estimate_prior_variance=bool(estimate_prior_variance),
                estimate_residual_variance=bool(estimate_residual_variance),
                prior_variance=options.get("component_prior_variance", options["prior_variance"]),
            )
        )
    if prepared is not None:
        # Initialization callbacks may mutate caller arrays. Fit and construct
        # credible sets from the same owned data whose signature was recorded.
        crossproduct, crossresponse = prepared.matrix, prepared.response
        XtX = prepared.cs_matrix
    params = dict(
        L=L,
        effective_L=effective_L,
        n=n,
        yty=yty,
        scaled_prior_variance=scaled_prior_variance,
        residual_variance=residual_variance,
        estimate_residual_variance=bool(estimate_residual_variance),
        estimate_prior_variance=bool(estimate_prior_variance),
        estimate_prior_method=estimate_prior_method,
        prior_weights=weights.tolist(),
        null_weight=null_weight,
        standardize=bool(standardize),
        max_iter=max_iter,
        tol=tol,
        coverage=coverage,
        min_abs_corr=min_abs_corr,
        n_purity=n_purity,
        prior_tol=prior_tol,
        check_null_threshold=check_null_threshold,
        check_input=bool(check_input),
        r_tol=r_tol,
        check_prior=bool(check_prior),
        s_init="posterior_state" if s_init is not None else None,
        refine=bool(refine),
        retain_data=bool(retain_data),
        track_fit=track_fit,
        convergence=control.options(),
        convergence_resolution=resolution,
        convergence_method=convergence_method,
        estimate_residual_method=estimate_residual_method,
        residual_variance_lowerbound=lower,
        residual_variance_upperbound=None if np.isinf(upper) else upper,
        prior_variance_grid=prior_variance_grid,
        mixture_weights=mixture_weights,
        alpha0=alpha0,
        beta0=beta0,
        L_greedy=L_greedy,
        greedy_lbf_cutoff=greedy_lbf_cutoff,
        slot_prior=slot_prior,
        median_abs_corr=median_abs_corr,
        cs_extension_corr=cs_extension_corr,
        verbose=verbose,
        init_only=init_only,
    )
    return _FitRequest(
        crossproduct=crossproduct,
        crossresponse=crossresponse,
        yty=yty,
        n=n,
        options=options,
        XtX=XtX,
        diagonal=diagonal,
        scales=scales,
        means=means,
        y_mean_value=y_mean_value,
        ids=ids,
        metadata=metadata,
        coverage=coverage,
        min_abs_corr=min_abs_corr,
        null_index=null_index,
        n_purity=n_purity,
        prior_tol=prior_tol,
        collected=collected,
        qc=qc,
        params=params,
        control=control,
        resolution=resolution,
        prepared=prepared,
        signature=signature,
        refine=refine,
        _rss_matrix_scale=_rss_matrix_scale,
        design=_design,
        dynamic=_dynamic,
        analysis_limitations=analysis_limitations,
        requested_convergence_method=convergence_method,
    )


def _fit(request: _FitRequest) -> SusieResult | SusieInputs:
    """Run the prepared model and assemble the common public result."""
    limitations = tuple(
        dict.fromkeys(
            request.analysis_limitations
            + _legacy_analysis_limitations(
                request.options,
                design=request.design,
                dynamic=request.dynamic,
                L_greedy=request.params.get("L_greedy"),
                median_abs_corr=request.params.get("median_abs_corr"),
                cs_extension_corr=request.params.get("cs_extension_corr"),
            )
        )
    )
    if limitations:
        # Dynamic models can be attached after ordinary preparation. Never expose
        # their static snapshot to Gaussian-only objective/replay helpers.
        request = replace(request, prepared=None, signature=None, analysis_limitations=limitations)
    crossproduct = request.crossproduct
    crossresponse = request.crossresponse
    yty = request.yty
    n = request.n
    options = request.options
    XtX = request.XtX
    diagonal = request.diagonal
    scales = request.scales
    means = request.means
    y_mean_value = request.y_mean_value
    ids = request.ids
    metadata = request.metadata
    coverage = request.coverage
    min_abs_corr = request.min_abs_corr
    null_index = request.null_index
    n_purity = request.n_purity
    prior_tol = request.prior_tol
    collected = request.collected
    qc = request.qc
    params = request.params
    control = request.control
    resolution = request.resolution
    prepared = request.prepared
    signature = request.signature
    refine = request.refine
    _rss_matrix_scale = request._rss_matrix_scale
    if params.get("init_only", False):
        return SusieInputs(
            data={
                "XtX": XtX,
                "Xty": crossresponse,
                "yty": yty,
                "n": n,
                "X": None if request.design is None else request.design.X,
            },
            params=params.copy(),
            _request=request,
        )
    if request.deferred_model_init is not None:
        raise ValueError(
            "model_init must be a fitted SusieResult; prepared SusieInputs has no posterior state"
        )
    # refine=True on these models uses the refitting search below; a
    # RefinementConfig still requires the retained-analysis engine.
    _require_legacy_analysis(
        bool(params.get("retain_data")),
        refine if isinstance(refine, RefinementConfig) else False,
        limitations,
    )
    if params.get("L_greedy") is not None:
        return _fit_greedy(request)
    if request.dynamic is not None:
        from .rss_models import fit_dynamic

        native = fit_dynamic(request)
    elif request.design is None:
        native = _native.fit(crossproduct, crossresponse, yty, n, options)
    else:
        design = request.design
        arguments = (
            np.ascontiguousarray(design.means),
            np.ascontiguousarray(design.scales),
            crossresponse,
            yty,
            n,
            options,
        )
        from ._trend import TrendDesign

        if isinstance(design.X, TrendDesign):
            native = _native.fit_design_trend(design.X.n, design.X.order, *arguments)
        elif hasattr(design.X, "indptr"):
            native = _native.fit_design_csc(
                design.X.data,
                design.X.indices.astype(np.int64),
                design.X.indptr.astype(np.int64),
                design.X.shape[0],
                *arguments,
            )
        else:
            native = _native.fit_design_dense(design.X, *arguments)
    for message in native["warnings"]:
        _warn(message, collected)
    if not native["converged"]:
        _warn(
            nonconvergence_message(native),
            collected,
        )
    alpha, V = native["alpha"], native["V"]
    cs_audit = []
    sets = credible_sets(
        alpha,
        V,
        XtX,
        coverage=coverage,
        min_abs_corr=min_abs_corr,
        median_abs_corr=params.get("median_abs_corr"),
        cs_extension_corr=params.get("cs_extension_corr"),
        null_index=null_index,
        n_purity=n_purity,
        X=None if request.design is None else request.design.X,
        covariance_scales=None if request.design is not None else np.sqrt(diagonal),
        covariance_multiplier=_rss_matrix_scale,
        audit=cs_audit,
    )
    intercept = y_mean_value - float(np.sum(means * np.sum(alpha * native["mu"], axis=0) / scales))
    summary, trace = convergence_summary(native, options, control)
    summary["config_resolution"] = resolution.copy()
    summary["legacy_requested_config"] = control.options()
    summary["requested_convergence_method"] = request.requested_convergence_method
    summary["requested_config"] = {
        name: options[name]
        for name in ("criterion", "alpha_tol", "pip_tol", "min_iter", "cycle_lag")
    }
    summary["requested_criterion"] = options["criterion"]
    summary["alpha_tol"] = options["alpha_tol"]
    summary["pip_tol"] = options["pip_tol"]
    summary["retained_analysis_available"] = prepared is not None
    summary["retained_analysis_limitations"] = list(limitations)
    fit = SusieResult(
        variant_ids=ids,
        alpha=alpha,
        mu=native["mu"],
        mu2=native["mu2"],
        lbf_variable=native["lbf_variable"],
        lbf=native["lbf"],
        V=V,
        sigma2=native["sigma2"],
        pip=marginal_pip(
            alpha,
            V,
            null_index=null_index,
            prior_tol=prior_tol,
            slot_weights=native.get("slot_weights"),
        ),
        elbo=native["elbo"],
        niter=native["niter"],
        converged=native["converged"],
        sets=sets,
        null_index=null_index,
        params=params,
        backend_version=_native.backend_version(),
        warnings=collected,
        qc=qc,
        variant_metadata=metadata,
        X_column_scale_factors=scales,
        intercept=intercept,
        KL=native["KL"],
        XtXr=native["XtXr"],
        convergence=summary,
        trace=trace,
        cs_audit=cs_audit,
        data_signature=signature,
        _prepared=prepared,
        slot_weights=native.get("slot_weights"),
        lbf_grid=native.get("lbf_grid"),
        rv=native.get("rv"),
        marginal_loglik=native.get("marginal_loglik"),
        R_finite_diagnostics=native.get("R_finite_diagnostics", {}),
    )
    if fit.R_finite_diagnostics:
        if "R_bf_attenuation" in native:
            from ._rss_qc import summarize_attenuation

            summarize_attenuation(
                fit, native["R_bf_attenuation"], request.dynamic["control"].sensitivity_threshold
            )
        else:
            fit.R_finite_diagnostics["R_reliability_flag"] = bool(
                fit.R_finite_diagnostics.get("artifact_flag", False)
            )
        if fit.R_finite_diagnostics["R_reliability_flag"]:
            _warn(
                "Fine-mapping is sensitive to LD-reference uncertainty; "
                "inspect R_finite_diagnostics",
                collected,
            )
    if request.design is not None and request.design.fitted_response:
        fit.fitted = np.asarray(request.design.X @ fit.coef).ravel() + fit.intercept
    if params.get("verbose"):
        import logging

        logging.getLogger("prusie").info(
            "SuSiE: %d sweeps, converged=%s, %d credible sets",
            fit.niter,
            fit.converged,
            len(fit.sets["cs"]),
        )
    if refine is not False:
        if not isinstance(refine, (bool, RefinementConfig)):
            raise TypeError("refine must be False, True, or a RefinementConfig")
        if limitations:
            fit = _refine_by_refitting(request, fit)
        else:
            fit = refine_fit(fit, refine if isinstance(refine, RefinementConfig) else None)
    return fit


def _refine_by_refitting(request: _FitRequest, model: SusieResult) -> SusieResult:
    """Credible-set refinement by block coordinate ascent, as in susieR ``run_refine``.

    For each credible set, refit with its members' prior weights set to zero,
    then refit from that posterior with the original weights. The candidate
    with the largest ELBO is accepted if it improves the objective by more than
    ``tol``; the search stops when no candidate does or when the relative ELBO
    change falls below ``tol``.
    """
    tol = float(request.params["tol"])
    null_index = request.null_index
    null_weight = float(request.params["null_weight"])
    weights = np.asarray(request.options["prior_weights"], dtype=np.float64)
    base = weights if null_index is None else np.delete(weights, null_index) / (1 - null_weight)

    def full_weights(values: NDArray[np.float64]) -> NDArray[np.float64]:
        if null_index is not None:
            values = np.r_[values * (1 - null_weight), null_weight]
        return values / values.sum()

    def refit(prior_weights: NDArray[np.float64], init: SusieResult | None) -> SusieResult:
        options = {k: v for k, v in request.options.items() if not k.startswith("init_")}
        options.update(prior_weights=prior_weights, track_level=tracking_level(False))
        if init is not None:
            options.update(
                initial_options(
                    {"alpha": init.alpha, "mu": init.mu, "mu2": init.mu2},
                    ids=request.ids,
                    metadata=request.metadata,
                    scales=request.scales,
                    null_index=null_index,
                    signature=None,
                    L=options["l"],
                    weights=prior_weights,
                    estimate_prior_variance=options["estimate_prior_variance"],
                    estimate_residual_variance=options["estimate_residual_variance"],
                    prior_variance=options.get(
                        "component_prior_variance", options["prior_variance"]
                    ),
                )
            )
        params = {
            **request.params,
            "prior_weights": prior_weights.tolist(),
            "refine": False,
            "track_fit": False,
            "verbose": False,
            "s_init": None if init is None else "posterior_state",
        }
        return _fit(
            replace(
                request,
                options=options,
                params=params,
                refine=False,
                collected=[],
                qc=dict(request.qc),
            )
        )

    if not model.sets["cs"]:
        return model
    history: list[dict[str, Any]] = []
    total_fits, converged, stop_reason, accepted = 0, False, "max_iter", False
    original_weights = full_weights(base)
    for iteration in range(1, 101):
        candidates = []
        for members in model.sets["cs"]:
            excluded = base.copy()
            excluded[np.asarray(members, dtype=np.intp)] = 0.0
            if not np.any(excluded):
                break
            explored = refit(full_weights(excluded), None)
            candidates.append(refit(original_weights, explored))
            total_fits += 2
        current = float(model.elbo[-1])
        if not candidates:
            converged, stop_reason = True, "no_credible_sets"
            break
        objectives = [float(candidate.elbo[-1]) for candidate in candidates]
        best = int(np.argmax(objectives))
        history.append(dict(iteration=iteration, elbo=current, candidate_elbos=objectives))
        if objectives[best] - current <= tol:
            converged, stop_reason = True, "no_improvement"
            break
        change = objectives[best] - current
        model, accepted = candidates[best], True
        if change / max(1.0, abs(objectives[best])) < tol:
            converged, stop_reason = True, "relative_elbo_change"
            break
    if accepted:
        model.warnings[:0] = request.collected
    model.params["refine"] = True
    model.refinement = dict(
        status="converged" if converged else "not_converged",
        stop_reason=stop_reason,
        method="block_coordinate_ascent",
        history=history,
        total_fits=total_fits,
    )
    if not converged:
        _warn("Credible-set refinement did not converge in 100 iterations", model.warnings)
    return model


def susie_rss(
    z: ArrayLike | None = None,
    R: ArrayLike | None = None,
    n: float | None = None,
    *,
    X: Any = None,
    z_method: str = "wald",
    maf: ArrayLike | None = None,
    maf_thresh: float = 0.0,
    R_finite: bool | float | Sequence[float] | None = None,
    R_mismatch: str = "none",
    control: RSSControl | Mapping[str, Any] | None = None,
    bhat: ArrayLike | None = None,
    shat: ArrayLike | None = None,
    var_y: float | None = None,
    z_ld_weight: float = 0.0,
    estimate_residual_variance: bool = False,
    prior_variance: float = 50.0,
    check_prior: bool = True,
    L: int = 10,
    scaled_prior_variance: float | ArrayLike = 0.2,
    residual_variance: float | None = None,
    estimate_prior_variance: bool = True,
    estimate_prior_method: str = "optim",
    check_null_threshold: float = 0.0,
    prior_tol: float = 1e-9,
    r_tol: float = 1e-8,
    prior_weights: ArrayLike | None = None,
    null_weight: float = 0.0,
    standardize: bool = True,
    max_iter: int = 50,
    s_init: SusieResult | Mapping[str, Any] | None = None,
    model_init: SusieResult | Mapping[str, Any] | None = None,
    estimate_residual_method: str = "MoM",
    residual_variance_lowerbound: float = 0.0,
    residual_variance_upperbound: float | None = np.inf,
    prior_variance_grid: ArrayLike | None = None,
    mixture_weights: ArrayLike | None = None,
    convergence_method: str | None = None,
    alpha0: float | None = None,
    beta0: float | None = None,
    L_greedy: int | None = None,
    greedy_lbf_cutoff: float = 0.1,
    slot_prior: SlotPrior | Mapping[str, Any] | None = None,
    median_abs_corr: float | None = None,
    cs_extension_corr: float | None = None,
    unmappable_effects: str = "none",
    init_only: bool = False,
    coverage: float | None = 0.95,
    min_abs_corr: float | None = 0.5,
    tol: float | None = 1e-4,
    verbose: bool = False,
    track_fit: bool | str = False,
    check_input: bool = False,
    refine: bool | RefinementConfig = False,
    convergence: ConvergenceConfig | None = None,
    retain_data: bool = False,
    n_purity: int | str | None = None,
    variant_ids: Sequence[str] | None = None,
    variant_metadata: Mapping[str, ArrayLike] | None = None,
) -> SusieResult | SusieInputs:
    """Fit SuSiE-RSS from marginal association statistics and signed LD.

    Parameters
    ----------
    z : array_like, shape (p,), optional
        Marginal z-scores for p variants. Mutually exclusive with ``bhat`` and
        ``shat``.
    R : array_like, shape (p, p)
        Signed LD correlation matrix in the variant and effect-allele order of
        the summary statistics. Singular matrices are accepted; no ridge or
        eigenvalue adjustment is applied.
    n : float, optional
        Sample size of the association study; must exceed one. If omitted, the
        large-sample likelihood of susieR is used (XᵀX = R, Xᵀy = z, yᵀy = 1,
        internal n = 2) and a warning is issued.
    bhat, shat : array_like, optional
        Marginal effect estimates and positive standard errors, used instead
        of z (z = bhat/shat). ``shat`` may be a scalar; positive infinity
        produces a zero z-score for a finite effect estimate.
    var_y : float, optional
        Positive sample variance of the phenotype. With known ``n``, sets
        yᵀy = (n - 1) var_y; with ``bhat`` and ``shat`` it also reconstructs
        the predictor crossproducts for original-scale effects. Without
        ``n``, it is validated and recorded but does not affect the fit.
    z_ld_weight : float, default 0
        Legacy argument removed upstream in susieR 0.16.6; must be zero.
    estimate_residual_variance : bool, default False
        Estimate σ². Appropriate only for in-sample LD consistent with the
        summary statistics; requesting it issues a warning.
    prior_variance : float, default 50
        Initial prior variance V when ``n`` is omitted. Ignored otherwise.
    check_prior : bool, default True
        See :func:`susie_suff_stat`.
    max_iter : int, default 50
        Maximum number of IBSS sweeps.

    Other Parameters
    ----------------
    L, scaled_prior_variance, residual_variance, estimate_prior_variance
        As in :func:`susie_suff_stat`.
    estimate_prior_method, check_null_threshold, prior_tol, r_tol
        As in :func:`susie_suff_stat`.
    prior_weights, null_weight, standardize, s_init, coverage, min_abs_corr
        As in :func:`susie_suff_stat`.
    tol, verbose, track_fit, check_input, refine, convergence, retain_data
        As in :func:`susie_suff_stat`, including retained-analysis limitations.
        Ordinary static R-based RSS supports retained helpers in that scope.
        Operator X inputs and dynamic RSS do not; the X/bhat/shat/known-var_y
        route explicitly materializes its Gram and is the static exception.
    X : array_like or scipy.sparse matrix, optional
        Reference genotypes instead of R. Centering/scaling and products remain
        implicit; reference sample size may differ from n. Lists of LD or genotype
        panels fit an adaptively weighted LD mixture and require n.
    z_method : {'wald', 'score'}, default 'wald'
        Apply the finite-n Wald-to-score adjustment or use scores directly.
    R_finite : bool or float or sequence, optional
        Finite LD reference sample size; True uses reference genotype row counts.
    R_mismatch : {'none', 'eb', 'eb_mix', 'eb_ser_init', 'eb_force_init', 'eb_no_init'}
        LD mismatch estimator and initialization mode.
    control : RSSControl or mapping, optional
        Mismatch-estimation and input-quality controls from susie_rss_control.
    model_init, estimate_residual_method, prior_variance_grid, mixture_weights
        As in :func:`susie_suff_stat`.
    residual_variance_lowerbound, residual_variance_upperbound, alpha0, beta0
        As in :func:`susie_suff_stat`.
    L_greedy, greedy_lbf_cutoff, slot_prior, convergence_method, init_only
        As in :func:`susie_suff_stat`.
    median_abs_corr, cs_extension_corr, maf, maf_thresh, unmappable_effects
        As in :func:`susie_suff_stat`.
    n_purity, variant_ids, variant_metadata
        As in :func:`susie_suff_stat`. All apply to the working sufficient
        statistics described in Notes.

    Returns
    -------
    SusieResult or SusieInputs
        Posterior fit, or prepared inputs when init_only=True. ``params`` also
        records the input route
        (``input_kind``), the supplied n and var_y, and the effect scale
        (``effect_scale``: 'standardized', 'original' or 'noncentrality').

    Raises
    ------
    ValueError
        Ambiguous or incomplete input, an invalid correlation matrix,
        invalid summary values, invalid parameter ranges, or a nonzero legacy
        ``z_ld_weight`` absent from susieR 0.16.6.

    Warns
    -----
    RuntimeWarning
        Missing n, residual-variance estimation with RSS input, and the
        fitting warnings of :func:`susie_suff_stat`.

    Notes
    -----
    For known n, each z-score is adjusted for the residual variance used in
    its Wald statistic, z̃_j = z_j ((n - 1)/(z_j² + n - 2))^½, and the
    working statistics are XᵀX = (n - 1)R, Xᵀy = (n - 1)^½ z̃ and
    yᵀy = n - 1, which places effects on a standardized scale. With z input
    and var_y, yᵀy = (n - 1) var_y while XᵀX and Xᵀy are unchanged.
    Original-scale coefficients therefore require ``bhat``, ``shat``, ``n``
    and ``var_y``.
    When n is omitted, a supplied var_y is validated and recorded but does
    not enter the working statistics, and effects are on the noncentrality
    scale. For ordinary z/R input, symmetry is checked exactly on the working
    XtX after scaling by n − 1 (or on R if n is omitted). Inequality triggers
    a warning and internal averaging of XtX with its transpose, as in susieR's
    dense numeric path. The bhat/shat/var_y route always averages its reconstructed
    XtX. Nonfinite inputs and averaging overflow are rejected. Input arrays
    remain unchanged. If R is an external reference panel, its agreement with the study
    LD is a modeling assumption that this function does not test.

    References
    ----------
    Zou Y, Carbonetto P, Wang G, Stephens M (2022). Fine-mapping from summary
    data with the "Sum of Single Effects" model. PLoS Genet 18:e1010299.
    Reference implementation: susieR 0.16.6, R/susie_constructors.R.

    Examples
    --------
    >>> fit = susie_rss([4., 0.], [[1., 0.], [0., 1.]], 100, L=1)
    >>> fit.pip.shape
    (2,)
    """
    requested_convergence_method = convergence_method
    if (R is None) == (X is None):
        raise ValueError("Supply exactly one of R or X")
    if z_method not in ("wald", "score"):
        raise ValueError("z_method must be 'wald' or 'score'")
    if R_mismatch not in ("none", "eb", "eb_mix", "eb_ser_init", "eb_force_init", "eb_no_init"):
        raise ValueError("Unknown R_mismatch option")
    rss_control = (
        susie_rss_control()
        if control is None
        else (control if isinstance(control, RSSControl) else susie_rss_control(**dict(control)))
    )
    if control is not None:
        check_input, check_prior, r_tol = (
            rss_control.check_input,
            rss_control.check_prior,
            rss_control.r_tol,
        )
    panels = R if R is not None else X
    multipanel = isinstance(panels, (list, tuple)) and len(panels) > 0 and np.ndim(panels[0]) == 2
    if multipanel:
        from .rss_models import prepare_multipanel

        # Panel selection fits candidate models before the final shared fitter.
        if not init_only:
            _require_legacy_analysis(retain_data, refine, ("multi-panel RSS",))
        return prepare_multipanel(dict(locals()))
    design_data = None
    if R_finite is True:
        if X is None:
            raise ValueError("R_finite=True requires X; supply a numeric panel size with R")
        R_finite = float(X.shape[0]) if hasattr(X, "shape") else float(len(X))
    if R_finite is not None and R_finite is not False:
        R_finite = _scalar(R_finite, "R_finite", strict=True)
    else:
        R_finite = None
    if z_ld_weight != 0:
        raise ValueError("z_ld_weight is absent from susieR 0.16.6; only legacy zero is accepted")
    if z is not None and (bhat is not None or shat is not None):
        raise ValueError("Provide either z or bhat/shat, not both")
    has_effects = z is None
    if has_effects:
        if bhat is None or shat is None:
            raise ValueError("Provide either z or both bhat and shat")
        beta = _vector(bhat, "bhat")
        se = (
            np.full(beta.size, shat, dtype=float)
            if np.ndim(shat) == 0
            else np.asarray(shat, dtype=np.float64)
        )
        if se.shape != beta.shape:
            raise ValueError(f"shat must have length {beta.size}")
        if np.any(np.isnan(se)) or np.any(se <= 0):
            raise ValueError("shat must be strictly positive without missing values")
        z = beta / se
    z = _summary_vector(z, "z")
    if estimate_residual_method == "NIG" and n is None:
        raise ValueError("NIG RSS fitting requires n")
    if X is not None:
        from .individual import _column_stats, _DesignData, _validated_design

        raw_rows = X.shape[0] if hasattr(X, "shape") else len(X)
        raw, _ = _validated_design(X, np.zeros(raw_rows), na_rm=False)
        if raw.shape[1] != len(z):
            raise ValueError("X must have one column per z-score")
        raw_means, norm2 = _column_stats(raw)
        target_norm = 1.0 if n is None else _scalar(n, "n", lower=1, strict=True) - 1
        factor = np.sqrt(np.divide(target_norm, norm2, out=np.zeros_like(norm2), where=norm2 > 0))
        if hasattr(raw, "multiply"):
            normalized = raw.multiply(factor).tocsc()
        else:
            normalized = np.asfortranarray(raw * factor)
        normalized_means = raw_means * factor
        means = normalized_means
        design_data = _DesignData(
            normalized,
            means,
            np.ones(len(z)),
            np.where(norm2 > 0, target_norm, 0),
            1.0 if var_y is None else float(var_y),
            False,
        )
        if has_effects and var_y is not None:
            from .posterior import _design_correlation

            R = _design_correlation(raw, np.arange(len(z)), np.arange(len(z)))
            design_data = None
    prevalidated_scale = None
    if n is None:
        prevalidated_scale = 1.0
    elif not (has_effects and var_y is not None):
        n = _scalar(n, "n", lower=1.0, strict=True)
        prevalidated_scale = n - 1
    extra_warnings = []
    if design_data is None:
        R, symmetric = _validated_matrix(R, "R", len(z), rss_scale=prevalidated_scale)
    else:
        R, symmetric = None, True
    rss_matrix_scale = 1.0
    if var_y is not None:
        var_y = _scalar(var_y, "var_y", strict=True)
    if n is None:
        _warn(
            ("n is missing: using the upstream large-sample noncentrality-parameter likelihood"),
            extra_warnings,
        )
        XtX, Xty, yty, effective_n = R, z, 1.0, 2.0
        effective_scaled_prior, effective_standardize = prior_variance, False
        scale_kind = "noncentrality"
    else:
        n = _scalar(n, "n", lower=1.0, strict=True)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            denominator = z * z + n - 2
            adj = (n - 1) / denominator if z_method == "wald" else np.ones_like(z)
            adjusted_z = np.sqrt(adj) * z
        if has_effects and var_y is not None:
            diagonal = var_y * adj / (se * se)
            sd = np.sqrt(diagonal)
            XtX = (R * sd[:, None]).T * sd[:, None]
            XtX = (XtX + XtX.T) / 2
            Xty = adjusted_z * np.sqrt(adj) * var_y / se
            effective_var_y, scale_kind = var_y, "original"
        else:
            XtX, Xty = R, np.sqrt(n - 1) * adjusted_z
            rss_matrix_scale = n - 1 if design_data is None else 1.0
            effective_var_y, scale_kind = (
                (1.0 if var_y is None else var_y),
                "standardized",
            )
        yty, effective_n = (n - 1) * effective_var_y, n
        effective_scaled_prior, effective_standardize = (
            scaled_prior_variance,
            standardize,
        )
    if prevalidated_scale is not None and not symmetric:
        XtX = _symmetrize_matrix(XtX, "XtX", rss_matrix_scale, extra_warnings)
        rss_matrix_scale = 1.0
    if estimate_residual_variance:
        _warn(
            (
                "Estimating residual variance requires in-sample LD consistent "
                "with the summary statistics"
            ),
            extra_warnings,
        )
    dynamic = None
    if R_finite is not None or R_mismatch != "none":
        dynamic = dict(
            kind="uncertain",
            reference_n=np.inf if R_finite is None else R_finite,
            mismatch=R_mismatch,
            control=rss_control,
            z=adjusted_z.copy(),
        )
        if convergence is None:
            convergence_method = "pip"
    if dynamic is not None:
        selected_z = adjusted_z if maf is None else adjusted_z[np.asarray(maf) > maf_thresh]
        dynamic["z"] = np.r_[selected_z, 0.0] if null_weight > 0 else selected_z.copy()
    request = _prepare_suff_stat(
        XtX,
        Xty,
        yty,
        effective_n,
        L=L,
        scaled_prior_variance=effective_scaled_prior,
        residual_variance=residual_variance,
        estimate_residual_variance=estimate_residual_variance,
        estimate_prior_variance=estimate_prior_variance,
        estimate_prior_method=estimate_prior_method,
        check_null_threshold=check_null_threshold,
        prior_tol=prior_tol,
        r_tol=r_tol,
        prior_weights=prior_weights,
        null_weight=null_weight,
        standardize=effective_standardize,
        max_iter=max_iter,
        s_init=s_init,
        coverage=coverage,
        min_abs_corr=min_abs_corr,
        tol=tol,
        verbose=verbose,
        track_fit=track_fit,
        check_input=check_input,
        refine=refine,
        convergence=convergence,
        retain_data=retain_data,
        check_prior=check_prior,
        n_purity=n_purity,
        variant_ids=variant_ids,
        variant_metadata=variant_metadata,
        model_init=model_init,
        estimate_residual_method=estimate_residual_method,
        residual_variance_lowerbound=residual_variance_lowerbound,
        residual_variance_upperbound=residual_variance_upperbound,
        prior_variance_grid=prior_variance_grid,
        mixture_weights=mixture_weights,
        convergence_method=convergence_method,
        alpha0=alpha0,
        beta0=beta0,
        L_greedy=L_greedy,
        greedy_lbf_cutoff=greedy_lbf_cutoff,
        slot_prior=slot_prior,
        median_abs_corr=median_abs_corr,
        cs_extension_corr=cs_extension_corr,
        unmappable_effects=unmappable_effects,
        init_only=init_only,
        _rss_matrix_scale=rss_matrix_scale,
        _rss_matrix_validated=prevalidated_scale is not None,
        _design=design_data,
        _dynamic=dynamic,
        _summary_missing_policy="reject" if has_effects else "zero",
        maf=maf,
        maf_thresh=maf_thresh,
    )
    request = replace(request, requested_convergence_method=requested_convergence_method)
    fit = _fit(request)
    if init_only:
        return fit
    fit.params.update(
        interface="susie_rss",
        input_n=n,
        input_kind="bhat_shat" if has_effects else "z",
        input_var_y=var_y,
        effect_scale=scale_kind,
        prior_variance=prior_variance,
        z_ld_weight=0.0,
        z_method=z_method,
        R_finite=R_finite,
        R_mismatch=R_mismatch,
        control=dict(rss_control),
    )
    fit.warnings[:0] = extra_warnings
    return fit


def _model_init_state(model: SusieResult | Mapping[str, Any]) -> dict[str, Any]:
    """Extract initialization fields, clearing CS metadata as susieR does."""
    if isinstance(model, Mapping):
        return {**model, "sets": None}
    return dict(
        alpha=model.alpha,
        mu=model.mu,
        mu2=model.mu2,
        V=model.V,
        sigma2=model.sigma2,
        variant_ids=model.variant_ids,
        variant_metadata=model.variant_metadata,
        scales=model.X_column_scale_factors,
        null_index=model.null_index,
        data_signature=model.data_signature,
        sets=None,
    )


def _fit_greedy(request: _FitRequest) -> SusieResult:
    """Grow L in fixed increments, matching susieR's workhorse outer loop."""
    maximum, step = request.options["l"], request.params["L_greedy"]
    current, previous, history = min(step, maximum), None, []
    while True:
        options = {
            key: value for key, value in request.options.items() if not key.startswith("init_")
        }
        options["l"] = current
        if "component_prior_variance" in options:
            options["component_prior_variance"] = options["component_prior_variance"][:current]
        if previous is not None:
            options.update(
                initial_options(
                    _model_init_state(previous),
                    ids=request.ids,
                    metadata=request.metadata,
                    scales=request.scales,
                    null_index=request.null_index,
                    signature=request.signature,
                    L=current,
                    weights=options["prior_weights"],
                    estimate_prior_variance=options["estimate_prior_variance"],
                    estimate_residual_variance=options["estimate_residual_variance"],
                    prior_variance=options.get(
                        "component_prior_variance", options["prior_variance"]
                    ),
                )
            )
        else:
            for key, value in request.options.items():
                if key.startswith("init_"):
                    options[key] = value
        params = {**request.params, "L_greedy": None, "effective_L": current}
        previous = _fit(replace(request, options=options, params=params))
        minimum = float(np.nanmin(previous.lbf))
        action = (
            "L_max reached"
            if current >= maximum
            else "saturated"
            if minimum < request.params["greedy_lbf_cutoff"]
            else "grow"
        )
        history.append(dict(L=current, min_lbf=minimum, action=action))
        if action != "grow":
            previous.params.update(L_greedy=step, greedy_history=history)
            return previous
        current = min(current + step, maximum)


susie_ss = susie_suff_stat
