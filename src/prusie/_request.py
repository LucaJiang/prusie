"""Construct validated fitting requests for the shared SuSiE workhorse."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _native
from ._types import FitParameters, InputChecks, RetainedInputs
from ._validation import (
    _integer,
    _metadata,
    _scalar,
    _summary_vector,
    _symmetrize_matrix,
    _thread_count,
    _validated_matrix,
    _vector,
    _warn,
)
from .control import SlotPrior
from .diagnostics import (
    ConvergenceConfig,
    resolve_convergence,
    tracking_level,
)

if TYPE_CHECKING:
    from .workhorse import SusieInputs
from .refinement import RefinementConfig, initial_options, make_prepared
from .result import SusieResult


def _replay_limitations(
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


def _require_replayable_analysis(
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
    qc: InputChecks
    params: FitParameters
    control: Any
    resolution: dict[str, Any]
    prepared: RetainedInputs | None
    signature: str | None
    refine: bool | RefinementConfig
    _rss_matrix_scale: float
    design: Any = None
    dynamic: Any = None
    deferred_model_init: SusieInputs | None = None
    analysis_limitations: tuple[str, ...] = ()
    requested_convergence_method: str | None = None


def _prepare_request(
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
    n_threads: int = 1,
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
    # Model-state conversion belongs to the workhorse, which imports this constructor.
    from .workhorse import _model_init_state

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
    n_threads = _thread_count(n_threads)
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
    if not isinstance(refine, (bool, RefinementConfig)):
        raise TypeError("refine must be False, True, or a RefinementConfig")
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
    qc: InputChecks = {
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
        n_threads=n_threads,
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
    if convergence is None and control.options() == ConvergenceConfig().options():
        # susieR checks a sweep whose ELBO is NaN with its 'pip' rule.
        options["nan_elbo_pip"] = True
    if slot_prior is not None or (estimate_residual_method == "NIG" and effective_L > 1):
        options.update(criterion="pip", alpha_tol=tol, pip_tol=tol)
    # Only an owned snapshot can supply a stable identity for later reuse.
    # Ordinary fits do not retain caller buffers or scan an unused hash.
    signature = None
    prepared = None
    analysis_limitations = _replay_limitations(
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
    params: FitParameters = dict(
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
        n_threads=n_threads,
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
