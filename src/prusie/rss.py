"""SuSiE-RSS input conversion and multi-panel fitting orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from . import _native
from ._array_layout import native_array
from ._execution import close_executor, create_executor
from ._overwrite import _overwrite_prepare, _overwrite_recheck
from ._request import (
    _prepare_request,
    _require_legacy_analysis,
)
from ._validation import (
    _scalar,
    _summary_vector,
    _symmetrize_matrix,
    _thread_count,
    _validated_matrix,
    _vector,
    _warn,
)
from .control import RSSControl, SlotPrior, susie_rss_control
from .diagnostics import (
    ConvergenceConfig,
)
from .individual import _column_stats, _DesignData, _validated_design
from .posterior import _design_correlation
from .refinement import RefinementConfig
from .result import SusieResult
from .workhorse import (
    SusieInputs,
    _fit,
)

_FUSED_RSS_DTYPE = np.dtype(np.float64)


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
    n_threads: int = 1,
    overwrite_ld: bool = False,
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
    overwrite_ld : bool, default False
        Allow R and its external aliases to be consumed as workspace. Supported
        only for plain writable native float64 C-order NumPy-owned R with
        ordinary z/R/n inputs and no retained/advanced model options. A finite
        scaled-asymmetry witness enables ordered in-place symmetrization into
        XtX; exact-symmetric and no-witness inputs stay unchanged. Unsupported
        storage, options, custom input-conversion hooks and overlaps with other
        arguments raise before writes. Once writes begin, later errors or warning-as-error may
        leave R changed. Reload the original LD before another call, including
        after an aborted call. The default never overwrites caller arrays.
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
    n_threads : int, default 1
        Per-fit native thread limit from 1 to 32, as in :func:`susie_suff_stat`.
        Eligible C-order float64 R inputs also use a private executor for exact
        validation and owned symmetry repair when p ≥ 1024. Solver and output
        options do not control eligibility for this matrix work. Preparation
        workers join before later Python transformations or warning dispatch;
        fitting then uses its own per-call executor. Default one and overwrite
        routes retain their ordinary preparation.

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
    a warning and internal averaging of XtX with its transpose, matching the
    dense numeric path. The bhat/shat/var_y route always averages its reconstructed
    XtX. Nonfinite inputs and averaging overflow are rejected. Input arrays
    remain unchanged with ``overwrite_ld=False``. If R is an external reference panel,
    its agreement with the study
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
    if type(overwrite_ld) not in (bool, np.bool_):
        raise ValueError("overwrite_ld must be bool or numpy.bool_")
    overwrite_ld = bool(overwrite_ld)
    # Snapshot public arguments only for the opt-in destructive path, before coercion.
    overwrite_state = _overwrite_prepare(locals()) if overwrite_ld else None
    n_threads = _thread_count(n_threads)
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
    uncertain_rss = (R_finite is not None and R_finite is not False) or R_mismatch != "none"
    if (multipanel or uncertain_rss) and isinstance(convergence, ConvergenceConfig):
        # The Python sweep loop of these models has susieR's ELBO and PIP rules only.
        requested = convergence.options()
        if (
            requested["min_iter"] != 2
            or requested["cycle_lag"] != 5
            or (uncertain_rss and requested["criterion"] != "elbo")
        ):
            raise ValueError(
                "R_finite, R_mismatch and multi-panel RSS fits support only the default "
                "min_iter=2 and cycle_lag=5, and R_finite or R_mismatch fits accept only "
                "criterion='elbo' from ConvergenceConfig; omit convergence to use "
                "susieR's PIP rule"
            )
    if multipanel:
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
    extra_warnings = []
    z = _summary_vector(z, "z", extra_warnings)
    if estimate_residual_method == "NIG" and n is None:
        raise ValueError("NIG RSS fitting requires n")
    if X is not None:
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
            R = _design_correlation(raw, np.arange(len(z)), np.arange(len(z)))
            design_data = None
    prevalidated_scale = None
    if n is None:
        prevalidated_scale = 1.0
    elif not (has_effects and var_y is not None):
        n = _scalar(n, "n", lower=1.0, strict=True)
        prevalidated_scale = n - 1
    repaired_rss = None
    repair_error = None
    # Native preparation returns warning predicates and deferred repair errors
    # as data. Its workers are joined before any later Python transformation or
    # warning dispatch, independently of tracing, display or NumPy error hooks.
    # Callers must not mutate inputs during the call.
    fused_rss = (
        n is not None
        and var_y is None
        and type(R) is np.ndarray
        and R.dtype is _FUSED_RSS_DTYPE
        and R.flags.c_contiguous
        and R.flags.aligned
        and type(z_method) is str
    )
    _executor = None
    try:
        if design_data is None:
            if overwrite_ld and (prevalidated_scale is None or not fused_rss):
                raise ValueError("overwrite_ld requires the ordinary contiguous z/R/n route")
            if prevalidated_scale is None or not fused_rss:
                R, symmetric = _validated_matrix(R, "R", len(z), rss_scale=prevalidated_scale)
            else:
                R = np.asarray(R, dtype=np.float64)
                if R.shape != (len(z), len(z)):
                    raise ValueError(f"R must have shape ({len(z)}, {len(z)})")
                R = native_array(R)
                if overwrite_ld:
                    _overwrite_recheck(overwrite_state)
                    symmetric, repaired_rss, repair_error, overwrite_action = (
                        _native.validate_and_overwrite_rss(R, prevalidated_scale, proof_threads=1)
                    )
                else:
                    # Eligibility uses normalized preparation inputs only; solver
                    # and output options cannot disable independent matrix work.
                    if n_threads > 1 and len(z) >= 1024:
                        _executor = create_executor(R, n_threads)
                    if _executor is None:
                        symmetric, repaired_rss, repair_error = _native.validate_and_repair_rss(
                            R, prevalidated_scale, proof_threads=1
                        )
                    else:
                        try:
                            symmetric, repaired_rss, repair_error = (
                                _native._validate_and_repair_rss_with_executor(
                                    R, prevalidated_scale, _executor
                                )
                            )
                        except BaseException:
                            # A helper failure takes precedence over cleanup.
                            close_executor(_executor, suppress_errors=True)
                            _executor = None
                            raise
                        else:
                            # Later NumPy operations may warn or invoke a caller
                            # error callback. Join before that work, and propagate
                            # close failures even inside a caller's except block.
                            try:
                                close_executor(_executor)
                            finally:
                                _executor = None
                R = np.ascontiguousarray(R)
        else:
            R, symmetric = None, True
        rss_matrix_scale = 1.0
        if var_y is not None:
            var_y = _scalar(var_y, "var_y", strict=True)
        if n is None:
            _warn(
                "n is missing: using the upstream large-sample "
                # Keep the warning text unchanged without joining this into a long line.
                "noncentrality-parameter likelihood",
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
            if fused_rss:
                # Keep the original helper stack depth for module-based warning
                # filters and the original warning-before-repair-error priority.
                XtX = _symmetrize_matrix(
                    XtX,
                    "XtX",
                    rss_matrix_scale,
                    extra_warnings,
                    _prepared=repaired_rss,
                    _repair_error=repair_error,
                )
            else:
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
        request = _prepare_request(
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
            n_threads=n_threads,
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
        request = replace(
            request,
            requested_convergence_method=requested_convergence_method,
            params={**request.params, "overwrite_ld": overwrite_ld},
        )
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
        if overwrite_ld:
            fit.qc["ld_overwrite"] = dict(
                action=overwrite_action,
                original_rss_scale=prevalidated_scale,
            )
        fit.warnings[:0] = extra_warnings
        return fit

    finally:
        if _executor is not None:
            close_executor(_executor)


def prepare_multipanel(arguments: dict[str, Any]) -> Any:
    """Construct panel sub-fits and the outcome-independent initial mixture."""
    import inspect

    panels = arguments["panels"]
    if arguments["n"] is None:
        raise ValueError("n is required for multi-panel RSS fitting")
    z = np.asarray(arguments["z"], dtype=float)
    if z.ndim != 1:
        raise ValueError("Multi-panel RSS requires a single shared z-score vector")
    collected: list[str] = []
    z = _summary_vector(z, "z", collected)
    accepted = set(inspect.signature(susie_rss).parameters) - {"R", "X", "R_finite", "init_only"}
    options = {key: value for key, value in arguments.items() if key in accepted}
    options["z"] = z
    sizes = arguments["R_finite"]
    if sizes is None or sizes is False:
        panel_sizes = None
    elif sizes is True:
        if arguments["X"] is None:
            raise ValueError("R_finite=True requires X panels")
        panel_sizes = np.array([np.shape(panel)[0] for panel in panels], dtype=float)
    else:
        panel_sizes = np.asarray(sizes, dtype=float)
        if panel_sizes.ndim == 0:
            panel_sizes = np.full(len(panels), float(panel_sizes))
        if (
            panel_sizes.shape != (len(panels),)
            or np.any(panel_sizes <= 0)
            or not np.isfinite(panel_sizes).all()
        ):
            raise ValueError("R_finite requires one positive size per panel")
    fits, correlations = [], []
    for k, panel in enumerate(panels):
        if arguments["R"] is not None:
            matrix = np.asarray(panel, dtype=float)
            denominator = np.sqrt(np.maximum(np.diag(matrix), 0))
            inverse = np.divide(
                1, denominator, out=np.zeros_like(denominator), where=denominator > 0
            )
            matrix = matrix * inverse[:, None] * inverse[None, :]
        else:
            raw, _ = _validated_design(panel, np.zeros(np.shape(panel)[0]), na_rm=False)
            matrix = _design_correlation(raw, np.arange(raw.shape[1]), np.arange(raw.shape[1]))
        correlations.append(matrix)
        fits.append(
            susie_rss(R=matrix, R_finite=None if panel_sizes is None else panel_sizes[k], **options)
        )
    scores = [float(fit.elbo[-1]) for fit in fits]
    selected = int(np.argmax(scores))
    options["convergence_method"] = "pip"
    inputs = susie_rss(R=correlations[selected], R_finite=None, init_only=True, **options)
    selected_variables = np.arange(len(z))
    if options.get("maf") is not None:
        selected_variables = np.flatnonzero(np.asarray(options["maf"]) > options["maf_thresh"])
    correlations = [
        matrix[np.ix_(selected_variables, selected_variables)] for matrix in correlations
    ]
    if inputs._request.null_index is not None:
        correlations = [np.pad(matrix, ((0, 1), (0, 1))) for matrix in correlations]
    omega = np.zeros(len(panels))
    omega[selected] = 1
    adjusted = z[selected_variables]
    if arguments["z_method"] == "wald":
        adjusted = adjusted * np.sqrt((arguments["n"] - 1) / (adjusted**2 + arguments["n"] - 2))
    if inputs._request.null_index is not None:
        adjusted = np.r_[adjusted, 0.0]
    dynamic = dict(
        kind="multipanel",
        panels=correlations,
        omega=omega,
        panel_sizes=panel_sizes,
        reference_n=np.inf if panel_sizes is None else panel_sizes[selected],
        mismatch=arguments["R_mismatch"],
        control=arguments["rss_control"],
        z=adjusted,
    )
    request = replace(
        inputs._request,
        dynamic=dynamic,
        prepared=None,
        signature=None,
        requested_convergence_method=arguments["convergence_method"],
        analysis_limitations=(*inputs._request.analysis_limitations, "multi-panel RSS"),
        params={**inputs._request.params, "init_only": arguments["init_only"]},
    )
    if arguments["init_only"]:
        return replace(inputs, _request=request)
    result = _fit(request)
    result.params.update(
        single_panel_elbo=scores, initial_panel=selected, omega=result.convergence.get("omega")
    )
    result.warnings[:0] = collected
    return result
