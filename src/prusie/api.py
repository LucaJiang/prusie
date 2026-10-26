"""Public SuSiE fitting functions for summary and sufficient statistics.

``susie_rss`` converts marginal association statistics and signed LD into
working sufficient statistics; ``susie_suff_stat`` validates them, runs the
native IBSS solver and assembles a :class:`~prusie.result.SusieResult`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from . import _native
from ._array_layout import native_array
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
from .refinement import RefinementConfig
from .result import SusieResult
from .workhorse import SusieInputs, _fit

_FUSED_RSS_DTYPE = np.dtype(np.float64)


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
    n_threads : int, default 1
        Per-fit native thread limit from 1 to 32. Dense Gram products can use
        fixed output partitions; component updates remain sequential. Small
        kernels and validation remain serial. Process-wide thread settings
        are unchanged; pass this option instead of PRUSIE_NUM_THREADS.
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
    return _fit(_prepare_request(**locals()))


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
    extra_warnings = []
    z = _summary_vector(z, "z", extra_warnings)
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
                        from ._execution import create_executor

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
                        finally:
                            # Later NumPy operations may warn or invoke a caller
                            # error callback. Do not keep preparation workers alive
                            # across any of that Python work, even when no package
                            # warning has been generated.
                            from ._execution import close_executor

                            close_executor(_executor)
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
            from ._execution import close_executor

            close_executor(_executor)


susie_ss = susie_suff_stat
