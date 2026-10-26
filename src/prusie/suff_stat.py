"""SuSiE entry points for sufficient statistics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ._request import (
    _prepare_request,
)
from .control import SlotPrior
from .diagnostics import (
    ConvergenceConfig,
)
from .refinement import RefinementConfig
from .result import SusieResult
from .workhorse import (
    SusieInputs,
    _fit,
)


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


susie_ss = susie_suff_stat
