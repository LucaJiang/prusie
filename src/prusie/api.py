"""Public SuSiE fitting functions for summary and sufficient statistics.

``susie_rss`` converts marginal association statistics and signed LD into
working sufficient statistics; ``susie_suff_stat`` validates them, runs the
native IBSS solver and assembles a :class:`~prusie.result.SusieResult`.
"""

from __future__ import annotations

import math
import warnings as warning_module
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _native
from ._array_layout import native_array
from .diagnostics import (
    convergence_summary,
    nonconvergence_message,
    resolve_convergence,
    tracking_level,
)
from .posterior import credible_sets, marginal_pip
from .refinement import RefinementConfig, data_signature, initial_options, make_prepared, refine_fit
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


def _matrix(
    value: ArrayLike,
    name: str,
    size: int,
    *,
    correlation=False,
    scale=1.0,
    rss_scale=None,
) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size})")
    array = native_array(array)
    # Finite, symmetry and diagonal checks run natively without p × p temporaries.
    if rss_scale is None:
        _native.validate_matrix(array, name, correlation, scale)
    else:
        _native.validate_rss_matrix(array, rss_scale)
    # native_array has already ensured alignment and element-compatible strides;
    # any copy made here is C-ordered, matching the public input orientation.
    return np.ascontiguousarray(array)


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
        ids = np.asarray(
            [str(i) for i in range(m)] if variant_ids is None else variant_ids, dtype=str
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


def susie_suff_stat(
    XtX: ArrayLike,
    Xty: ArrayLike,
    yty: float,
    n: float,
    *,
    X_colmeans: ArrayLike | None = None,
    y_mean: float | None = None,
    L: int = 10,
    scaled_prior_variance=0.2,
    residual_variance=None,
    estimate_residual_variance=True,
    estimate_prior_variance=True,
    estimate_prior_method="optim",
    check_null_threshold=0.0,
    prior_tol=1e-9,
    r_tol=1e-8,
    prior_weights=None,
    null_weight=0.0,
    standardize=True,
    max_iter=100,
    s_init=None,
    coverage=0.95,
    min_abs_corr=0.5,
    tol=1e-4,
    verbose=False,
    track_fit=False,
    check_input=False,
    refine=False,
    convergence=None,
    retain_data=False,
    check_prior=False,
    n_purity: int | None = None,
    variant_ids: Sequence[str] | None = None,
    variant_metadata: Mapping[str, ArrayLike] | None = None,
    maf=None,
    maf_thresh=0.0,
) -> SusieResult:
    return _susie_suff_stat(
        XtX,
        Xty,
        yty,
        n,
        X_colmeans=X_colmeans,
        y_mean=y_mean,
        L=L,
        scaled_prior_variance=scaled_prior_variance,
        residual_variance=residual_variance,
        estimate_residual_variance=estimate_residual_variance,
        estimate_prior_variance=estimate_prior_variance,
        estimate_prior_method=estimate_prior_method,
        check_null_threshold=check_null_threshold,
        prior_tol=prior_tol,
        r_tol=r_tol,
        prior_weights=prior_weights,
        null_weight=null_weight,
        standardize=standardize,
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
        maf=maf,
        maf_thresh=maf_thresh,
    )


def _susie_suff_stat(
    XtX: ArrayLike,
    Xty: ArrayLike,
    yty: float,
    n: float,
    *,
    X_colmeans: ArrayLike | None = None,
    y_mean: float | None = None,
    L: int = 10,
    scaled_prior_variance=0.2,
    residual_variance=None,
    estimate_residual_variance=True,
    estimate_prior_variance=True,
    estimate_prior_method="optim",
    check_null_threshold=0.0,
    prior_tol=1e-9,
    r_tol=1e-8,
    prior_weights=None,
    null_weight=0.0,
    standardize=True,
    max_iter=100,
    s_init=None,
    coverage=0.95,
    min_abs_corr=0.5,
    tol=1e-4,
    verbose=False,
    track_fit=False,
    check_input=False,
    refine=False,
    convergence=None,
    retain_data=False,
    check_prior=False,
    n_purity: int | None = None,
    variant_ids: Sequence[str] | None = None,
    variant_metadata: Mapping[str, ArrayLike] | None = None,
    maf=None,
    maf_thresh=0.0,
    _rss_matrix_scale=1.0,
    _rss_matrix_validated=False,
) -> SusieResult:
    """Fit SuSiE to centered sufficient statistics.

    Parameters
    ----------
    XtX : array_like, shape (p, p)
        Centered predictor crossproduct XᵀX for p variants. Must be finite and
        symmetric; converted to float64.
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
    scaled_prior_variance : float, default 0.2
        Initial prior effect variance V (σ₀²) divided by yᵀy/(n - 1). Must not
        exceed one when ``standardize=True``.
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
        Initial posterior state for the same data. Variant IDs, allele
        metadata, working scales, null column and data signature must agree.
        A mapping needs ``alpha``, ``mu``, ``mu2``, ``variant_ids``,
        ``scales``, ``null_index`` and ``data_signature``; ``V`` and
        ``sigma2`` are optional. The first min(L, L_init) components are
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
        Must be False.
    track_fit : bool or {'compact', 'dense'}, default False
        Record per-sweep convergence metrics. True or 'compact' stores exact
        distances and a thresholded alpha display; 'dense' also stores full
        posterior arrays and is intended for small problems.
    check_input : bool, default False
        Eigendecompose XtX, raise if its smallest eigenvalue is below -r_tol,
        and warn if Xty lies outside the span of the eigenvectors whose
        eigenvalues exceed machine epsilon.
    refine : bool or RefinementConfig, default False
        After a converged fit, run the credible-set exclusion search of
        :func:`refine_fit`; True uses the default configuration.
    convergence : ConvergenceConfig, optional
        Stopping rule (ELBO, alpha/PIP or joint). Defaults to the ELBO rule,
        or to the controls recorded in ``s_init`` when it is a fit.
    retain_data : bool, default False
        Retain read-only copies of the prepared inputs, as required by
        :func:`continue_fit`, :func:`refine_fit` and
        :func:`evaluate_objective`. Implied by ``track_fit``, ``s_init`` and
        ``refine``.
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
    maf, maf_thresh : optional
        Minor-allele-frequency filtering is not implemented; values other than
        None and 0 raise NotImplementedError.

    Returns
    -------
    SusieResult
        Variational posterior (alpha, mu, mu2), Bayes factors, prior and
        residual variances, ELBO trace, PIPs, credible sets and run metadata.

    Raises
    ------
    ValueError
        Invalid shapes, nonfinite values, out-of-range parameters,
        inconsistent initialization or a native numerical failure.
    TypeError
        ``s_init``, ``refine`` or ``convergence`` has an unsupported type.
    NotImplementedError
        ``verbose`` or MAF filtering is requested.

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
    Nonfinite or asymmetric inputs raise an error; they are not repaired.
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
    for name, requested in (
        ("verbose", verbose),
        ("maf", maf is not None or maf_thresh != 0),
    ):
        if requested:
            raise NotImplementedError(f"{name} is not implemented")
    n = _scalar(n, "n", lower=1.0, strict=True)
    yty = _scalar(yty, "yty", strict=True)
    Xty = _vector(Xty, "Xty")
    m = Xty.size
    if _rss_matrix_validated:
        # Only susie_rss sets this flag, after validating every raw and scaled
        # matrix entry; direct susie_suff_stat calls always validate XtX here.
        assert isinstance(XtX, np.ndarray) and XtX.shape == (m, m) and XtX.dtype == np.float64
    else:
        XtX = _matrix(XtX, "XtX", m, scale=_rss_matrix_scale)
    ids, metadata = _metadata(variant_ids, variant_metadata, m)
    L, max_iter = _integer(L, "L"), _integer(max_iter, "max_iter")
    tol = _scalar(tol, "tol")
    prior_tol = _scalar(prior_tol, "prior_tol")
    r_tol = _scalar(r_tol, "r_tol")
    check_null_threshold = _scalar(check_null_threshold, "check_null_threshold")
    scaled_prior_variance = _scalar(scaled_prior_variance, "scaled_prior_variance")
    if standardize and scaled_prior_variance > 1:
        raise ValueError("scaled_prior_variance must be <= 1 when standardize=True")
    sigma2 = (
        yty / (n - 1)
        if residual_variance is None
        else _scalar(residual_variance, "residual_variance", strict=True)
    )
    if estimate_prior_method not in ("optim", "EM", "simple"):
        raise ValueError("estimate_prior_method must be 'optim', 'EM', or 'simple'")
    for name, value in (("coverage", coverage), ("min_abs_corr", min_abs_corr)):
        if value is not None and (not np.isfinite(value) or not 0 < value <= 1):
            if name == "min_abs_corr" and value == 0:
                continue
            raise ValueError(f"{name} must be in (0, 1] (min_abs_corr also allows 0)")
    if n_purity is not None:
        n_purity = _integer(n_purity, "n_purity")
    collected = []
    qc: dict[str, Any] = {
        "matrix_symmetric": True,
        "psd_checked": bool(check_input),
        "input_columns": m,
        "input_missing_policy": "error",
        "variant_ids_are_positional": variant_ids is None,
    }
    if check_input:
        eigenvalues, eigenvectors = np.linalg.eigh(
            XtX if _rss_matrix_scale == 1.0 else XtX * _rss_matrix_scale
        )
        qc["minimum_eigenvalue"] = float(eigenvalues[0])
        if eigenvalues[0] < -r_tol:
            raise ValueError("XtX is not positive semidefinite")
        basis = eigenvectors[:, eigenvalues > np.finfo(float).eps]
        projection_error = float(np.linalg.norm(Xty - basis @ (basis.T @ Xty)))
        qc["projection_residual_norm"] = projection_error
        if projection_error > 1e-8 * max(1.0, np.linalg.norm(Xty)):
            _warn("Xty does not lie in the nonzero eigenspace of XtX", collected)
    weights = (
        np.full(m, 1.0 / m)
        if prior_weights is None
        else _vector(prior_weights, "prior_weights", m).copy()
    )
    if np.any(weights < 0) or not np.isfinite(weights.sum()) or weights.sum() <= 0:
        raise ValueError("prior_weights must be nonnegative with a finite positive sum")
    null_weight = _scalar(null_weight, "null_weight")
    if null_weight >= 1:
        raise ValueError("null_weight must be < 1")
    null_index = m if null_weight > 0 else None
    if null_index is not None:
        # As in susieR: augment the raw weights, then normalize the whole vector.
        weights = np.r_[weights * (1 - null_weight), null_weight]
        XtX = np.pad(XtX, ((0, 1), (0, 1)))
        Xty = np.r_[Xty, 0.0]
    # susieR 0.16.6 normalizes default and supplied weights alike.
    weights /= math.fsum(weights)
    p = len(Xty)
    means = np.full(p, np.nan) if X_colmeans is None else np.asarray(X_colmeans, dtype=float)
    if means.ndim == 0:
        means = np.full(p, means)
    elif null_index is not None and means.shape == (m,):
        means = np.r_[means, 0.0]
    if means.shape != (p,):
        raise ValueError("X_colmeans must be scalar or have one value per predictor")
    y_mean_value = np.nan if y_mean is None else float(y_mean)
    diagonal = XtX.diagonal().copy()
    if _rss_matrix_scale != 1.0:
        diagonal *= _rss_matrix_scale
    scales = np.sqrt(diagonal / (n - 1)) if standardize else np.ones(p)
    scales[scales == 0] = 1.0
    # Working crossproduct D⁻¹ XtX D⁻¹ in the column-major layout read by the
    # native column updates. The transpose and the order of the multiplication
    # and division determine the rounding of every entry; keep both.
    inverse_scales = 1.0 / scales
    matrix_operator = (
        getattr(_native, "MATRIX_OPERATOR_VERSION", 0) == 1
        and np.isfinite(_rss_matrix_scale)
        and _rss_matrix_scale > 0
        and bool(np.all(np.isfinite(scales) & (scales > 0)))
        and bool(np.all(np.isfinite(inverse_scales) & (inverse_scales > 0)))
    )
    if matrix_operator:
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
                # keeping the order of the subsequent operations.
                np.multiply(XtX.T, _rss_matrix_scale, out=crossproduct)
                np.multiply(crossproduct, inverse_scales[None, :], out=crossproduct)
            np.divide(crossproduct, scales[:, None], out=crossproduct)
    crossresponse = np.ascontiguousarray(Xty / scales)
    options = dict(
        l=min(L, p),
        prior_variance=scaled_prior_variance * (yty / (n - 1)),
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
    signature = data_signature(
        crossproduct, crossresponse, yty, n, scales, _rss_matrix_scale, ids, metadata
    )
    if s_init is not None:
        options.update(
            initial_options(
                s_init,
                ids=ids,
                metadata=metadata,
                scales=scales,
                null_index=null_index,
                signature=signature,
                L=min(L, p),
                weights=weights,
                estimate_prior_variance=bool(estimate_prior_variance),
                estimate_residual_variance=bool(estimate_residual_variance),
                prior_variance=options["prior_variance"],
            )
        )
    prepared = None
    if retain_data or level or refine is not False or s_init is not None:
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
            signature,
            _rss_matrix_scale,
            ids,
            metadata,
        )
    native = _native.fit(crossproduct, crossresponse, yty, n, options)
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
        null_index=null_index,
        n_purity=n_purity,
        covariance_scales=np.sqrt(diagonal),
        covariance_multiplier=_rss_matrix_scale,
        audit=cs_audit,
    )
    params = dict(
        L=L,
        effective_L=min(L, p),
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
        track_fit=track_fit,
        convergence=control.options(),
        convergence_resolution=resolution,
    )
    intercept = y_mean_value - float(np.sum(means * np.sum(alpha * native["mu"], axis=0) / scales))
    summary, trace = convergence_summary(native, options, control)
    summary["config_resolution"] = resolution.copy()
    fit = SusieResult(
        variant_ids=ids,
        alpha=alpha,
        mu=native["mu"],
        mu2=native["mu2"],
        lbf_variable=native["lbf_variable"],
        lbf=native["lbf"],
        V=V,
        sigma2=native["sigma2"],
        pip=marginal_pip(alpha, V, null_index=null_index, prior_tol=prior_tol),
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
    )
    if refine is not False:
        if not isinstance(refine, (bool, RefinementConfig)):
            raise TypeError("refine must be False, True, or a RefinementConfig")
        fit = refine_fit(fit, refine if isinstance(refine, RefinementConfig) else None)
    return fit


susie_suff_stat.__doc__ = _susie_suff_stat.__doc__


def susie_rss(
    z: ArrayLike | None = None,
    R: ArrayLike | None = None,
    n: float | None = None,
    *,
    bhat: ArrayLike | None = None,
    shat: ArrayLike | None = None,
    var_y: float | None = None,
    z_ld_weight=0.0,
    estimate_residual_variance=False,
    prior_variance=50.0,
    check_prior=True,
    L=10,
    scaled_prior_variance=0.2,
    residual_variance=None,
    estimate_prior_variance=True,
    estimate_prior_method="optim",
    check_null_threshold=0.0,
    prior_tol=1e-9,
    r_tol=1e-8,
    prior_weights=None,
    null_weight=0.0,
    standardize=True,
    max_iter=50,
    s_init=None,
    coverage=0.95,
    min_abs_corr=0.5,
    tol=1e-4,
    verbose=False,
    track_fit=False,
    check_input=False,
    refine=False,
    convergence=None,
    retain_data=False,
    n_purity: int | None = None,
    variant_ids: Sequence[str] | None = None,
    variant_metadata: Mapping[str, ArrayLike] | None = None,
) -> SusieResult:
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
        of z (z = bhat/shat). ``shat`` may be a scalar.
    var_y : float, optional
        Sample variance of the phenotype; requires ``n``. Sets yᵀy =
        (n - 1) var_y. With ``bhat`` and ``shat``, the predictor crossproducts
        are also reconstructed so that effects are on the original scale.
    z_ld_weight : float, default 0
        Must be 0; the earlier susieR LD modification is not implemented.
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
        As in :func:`susie_suff_stat`.
    n_purity, variant_ids, variant_metadata
        As in :func:`susie_suff_stat`. All apply to the working sufficient
        statistics described in Notes.

    Returns
    -------
    SusieResult
        Posterior fit. ``params`` additionally records the input route
        (``input_kind``), the supplied n and var_y, and the effect scale
        (``effect_scale``: 'standardized', 'original' or 'noncentrality').

    Raises
    ------
    ValueError
        Ambiguous or incomplete input, an invalid correlation matrix,
        nonfinite values, or z² + n - 2 ≤ 0 for known n.
    NotImplementedError
        ``verbose`` or a nonzero ``z_ld_weight`` is requested.

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
    yᵀy = n - 1, which places effects on a standardized scale. With var_y,
    yᵀy = (n - 1) var_y while XᵀX and Xᵀy are unchanged. Original-scale
    coefficients therefore require ``bhat``, ``shat``, ``n`` and ``var_y``.
    When n is omitted, a supplied var_y is validated and recorded but does
    not enter the working statistics, and effects are on the noncentrality
    scale. If R is an external reference panel, its agreement with the study
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
    if z_ld_weight != 0:
        raise NotImplementedError("Nonzero z_ld_weight is not implemented")
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
            else _vector(shat, "shat", beta.size)
        )
        if not np.all(np.isfinite(se)) or np.any(se <= 0):
            raise ValueError("shat must be finite and strictly positive")
        z = beta / se
    z = _vector(z, "z")
    prevalidated_scale = None
    if n is None:
        prevalidated_scale = 1.0
    elif not (has_effects and var_y is not None):
        n = _scalar(n, "n", lower=1.0, strict=True)
        prevalidated_scale = n - 1
    R = _matrix(R, "R", len(z), correlation=True, rss_scale=prevalidated_scale)
    extra_warnings = []
    rss_matrix_scale = 1.0
    if var_y is not None:
        var_y = _scalar(var_y, "var_y", strict=True)
    if n is None:
        _warn(
            ("n is missing: using the upstream large-sample noncentrality-parameter likelihood"),
            extra_warnings,
        )
        XtX, Xty, yty, effective_n = R, z, 1.0, 2.0
        if scaled_prior_variance != 0.2:
            raise ValueError("Missing-n mode uses prior_variance, not scaled_prior_variance")
        effective_scaled_prior, effective_standardize = prior_variance, False
        scale_kind = "noncentrality"
    else:
        n = _scalar(n, "n", lower=1.0, strict=True)
        denominator = z * z + n - 2
        if np.any(denominator <= 0):
            raise ValueError("Known-n PVE adjustment requires z²+n-2 > 0")
        adj = (n - 1) / denominator
        adjusted_z = np.sqrt(adj) * z
        if has_effects and var_y is not None:
            diagonal = var_y * adj / (se * se)
            sd = np.sqrt(diagonal)
            XtX = R * sd[:, None] * sd[None, :]
            XtX = (XtX + XtX.T) / 2
            Xty = adjusted_z * np.sqrt(adj) * var_y / se
            effective_var_y, scale_kind = var_y, "original"
        else:
            XtX, Xty = R, np.sqrt(n - 1) * adjusted_z
            rss_matrix_scale = n - 1
            effective_var_y, scale_kind = (
                (1.0 if var_y is None else var_y),
                "standardized",
            )
        yty, effective_n = (n - 1) * effective_var_y, n
        effective_scaled_prior, effective_standardize = (
            scaled_prior_variance,
            standardize,
        )
    if estimate_residual_variance:
        _warn(
            (
                "Estimating residual variance requires in-sample LD consistent "
                "with the summary statistics"
            ),
            extra_warnings,
        )
    fit = _susie_suff_stat(
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
        _rss_matrix_scale=rss_matrix_scale,
        _rss_matrix_validated=prevalidated_scale is not None,
    )
    fit.params.update(
        interface="susie_rss",
        input_n=n,
        input_kind="bhat_shat" if has_effects else "z",
        input_var_y=var_y,
        effect_scale=scale_kind,
        prior_variance=prior_variance,
        z_ld_weight=0.0,
    )
    fit.warnings[:0] = extra_warnings
    return fit
