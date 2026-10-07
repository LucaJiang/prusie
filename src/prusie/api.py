"""Validated RSS transformations and native sufficient-statistics inference."""

from __future__ import annotations

import warnings as warning_module
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _native
from ._array_layout import native_array
from .posterior import credible_sets, marginal_pip
from .result import SusieResult


def _vector(
    value: ArrayLike, name: str, size: int | None = None
) -> NDArray[np.float64]:
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
    # Complete finite/symmetry/diagonal checks without matrix-sized temporaries.
    if rss_scale is None:
        _native.validate_matrix(array, name, correlation, scale)
    else:
        _native.validate_rss_matrix(array, rss_scale)
    # native_array already established alignment and compatible byte strides.
    # The C copy, if needed, preserves the public preparation orientation.
    return np.ascontiguousarray(array)


def _scalar(
    value: float, name: str, *, lower: float = 0.0, strict: bool = False
) -> float:
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
    ids = np.asarray(
        [str(i) for i in range(m)] if variant_ids is None else variant_ids, dtype=str
    )
    if ids.shape != (m,) or len(set(ids.tolist())) != m or np.any(ids == ""):
        raise ValueError(
            "variant_ids must contain m distinct nonempty IDs in input order"
        )
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
    check_prior=False,
    n_purity: int | None = None,
    variant_ids: Sequence[str] | None = None,
    variant_metadata: Mapping[str, ArrayLike] | None = None,
    maf=None,
    maf_thresh=0.0,
    _rss_matrix_scale=1.0,
    _rss_matrix_validated=False,
) -> SusieResult:
    """Fit SuSiE to centered sufficient statistics using sequential Rust IBSS.

    Parameters
    ----------
    XtX : array_like, shape (p, p)
        Centered predictor crossproduct, float64, symmetric and finite.
    Xty : array_like, shape (p,)
        Centered predictor-response crossproduct in exactly the same order.
    yty : float
        Centered response sum of squares, strictly positive.
    n : float
        Sample size greater than one, never inferred from LD dimension.
    L : int, default 10
        Maximum single effects, capped at the number of columns including null.
    scaled_prior_variance : float, default 0.2
        Initial V divided by yty/(n-1); at most one when standardized.
    residual_variance : float or None
        Initial or fixed sigma2; None uses yty/(n-1).
    estimate_residual_variance, estimate_prior_variance : bool
        Update sigma2 and V respectively. Both default True for this interface.
    estimate_prior_method : {'optim', 'EM', 'simple'}
        'optim' searches the SER marginal likelihood; 'EM' updates from the
        posterior second moment; 'simple' only compares current V with zero.
    prior_weights : array_like, optional
        Nonnegative finite variant weights; normalized after optional null.
    null_weight : float, default 0
        Optional last, zero-crossproduct null column. Must be in [0, 1).
    standardize : bool, default True
        Scale predictors to unit sample variance before inference.
    max_iter, tol : int, float
        IBSS iteration limit (100 here; 50 for RSS) and finite nonnegative
        ELBO improvement threshold (1e-4).
    coverage, min_abs_corr : float or None
        Requested CS posterior mass (0.95) and minimum absolute correlation
        (0.5). Returned ``sets.coverage`` is each retained set's actual mass;
        ``sets.requested_coverage`` records the target.
        None disables CS construction; marginal PIP is still returned.
    n_purity : int or None
        Matrix-input purity is always complete, matching the pinned reference;
        a finite n_purity does not subsample the matrix.
    prior_tol : float, default 1e-9
        V threshold for final null-component trim and PIP. The CS threshold
        stays fixed at 1e-9. Trim also runs at the iteration cap.
    check_input, r_tol : bool, float
        Optional PSD/projection checks, eigenvalue tolerance 1e-8.
    check_prior, check_null_threshold : bool, float
        Large-V guard and log-BF threshold for setting V to zero.
    X_colmeans, y_mean : array_like or float, optional
        Means needed to recover intercept; without them intercept is NaN.
    variant_ids, variant_metadata : sequence, mapping, optional
        Ordered biological IDs and aligned chromosome/allele metadata.
        Omitted IDs are positional labels, not inferred biological identifiers.
    s_init, refine, track_fit, maf, maf_thresh, verbose
        Nondefault initialization, refinement, trace, MAF filtering and verbose
        mode raise NotImplementedError. These options are never silently ignored.

    Returns
    -------
    SusieResult
        Posterior arrays, BF, variance, ELBO, CS, PIP, and execution metadata.

    Raises
    ------
    ValueError
        Invalid shape, nonfinite input, invalid parameters or native failure.
    NotImplementedError
        An explicitly unsupported option is requested.

    Notes
    -----
    Supported Gaussian updates follow pinned susieR 0.16.6. Matrix-input CS
    purity uses the complete submatrix.
    Asymmetric or missing inputs error rather than being silently repaired.
    Component mu/mu2 remain on the fitted predictor scale; ``coef``
    applies the stored inverse scale transformation.

    References
    ----------
    Wang et al. (2020), JRSS B 82, 1273-1300; susieR 0.16.6 R/sufficient_stats_methods.R.

    Examples
    --------
    >>> fit = susie_suff_stat([[99., 0.], [0., 99.]], [30., 0.], 99., 100,
    ...                       L=1, estimate_residual_variance=False)
    >>> fit.alpha.shape
    (1, 2)
    """
    for name, requested in (
        ("s_init", s_init is not None),
        ("refine", refine),
        ("track_fit", track_fit),
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
        # Private RSS-only route: the caller checked every raw and scaled
        # matrix entry together. Public suff-stat calls cannot set this flag.
        assert (
            isinstance(XtX, np.ndarray)
            and XtX.shape == (m, m)
            and XtX.dtype == np.float64
        )
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
        # Preserve R's order: augment raw weights, then normalize the whole vector.
        weights = np.r_[weights * (1 - null_weight), null_weight]
        XtX = np.pad(XtX, ((0, 1), (0, 1)))
        Xty = np.r_[Xty, 0.0]
    # The current reference normalizes default and explicit priors alike.
    weights /= math.fsum(weights)
    p = len(Xty)
    means = (
        np.full(p, np.nan)
        if X_colmeans is None
        else np.asarray(X_colmeans, dtype=float)
    )
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
    # Centered crossproducts rescaled to predictor sample standard deviations.
    # Preserve the transpose and the two floating operations, writing directly
    # to the column-major layout consumed by the native column updates.
    inverse_scales = 1.0 / scales
    matrix_operator = (
        getattr(_native, "MATRIX_OPERATOR_VERSION", 0) == 1
        and np.isfinite(_rss_matrix_scale)
        and _rss_matrix_scale > 0
        and bool(np.all(np.isfinite(scales) & (scales > 0)))
        and bool(np.all(np.isfinite(inverse_scales) & (inverse_scales > 0)))
    )
    if matrix_operator:
        # The compatible native operator owns the exact diagonal/scalar factors
        # and complete materialization fallback. Keep the original orientation.
        crossproduct = XtX.T
    else:
        crossproduct = _native.prepare_crossproduct(
            XtX, inverse_scales, scales, _rss_matrix_scale
        )
        if crossproduct is None:
            # Portable NumPy fallback when runtime SIMD or contiguous input is absent.
            crossproduct = np.empty((p, p), dtype=np.float64, order="F")
            if _rss_matrix_scale == 1.0:
                np.multiply(XtX.T, inverse_scales[None, :], out=crossproduct)
            else:
                # Materialize the exact RSS multiplication directly into the sole
                # native matrix allocation. Subsequent operations retain their order.
                np.multiply(XtX.T, _rss_matrix_scale, out=crossproduct)
                np.multiply(crossproduct, inverse_scales[None, :], out=crossproduct)
            np.divide(crossproduct, scales[:, None], out=crossproduct)
    crossresponse = np.ascontiguousarray(Xty / scales)
    options = dict(
        l=min(L, p),
        prior_variance=scaled_prior_variance * (yty / (n - 1)),
        residual_variance=sigma2,
        prior_weights=weights.tolist(),
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
            matrix_inverse_scales=inverse_scales.tolist(),
            matrix_scales=scales.tolist(),
        )
    native = _native.fit(crossproduct, crossresponse, yty, n, options)
    for message in native["warnings"]:
        _warn(message, collected)
    if not native["converged"]:
        _warn(
            f"IBSS did not converge in {max_iter} iterations; check summary/LD consistency",
            collected,
        )
    alpha, V = native["alpha"], native["V"]
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
        s_init=None,
        refine=False,
        track_fit=False,
    )
    intercept = y_mean_value - float(
        np.sum(means * np.sum(alpha * native["mu"], axis=0) / scales)
    )
    return SusieResult(
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
    )


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
    n_purity: int | None = None,
    variant_ids: Sequence[str] | None = None,
    variant_metadata: Mapping[str, ArrayLike] | None = None,
) -> SusieResult:
    """Fit SuSiE-RSS using z + signed R + sample size, or bhat/shat + R.

    Parameters
    ----------
    z : array_like, shape (p,), optional
        Signed association statistics; mutually exclusive with bhat/shat.
    R : array_like, shape (p, p)
        Signed LD correlation r in the exact variant/effect-allele order.
        Singular matrices are allowed; no ridge or PSD clipping is applied.
    n : float, optional
        Known sample size > 1. If absent, use the upstream large-n likelihood
        (XtX=R, Xty=z, n_internal=2, yty=1), with a recorded warning.
    bhat, shat : array_like, optional
        Effect estimates and positive standard errors, instead of z. shat may
        be scalar. With var_y and n, reconstruct the original predictor scale.
    var_y : float, optional
        Sample phenotype variance for known n; sets the response variance.
        With bhat/shat also reconstructs the original predictor scale.
    estimate_residual_variance : bool, default False
        Fixed residual variance is recommended for external reference LD.
    prior_variance : float, default 50
        Initial prior variance for missing-n mode only.
    z_ld_weight : float, default 0
        Nonzero legacy LD modification is explicitly unsupported.
    L, scaled_prior_variance, residual_variance, estimate_prior_variance,
    estimate_prior_method, check_null_threshold, prior_tol, r_tol,
    prior_weights, null_weight, standardize, max_iter, s_init, coverage,
    min_abs_corr, tol, verbose, track_fit, check_input, refine, check_prior,
    n_purity, variant_ids, variant_metadata
        See ``susie_suff_stat``. check_prior defaults True in this interface.
        RSS defaults max_iter=50 and tol=1e-4; matrix purity is complete.

    Returns
    -------
    SusieResult
        Posterior fit with the working RSS transformation parameters recorded.

    Raises
    ------
    ValueError
        Ambiguous input, invalid correlation or nonfinite values.
    NotImplementedError
        Unsupported nondefault option (including initialization/refinement).

    Notes
    -----
    Known n applies z_tilde = z sqrt((n-1)/(z²+n-2)); this is not the
    infinite-sample approximation. With z and no var_y the effects use standardized response units.
    With z and var_y, yty changes to (n-1)*var_y but Xty is not rescaled.
    Original-scale coefficients require bhat/shat/n/var_y. Missing-n mode
    ignores a valid var_y in its working quantities and reports noncentrality
    effects. External-LD misspecification remains the user's modeling
    assumption; large prior and convergence warnings are retained.

    References
    ----------
    Zou et al. (2022), PLoS Genetics 18, e1010299; susieR 0.16.6 R/susie_constructors.R.

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
            "n is missing: using the upstream large-sample noncentrality-parameter likelihood",
            extra_warnings,
        )
        XtX, Xty, yty, effective_n = R, z, 1.0, 2.0
        if scaled_prior_variance != 0.2:
            raise ValueError(
                "Missing-n mode uses prior_variance, not scaled_prior_variance"
            )
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
            "Estimating residual variance requires in-sample LD consistent with the summary statistics",
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
