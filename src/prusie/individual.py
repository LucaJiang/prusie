"""Individual-level SuSiE fitting and regression data utilities.

Preprocessing and public defaults follow susieR 0.16.6 (BSD-3-Clause).
Dense and sparse predictor matrices are passed to matrix-free native operators.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._array_layout import native_array
from .control import SlotPrior
from .diagnostics import ConvergenceConfig
from .refinement import RefinementConfig
from .result import SusieResult


@dataclass(frozen=True)
class _DesignData:
    X: Any
    means: NDArray[np.float64]
    scales: NDArray[np.float64]
    diagonal: NDArray[np.float64]
    response_variance: float
    fitted_response: bool = True


@dataclass(frozen=True)
class SusieInputs:
    """Validated data and fitting parameters returned by ``init_only=True``.

    ``data`` and ``params`` are inspectable mappings. Pass this object to
    :func:`susie_workhorse` to fit the same prepared inputs. A trend-filter
    MAD preparation may retain another preparation as ``model_init``;
    this is not a fitted warm start and cannot be used by the workhorse.
    """

    data: dict[str, Any]
    params: dict[str, Any]
    _request: Any = field(repr=False)

    def __getitem__(self, key: str) -> dict[str, Any]:
        if key not in ("data", "params"):
            raise KeyError(key)
        return getattr(self, key)


def _validated_design(X: Any, y: ArrayLike, *, na_rm: bool) -> tuple[Any, NDArray[np.float64]]:
    try:
        from scipy import sparse

        is_sparse = sparse.issparse(X)
    except ImportError:
        is_sparse = False
    from ._trend import TrendDesign

    if isinstance(X, TrendDesign):
        pass
    elif is_sparse:
        X = X.tocsc(copy=True).astype(np.float64)
        X.sum_duplicates()
        X.sort_indices()
        if not np.isfinite(X.data).all():
            raise ValueError("X must contain finite values")
    else:
        X = native_array(np.asarray(X, dtype=np.float64))
        if X.ndim != 2 or not np.isfinite(X).all():
            raise ValueError("X must be a finite two-dimensional predictor matrix")
        X = np.asfortranarray(X)
    response = np.asarray(y, dtype=np.float64)
    if response.ndim != 1 or response.shape[0] != X.shape[0]:
        raise ValueError("y must have one entry for each row of X")
    if np.isinf(response).any():
        raise ValueError("y must not contain infinite values")
    missing = np.isnan(response)
    if missing.any():
        if not na_rm:
            raise ValueError("y contains missing values; use na_rm=True to omit these observations")
        response, X = response[~missing], X[~missing, :]
        if is_sparse:
            X = X.tocsc()
    if X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError("X needs at least two observations and one predictor")
    return X, np.ascontiguousarray(response)


def _column_stats(X: Any) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(X, "column_statistics"):
        return X.column_statistics()
    means = np.asarray(X.mean(axis=0)).ravel()
    if hasattr(X, "multiply"):
        # Implicit centering preserves sparsity; no centered n × p array is built.
        variance_sum = np.asarray(X.multiply(X).sum(axis=0)).ravel() - X.shape[0] * means**2
    else:
        variance_sum = np.sum((X - means) ** 2, axis=0)
    return means, np.maximum(variance_sum, 0)


def susie(
    X: Any,
    y: ArrayLike,
    *,
    L: int = 10,
    scaled_prior_variance: float | ArrayLike = 0.2,
    residual_variance: float | None = None,
    prior_weights: ArrayLike | None = None,
    null_weight: float = 0.0,
    standardize: bool = True,
    intercept: bool = True,
    estimate_residual_variance: bool = True,
    estimate_residual_method: str = "MoM",
    estimate_prior_variance: bool = True,
    estimate_prior_method: str = "optim",
    prior_variance_grid: ArrayLike | None = None,
    mixture_weights: ArrayLike | None = None,
    unmappable_effects: str = "none",
    check_null_threshold: float = 0.0,
    prior_tol: float = 1e-9,
    residual_variance_upperbound: float = np.inf,
    model_init: SusieResult | Mapping[str, Any] | None = None,
    s_init: SusieResult | Mapping[str, Any] | None = None,
    coverage: float | None = 0.95,
    min_abs_corr: float | None = 0.5,
    compute_univariate_zscore: bool = False,
    na_rm: bool = False,
    max_iter: int = 100,
    L_greedy: int | None = None,
    greedy_lbf_cutoff: float = 0.1,
    tol: float | None = None,
    convergence_method: Literal["elbo", "pip"] = "elbo",
    verbose: bool = False,
    track_fit: bool | Literal["compact", "dense"] = False,
    residual_variance_lowerbound: float | None = None,
    refine: bool | RefinementConfig = False,
    n_purity: int | float | Literal["auto"] = "auto",
    median_abs_corr: float | None = None,
    cs_extension_corr: float | None = None,
    alpha0: float | None = None,
    beta0: float | None = None,
    init_only: bool = False,
    slot_prior: SlotPrior | Mapping[str, Any] | None = None,
    convergence: ConvergenceConfig | None = None,
    retain_data: bool = False,
    variant_ids: Sequence[str] | None = None,
    variant_metadata: Mapping[str, ArrayLike] | None = None,
) -> SusieResult | SusieInputs:
    """Fit Gaussian SuSiE to individual observations with a matrix-free operator.

    Parameters
    ----------
    X : array_like or scipy sparse matrix, shape (n, p)
        Finite predictor matrix. Sparse inputs stay sparse throughout fitting;
        neither a dense predictor matrix nor a p × p crossproduct is constructed.
    y : array_like, shape (n,)
        Response vector.
    intercept : bool, default True
        Center X and y and recover the original-scale intercept.
    standardize : bool, default True
        Divide predictor columns by their sample standard deviation.
    na_rm : bool, default False
        Omit observations with missing responses. Missing X entries are rejected.
    compute_univariate_zscore : bool, default False
        Attach marginal regression z-scores to the result.
    residual_variance_lowerbound : float, optional
        Defaults to sample variance of y divided by 10000.
    init_only : bool, default False
        Return validated data and parameters without fitting.

    Other Parameters
    ----------------
    L, scaled_prior_variance, residual_variance, prior_weights, null_weight
        As in :func:`susie_suff_stat`.
    estimate_residual_variance, estimate_residual_method, estimate_prior_variance
        As in :func:`susie_suff_stat`.
    estimate_prior_method, prior_variance_grid, mixture_weights
        As in :func:`susie_suff_stat`.
    model_init, s_init, L_greedy, greedy_lbf_cutoff, slot_prior
        Initialization, greedy component growth and optional slot-activity prior.
    tol, convergence_method, convergence, max_iter, track_fit, verbose
        Fitting convergence and logging controls.
    coverage, min_abs_corr, median_abs_corr, cs_extension_corr, n_purity
        Credible-set controls; correlations are computed only as needed.
    alpha0, beta0, residual_variance_upperbound, check_null_threshold, prior_tol
        Residual and single-effect prior controls.
    variant_ids, variant_metadata, retain_data, refine, unmappable_effects
        As in :func:`susie_suff_stat`; unmappable models are excluded.

    Returns
    -------
    SusieResult or SusieInputs
        Fitted posterior or a validated no-fit request.
    """
    from .api import _fit, _prepare_suff_stat

    for name, value in (
        ("intercept", intercept),
        ("standardize", standardize),
        ("na_rm", na_rm),
        ("compute_univariate_zscore", compute_univariate_zscore),
    ):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be a boolean")
    X, response = _validated_design(X, y, na_rm=na_rm)
    n = len(response)
    means, centered_ss = _column_stats(X)
    scales = np.sqrt(centered_ss / (n - 1)) if standardize else np.ones(X.shape[1])
    scales[scales == 0] = 1
    effective_means = means if intercept else np.zeros(X.shape[1])
    y_mean = float(response.mean()) if intercept else 0.0
    working_y = response - y_mean
    response_variance = float(np.var(response, ddof=1))
    diagonal = centered_ss if intercept else centered_ss + n * means**2
    crossresponse = np.asarray(X.T @ working_y).ravel() - effective_means * working_y.sum()
    design = _DesignData(X, effective_means, scales, diagonal, response_variance)
    lower = (
        response_variance / 1e4
        if residual_variance_lowerbound is None
        else residual_variance_lowerbound
    )
    kwargs = dict(locals())
    accepted = (
        "L",
        "scaled_prior_variance",
        "residual_variance",
        "prior_weights",
        "null_weight",
        "standardize",
        "estimate_residual_variance",
        "estimate_residual_method",
        "estimate_prior_variance",
        "estimate_prior_method",
        "prior_variance_grid",
        "mixture_weights",
        "unmappable_effects",
        "check_null_threshold",
        "prior_tol",
        "residual_variance_upperbound",
        "model_init",
        "s_init",
        "coverage",
        "min_abs_corr",
        "max_iter",
        "L_greedy",
        "greedy_lbf_cutoff",
        "tol",
        "verbose",
        "track_fit",
        "refine",
        "n_purity",
        "median_abs_corr",
        "cs_extension_corr",
        "alpha0",
        "beta0",
        "init_only",
        "slot_prior",
        "convergence",
        "retain_data",
        "variant_ids",
        "variant_metadata",
    )
    options = {name: kwargs[name] for name in accepted}
    options.update(
        residual_variance_lowerbound=lower,
        X_colmeans=effective_means,
        y_mean=y_mean,
        convergence_method=None if convergence is not None else convergence_method,
        _design=design,
    )
    request = _prepare_suff_stat(None, crossresponse, float(working_y @ working_y), n, **options)
    result = _fit(request)
    if isinstance(result, SusieResult):
        result.params.update(interface="susie", intercept=bool(intercept), na_rm=bool(na_rm))
        if compute_univariate_zscore:
            regression = univariate_regression(X, response)
            with np.errstate(divide="ignore", invalid="ignore"):
                result.univariate_zscore = regression["betahat"] / regression["sebetahat"]
    return result


def susie_workhorse(inputs: SusieInputs) -> SusieResult:
    """Fit a request returned by an entry point with ``init_only=True``.

    Parameters
    ----------
    inputs : SusieInputs
        Validated data and model options returned by a public fitter.

    Returns
    -------
    SusieResult
        Posterior fit using the stored request without repeating preprocessing.

    Raises
    ------
    ValueError
        A deferred ``model_init`` contains prepared inputs instead of a fitted
        posterior, as in the default-MAD trend-filter preparation-only path.
    """
    from dataclasses import replace

    from .api import _fit

    if not isinstance(inputs, SusieInputs):
        raise TypeError("inputs must be SusieInputs returned by init_only=True")
    request = replace(inputs._request, params={**inputs._request.params, "init_only": False})
    return _fit(request)


def compute_suff_stat(X: Any, y: ArrayLike, *, standardize: bool = False) -> dict[str, Any]:
    """Compute centered sufficient statistics and original predictor/response means.

    Parameters
    ----------
    X : array_like or scipy sparse matrix, shape (n, p)
        Finite individual predictor observations.
    y : array_like, shape (n,)
        Finite response vector.
    standardize : bool, default False
        Divide predictors by sample standard deviations before crossproducts.

    Returns
    -------
    dict
        XtX, Xty, yty, n, X_colmeans and y_mean; XtX is explicitly dense.
        To avoid this p × p allocation use :func:`susie` directly.
    """
    X, response = _validated_design(X, y, na_rm=False)
    n, y_mean = len(response), float(response.mean())
    means, diagonal = _column_stats(X)
    centered = response - y_mean
    gram = X.T @ X
    if hasattr(gram, "toarray"):
        gram = gram.toarray()
    gram = np.asarray(gram) - n * means[:, None] * means[None, :]
    crossresponse = np.asarray(X.T @ centered).ravel()
    if standardize:
        scales = np.sqrt(diagonal / (n - 1))
        scales[scales == 0] = 1
        gram = gram / scales[:, None] / scales[None, :]
        crossresponse = crossresponse / scales
    return dict(
        XtX=gram,
        Xty=crossresponse,
        yty=float(centered @ centered),
        n=n,
        X_colmeans=means,
        y_mean=y_mean,
    )


def univariate_regression(
    X: Any,
    y: ArrayLike,
    *,
    Z: ArrayLike | None = None,
    center: bool = True,
    scale: bool = False,
    return_residuals: bool = False,
    method: Literal["lmfit", "sumstats"] = "lmfit",
) -> dict[str, NDArray[np.float64]]:
    """Regress y on each predictor separately, including an intercept.

    Parameters
    ----------
    X : array_like or scipy sparse matrix
        Predictor observations.
    y : array_like
        Response; observations with missing responses are omitted.
    Z : array_like, optional
        Covariates removed from y before marginal fitting, following susieR.
    center, scale : bool
        Center input columns and optionally divide by sample standard deviation.
    return_residuals : bool, default False
        Return covariate-adjusted y when Z is supplied.
    method : {'lmfit', 'sumstats'}, default 'lmfit'
        Accepted equivalent least-squares formulations.

    Returns
    -------
    dict
        Marginal ``betahat``, ``sebetahat`` and optional ``residuals``.
    """
    if method not in ("lmfit", "sumstats"):
        raise ValueError("method must be 'lmfit' or 'sumstats'")
    response_before = np.asarray(y, dtype=float)
    X, response = _validated_design(X, y, na_rm=True)
    means, diagonal = _column_stats(X)
    n = len(response)
    if center:
        response = response - response.mean()
    if Z is not None:
        covariates = np.asarray(Z, dtype=float)
        if covariates.ndim == 1:
            covariates = covariates[:, None]
        covariates = covariates[~np.isnan(response_before)]
        if covariates.shape[0] != n or not np.isfinite(covariates).all():
            raise ValueError("Z must be finite with one row per observation")
        if center:
            covariates = covariates - covariates.mean(axis=0)
        if scale and center:
            cov_scale = np.std(covariates, axis=0, ddof=1)
            cov_scale[cov_scale == 0] = 1
            covariates = covariates / cov_scale
        response = response - covariates @ np.linalg.lstsq(covariates, response, rcond=None)[0]
    y_centered = response - response.mean()
    xy = np.asarray(X.T @ y_centered).ravel()
    slope = np.divide(xy, diagonal, out=np.zeros_like(xy), where=diagonal > 0)
    residual_ss = np.maximum(float(y_centered @ y_centered) - slope * xy, 0)
    se = np.sqrt(
        np.divide(residual_ss, (n - 2) * diagonal, out=np.zeros_like(diagonal), where=diagonal > 0)
    )
    if scale:
        # R scale(center=False) uses root mean squares with denominator n-1.
        norms = diagonal if center else diagonal + n * means**2
        factor = np.sqrt(norms / (n - 1))
        slope, se = slope * factor, se * factor
    result = dict(betahat=slope, sebetahat=se)
    if return_residuals and Z is not None:
        result["residuals"] = response
    return result


def calc_z(X: Any, Y: ArrayLike, *, center: bool = False, scale: bool = False) -> np.ndarray:
    """Return marginal regression z-scores for a response vector or matrix.

    Parameters
    ----------
    X : array_like or scipy.sparse matrix, shape (n, p)
        Predictor observations.
    Y : array_like, shape (n,) or (n, t)
        One or more response vectors.
    center, scale : bool
        Center and scale inputs as in :func:`univariate_regression`.

    Returns
    -------
    ndarray, shape (p,) or (p, t)
        Marginal coefficient estimates divided by their standard errors.
    """
    response = np.asarray(Y, dtype=float)
    if response.ndim == 1:
        result = univariate_regression(X, response, center=center, scale=scale)
        with np.errstate(divide="ignore", invalid="ignore"):
            return result["betahat"] / result["sebetahat"]
    if response.ndim != 2:
        raise ValueError("Y must be a response vector or matrix")
    return np.column_stack(
        [calc_z(X, response[:, i], center=center, scale=scale) for i in range(response.shape[1])]
    )
