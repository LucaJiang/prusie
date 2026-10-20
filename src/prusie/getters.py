"""Public posterior summaries, following susieR 0.16.6.

These definitions are ported from susieR's BSD-3-Clause getter methods; see
THIRD_PARTY_NOTICES.md. Variable and component indices are zero-based.
"""

from __future__ import annotations

import math
import warnings
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .posterior import _design_correlation, credible_sets, marginal_pip
from .result import SusieResult


def susie_get_objective(
    res: SusieResult,
    *,
    last_only: bool = True,
    warning_tol: float = 1e-6,
) -> float | NDArray[np.float64]:
    """Return the final objective or its iteration trace.

    Parameters
    ----------
    res : SusieResult
        Fitted model.
    last_only : bool, default True
        Return only the final objective.
    warning_tol : float, default 1e-6
        Warn about objective decreases larger than this amount.

    Returns
    -------
    float or ndarray
        Final objective or a copy of the complete recorded trace.
    """
    if not np.isfinite(warning_tol) or warning_tol < 0:
        raise ValueError("warning_tol must be finite and nonnegative")
    objective = np.asarray(res.elbo)
    if np.any(np.diff(objective) < -warning_tol):
        warnings.warn("Objective is decreasing", RuntimeWarning, stacklevel=2)
    return (
        float(objective[-1])
        if last_only and objective.size
        else (float("nan") if last_only else objective.copy())
    )


def susie_get_niter(res: SusieResult) -> int:
    """Return the number of completed IBSS sweeps.

    Parameters
    ----------
    res : SusieResult
        Fitted model.

    Returns
    -------
    int
        Number of recorded fitting sweeps.
    """
    return res.niter


def susie_get_prior_variance(res: SusieResult) -> NDArray[np.float64]:
    """Return a copy of the fitted single-effect prior variances.

    Parameters
    ----------
    res : SusieResult
        Fitted model.

    Returns
    -------
    ndarray, shape (L,)
        Working-scale single-effect variances.
    """
    return res.V.copy()


def susie_get_residual_variance(res: SusieResult) -> float:
    """Return the fitted residual variance.

    Parameters
    ----------
    res : SusieResult
        Fitted model.

    Returns
    -------
    float
        Residual variance on the working response scale.
    """
    return float(res.sigma2)


def susie_get_posterior_mean(
    res: SusieResult,
    *,
    prior_tol: float = 1e-9,
) -> NDArray[np.float64]:
    """Return posterior mean coefficients from active effects, including a null column.

    Parameters
    ----------
    res : SusieResult
        Fitted model with moments on the working predictor scale.
    prior_tol : float, default 1e-9
        Include only components whose prior variance exceeds this value.

    Returns
    -------
    ndarray, shape (q,)
        Coefficients on the original predictor scale, without an intercept.
    """
    keep = prior_tol < res.V
    return np.sum((res.alpha * res.mu)[keep], axis=0) / res.X_column_scale_factors


def susie_get_posterior_sd(
    res: SusieResult,
    *,
    prior_tol: float = 1e-9,
) -> NDArray[np.float64]:
    """Return coefficient posterior standard deviations over active effects.

    Parameters
    ----------
    res : SusieResult
        Fitted model.
    prior_tol : float, default 1e-9
        Minimum active prior variance.

    Returns
    -------
    ndarray, shape (q,)
        Standard deviations on the original predictor scale.
    """
    keep = prior_tol < res.V
    variance = np.sum((res.alpha * res.mu2 - (res.alpha * res.mu) ** 2)[keep], axis=0)
    # Roundoff can make the difference of equal nonnegative moments negative.
    return np.sqrt(np.maximum(variance, 0)) / res.X_column_scale_factors


def susie_get_pip(
    res: SusieResult | ArrayLike,
    *,
    prune_by_cs: bool = False,
    prior_tol: float = 1e-9,
) -> NDArray[np.float64]:
    """Return marginal inclusion probabilities, optionally restricted to CS effects.

    Parameters
    ----------
    res : SusieResult or array_like, shape (L, q)
        Fit or raw single-effect probabilities. A fit's null column is removed.
    prune_by_cs : bool, default False
        Use only components with reported credible sets.
    prior_tol : float, default 1e-9
        Minimum active prior variance when a fit is supplied.

    Returns
    -------
    ndarray
        Marginal inclusion probabilities in original variable order.
    """
    if not isinstance(res, SusieResult):
        alpha = np.asarray(res, dtype=np.float64)
        if alpha.ndim != 2:
            raise ValueError("alpha must be a two-dimensional matrix")
        return 1 - np.prod(1 - alpha, axis=0)
    variances = res.V.copy()
    if prune_by_cs:
        retain = np.zeros(len(variances), dtype=bool)
        retain[np.asarray(res.sets.get("cs_index", []), dtype=int)] = True
        variances[~retain] = 0
    return marginal_pip(
        res.alpha,
        variances,
        null_index=res.null_index,
        prior_tol=prior_tol,
        slot_weights=res.slot_weights,
    )


def susie_get_lfsr(res: SusieResult) -> NDArray[np.float64]:
    """Return the alpha-weighted local false-sign rate for each single effect.

    Parameters
    ----------
    res : SusieResult
        Fitted model with conditional Gaussian posterior moments.

    Returns
    -------
    ndarray, shape (L,)
        Average posterior probability of the less likely sign per effect.
    """
    sd = np.sqrt(np.maximum(res.mu2 - res.mu**2, 0))
    ratio = np.divide(-res.mu, sd, out=np.zeros_like(sd), where=sd > 0)
    cdf = np.fromiter(
        (0.5 * math.erfc(-x / math.sqrt(2)) for x in ratio.flat), dtype=float, count=ratio.size
    ).reshape(ratio.shape)
    cdf = np.where(sd == 0, np.asarray(res.mu <= 0, dtype=float), cdf)
    return 1 - np.sum(res.alpha * np.maximum(cdf, 1 - cdf), axis=1)


def susie_get_posterior_samples(
    susie_fit: SusieResult,
    num_samples: int,
    *,
    seed: int,
) -> dict[str, NDArray[np.float64]]:
    """Draw posterior coefficients using a local explicitly seeded NumPy RNG.

    Parameters
    ----------
    susie_fit : SusieResult
        Fitted model.
    num_samples : int
        Positive number of posterior draws.
    seed : int
        Seed for ``numpy.random.default_rng``. R and NumPy use different RNG
        streams, so identical seeds do not imply identical individual draws.

    Returns
    -------
    dict
        ``b`` contains original-scale coefficient draws of shape (q, num_samples);
        ``gamma`` records whether each total sampled coefficient is nonzero.
    """
    if isinstance(num_samples, bool) or int(num_samples) != num_samples or num_samples < 1:
        raise ValueError("num_samples must be a positive integer")
    rng = np.random.default_rng(seed)
    p = susie_fit.alpha.shape[1]
    samples = np.zeros((p, int(num_samples)))
    sd = np.sqrt(np.maximum(susie_fit.mu2 - susie_fit.mu**2, 0))
    scales = susie_fit.X_column_scale_factors
    for component in np.flatnonzero(susie_fit.V > 1e-9):
        locations = rng.choice(p, size=int(num_samples), p=susie_fit.alpha[component])
        values = rng.normal(
            susie_fit.mu[component, locations] / scales[locations],
            sd[component, locations] / scales[locations],
        )
        samples[locations, np.arange(int(num_samples))] += values
    return {"b": samples, "gamma": np.asarray(samples != 0, dtype=float)}


def susie_get_cs(
    res: SusieResult | ArrayLike,
    *,
    X: Any = None,
    Xcorr: ArrayLike | None = None,
    coverage: float = 0.95,
    min_abs_corr: float | None = 0.5,
    dedup: bool = True,
    squared: bool = False,
    check_symmetric: bool = True,
    n_purity: int | float | Literal["auto"] = "auto",
    median_abs_corr: float | None = None,
    cs_extension_corr: float | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Compute credible sets, with optional LD purity and proxy extension.

    Parameters
    ----------
    res : SusieResult or array_like
        Fitted model or complete component probability matrix.
    X : array_like or scipy sparse matrix, optional
        Individual data used to calculate only the required correlation blocks.
    Xcorr : array_like, optional
        Correlation matrix, mutually exclusive with X.
    coverage : float, default 0.95
        Target posterior mass.
    min_abs_corr, median_abs_corr : float or None
        Minimum and median purity thresholds, combined with OR.
    dedup : bool, default True
        Remove duplicate sets before purity filtering.
    squared : bool, default False
        Report squared rather than absolute correlation summaries.
    check_symmetric : bool, default True
        Warn and symmetrize asymmetric Xcorr input.
    n_purity : number or 'auto', default 'auto'
        Resource-aware cap for X-based purity; Xcorr always uses all pairs.
    cs_extension_corr : float, optional
        Include proxies with absolute correlation strictly above this threshold.
    seed : int, optional
        Local RNG seed for X-based purity subsampling.

    Returns
    -------
    dict
        Sets, attained coverage, purity and original zero-based component IDs.
    """
    if X is not None and Xcorr is not None:
        raise ValueError("Only one of X or Xcorr should be specified")
    if isinstance(res, SusieResult) or hasattr(res, "alpha"):
        alpha, variances, null_index = res.alpha, res.V, res.null_index
    else:
        alpha = np.asarray(res, dtype=np.float64)
        if alpha.ndim != 2:
            raise ValueError("alpha must be a two-dimensional matrix")
        variances, null_index = np.ones(len(alpha)), None
    if Xcorr is not None:
        Xcorr = np.asarray(Xcorr, dtype=float)
        if Xcorr.shape != (alpha.shape[1], alpha.shape[1]):
            raise ValueError("Xcorr shape must match the alpha columns")
        if not np.isfinite(Xcorr).all():
            raise ValueError("Xcorr must contain finite values")
        if check_symmetric and not np.array_equal(Xcorr, Xcorr.T):
            warnings.warn(
                "Xcorr is not symmetric; averaging with its transpose", RuntimeWarning, stacklevel=2
            )
            Xcorr = (Xcorr + Xcorr.T) / 2
    elif X is None:
        warnings.warn(
            "Neither X nor Xcorr was provided; purity filtering is skipped",
            RuntimeWarning,
            stacklevel=2,
        )
    return credible_sets(
        alpha,
        variances,
        Xcorr,
        coverage=coverage,
        min_abs_corr=min_abs_corr,
        median_abs_corr=median_abs_corr,
        cs_extension_corr=cs_extension_corr,
        null_index=null_index,
        n_purity=n_purity,
        X=X,
        dedup=dedup,
        squared=squared,
        seed=seed,
        disable_without_threshold=False,
    )


def susie_get_cs_attainable(
    res: SusieResult,
    *,
    coverage: float = 0.95,
    ethres: float | None = None,
    dedup: bool = True,
    X: Any = None,
    Xcorr: ArrayLike | None = None,
    min_abs_corr: float | None = 0.5,
    squared: bool = False,
    check_symmetric: bool = True,
    n_purity: int | float | Literal["auto"] = "auto",
    median_abs_corr: float | None = None,
    cs_extension_corr: float | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Return credible sets retained by attainable coverage and entropy.

    Parameters
    ----------
    res : SusieResult
        Fitted model.
    coverage : float, default 0.95
        Strict minimum attainable probability mass and target CS mass.
    ethres : float, optional
        Effective-number-of-variables entropy threshold; defaults to
        ``max(100, 0.1 * p)``. Positive infinity disables the entropy filter.
    dedup : bool, default True
        Remove duplicate selected sets.
    X, Xcorr : array_like, optional
        Accepted for compatibility and ignored; attainable sets use no LD.
    min_abs_corr, squared, check_symmetric, n_purity, median_abs_corr
        Passed to :func:`susie_get_cs` without any correlation data.
    cs_extension_corr, seed
        Compatibility controls passed to :func:`susie_get_cs`; no LD extension
        or purity subsampling occurs without correlation data.

    Returns
    -------
    dict
        Unfiltered-by-LD sets with original component IDs.
    """
    p = res.alpha.shape[1]
    threshold = max(100, 0.1 * p) if ethres is None else float(ethres)
    if np.isnan(threshold) or threshold <= 0:
        raise ValueError("ethres must be positive and not NaN")
    attainable = res.alpha * (res.alpha == np.max(res.alpha, axis=0))
    mass = attainable.sum(axis=1)
    projection = np.divide(
        attainable, mass[:, None], out=np.zeros_like(attainable), where=mass[:, None] > 0
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.sum(np.where(projection > 0, projection * np.log(projection), 0), axis=1)
    entropy[mass == 0] = np.inf
    keep = np.flatnonzero((mass > coverage) & (entropy < np.log(threshold)))
    view = SimpleNamespace(alpha=res.alpha[keep], V=res.V[keep], null_index=res.null_index)
    out = susie_get_cs(
        view,
        X=None,
        Xcorr=None,
        coverage=coverage,
        min_abs_corr=min_abs_corr,
        dedup=dedup,
        squared=squared,
        check_symmetric=check_symmetric,
        n_purity=n_purity,
        median_abs_corr=median_abs_corr,
        cs_extension_corr=cs_extension_corr,
        seed=seed,
    )
    out["cs_index"] = keep[out["cs_index"]]
    return out


def get_cs_correlation(
    model: SusieResult,
    *,
    X: Any = None,
    Xcorr: ArrayLike | None = None,
    max: bool = False,
) -> float | NDArray[np.float64]:
    """Correlate the maximum-PIP variable from each credible set.

    Parameters
    ----------
    model : SusieResult
        Model with credible sets.
    X, Xcorr : array_like, optional
        Supply exactly one source of correlations.
    max : bool, default False
        Return maximum absolute off-diagonal correlation instead of the matrix.

    Returns
    -------
    float or ndarray
        Correlations, or NaN when fewer than two credible sets are available.
    """
    sets = model.sets.get("cs", [])
    if len(sets) < 2:
        return float("nan")
    if (X is None) == (Xcorr is None):
        raise ValueError("Supply exactly one of X or Xcorr")
    selected = np.array([members[np.argmax(model.pip[members])] for members in sets])
    if Xcorr is None:
        matrix = _design_correlation(X, selected, selected)
        np.fill_diagonal(matrix, 1.0)
    else:
        correlation = np.asarray(Xcorr, dtype=float)
        if not np.array_equal(correlation, correlation.T):
            warnings.warn(
                "Xcorr is not symmetric; averaging with its transpose", RuntimeWarning, stacklevel=2
            )
            correlation = (correlation + correlation.T) / 2
        matrix = correlation[np.ix_(selected, selected)]
    return float(np.max(np.abs(matrix[np.triu_indices(len(selected), 1)]))) if max else matrix


def susie_init_coef(
    coef_index: ArrayLike,
    coef_value: ArrayLike,
    p: int,
) -> dict[str, NDArray[np.float64]]:
    """Create a single-effect initialization from nonzero coefficients.

    Parameters
    ----------
    coef_index : array_like of int
        Zero-based variable indices, one per effect.
    coef_value : array_like
        Nonzero initial coefficients on the working predictor scale.
    p : int
        Number of predictor columns, including a null column if applicable.

    Returns
    -------
    dict
        ``alpha``, ``mu`` and ``mu2`` arrays suitable for ``model_init``.
    """
    indices = np.asarray(coef_index)
    values = np.asarray(coef_value, dtype=float)
    if (
        indices.ndim != 1
        or not indices.size
        or values.shape != indices.shape
        or not np.issubdtype(indices.dtype, np.integer)
    ):
        raise ValueError("coef_index and coef_value must be nonempty equal-length vectors")
    if isinstance(p, bool) or int(p) != p or p < 1 or np.any((indices < 0) | (indices >= p)):
        raise ValueError("coef_index must be within the p columns")
    if not np.isfinite(values).all() or np.any(values == 0):
        raise ValueError("coef_value must contain finite nonzero values")
    alpha, mu = np.zeros((len(indices), int(p))), np.zeros((len(indices), int(p)))
    alpha[np.arange(len(indices)), indices] = 1
    mu[np.arange(len(indices)), indices] = values
    return {"alpha": alpha, "mu": mu, "mu2": mu**2}


def coef(res: SusieResult) -> NDArray[np.float64]:
    """Return the intercept followed by original-scale posterior mean coefficients.

    Parameters
    ----------
    res : SusieResult
        Fitted model.

    Returns
    -------
    ndarray, shape (q + 1,)
        Intercept followed by posterior mean effects.

    Notes
    -----
    Includes every returned component and an optional null column, following
    susieR's coefficient extractor. ``res.coef`` remains the historical
    predictor-only accessor and excludes the null column.
    """
    intercept = res.intercept
    if not np.isfinite(intercept):
        warnings.warn("Intercept is missing; returning 0", RuntimeWarning, stacklevel=2)
        intercept = 0.0
    effects = np.sum(res.alpha * res.mu, axis=0) / res.X_column_scale_factors
    return np.r_[intercept, effects]


def predict(
    res: SusieResult,
    newx: Any = None,
    *,
    type: Literal["response", "coefficients"] = "response",
) -> NDArray[np.float64] | None:
    """Predict responses from new observations or extract coefficients.

    Parameters
    ----------
    res : SusieResult
        Fitted model.
    newx : array_like or scipy sparse matrix, optional
        Observations with the same predictor columns as the original data.
        Omit to return stored fitted values, when available.
    type : {'response', 'coefficients'}, default 'response'
        Requested output. Do not pass newx for coefficients.

    Returns
    -------
    ndarray or None
        Predicted responses, coefficients including intercept, or stored fitted
        values. Summary-statistic fits do not contain fitted responses.
    """
    if type not in ("response", "coefficients"):
        raise ValueError("type must be 'response' or 'coefficients'")
    if type == "coefficients":
        if newx is not None:
            raise ValueError("Do not supply newx when predicting coefficients")
        return coef(res)
    if newx is None:
        return None if res.fitted is None else res.fitted.copy()
    if not hasattr(newx, "shape"):
        newx = np.asarray(newx)
    if len(newx.shape) != 2 or newx.shape[1] != len(res.variant_ids):
        raise ValueError("newx must have one column per original predictor")
    coefficients = coef(res)
    effects = (
        np.delete(coefficients[1:], res.null_index)
        if res.null_index is not None
        else coefficients[1:]
    )
    return np.asarray(newx @ effects).ravel() + coefficients[0]


def summary(res: SusieResult) -> dict[str, list[dict[str, Any]]]:
    """Return variable and credible-set summary records with zero-based indices.

    Parameters
    ----------
    res : SusieResult
        Fitted model.

    Returns
    -------
    dict
        Variable and credible-set tables represented by lists of records.

    Notes
    -----
    Returns ``vars`` records (variable, variable_prob, cs) and ``cs`` records
    (cs, cs_log10bf, cs_avg_r2, cs_min_r2, variable), matching susieR's fields.
    A variable not in a reported set has ``cs=-1``.
    """
    variables = [
        dict(variable=i, variable_prob=float(probability), cs=-1)
        for i, probability in enumerate(res.pip)
    ]
    credible = []
    purity = res.sets.get("purity", {})
    for position, (component, members) in enumerate(
        zip(res.sets.get("cs_index", []), res.sets.get("cs", []), strict=True)
    ):
        for member in members:
            if member < len(variables):
                variables[int(member)]["cs"] = int(component)
        mean = purity.get("mean_abs_corr", np.full(position + 1, np.nan))[position]
        minimum = purity.get("min_abs_corr", np.full(position + 1, np.nan))[position]
        credible.append(
            dict(
                cs=int(component),
                cs_log10bf=float(res.lbf[component] / np.log(10)),
                cs_avg_r2=float(mean**2),
                cs_min_r2=float(minimum**2),
                variable=",".join(map(str, members)),
            )
        )
    if credible:
        variables.sort(key=lambda row: -row["variable_prob"])
    return {"vars": variables, "cs": credible}


get_objective = susie_get_objective
