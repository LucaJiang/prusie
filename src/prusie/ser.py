"""Linear-time one-effect posterior from marginal association statistics.

Ported from susieR 0.16.6 (BSD-3-Clause).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from . import _native
from ._validation import _scalar, _vector
from .getters import susie_get_cs_attainable
from .posterior import marginal_pip
from .result import SusieResult


def susie_ser(
    z: ArrayLike | None = None,
    *,
    bhat: ArrayLike | None = None,
    shat: ArrayLike | None = None,
    n: float | None = None,
    var_y: float | None = None,
    prior_variance: float = 50.0,
    scaled_prior_variance: float = 0.2,
    prior_weights: ArrayLike | None = None,
    null_weight: float = 0.0,
    estimate_prior_method: str = "optim",
    check_null_threshold: float = 0.0,
    prior_tol: float = 1e-9,
    coverage: float | None = 0.95,
    ethres: float | None = None,
    variant_ids: Sequence[str] | None = None,
) -> SusieResult:
    """Fit a single effect without an LD reference in O(p) memory and time.

    Parameters
    ----------
    z : array_like, optional
        Marginal z-scores. Supply either z or both bhat and shat.
    bhat, shat : array_like, optional
        Marginal estimates and standard errors.
    n : float, optional
        GWAS sample size, enabling the Wald-to-score adjustment.
    var_y : float, optional
        Response sample variance; with n and shat enables original-scale effects.
    prior_variance, scaled_prior_variance : float
        Initial prior variance without n, or relative to response variance with n.
    prior_weights : array_like, optional
        Nonnegative variant prior weights.
    null_weight : float, default 0
        Prior mass on a virtual null variant.
    estimate_prior_method : {'optim', 'simple'}, default 'optim'
        Optimize the marginal likelihood, or compare the supplied prior variance
        with zero using the null-model threshold.
    check_null_threshold, prior_tol : float
        Null-model evidence threshold and negligible prior-variance threshold.
    coverage : float, optional
        Target attainable credible-set coverage. None disables credible sets.
    ethres : float, optional
        Effective predictor-count entropy threshold for attainable sets.
    variant_ids : sequence of str, optional
        Ordered unique variant identifiers.

    Returns
    -------
    SusieResult
        One-effect posterior, with attainable credible sets without LD filtering.
    """

    if estimate_prior_method not in ("optim", "simple"):
        raise ValueError("estimate_prior_method must be 'optim' or 'simple'")
    if z is None:
        if bhat is None or shat is None:
            raise ValueError("Supply z or both bhat and shat")
        effects = _vector(bhat, "bhat")
        errors = _vector(np.full_like(effects, shat) if np.ndim(shat) == 0 else shat, "shat")
        if effects.shape != errors.shape or np.any(errors <= 0):
            raise ValueError("bhat and positive shat must have identical shape")
        scores = effects / errors
    else:
        if bhat is not None or shat is not None:
            raise ValueError("Supply z or bhat/shat, not both")
        scores = np.array(z, dtype=float, copy=True)
        if scores.ndim != 1 or not scores.size or np.isinf(scores).any():
            raise ValueError("z must be a nonempty vector without infinite values")
        if np.isnan(scores).any():
            warnings.warn("Missing z-scores are replaced by zero", RuntimeWarning, stacklevel=2)
            scores[np.isnan(scores)] = 0
        errors = None
    p = len(scores)
    ids = np.asarray(
        [str(index) for index in range(p)] if variant_ids is None else variant_ids, dtype=str
    )
    if ids.shape != (p,) or len(np.unique(ids)) != p:
        raise ValueError("variant_ids must be unique and match z")
    prior_variance = _scalar(prior_variance, "prior_variance", lower=0, strict=True)
    scaled_prior_variance = _scalar(
        scaled_prior_variance, "scaled_prior_variance", lower=0, strict=True
    )
    check_null_threshold = _scalar(check_null_threshold, "check_null_threshold")
    prior_tol = _scalar(prior_tol, "prior_tol")
    scales = np.ones(p)
    if n is None:
        sigma2, diagonal, response, initial = 1.0, np.ones(p), scores, prior_variance
    else:
        n = _scalar(n, "n", lower=1, strict=True)
        adjustment = (n - 1) / (scores**2 + n - 2)
        scores = scores * np.sqrt(adjustment)
        sigma2 = 1.0 if var_y is None else _scalar(var_y, "var_y", lower=0, strict=True)
        diagonal = np.full(p, n - 1)
        response = np.sqrt(n - 1) * scores
        initial = scaled_prior_variance * sigma2
        if errors is not None and var_y is not None:
            scales = np.sqrt(var_y * adjustment / errors**2 / (n - 1))
            response = scores * np.sqrt(adjustment) * var_y / errors / scales
    weights = (
        np.full(p, 1 / p) if prior_weights is None else _vector(prior_weights, "prior_weights")
    )
    if weights.shape != (p,) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("prior_weights must match z, be nonnegative and have positive sum")
    null_weight = _scalar(null_weight, "null_weight")
    if null_weight >= 1:
        raise ValueError("null_weight must be less than one")
    weights = weights * (1 - null_weight)
    null_index = p if null_weight > 0 else None
    if null_index is not None:
        diagonal, response, scales, weights = (
            np.r_[diagonal, 0],
            np.r_[response, 0],
            np.r_[scales, 1],
            np.r_[weights, null_weight],
        )
    weights = weights / weights.sum()
    settings = dict(
        l=1,
        prior_variance=initial,
        residual_variance=sigma2,
        prior_weights=weights,
        estimate_prior_variance=True,
        estimate_prior_method=estimate_prior_method,
        estimate_residual_variance=False,
        check_null_threshold=check_null_threshold,
        max_iter=1,
        tol=1e-3,
        check_prior=True,
        prior_tol=prior_tol,
        null_index=null_index,
    )
    update = _native.ser_update(
        np.ascontiguousarray(response), diagonal, 1.0, 2 if n is None else n, settings
    )
    alpha, mu, mu2 = [np.atleast_2d(update[name]) for name in ("alpha", "mu", "mu2")]
    variance = np.atleast_1d(update["V"])
    kl = -update["lbf"] + np.sum(
        (2 * response * alpha[0] * mu[0] - diagonal * alpha[0] * mu2[0]) / (2 * sigma2)
    )
    result = SusieResult(
        variant_ids=ids,
        alpha=alpha,
        mu=mu,
        mu2=mu2,
        lbf_variable=np.atleast_2d(update["lbf_variable"]),
        lbf=np.atleast_1d(update["lbf"]),
        V=variance,
        sigma2=sigma2,
        pip=marginal_pip(alpha, variance, null_index=null_index, prior_tol=prior_tol),
        elbo=np.atleast_1d(update["lbf"]),
        niter=1,
        converged=True,
        sets={},
        null_index=null_index,
        params={"method": "susie_ser", "n": n, "prior_weights": weights, "prior_tol": prior_tol},
        backend_version=_native.backend_version(),
        warnings=[],
        qc={
            "p": p,
            "variant_ids_positional": variant_ids is None,
            "input_missing_policy": "reject" if errors is not None else "zero",
        },
        variant_metadata={},
        X_column_scale_factors=scales,
        intercept=np.nan,
        KL=np.atleast_1d(kl),
        XtXr=diagonal * alpha[0] * mu[0],
    )
    if coverage is not None:
        if not 0 < coverage < 1:
            raise ValueError("coverage must be between zero and one")
        result.sets = susie_get_cs_attainable(result, coverage=coverage, ethres=ethres)
    return result
