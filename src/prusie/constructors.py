"""Adaptive component selection and trend-filtering entry points.

Algorithms follow susieR 0.16.6, with its BSD-3-Clause attribution retained.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .individual import SusieInputs, susie
from .result import SusieResult


def _expanded_state(fit: SusieResult, additional: int, sigma2: float) -> dict[str, Any]:
    p = fit.alpha.shape[1]
    return dict(
        alpha=np.vstack((fit.alpha, np.full((additional, p), 1 / p))),
        mu=np.vstack((fit.mu, np.zeros((additional, p)))),
        mu2=np.vstack((fit.mu2, np.zeros((additional, p)))),
        V=np.r_[fit.V, np.ones(additional)],
        sigma2=sigma2,
        sets=fit.sets,
    )


def susie_auto(
    X: Any,
    y: ArrayLike,
    *,
    L_init: int = 1,
    L_max: int = 512,
    verbose: bool = False,
    init_tol: float = 1.0,
    standardize: bool = True,
    intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-2,
    **kwargs: Any,
) -> SusieResult:
    """Select the number of single effects by repeated component doubling.

    Parameters
    ----------
    X, y : array_like
        Individual predictor matrix and response vector.
    L_init : int, default 1
        Initial component count.
    L_max : int, default 512
        Stop growing after a round starts above this count, following susieR.
    verbose : bool, default False
        Log attempted component counts.
    init_tol : float, default 1
        Tolerance during staged variance estimation.
    standardize, intercept : bool
        Predictor scaling and centering controls.
    max_iter : int, default 100
        Maximum sweeps per stage.
    tol : float, default 0.01
        Final fit tolerance.
    **kwargs
        Additional options passed to :func:`susie`. ``init_only=True`` is
        unsupported because each stage requires a fitted posterior.

    Returns
    -------
    SusieResult
        Final model after staged fitting and a final tolerance refinement.
    """
    from ._validation import _integer

    count, maximum = _integer(L_init, "L_init"), _integer(L_max, "L_max")
    response = np.asarray(y, dtype=float)
    low_variance = 0.01 * float(np.var(response, ddof=1))
    controls = dict(
        standardize=standardize, intercept=intercept, max_iter=max_iter, verbose=verbose, **kwargs
    )
    if verbose:
        logging.getLogger("prusie").info("Trying L=%d", count)
    first = susie(
        X,
        y,
        L=count,
        residual_variance=low_variance,
        tol=init_tol,
        scaled_prior_variance=1,
        estimate_residual_variance=False,
        estimate_prior_variance=False,
        **controls,
    )
    second = susie(
        X,
        y,
        L=first.alpha.shape[0],
        model_init=first,
        tol=init_tol,
        estimate_residual_variance=True,
        estimate_prior_variance=False,
        **controls,
    )
    fit = susie(
        X,
        y,
        L=second.alpha.shape[0],
        model_init=second,
        tol=init_tol,
        estimate_residual_variance=True,
        estimate_prior_variance=True,
        **controls,
    )
    while np.all(fit.V > 0) and count <= maximum:
        initial = _expanded_state(fit, count, low_variance)
        count *= 2
        if verbose:
            logging.getLogger("prusie").info("Trying L=%d", count)
        first = susie(
            X,
            y,
            L=len(initial["alpha"]),
            model_init=initial,
            tol=init_tol,
            estimate_residual_variance=False,
            estimate_prior_variance=False,
            **controls,
        )
        second = susie(
            X,
            y,
            L=first.alpha.shape[0],
            model_init=first,
            tol=init_tol,
            estimate_residual_variance=True,
            estimate_prior_variance=False,
            **controls,
        )
        fit = susie(
            X,
            y,
            L=second.alpha.shape[0],
            model_init=second,
            tol=init_tol,
            estimate_residual_variance=True,
            estimate_prior_variance=True,
            **controls,
        )
    return susie(
        X,
        y,
        L=fit.alpha.shape[0],
        model_init=fit,
        tol=tol,
        estimate_residual_variance=True,
        estimate_prior_variance=True,
        **controls,
    )


def susie_trendfilter(
    y: ArrayLike,
    *,
    order: int = 0,
    standardize: bool = False,
    use_mad: bool = True,
    **kwargs: Any,
) -> SusieResult | SusieInputs:
    """Fit piecewise polynomial trends using SuSiE's trend-filter design.

    Parameters
    ----------
    y : array_like
        Ordered response observations.
    order : int, default 0
        Nonnegative trend order; order zero gives piecewise constant trends.
    standardize : bool, default False
        Scale trend-design columns by their sample standard deviations.
    use_mad : bool, default True
        Start with a fixed variance estimated from adjacent differences, then
        warm-start the requested variance-estimating fit.
    **kwargs
        Additional options passed to :func:`susie`.

    Returns
    -------
    SusieResult or SusieInputs
        Fitted trend model, or prepared inputs with ``init_only=True``.
        With MAD initialization, the prepared result retains the first
        preparation in ``params["model_init"]``. This nested object is not a
        fitted posterior, so :func:`susie_workhorse` rejects it.
    """
    if isinstance(order, bool) or int(order) != order or order < 0:
        raise ValueError("order must be a nonnegative integer")
    if order > 0:
        warnings.warn("order > 0 is not recommended", RuntimeWarning, stacklevel=2)
    response = np.asarray(y, dtype=float)
    if response.ndim != 1 or len(response) < 2 or not np.isfinite(response).all():
        raise ValueError("y must be a finite vector with at least two observations")
    from ._trend import TrendDesign

    design = TrendDesign(len(response), int(order))
    controls = dict(kwargs)
    if use_mad and "model_init" not in controls and "s_init" not in controls:
        variance = 0.5 * float(np.median(np.abs(np.diff(response)) / 0.6745) ** 2)
        if variance == 0:
            raise ValueError("MAD residual variance is zero; set use_mad=False")
        initial_controls = {
            **controls,
            "estimate_residual_variance": False,
            "residual_variance": variance,
        }
        initial = susie(design, response, standardize=standardize, **initial_controls)
        if isinstance(initial, SusieInputs):
            prepared = susie(design, response, standardize=standardize, **controls)
            assert isinstance(prepared, SusieInputs)
            params = {**prepared.params, "model_init": initial}
            request = replace(prepared._request, params=params, deferred_model_init=initial)
            return replace(prepared, params=params.copy(), _request=request)
        controls["model_init"] = initial
    return susie(design, response, standardize=standardize, **controls)
