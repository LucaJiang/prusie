"""Summary-statistic and predictor-order helpers from susieR (BSD-3-Clause)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .result import SusieResult


def absolute_order(beta: ArrayLike) -> NDArray[np.int64]:
    """Return zero-based predictor indices by decreasing absolute coefficient.

    Parameters
    ----------
    beta : array_like, shape (p,)
        Predictor coefficients.

    Returns
    -------
    ndarray of int, shape (p,)
        Stable descending order of absolute values.
    """
    values = np.asarray(beta, dtype=float)
    if values.ndim != 1:
        raise ValueError("beta must be a vector")
    return np.argsort(-np.abs(values), kind="stable")


def univar_order(X: Any, y: ArrayLike) -> NDArray[np.int64]:
    """Rank predictors by absolute uncentered marginal regression coefficient.

    Parameters
    ----------
    X : array_like or scipy.sparse matrix, shape (n, p)
        Predictor observations.
    y : array_like, shape (n,)
        Response observations.

    Returns
    -------
    ndarray of int, shape (p,)
        Stable descending order of absolute univariate coefficients.
    """
    from .individual import _validated_design

    X, response = _validated_design(X, y, na_rm=False)
    squared = (
        np.asarray(X.multiply(X).sum(axis=0)).ravel()
        if hasattr(X, "multiply")
        else np.sum(X**2, axis=0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        coefficients = np.asarray(X.T @ response).ravel() / squared
    return absolute_order(coefficients)


def path_order(fit: Any) -> NDArray[np.int64]:
    """Rank predictors by first nonzero coefficient along a regularization path.

    Parameters
    ----------
    fit : array_like, mapping, or object
        A coefficient matrix, a ``coefficients`` mapping entry, or a ``coef()``
        method. The first row is an intercept; columns follow decreasing penalty.

    Returns
    -------
    ndarray of int
        Zero-based predictor order. Unselected predictors are appended in input
        order, and predictors first selected together retain input order.
    """
    if isinstance(fit, Mapping):
        values = fit["coefficients"]
    elif hasattr(fit, "coef"):
        values = fit.coef() if callable(fit.coef) else fit.coef
    else:
        values = fit
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("The coefficient path must be a matrix including an intercept row")
    selected: list[int] = []
    for column in matrix[1:].T:
        selected.extend(index for index in np.flatnonzero(column) if index not in selected)
    selected.extend(index for index in range(matrix.shape[0] - 1) if index not in selected)
    return np.asarray(selected, dtype=np.int64)


def is_symmetric_matrix(x: Any) -> bool:
    """Return whether a real square matrix equals its transpose exactly.

    Parameters
    ----------
    x : array_like or scipy.sparse matrix
        Candidate matrix.

    Returns
    -------
    bool
        Whether x is square and exactly symmetric, with matching NaNs allowed.
    """
    shape = np.shape(x)
    if len(shape) != 2 or shape[0] != shape[1]:
        return False
    if hasattr(x, "nnz"):
        difference = x - x.T
        return bool(difference.nnz == 0 or np.all(difference.data == 0))
    values = np.asarray(x)
    return bool(np.array_equal(values, values.T, equal_nan=True))


def compute_marginal_bhat_shat(
    X: Any,
    Y: ArrayLike,
    *,
    predictor_weights: ArrayLike | None = None,
    sigma2: float | ArrayLike | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Compute uncentered marginal effects and their residual standard errors.

    Parameters
    ----------
    X : array_like or scipy.sparse matrix, shape (n, p)
        Predictor observations. Center or standardize before calling if needed.
    Y : array_like, shape (n,) or (n, t)
        One or more responses.
    predictor_weights : array_like, shape (p,), optional
        Column sums of squares; computed from X by default.
    sigma2 : float or array_like, optional
        Known residual variance. When omitted, use the sample variance of each
        marginal residual divided by ``n - 1``, following susieR.

    Returns
    -------
    dict
        ``Bhat`` and ``Shat`` matrices with shape (p, t).
    """
    from .individual import _column_stats, _validated_design

    response = np.asarray(Y, dtype=float)
    if response.ndim == 1:
        response = response[:, None]
    if response.ndim != 2 or not np.isfinite(response).all():
        raise ValueError("Y must be a finite vector or response matrix")
    X, _ = _validated_design(X, response[:, 0], na_rm=False)
    n, p = X.shape
    means, centered_ss = _column_stats(X)
    weights = (
        centered_ss + n * means**2
        if predictor_weights is None
        else np.asarray(predictor_weights, dtype=float)
    )
    if weights.shape != (p,) or np.any(weights <= 0) or not np.isfinite(weights).all():
        raise ValueError("predictor_weights must be positive, finite, and match X columns")
    cross = np.asarray(X.T @ response)
    bhat = cross / weights[:, None]
    if sigma2 is not None:
        known = np.asarray(sigma2, dtype=float)
        if not np.isfinite(known).all() or np.any(known <= 0):
            raise ValueError("sigma2 must be positive and finite")
        shat = np.broadcast_to(np.sqrt(known / weights[:, None]), bhat.shape).copy()
    else:
        response_centered = response - response.mean(axis=0)
        xy_centered = cross - n * means[:, None] * response.mean(axis=0)
        residual_ss = (
            np.sum(response_centered**2, axis=0)
            - 2 * bhat * xy_centered
            + bhat**2 * centered_ss[:, None]
        )
        shat = np.sqrt(np.maximum(residual_ss / (n - 1), 1e-64)) / np.sqrt(n - 1)
    return dict(Bhat=bhat, Shat=shat)


def block_coordinate_ascent(
    model: SusieResult,
    data: Any,
    step_fn: Callable[[SusieResult, Any, int], Mapping[str, Any]],
    *,
    max_iter: int = 100,
    tol: float = 1e-3,
    verbose: bool = False,
) -> SusieResult:
    """Apply objective-monotone block updates to an existing SuSiE result.

    Parameters
    ----------
    model : SusieResult
        Initial posterior with a finite final objective.
    data : object
        Caller data passed unchanged to each update unless replaced by the callback.
    step_fn : callable
        Called as ``step_fn(model, data, iteration)`` with iterations starting at
        one. Return a mapping with ``model`` and optional ``data``, ``converged``
        and ``log_msg`` entries. A decrease rejects the update and stops.
    max_iter : int, default 100
        Maximum block updates.
    tol : float, default 0.001
        Relative objective-improvement convergence threshold.
    verbose : bool, default False
        Log objective changes through the prusie logger.

    Returns
    -------
    SusieResult
        Last accepted posterior, with the block-ascent convergence flag.
    """
    import copy
    import logging
    import warnings

    from .api import _integer, _scalar
    from .getters import susie_get_objective
    from .result import SusieResult

    if not isinstance(model, SusieResult) or not callable(step_fn):
        raise TypeError("model must be a SusieResult and step_fn must be callable")
    max_iter, tol = _integer(max_iter, "max_iter"), _scalar(tol, "tol")
    previous_objective = float(susie_get_objective(model))
    converged = False
    for iteration in range(1, max_iter + 1):
        # Value semantics preserve the accepted state if a callback edits its input.
        previous = SusieResult(**copy.deepcopy(model.as_dict()))
        previous._prepared = model._prepared
        update = step_fn(model, data, iteration)
        model = update["model"]
        if update.get("data") is not None:
            data = update["data"]
        objective = float(susie_get_objective(model))
        change = objective - previous_objective
        if change < 0:
            model, converged = previous, True
            break
        if verbose:
            logging.getLogger("prusie").info(
                "Block ascent iteration %d: ELBO=%.6g, change=%.6g %s",
                iteration,
                objective,
                change,
                update.get("log_msg", ""),
            )
        if update.get("converged") or abs(change) / max(1, abs(objective)) < tol:
            converged = True
            break
        previous_objective = objective
    model.converged = converged
    if not converged:
        warnings.warn(
            f"Block coordinate ascent did not converge in {max_iter} iterations",
            RuntimeWarning,
            stacklevel=2,
        )
    return model
