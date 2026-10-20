"""Posterior inclusion probabilities and credible sets.

The credible-set definitions follow susieR 0.16.6 (BSD-3-Clause); attribution
is retained in THIRD_PARTY_NOTICES.md. All indices are zero-based.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _native
from ._array_layout import native_array


def marginal_pip(
    alpha: NDArray[np.float64],
    V: ArrayLike,
    *,
    null_index: int | None = None,
    prior_tol: float = 1e-9,
    slot_weights: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Return marginal inclusion probabilities over active single effects.

    Parameters
    ----------
    alpha : ndarray, shape (L, q)
        Single-effect inclusion probabilities, including an optional null column.
    V : array_like, shape (L,)
        Prior variances; only effects with variance greater than ``prior_tol``
        contribute.
    null_index : int, optional
        Zero-based null column to omit.
    prior_tol : float, default 1e-9
        Activity threshold.
    slot_weights : array_like, shape (L,), optional
        Posterior slot activity probabilities, when a slot prior was fitted.

    Returns
    -------
    ndarray
        Marginal inclusion probabilities in input variable order.
    """
    keep = np.asarray(V) > prior_tol
    probabilities = np.asarray(alpha)[keep]
    if slot_weights is not None:
        probabilities = probabilities * np.asarray(slot_weights)[keep, None]
    pip = 1 - np.prod(1 - probabilities, axis=0)
    return np.delete(pip, null_index) if null_index is not None else pip


def _purity_cap(n_purity: int | float | Literal["auto"] | None, n: int, p: int) -> int:
    if n_purity == "auto":
        budget = 512 * 1024**2
        return max(
            1,
            int(
                min(
                    p,
                    max(500, np.sqrt(2e10 / max(n, 1))),
                    budget / (16 * max(n, 1)),
                    np.sqrt(budget / 32),
                )
            ),
        )
    if n_purity is None:
        return p
    if isinstance(n_purity, (bool, np.bool_)) or not np.isscalar(n_purity):
        raise ValueError('n_purity must be "auto" or a finite number')
    try:
        value = float(n_purity)
    except (TypeError, ValueError) as error:
        raise ValueError('n_purity must be "auto" or a finite number') from error
    if not np.isfinite(value):
        raise ValueError('n_purity must be "auto" or a finite number')
    return max(1, min(p, int(np.floor(value)))) if value > 0 else p


def _design_correlation(X: Any, rows: NDArray[np.int64], columns: NDArray[np.int64]) -> np.ndarray:
    """Compute only a requested correlation block, including for sparse X."""
    left, right = X[:, rows], X[:, columns]
    n = X.shape[0]
    lm, rm = np.asarray(left.mean(axis=0)).ravel(), np.asarray(right.mean(axis=0)).ravel()
    product = left.T @ right
    if hasattr(product, "toarray"):
        product = product.toarray()
    product = np.asarray(product) - n * lm[:, None] * rm[None, :]
    if hasattr(left, "multiply"):
        lv = np.asarray(left.multiply(left).sum(axis=0)).ravel() - n * lm**2
        rv = np.asarray(right.multiply(right).sum(axis=0)).ravel() - n * rm**2
    else:
        lv = np.sum((np.asarray(left) - lm) ** 2, axis=0)
        rv = np.sum((np.asarray(right) - rm) ** 2, axis=0)
    denominator = np.sqrt(np.maximum(lv, 0))[:, None] * np.sqrt(np.maximum(rv, 0))[None, :]
    return np.divide(product, denominator, out=np.zeros_like(product), where=denominator != 0)


def credible_sets(
    alpha: NDArray[np.float64],
    V: ArrayLike,
    correlation: ArrayLike | None,
    *,
    coverage: float | None = 0.95,
    min_abs_corr: float | None = 0.5,
    median_abs_corr: float | None = None,
    cs_extension_corr: float | None = None,
    null_index: int | None = None,
    n_purity: int | float | Literal["auto"] | None = None,
    covariance_scales: ArrayLike | None = None,
    covariance_multiplier: float = 1.0,
    X: Any = None,
    dedup: bool = True,
    squared: bool = False,
    seed: int | None = None,
    disable_without_threshold: bool = True,
    audit: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Construct credible sets and optionally filter or extend them by correlation.

    Parameters
    ----------
    alpha : ndarray, shape (L, q)
        Single-effect probabilities, in original component and variable order.
    V : array_like, shape (L,)
        Effects with prior variance greater than 1e-9 are active.
    correlation : array_like or None
        Correlation matrix, or covariance matrix with ``covariance_scales``.
    coverage : float or None, default 0.95
        Target mass. None disables construction.
    min_abs_corr, median_abs_corr : float or None
        Purity thresholds combined with OR when both are given.
    cs_extension_corr : float or None
        Add variables with absolute correlation strictly above this threshold
        with any initially selected member.
    null_index : int, optional
        Null column, which is never added by extension.
    n_purity : number, 'auto' or None
        Cap for purity computed from X; correlation matrices use all members.
    covariance_scales : array_like, optional
        Square roots of covariance diagonal entries.
    covariance_multiplier : float, default 1
        Scalar applied before covariance normalization.
    X : array_like or scipy sparse matrix, optional
        Individual observations, used without forming a full correlation matrix.
    dedup : bool, default True
        Remove repeated sets before purity filtering and extension.
    squared : bool, default False
        Summarize squared instead of absolute correlations.
    seed : int, optional
        Local random seed for subsampling X-based purity calculations.
    disable_without_threshold : bool, default True
        Fitting entry points disable CS construction if both thresholds are None;
        standalone getters may request unfiltered sets with False.
    audit : list of dict, optional
        Append decisions for each original component.

    Returns
    -------
    dict
        Credible sets, original component indices, attained coverage and purity.
        Empty results contain empty arrays; component and variable indices are
        zero-based. ``coverage`` and ``actual_coverage`` share the same array.
    """
    keys = ("min_abs_corr", "mean_abs_corr", "median_abs_corr")
    empty = dict(
        cs=[],
        cs_index=np.empty(0, dtype=np.int64),
        coverage=np.empty(0),
        actual_coverage=np.empty(0),
        requested_coverage=coverage,
        purity={key: np.empty(0) for key in keys},
    )
    if coverage is None or (
        disable_without_threshold and min_abs_corr is None and median_abs_corr is None
    ):
        if audit is not None:
            audit.extend(
                dict(
                    component_id=i,
                    candidate_members=[],
                    reasons=["disabled"],
                    stages=["disabled"],
                    requested_coverage=coverage,
                    min_abs_corr=min_abs_corr,
                )
                for i in range(len(alpha))
            )
        return empty
    for name, value in (
        ("coverage", coverage),
        ("min_abs_corr", min_abs_corr),
        ("median_abs_corr", median_abs_corr),
        ("cs_extension_corr", cs_extension_corr),
    ):
        if value is not None and (
            not np.isscalar(value) or not np.isfinite(value) or not 0 <= value <= 1
        ):
            raise ValueError(f"{name} must be None or a finite number in [0, 1]")
    if X is not None and correlation is not None:
        raise ValueError("Supply only X or correlation")
    a = np.asarray(alpha, dtype=np.float64)
    variances = np.asarray(V)
    matrix = None
    inverse = None
    if covariance_scales is not None:
        scale = np.asarray(covariance_scales)
        if scale.dtype.kind != "f":
            scale = scale.astype(float)
        inverse = np.divide(1.0, scale, out=np.zeros_like(scale), where=scale != 0)
        inverse = np.asarray(native_array(inverse), dtype=np.float64)
    generator = np.random.default_rng(seed)
    seen: set[tuple[int, ...]] = set()
    retained = []
    for i, row in enumerate(a):
        order = np.argsort(-row, kind="stable")
        count = min(len(row), int(np.sum(np.cumsum(row[order]) < coverage)) + 1)
        members = np.sort(order[:count]).astype(np.int64)
        key = tuple(members)
        duplicate = dedup and key in seen
        seen.add(key)
        reasons = []
        if duplicate:
            reasons.append("duplicate")
        if not variances[i] > 1e-9:
            reasons.append("inactive")
        record = dict(
            component_id=i,
            candidate_members=members.tolist(),
            reasons=reasons,
            stages=["candidate", "deduplicate", "activity"],
            requested_coverage=coverage,
            min_abs_corr=min_abs_corr,
            active_threshold=1e-9,
            actual_coverage=float(row[members].sum()),
        )
        if audit is not None:
            audit.append(record)
        if reasons:
            continue
        if cs_extension_corr is not None and (correlation is not None or X is not None):
            if correlation is not None:
                if matrix is None:
                    matrix = native_array(np.asarray(correlation, dtype=np.float64))
                block = matrix[members, :] * covariance_multiplier
                if inverse is not None:
                    block = block * inverse[members, None] * inverse[None, :]
            else:
                block = _design_correlation(X, members[members < X.shape[1]], np.arange(X.shape[1]))
            proxies = np.flatnonzero(np.any(np.abs(block) > cs_extension_corr, axis=0))
            if null_index is not None:
                proxies = proxies[proxies != null_index]
            members = np.union1d(members, proxies).astype(np.int64)
            record["extended_members"] = members.tolist()
        record["stages"].append("purity")
        if correlation is None and X is None:
            purity = (np.nan, np.nan, np.nan)
            keep = True
        elif null_index is not None and null_index in members:
            purity, keep = (-9.0, -9.0, -9.0), False
            reasons.append("contains_null")
        elif len(members) == 1:
            purity, keep = (1.0, 1.0, 1.0), True
        else:
            if correlation is not None:
                if matrix is None:
                    matrix = native_array(np.asarray(correlation, dtype=np.float64))
                inv = None if inverse is None else inverse[members]
                threshold = min_abs_corr if median_abs_corr is None else None
                values = _native.cs_pairwise_abs(
                    matrix, members, inv, threshold, covariance_multiplier
                )
            else:
                selected = members
                cap = _purity_cap(n_purity, X.shape[0], len(members))
                if len(members) > cap:
                    selected = generator.choice(members, size=cap, replace=False)
                block = _design_correlation(X, selected, selected)
                values = np.abs(block[np.triu_indices(len(selected), 1)])
            if values is None:
                reasons.append("low_purity_or_unavailable")
                record["purity_evaluation"] = "threshold_short_circuit"
                continue
            if squared:
                values = values**2
            purity = (
                (1.0, 1.0, 1.0)
                if not len(values)
                else (float(values.min()), float(values.mean()), float(np.median(values)))
            )
            if not np.all(np.isfinite(purity)):
                reasons.append("nonfinite_purity")
                continue
            minimum = None if min_abs_corr is None else min_abs_corr ** (2 if squared else 1)
            median = None if median_abs_corr is None else median_abs_corr ** (2 if squared else 1)
            keep = (
                (minimum is not None and purity[0] >= minimum)
                or (median is not None and purity[2] >= median)
                or (minimum is None and median is None)
            )
        if keep:
            reasons.append("retained")
            record["purity"] = dict(zip(keys, purity, strict=True))
            retained.append((i, members, purity))
        else:
            reasons.append("low_purity")
    if not retained:
        return empty
    primary = 0 if min_abs_corr is not None else 2
    ordering = np.argsort([-item[2][primary] for item in retained], kind="stable")
    result = [retained[i] for i in ordering]
    mass = np.array([a[i, members].sum() for i, members, _ in result])
    return dict(
        cs=[item[1] for item in result],
        cs_index=np.array([item[0] for item in result], dtype=np.int64),
        coverage=mass,
        actual_coverage=mass,
        requested_coverage=coverage,
        purity={key: np.array([item[2][j] for item in result]) for j, key in enumerate(keys)},
    )
