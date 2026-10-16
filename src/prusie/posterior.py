"""Posterior inclusion probabilities and credible sets, following susieR 0.16.6."""

import numpy as np

from . import _native
from ._array_layout import native_array


def marginal_pip(alpha, V, *, null_index=None, prior_tol=1e-9):
    """Compute posterior inclusion probabilities from the active single effects.

    Parameters
    ----------
    alpha : ndarray, shape (L, q)
        Single-effect inclusion probabilities α_ℓj, including any null column.
    V : array_like, shape (L,)
        Prior effect variances on the working scale.
    null_index : int, optional
        Zero-based null column to omit from the result.
    prior_tol : float, default 1e-9
        Effects with V_ℓ > prior_tol are active.

    Returns
    -------
    ndarray, shape (p,)
        PIP_j = 1 - ∏_{ℓ active} (1 - α_ℓj), in input variant order. All
        entries are zero when no effect is active.
    """
    keep = np.asarray(V) > prior_tol
    pip = 1 - np.prod(1 - alpha[keep], axis=0)
    return np.delete(pip, null_index) if null_index is not None else pip


def credible_sets(
    alpha,
    V,
    correlation,
    *,
    coverage=0.95,
    min_abs_corr=0.5,
    null_index=None,
    n_purity=None,
    covariance_scales=None,
    covariance_multiplier=1.0,
    audit=None,
):
    """Construct one credible set per single effect and filter by LD purity.

    Parameters
    ----------
    alpha : ndarray, shape (L, q)
        Single-effect inclusion probabilities, including any null column.
    V : array_like, shape (L,)
        Prior effect variances; an effect is active if V_ℓ > 1e-9.
    correlation : array_like, shape (q, q)
        Signed correlation matrix, or a covariance matrix together with
        ``covariance_scales`` and ``covariance_multiplier``.
    coverage : float or None, default 0.95
        Requested posterior mass of each set. None disables construction.
    min_abs_corr : float or None, default 0.5
        Minimum absolute pairwise correlation (purity) of a retained set.
        None disables construction.
    null_index : int, optional
        Null column; a candidate set containing it is rejected.
    n_purity : int, optional
        Accepted for compatibility; purity always uses all member pairs.
    covariance_scales : array_like, shape (q,), optional
        Square roots of the covariance diagonal, used to convert covariance
        entries to correlations.
    covariance_multiplier : float, default 1
        Scalar applied to covariance entries before that conversion.
    audit : list, optional
        If given, one record per effect is appended describing each
        construction decision. The selection itself is unaffected.

    Returns
    -------
    dict
        ``cs`` (member index arrays), ``cs_index`` (original effect indices),
        ``coverage`` and ``actual_coverage`` (posterior masses, sharing one
        array for nonempty results), ``requested_coverage`` and ``purity``
        (minimum, mean and median absolute correlation). Empty results keep
        an empty ``cs`` list and empty arrays; ``requested_coverage`` retains
        the supplied target.

    Notes
    -----
    For each effect, variants are added in decreasing α_ℓj (stable ordering
    of ties) until the cumulative mass reaches ``coverage``. A membership
    identical to an earlier candidate is dropped before activity and purity
    are checked, as in susieR. Retained sets are ordered by decreasing
    minimum purity. A NaN purity fails the threshold comparison.
    """
    empty = dict(
        cs=[],
        cs_index=np.empty(0, dtype=np.int64),
        coverage=np.empty(0),
        actual_coverage=np.empty(0),
        requested_coverage=coverage,
        purity={key: np.empty(0) for key in ("min_abs_corr", "mean_abs_corr", "median_abs_corr")},
    )
    if coverage is None or min_abs_corr is None:
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
    candidates, seen, records = [], set(), {}
    for i, row in enumerate(alpha):
        order = np.argsort(-row, kind="stable")
        count = min(len(row), int(np.sum(np.cumsum(row[order]) < coverage)) + 1)
        members = np.sort(order[:count]).astype(np.int64)
        key = tuple(members)
        duplicate = key in seen
        seen.add(key)
        if audit is not None:
            reasons = []
            if duplicate:
                reasons.append("duplicate")
            if not V[i] > 1e-9:
                reasons.append("inactive")
            if not len(members):
                reasons.append("unavailable")
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
            audit.append(record)
            records[i] = record
        if V[i] > 1e-9 and len(members) and not duplicate:
            candidates.append((i, members))
    retained = []
    native_correlation = None
    for i, members in candidates:
        record = records.get(i)
        if record is not None:
            record["stages"].append("purity")
        if null_index is not None and null_index in members:
            purity = (-9.0, -9.0, -9.0)
            if record is not None:
                record["reasons"].append("contains_null")
        elif len(members) == 1:
            purity = (1.0, 1.0, 1.0)
        else:
            inv = None
            if covariance_scales is not None:
                scales = np.asarray(covariance_scales)[members]
                inv = np.divide(1.0, scales, out=np.zeros_like(scales), where=scales != 0)
                # Compute the reciprocal in the input dtype, then convert for the
                # float64 native interface (a no-op for fitted results).
                inv = np.asarray(inv, dtype=np.float64)
            if native_correlation is None:
                native_correlation = native_array(np.asarray(correlation, dtype=np.float64))
            values = _native.cs_pairwise_abs(
                native_correlation, members, inv, min_abs_corr, covariance_multiplier
            )
            if values is None:
                if record is not None:
                    record["reasons"].append("low_purity_or_unavailable")
                    record["purity_evaluation"] = "threshold_short_circuit"
                continue
            minimum = float(values.min())
            # Rejected sets report no purity, so their mean and median are not
            # computed; the decision depends on the minimum alone.
            if not minimum >= min_abs_corr:
                if record is not None:
                    record["reasons"].append(
                        "low_purity" if np.isfinite(minimum) else "unavailable"
                    )
                continue
            purity = (minimum, float(values.mean()), float(np.median(values)))
        if purity[0] >= min_abs_corr:
            retained.append((i, members, purity))
            if record is not None:
                record["reasons"].append("retained")
                record["purity"] = dict(
                    zip(("min_abs_corr", "mean_abs_corr", "median_abs_corr"), purity, strict=True)
                )
        elif record is not None:
            record["reasons"].append("low_purity")
    if not retained:
        return empty
    order = np.argsort([-item[2][0] for item in retained], kind="stable")
    result = [retained[i] for i in order]
    # Masses are summed after selection and ordering, by original effect index.
    mass = np.array([alpha[i, members].sum() for i, members, _ in result])
    return dict(
        cs=[x[1] for x in result],
        cs_index=np.array([x[0] for x in result], dtype=np.int64),
        coverage=mass,
        actual_coverage=mass,
        requested_coverage=coverage,
        purity={
            key: np.array([x[2][j] for x in result])
            for j, key in enumerate(("min_abs_corr", "mean_abs_corr", "median_abs_corr"))
        },
    )
