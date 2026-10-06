"""Posterior summaries mapped to pinned susieR 0.16.6 Gaussian scope."""
import numpy as np
from . import _native
from ._array_layout import native_array


def marginal_pip(alpha, V, *, null_index=None, prior_tol=1e-9):
    """Marginal inclusion probabilities, excluding small-V effects and null."""
    keep = np.asarray(V) > prior_tol
    pip = 1 - np.prod(1 - alpha[keep], axis=0)
    return np.delete(pip, null_index) if null_index is not None else pip


def credible_sets(alpha, V, correlation, *, coverage=0.95,
                  min_abs_corr=0.5, null_index=None, n_purity=None,
                  covariance_scales=None, covariance_multiplier=1.):
    """Construct deduplicated CSs with complete pairwise absolute-r purity.

    The current matrix-input reference uses complete purity regardless of
    ``n_purity``. None for coverage or min_abs_corr disables set construction,
    following the high-level sufficient-statistics route.
    Returned ``coverage[k]`` is the posterior mass of ``cs[k]`` in its original
    component ``cs_index[k]``. ``actual_coverage`` is an equal alias; the target
    threshold is recorded separately as ``requested_coverage``.
    """
    empty = dict(cs=[], cs_index=np.empty(0, dtype=np.int64),
                 coverage=np.empty(0), actual_coverage=np.empty(0),
                 requested_coverage=coverage,
                 purity={key: np.empty(0) for key in
                         ("min_abs_corr", "mean_abs_corr", "median_abs_corr")})
    if coverage is None or min_abs_corr is None:
        return empty
    candidates, seen = [], set()
    for i, row in enumerate(alpha):
        order = np.argsort(-row, kind="stable")
        count = min(len(row), int(np.sum(np.cumsum(row[order]) < coverage)) + 1)
        members = np.sort(order[:count]).astype(np.int64)
        key = tuple(members)
        duplicate = key in seen
        seen.add(key)
        if V[i] > 1e-9 and len(members) and not duplicate:
            candidates.append((i, members))
    retained = []
    native_correlation = None
    for i, members in candidates:
        if null_index is not None and null_index in members:
            purity = (-9., -9., -9.)
        elif len(members) == 1:
            purity = (1., 1., 1.)
        else:
            inv = None
            if covariance_scales is not None:
                scales = np.asarray(covariance_scales)[members]
                inv = np.divide(1., scales, out=np.zeros_like(scales), where=scales != 0)
                # Preserve the original reciprocal dtype/rounding, then satisfy
                # the native f64 boundary (a no-op for ordinary fit outputs).
                inv = np.asarray(inv, dtype=np.float64)
            if native_correlation is None:
                native_correlation = native_array(np.asarray(correlation, dtype=np.float64))
            values = _native.cs_pairwise_abs(
                native_correlation, members, inv,
                min_abs_corr, covariance_multiplier)
            if values is None:
                continue
            minimum = float(values.min())
            # Rejected sets have no returned purity row; avoid their unused
            # mean and median while retaining the same minimum-r decision.
            if min_abs_corr is not None and not minimum >= min_abs_corr:
                continue
            purity = (minimum, float(values.mean()), float(np.median(values)))
        if purity[0] >= (min_abs_corr if min_abs_corr is not None else 0):
            retained.append((i, members, purity))
    if not retained:
        return empty
    order = np.argsort([-item[2][0] for item in retained], kind="stable")
    result = [retained[i] for i in order]
    # Sum only after selection and sorting, using original component identities.
    mass = np.array([alpha[i, members].sum() for i, members, _ in result])
    return dict(cs=[x[1] for x in result],
                cs_index=np.array([x[0] for x in result], dtype=np.int64),
                coverage=mass,
                actual_coverage=mass,
                requested_coverage=coverage,
                purity={key: np.array([x[2][j] for x in result]) for j, key in
                        enumerate(("min_abs_corr", "mean_abs_corr", "median_abs_corr"))})
