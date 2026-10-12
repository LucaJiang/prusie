"""Explicit LD diagnostics mapped to susieR 0.16.6 RSS diagnostics.

The model is z ~ N(0, (1-s) R + s I), with the upstream finite-n PVE
adjustment. Kriging uses a maximum-likelihood normal scale mixture for allele
flip likelihood ratios. These checks neither modify LD nor refit SuSiE.
"""
from __future__ import annotations

import hashlib
import math
import time
import numpy as np
from .analysis import _correlation

_REFERENCE = "stephenslab/susieR@8e56a8e038e989856d106d9ca5175cc664fea9d2:R/susie_rss_utils.R"


def diagnostic_input_missing(row):
    """Read explicit or legacy missing-input markers without guessing from zero."""
    return bool(row.get("input_missing", False) or
                ("input_z" in row and row["input_z"] is None) or
                str(row.get("unavailable_reason", "")).startswith("missing_z"))


def diagnostic_status(row):
    """Classify flip assessment, including older saved records without status.

    Finite conditional moments alone never establish an assessed flip test.
    """
    if diagnostic_input_missing(row) or row.get("status") == "not_assessed":
        return "not_assessed"
    if row.get("status") == "conditional_only":
        return "conditional_only"
    if isinstance(row.get("flag"), (bool, np.bool_)) and row.get("status") in (None, "assessed"):
        return "flagged" if row["flag"] else "not_flagged"
    return "not_assessed"


def diagnostic_counts(diagnostics):
    """Counts of completed flip tests and separately available conditional data."""
    states = [diagnostic_status(row) for row in diagnostics.get("variants", [])]
    return dict(total=len(states), assessed=states.count("flagged") + states.count("not_flagged"),
                flagged=states.count("flagged"), not_flagged=states.count("not_flagged"),
                conditional_only=states.count("conditional_only"), unassessed=states.count("not_assessed"))


def _scipy():
    try:
        from scipy import optimize, special
    except ImportError as exc:
        raise ImportError("LD diagnostics require scipy; install prusie[analysis]") from exc
    return optimize, special


def prepare_ld(R, *, r_tol=1e-8, cache=None):
    """Explicitly decompose LD, optionally reusing a caller-owned content cache.

    Small/negative eigenvalues are set to zero *for this diagnostic only*, as
    in the pinned reference. The input is unchanged. Indefinite input produces
    a warning and a recorded eigenvalue, not an silently repaired fitting LD.
    """
    start = time.perf_counter()
    raw = np.asarray(R)
    if raw.ndim != 2 or raw.shape[0] == 0:
        raise ValueError("R must be a nonempty square correlation matrix")
    R = _correlation(raw, raw.shape[0])
    if not math.isfinite(r_tol) or r_tol <= 0:
        raise ValueError("r_tol must be finite and positive")
    digest = hashlib.sha256(np.ascontiguousarray(R, dtype='<f8').tobytes()).hexdigest()
    key = f"prusie.ld.v1:{R.shape[0]}:{r_tol.hex() if isinstance(r_tol, float) else float(r_tol).hex()}:{digest}"
    hit = cache is not None and key in cache
    if hit:
        values, vectors, minimum = cache[key]
    else:
        values, vectors = np.linalg.eigh(R)
        minimum = float(values[0])
        values[values < r_tol] = 0
        values.setflags(write=False)
        vectors.setflags(write=False)
        if cache is not None:
            cache[key] = (values, vectors, minimum)
    messages = []
    if minimum < -r_tol:
        messages.append("R is not positive semidefinite; negative eigenvalues are zeroed only in this diagnostic.")
    return dict(eigenvalues=values, eigenvectors=vectors, minimum_eigenvalue=minimum,
                rank=int(np.sum(values > 0)), matrix_sha256=digest, cache_key=key,
                cache_hit=hit, r_tol=r_tol, warnings=messages,
                decomposition_seconds=time.perf_counter()-start)


def _adjust(z, n, p):
    z = np.asarray(z, dtype=float)
    if z.shape != (p,) or np.any(np.isinf(z)):
        raise ValueError("z must be a length-p vector without infinite values")
    missing = np.isnan(z)
    z = np.where(missing, 0., z)
    if n is not None and not math.isinf(n):
        if not math.isfinite(n) or n <= 1:
            raise ValueError("n must be greater than 1")
        denominator = z*z+n-2
        if np.any(denominator <= 0):
            raise ValueError("finite-n z adjustment is undefined for these n and z")
        z = z * np.sqrt((n-1) / denominator)
    elif n is not None and n < 0:
        raise ValueError("n must be greater than 1")
    return z, missing


def _estimate(z, decomposition, method):
    optimize, _ = _scipy()
    d, U = decomposition["eigenvalues"], decomposition["eigenvectors"]
    projected = U.T @ z
    if method == "null-partialmle":
        null = d == 0
        return float(np.mean(projected[null]**2)) if np.any(null) else 0.
    if method == "null-mle":
        def objective(s):
            variance = (1-s)*d+s
            with np.errstate(divide="ignore", invalid="ignore"):
                value = .5 * np.sum(np.log(variance) + projected**2 / variance)
            return float(value) if np.isfinite(value) else np.inf
    elif method == "null-pseudomle":
        def objective(s):
            precision = (U * (1/((1-s)*d+s))) @ U.T
            diagonal = np.diag(precision)
            residual = (precision @ z) / diagonal
            return float(.5*np.sum(math.log(2*math.pi) - np.log(diagonal) + residual**2*diagonal))
    else:
        raise ValueError("method must be null-mle, null-partialmle or null-pseudomle")
    result = optimize.minimize_scalar(objective, bounds=(0., 1.), method="bounded",
                                      options={"xatol": 1e-12, "maxiter": 1000})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("LD regularization likelihood optimization did not converge")
    # Like R's Brent routine, return its interior minimizer, not a clipped endpoint.
    return float(result.x)


def estimate_s_rss(z, R, n=None, *, r_tol=1e-8, method="null-mle", cache=None,
                   return_info=False):
    """Estimate upstream RSS null-model regularization (explicit O(p³) work).

    ``null-partialmle`` can exceed one. Missing z is replaced by zero, as in
    susieR, and reported when ``return_info=True``. No-n uses unadjusted z.
    """
    decomposition = prepare_ld(R, r_tol=r_tol, cache=cache)
    adjusted, missing = _adjust(z, n, len(decomposition["eigenvalues"]))
    s = _estimate(adjusted, decomposition, method)
    if not return_info:
        return s
    return dict(s=s, method=method, n=n, adjusted_z=adjusted,
                missing_z_indices=np.flatnonzero(missing).tolist(), reference=_REFERENCE,
                **{k: v for k, v in decomposition.items() if not k.startswith("eigen")})


def _mixture(log_likelihood):
    """Convex likelihood simplex optimization, with an explicit KKT audit."""
    optimize, special = _scipy()
    factors = log_likelihood.max(axis=1)
    likelihood = np.exp(log_likelihood-factors[:, None])
    k = likelihood.shape[1]
    def objective(w):
        fitted = likelihood @ w
        if np.any(fitted <= 0):
            return np.inf
        return -float(np.sum(np.log(fitted)))
    def gradient(w):
        return -np.sum(likelihood / (likelihood @ w)[:, None], axis=0)
    result = optimize.minimize(objective, np.full(k, 1/k), jac=gradient,
        method="SLSQP", bounds=[(0., 1.)]*k,
        constraints={"type": "eq", "fun": lambda w: w.sum()-1,
                     "jac": lambda w: np.ones(k)},
        options={"ftol": 1e-12, "maxiter": 2000})
    w = np.maximum(result.x, 0.)
    w /= w.sum()
    g = -gradient(w)
    # At an optimum g_j <= p, with equality at positive mixture weights.
    kkt = float(max(0., np.max(g)-len(likelihood)))
    if not result.success or kkt > 1e-5:
        raise RuntimeError(f"LD mixture likelihood optimization failed: {result.message}; KKT error {kkt:g}")
    log_mix = special.logsumexp(log_likelihood + np.log(w+1e-15), axis=1)
    return w, log_mix, dict(optimizer="scipy SLSQP convex simplex likelihood", iterations=int(result.nit),
                            kkt_max_violation=kkt, converged=True)


def kriging_rss(z, R, n=None, *, s=None, r_tol=1e-8, method="null-mle",
                variant_ids=None, cache=None, flip_log_lr=2., flip_abs_z=2.):
    """Upstream conditional z and scale-mixture allele-flip diagnostics.

    Flags mean *suspected* mismatch. No allele is flipped or variant removed.
    Singular ``s=0`` uses the upstream spectral pseudoinverse; it is labelled
    explicitly and must not be mistaken for a nonsingular conditional model.
    Entirely undefined conditional variances return not_assessed rows.
    """
    start = time.perf_counter()
    _, special = _scipy()
    decomposition = prepare_ld(R, r_tol=r_tol, cache=cache)
    d, U = decomposition["eigenvalues"], decomposition["eigenvectors"]
    p = len(d)
    adjusted, missing = _adjust(z, n, p)
    ids = [str(x) for x in (variant_ids if variant_ids is not None else range(p))]
    if len(ids) != p or len(set(ids)) != p:
        raise ValueError("variant_ids must be unique and have length p")
    if not np.isfinite(flip_log_lr) or not np.isfinite(flip_abs_z) or flip_abs_z < 0:
        raise ValueError("flip thresholds must be finite and abs-z nonnegative")
    estimated = s is None
    s = _estimate(adjusted, decomposition, method) if estimated else float(s)
    requested_s = s
    messages = list(decomposition["warnings"])
    if not np.isfinite(s) or s < 0:
        raise ValueError("s must be finite and nonnegative")
    if s > 1:
        s = .8
        messages.append("Requested s > 1 replaced by 0.8, following pinned susieR.")
    variance = (1-s)*d+s
    inverse = np.divide(1., variance, out=np.zeros_like(d), where=variance != 0)
    precision = (U * inverse) @ U.T
    diagonal = np.diag(precision)
    valid = np.isfinite(diagonal) & (diagonal > 0)
    condvar = np.divide(1., diagonal, out=np.full(p, np.nan), where=valid)
    # Match explicit off-diagonal conditional formula; avoid cancellation z-Pz/Pjj.
    offdiagonal = precision.copy()
    np.fill_diagonal(offdiagonal, 0.)
    condmean = -condvar * (offdiagonal @ adjusted)
    residual = (adjusted-condmean)/np.sqrt(condvar)
    rows = [dict(variant_id=ids[j], index=j, input_z=None if missing[j] else float(np.asarray(z)[j]),
                 input_missing=bool(missing[j]), observed_z=None if missing[j] else float(adjusted[j]),
                 working_z=float(adjusted[j]), expected_z=float(condmean[j]) if valid[j] else None,
                 conditional_variance=float(condvar[j]) if valid[j] else None,
                 standardized_residual=float(residual[j]) if valid[j] and not missing[j] else None,
                 working_standardized_residual=float(residual[j]) if valid[j] else None,
                 log_likelihood_ratio=None, flag=None, status="assessed" if valid[j] and not missing[j] else "not_assessed",
                 unavailable_reason="missing_z_replaced_by_zero_upstream" if missing[j] else
                    (None if valid[j] else "nonpositive_precision_diagonal")) for j in range(p)]
    mixture = {"status": "not_assessed", "reason": "undefined_conditional_variance"}
    if np.all(valid) and np.all(np.isfinite(residual)):
        amax = 2. if np.max(residual**2) < 1 else 2*float(np.max(np.abs(residual)))
        points = int(np.ceil(np.log(amax/.8)/np.log(1.05)))
        grid = 1.05**np.arange(-points, 1)*amax
        sd = np.sqrt(condvar)[:, None]*grid
        ll0 = -.5*((adjusted-condmean)[:, None]/sd)**2-np.log(sd)-.5*math.log(2*math.pi)
        try:
            weights, log0, optimizer = _mixture(ll0)
        except RuntimeError as exc:
            mixture = dict(status="not_assessed", reason="mixture_optimization_failed", detail=str(exc), scale_grid=grid)
            messages.append(str(exc))
            for row in rows:
                if row["status"] == "assessed":
                    row.update(status="conditional_only", unavailable_reason="mixture_optimization_failed")
        else:
            ll1 = -.5*((adjusted+condmean)[:, None]/sd)**2-np.log(sd)-.5*math.log(2*math.pi)
            log1 = special.logsumexp(ll1+np.log(weights+1e-15), axis=1)
            for j, row in enumerate(rows):
                row.update(log_likelihood_ratio=float(log1[j]-log0[j]),
                           log_likelihood_original=float(log0[j]), log_likelihood_flipped=float(log1[j]),
                           flag=bool(log1[j]-log0[j] > flip_log_lr and abs(adjusted[j]) > flip_abs_z) if not missing[j] else None)
            mixture = dict(status="assessed", scale_grid=grid, weights=weights, **optimizer)
    for row in rows:
        if row["status"] == "assessed" and row["flag"] is None:
            row.update(status="conditional_only", unavailable_reason=mixture["reason"])
    counts = diagnostic_counts({"variants": rows})
    return dict(schema_version="prusie.ld.v1", method="susieR RSS kriging normal scale mixture",
                reference=_REFERENCE, regularization=s, requested_regularization=requested_s,
                regularization_method=method if estimated else "user_fixed", n=n,
                z_adjustment="sqrt((n-1)/(z²+n-2))" if n is not None and np.isfinite(n) else "none",
                variants=rows, mixture=mixture, warnings=messages, status_counts=counts,
                singular_pseudoinverse=bool(s == 0 and np.any(d == 0)),
                thresholds=dict(flip_log_lr=flip_log_lr, flip_abs_z=flip_abs_z, strict=True),
                matrix_sha256=decomposition["matrix_sha256"], cache_hit=decomposition["cache_hit"],
                minimum_eigenvalue=decomposition["minimum_eigenvalue"], rank=decomposition["rank"],
                r_tol=r_tol, decomposition_seconds=decomposition["decomposition_seconds"],
                elapsed_seconds=time.perf_counter()-start,
                limitations=["Diagnostic null model does not prove LD calibration or causal identity.",
                             "Scale-mixture weights use a full-likelihood constrained solver with KKT checks; pinned R mixsqp may stop at suboptimal boundary weights and produce different flip likelihood ratios.",
                             "Flags suggest possible allele mismatch; no automatic flip or filtering."])


def diagnose_ld(z, R, n=None, **kwargs):
    """Alias for the explicit kriging diagnostic, including regularization."""
    return kriging_rss(z, R, n, **kwargs)
