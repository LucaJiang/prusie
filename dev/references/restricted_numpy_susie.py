"""Restricted independent Gaussian SuSiE reference for developer checks.

Adapted from the user-supplied numpy_reference_susie.py (preserved separately).
This is not a supported solver: no check_prior, null column, warm starts,
refinement, missing n, variance bounds, alternative convergence or zero diagonal.
Only positive prior weights and moderate finite inputs are admitted. It uses
exact normalized priors, without the pinned R implementation's sqrt(eps) term;
that small known likelihood difference is retained in comparisons. No epsilon
floor modifies sufficient statistics or sampling variances. Optim uses SciPy's
bounded search, not R's optimizer; compare numerically, never assume identity.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


def _lbf(betahat, shat2, v):
    if not np.all(np.isfinite(shat2)) or np.any(shat2 <= np.finfo(float).eps):
        raise ValueError("Reference excludes sampling variances at or below machine epsilon")
    return (-0.5 * np.log1p(v / shat2)
            + 0.5 * (betahat / np.sqrt(shat2))**2 * (v / (v + shat2)))


def susie_ss_ref(XtX, Xty, yty, n, L=10, scaled_prior_variance=0.2,
                 residual_variance=None, estimate_residual_variance=True,
                 estimate_prior_variance=True, method="optim", max_iter=100,
                 tol=1e-4, prior_weights=None, standardize=True,
                 prior_tol=1e-9, check_null_threshold=0.0):
    XtX = np.asarray(XtX, float); Xty = np.asarray(Xty, float)
    p = Xty.size
    if Xty.ndim != 1 or not 1 <= p <= 2000 or XtX.shape != (p, p):
        raise ValueError("Reference requires a vector and matching dense matrix, p<=2000")
    if not np.all(np.isfinite(XtX)) or not np.all(np.isfinite(Xty)):
        raise ValueError("Reference rejects nonfinite inputs")
    if (not np.allclose(XtX, XtX.T, atol=1e-12, rtol=1e-12)
            or np.any(np.diag(XtX) <= 1e-12)
            or max(np.max(np.abs(XtX)), np.max(np.abs(Xty))) > 1e12):
        raise ValueError("Reference rejects asymmetric, zero-diagonal or extreme inputs")
    if np.linalg.eigvalsh(XtX)[0] < -1e-10 * np.max(np.diag(XtX)):
        raise ValueError("Reference requires a positive semidefinite crossproduct matrix")
    scalar_values = (yty, n, scaled_prior_variance, tol, prior_tol, check_null_threshold)
    if not np.isfinite(scalar_values).all() or not 1e-12 <= yty <= 1e12 or not 3 <= n <= 1e9:
        raise ValueError("Reference requires moderate finite scalars and n>=3")
    if (not 0 <= scaled_prior_variance <= (1 if standardize else 1e6)
            or tol <= 0 or prior_tol < 0 or not isinstance(L, int) or L < 1
            or not isinstance(max_iter, int) or max_iter < 1):
        raise ValueError("Unsupported reference parameter range")
    if method not in ("optim", "EM", "simple"):
        raise ValueError("Reference supports only optim, EM, simple or fixed V")
    var_y = yty / (n - 1)
    csd = np.sqrt(np.diag(XtX) / (n - 1)) if standardize else np.ones(p)
    csd[csd == 0] = 1
    A = XtX / csd[:, None] / csd[None, :]
    b = Xty / csd
    d = np.diag(A).copy()
    pi = np.full(p, 1 / p) if prior_weights is None else np.asarray(prior_weights, float)
    if pi.shape != (p,) or not np.isfinite(pi).all() or np.any(pi <= 0) or not np.isfinite(pi.sum()):
        raise ValueError("Reference requires finite strictly positive prior weights")
    pi = pi / pi.sum()
    if np.min(pi) < 1e-8:
        raise ValueError("Reference rejects extreme normalized prior weights")
    logpi = np.log(pi)
    sigma2 = var_y if residual_variance is None else residual_variance
    if not np.isfinite(sigma2) or not 1e-12 <= sigma2 <= 1e12:
        raise ValueError("Reference requires moderate positive residual variance")
    L = min(L, p)
    V = np.full(L, scaled_prior_variance * var_y)
    alpha = np.full((L, p), 1 / p); mu = np.zeros((L, p)); mu2 = np.zeros((L, p))
    lbf = np.zeros(L); lbfv = np.zeros((L, p)); KL = np.zeros(L)
    XtXr = np.zeros(p)
    elbos = []; prev = -np.inf; converged = False
    for it in range(max_iter):
        for component in range(L):
            XtXr -= A @ (alpha[component] * mu[component])
            r = b - XtXr
            bh = r / d; s2 = sigma2 / d
            def ll(v, bh=bh, s2=s2):
                return logsumexp(_lbf(bh, s2, v) + logpi)
            v = V[component]
            if estimate_prior_variance and method == "optim":
                res = minimize_scalar(lambda lv, ll=ll: -ll(np.exp(lv)), bounds=(-30, 15),
                                      method="bounded", options={"xatol": 1e-10})
                cand = np.exp(res.x)
                if -ll(cand) <= -ll(v):
                    v = cand
            if (estimate_prior_variance and method in ("optim", "simple")
                    and ll(0.0) + check_null_threshold >= ll(v)):
                v = 0.0
            lb = _lbf(bh, s2, v)
            lpo = lb + logpi
            m = lpo.max(); w = np.exp(lpo - m)
            alpha[component] = w / w.sum()
            lbf[component] = m + np.log(w.sum()); lbfv[component] = lb
            pv = v * s2 / (v + s2)
            mu[component] = pv / s2 * bh; mu2[component] = pv + mu[component] ** 2
            if estimate_prior_variance and method == "EM":
                v = np.sum(alpha[component] * mu2[component])
            V[component] = v
            KL[component] = -lbf[component] - 0.5 * np.sum((-2 * alpha[component] * mu[component] * bh + alpha[component] * mu2[component]) / s2)
            XtXr += A @ (alpha[component] * mu[component])
        B = alpha * mu
        bbar = B.sum(0)
        erss = (yty - 2 * bbar @ b + bbar @ A @ bbar
                - np.sum((B @ A) * B) + np.sum(d * (alpha * mu2)))
        elbo = -n / 2 * np.log(2 * np.pi * sigma2) - erss / (2 * sigma2) - KL.sum()
        if not np.isfinite(elbo):
            raise ValueError("Objective left the restricted finite reference domain")
        elbos.append(elbo)
        if it > 0 and 0 <= elbo - prev < tol:
            converged = True
            break
        prev = elbo
        if estimate_residual_variance:
            sigma2 = erss / n
            if not np.isfinite(sigma2) or not 1e-12 <= sigma2 <= 1e12:
                raise ValueError("Residual variance left the restricted reference domain")
    keep = prior_tol <= V
    for component in np.where(~keep)[0]:
        V[component] = 0; alpha[component] = pi; mu[component] = 0; mu2[component] = 0; lbfv[component] = 0; lbf[component] = 0; KL[component] = 0
    act = prior_tol < V
    pip = 1 - np.prod(1 - alpha[act], axis=0) if act.any() else np.zeros(p)
    return dict(alpha=alpha, mu=mu, mu2=mu2, V=V, sigma2=sigma2, elbo=np.array(elbos),
                niter=len(elbos), converged=converged, pip=pip, lbf=lbf, KL=KL,
                lbf_variable=lbfv)


def susie_rss_ref(z, R, n, **kw):
    z = np.asarray(z, float)
    if not np.isfinite(z).all() or np.max(np.abs(z)) > 1e6 or not np.isfinite(n) or n < 3:
        raise ValueError("Reference rejects nonfinite/extreme RSS inputs or n<3")
    R = np.asarray(R, float)
    if R.shape != (z.size, z.size) or not np.allclose(np.diag(R), 1, atol=1e-12, rtol=0):
        raise ValueError("Reference requires a unit-diagonal correlation matrix")
    adj = (n - 1) / (z**2 + n - 2)
    zt = np.sqrt(adj) * z
    XtX = (n - 1) * np.asarray(R, float)
    Xty = np.sqrt(n - 1) * zt
    kw.setdefault("estimate_residual_variance", False)
    kw.setdefault("max_iter", 50)
    return susie_ss_ref(XtX, Xty, n - 1, n, **kw)


def credible_sets_ref(alpha, V, R, coverage=0.95, min_abs_corr=0.5):
    if not 0 < coverage <= 1 or not 0 <= min_abs_corr <= 1:
        raise ValueError("Invalid credible-set coverage or correlation threshold")
    out = []
    for component in range(alpha.shape[0]):
        if V[component] <= 1e-9:
            continue
        order = np.argsort(-alpha[component], kind="stable")
        k = int(np.searchsorted(np.cumsum(alpha[component][order]), coverage) + 1)
        members = np.sort(order[:k])
        sub = np.abs(R[np.ix_(members, members)])
        purity = sub.min()
        if purity >= min_abs_corr:
            out.append((component, members.tolist(), float(alpha[component][members].sum()), purity))
    # susieR also drops duplicate sets (same members)
    seen, uniq = set(), []
    for cs in out:
        key = tuple(cs[1])
        if key not in seen:
            seen.add(key); uniq.append(cs)
    return uniq
