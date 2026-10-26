"""RSS-lambda likelihood with an explicit isotropic covariance component.

Ported from susieR 0.16.6 (BSD-3-Clause).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .control import SlotPrior
from .individual import SusieInputs
from .result import SusieResult


def susie_rss_lambda(
    z: ArrayLike,
    R: ArrayLike | None = None,
    n: float | None = None,
    *,
    X: Any = None,
    lambda_: float | str,
    L: int | None = 10,
    maf: ArrayLike | None = None,
    maf_thresh: float = 0.0,
    prior_variance: float = 50.0,
    residual_variance: float | None = None,
    prior_weights: ArrayLike | None = None,
    null_weight: float = 0.0,
    intercept_value: float = 0.0,
    estimate_residual_variance: bool = False,
    estimate_residual_method: str = "MLE",
    z_method: str = "wald",
    estimate_prior_variance: bool = True,
    estimate_prior_method: str = "optim",
    prior_variance_grid: ArrayLike | None = None,
    mixture_weights: ArrayLike | None = None,
    check_null_threshold: float = 0.0,
    prior_tol: float = 1e-9,
    residual_variance_lowerbound: float = 0.0,
    model_init: SusieResult | Mapping[str, Any] | None = None,
    s_init: SusieResult | Mapping[str, Any] | None = None,
    coverage: float | None = 0.95,
    min_abs_corr: float | None = 0.5,
    max_iter: int = 50,
    L_greedy: int | None = None,
    greedy_lbf_cutoff: float = 0.1,
    tol: float = 1e-3,
    convergence_method: str = "elbo",
    verbose: bool = False,
    track_fit: bool | str = False,
    check_prior: bool = True,
    check_R: bool = True,
    check_z: bool = False,
    n_purity: int | str | None = "auto",
    median_abs_corr: float | None = None,
    cs_extension_corr: float | None = None,
    r_tol: float = 1e-8,
    refine: bool = False,
    init_only: bool = False,
    slot_prior: SlotPrior | Mapping[str, Any] | None = None,
    variant_ids: Sequence[str] | None = None,
    variant_metadata: Mapping[str, ArrayLike] | None = None,
) -> SusieResult | SusieInputs:
    """Fit the RSS likelihood with covariance ``sigma2 * R + lambda_ * I``.

    Parameters
    ----------
    z : array_like, shape (p,)
        Marginal association z-scores in predictor order.
    R : array_like, shape (p, p), optional
        Signed LD matrix. Supply exactly one of ``R`` and ``X``.
    n : float, optional
        GWAS sample size for the Wald-to-score adjustment.
    X : array_like or scipy.sparse matrix, optional
        Reference genotype matrix; its centered correlation defines LD.
    lambda_ : float or {'estimate'}
        Nonnegative isotropic covariance. ``'estimate'`` uses z-score energy
        in the null space of LD. This is susieR's ``lambda`` argument.
    L : int or None, default 10
        Maximum number of single effects, capped at the number of retained
        variants after MAF filtering, excluding any added null column.
        Explicit ``None`` is accepted as an alias for 10.
    prior_variance : float, default 50
        Initial effect variance, without phenotype-variance scaling.
    residual_variance : float, optional
        Total initial variance; ``lambda_`` is subtracted before fitting.
    intercept_value : float, default 0
        Intercept returned with coefficient estimates.
    check_R, check_z : bool
        Check LD positive semidefiniteness and z-score projection, respectively.
    **options
        Other options have the same meaning as in :func:`susie_rss`.
        ``refine=True`` refits the model by credible-set block coordinate
        ascent; ``retain_data=True`` and a RefinementConfig are
        unavailable for this dynamic likelihood.

    Returns
    -------
    SusieResult
        The fitted RSS-lambda posterior. Initialization-only calls return
        :class:`SusieInputs`, accepted by :func:`susie_workhorse`.
    """
    from .api import _fit, _integer, _prepare_suff_stat
    from .individual import _validated_design
    from .posterior import _design_correlation

    if (R is None) == (X is None):
        raise ValueError("Supply exactly one of R and X")
    if estimate_residual_method != "MLE":
        raise ValueError("RSS-lambda supports estimate_residual_method='MLE' only")
    if z_method not in ("wald", "score"):
        raise ValueError("z_method must be 'wald' or 'score'")
    scores = np.array(z, dtype=float, copy=True)
    if scores.ndim != 1 or not scores.size or np.isinf(scores).any():
        raise ValueError("z must be a nonempty vector without infinite values")
    if np.isnan(scores).any():
        warnings.warn("Missing z-scores are replaced by zero", RuntimeWarning, stacklevel=2)
        scores[np.isnan(scores)] = 0
    if n is not None:
        if not np.isscalar(n) or not np.isfinite(n) or n <= 1:
            raise ValueError("n must be finite and exceed one")
        if z_method == "wald":
            scores *= np.sqrt((n - 1) / (scores**2 + n - 2))
    if X is None:
        matrix = np.asarray(R, dtype=float)
        if matrix.shape != (len(scores), len(scores)) or not np.isfinite(matrix).all():
            raise ValueError("R must be a finite square matrix matching z")
        if not np.array_equal(matrix, matrix.T):
            warnings.warn(
                "R is not symmetric; averaging its triangles", RuntimeWarning, stacklevel=2
            )
            matrix = (matrix + matrix.T) / 2
    else:
        raw, _ = _validated_design(X, np.zeros(np.shape(X)[0]), na_rm=False)
        if raw.shape[1] != len(scores):
            raise ValueError("X columns must match z")
        matrix = _design_correlation(raw, np.arange(raw.shape[1]), np.arange(raw.shape[1]))
    if maf is not None:
        frequencies = np.asarray(maf, dtype=float)
        if frequencies.shape != scores.shape or not np.isfinite(frequencies).all():
            raise ValueError("maf must be finite and match z")
        keep = frequencies > maf_thresh
        matrix, scores = matrix[np.ix_(keep, keep)], scores[keep]
        if prior_weights is not None:
            prior_weights = np.asarray(prior_weights)[keep]
        if variant_ids is not None:
            variant_ids = np.asarray(variant_ids)[keep].tolist()
        if variant_metadata is not None:
            variant_metadata = {
                key: np.asarray(value)[keep] for key, value in variant_metadata.items()
            }
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if X is None and check_R and np.any(eigenvalues < -r_tol):
        raise ValueError(
            "R must be positive semidefinite; use check_R=False to project negative eigenvalues"
        )
    positive = eigenvalues >= r_tol
    if X is None and check_z and not np.all(positive):
        projection = eigenvectors[:, ~positive].T @ scores
        if np.max(np.abs(projection), initial=0) > r_tol:
            warnings.warn(
                "z does not lie in the nonzero eigenspace of R", RuntimeWarning, stacklevel=2
            )
    if isinstance(lambda_, str):
        if lambda_ != "estimate":
            raise ValueError("lambda_ must be nonnegative or 'estimate'")
        null_scores = eigenvectors[:, ~positive].T @ scores
        ridge = float(np.mean(null_scores**2)) if null_scores.size else 0.0
    else:
        ridge = float(lambda_)
        if not np.isfinite(ridge) or ridge < 0:
            raise ValueError("lambda_ must be nonnegative and finite")
    sigma2 = (1.0 if residual_variance is None else residual_variance) - ridge
    if sigma2 <= 0:
        raise ValueError("Initial residual variance must exceed lambda_")
    # Validate before capping so an invalid request cannot become a valid count.
    L = min(_integer(10 if L is None else L, "L"), len(scores))
    effective_n = 2 if n is None else float(n)
    request = _prepare_suff_stat(
        matrix,
        scores,
        effective_n - 1,
        effective_n,
        L=L,
        scaled_prior_variance=prior_variance,
        residual_variance=sigma2,
        estimate_residual_variance=estimate_residual_variance,
        estimate_residual_method="MLE",
        prior_weights=prior_weights,
        null_weight=null_weight,
        standardize=False,
        estimate_prior_variance=estimate_prior_variance,
        estimate_prior_method=estimate_prior_method,
        prior_variance_grid=prior_variance_grid,
        mixture_weights=mixture_weights,
        check_null_threshold=check_null_threshold,
        prior_tol=prior_tol,
        residual_variance_lowerbound=residual_variance_lowerbound,
        residual_variance_upperbound=1,
        model_init=model_init,
        s_init=s_init,
        coverage=coverage,
        min_abs_corr=min_abs_corr,
        max_iter=max_iter,
        L_greedy=L_greedy,
        greedy_lbf_cutoff=greedy_lbf_cutoff,
        tol=tol,
        convergence_method=convergence_method,
        verbose=verbose,
        track_fit=track_fit,
        check_prior=check_prior,
        n_purity=n_purity,
        median_abs_corr=median_abs_corr,
        cs_extension_corr=cs_extension_corr,
        r_tol=r_tol,
        refine=refine,
        init_only=init_only,
        slot_prior=slot_prior,
        variant_ids=variant_ids,
        variant_metadata=variant_metadata,
        y_mean=intercept_value,
    )
    if request.null_index is not None:
        matrix, scores = np.pad(matrix, ((0, 1), (0, 1))), np.r_[scores, 0.0]
    request = replace(
        request,
        dynamic={
            "kind": "lambda",
            "R": matrix,
            "z": scores,
            "lambda": ridge,
            "r_tol": r_tol,
            "mismatch": "none",
        },
        params={
            **request.params,
            "method": "susie_rss_lambda",
            "lambda": ridge,
            "z_method": z_method,
            "n": n,
        },
    )
    return _fit(request)
