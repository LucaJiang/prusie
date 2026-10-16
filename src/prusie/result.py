"""Result container returned by the SuSiE fitting functions."""

from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np


@dataclass
class SusieResult:
    """Variational SuSiE posterior and fitting record.

    Let p be the number of variants and q = p + 1 when a null column is
    included (``null_index`` is not None), otherwise q = p. Component arrays
    have shape (L, q) and dtype float64. Indices are zero-based.

    Attributes
    ----------
    variant_ids : ndarray of str, shape (p,)
        Variant identifiers in input order.
    alpha : ndarray, shape (L, q)
        Posterior inclusion probabilities α_ℓj of each single effect; each row
        sums to one.
    mu, mu2 : ndarray, shape (L, q)
        Conditional posterior mean and second moment of effect ℓ given that it
        is at variant j, on the working predictor scale. ``mu2`` includes the
        conditional posterior variance.
    lbf_variable : ndarray, shape (L, q)
        Natural-log Bayes factors of each variant within each single effect,
        before prior weighting.
    lbf : ndarray, shape (L,)
        Natural-log prior-weighted marginal likelihood ratio of each effect.
    V : ndarray, shape (L,)
        Prior effect variances after final trimming.
    sigma2 : float
        Residual variance σ² on the working response scale.
    pip : ndarray, shape (p,)
        Posterior inclusion probabilities, 1 - ∏_ℓ (1 - α_ℓj) over active
        effects; the null column is excluded.
    elbo : ndarray, shape (niter,)
        Evidence lower bound after each sweep, before final trimming.
    niter : int
        Number of IBSS sweeps executed.
    converged : bool
        Whether IBSS satisfied its effective stopping rule.
    sets : dict
        Credible sets. ``cs`` lists member indices, ``cs_index`` the original
        component of each set (not necessarily consecutive), ``coverage``
        (alias ``actual_coverage``) the posterior mass of each set within its
        component, ``requested_coverage`` the target mass and ``purity`` the
        minimum, mean and median absolute pairwise correlation.
    null_index : int or None
        Column index of the null variant, if included.
    params : dict
        Resolved fitting options and normalized prior weights.
    backend_version : str
        Native backend and reference identifiers.
    warnings : list of str
        Messages also emitted as ``RuntimeWarning``.
    qc : dict
        Input checks performed for this call.
    variant_metadata : dict of str to ndarray
        Caller-supplied length-p columns.
    X_column_scale_factors : ndarray, shape (q,)
        Predictor scales d_j applied before fitting.
    intercept : float
        Intercept on the original scale, or NaN if means were not supplied.
    KL : ndarray, shape (L,)
        Kullback-Leibler divergence of each single-effect posterior from its
        prior.
    XtXr : ndarray, shape (q,)
        Fitted crossproduct XᵀX b̄ from the last sweep, before trimming.
    convergence : dict
        Requested and effective stopping rule, stop reason and last distances.
    trace : list of dict
        Per-sweep metrics when ``track_fit`` was requested.
    refinement : dict
        Record of an explicit refinement search, if any.
    cs_audit : list of dict
        Decisions taken while constructing each candidate credible set.
    data_signature : str
        SHA-256 digest of the working inputs, used to validate warm starts.
    """

    variant_ids: np.ndarray
    alpha: np.ndarray
    mu: np.ndarray
    mu2: np.ndarray
    lbf_variable: np.ndarray
    lbf: np.ndarray
    V: np.ndarray
    sigma2: float
    pip: np.ndarray
    elbo: np.ndarray
    niter: int
    converged: bool
    sets: dict[str, Any]
    null_index: int | None
    params: dict[str, Any]
    backend_version: str
    warnings: list[str]
    qc: dict[str, Any]
    variant_metadata: dict[str, np.ndarray]
    X_column_scale_factors: np.ndarray
    intercept: float
    KL: np.ndarray
    XtXr: np.ndarray
    convergence: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    refinement: dict[str, Any] = field(
        default_factory=lambda: dict(
            status="not_executed", stop_reason="not_requested", history=[], total_fits=0
        )
    )
    cs_audit: list[dict[str, Any]] = field(default_factory=list)
    data_signature: str = ""
    _prepared: Any = field(default=None, repr=False, compare=False)

    def __repr__(self):
        """Summarize dimensions and stopping status."""
        reason = self.convergence.get("stop_reason", "not_available")
        return (
            f"SusieResult(variables={len(self.variant_ids)}, "
            f"components={self.alpha.shape[0]}, "
            f"credible_sets={len(self.sets.get('cs') or {})}, "
            f"niter={self.niter}, converged={self.converged}, "
            f"stop_reason={reason!r})"
        )

    @property
    def ibss_converged(self):
        """Convergence of the returned IBSS fit, independent of any refinement."""
        return self.converged

    @property
    def refinement_status(self):
        """Outcome of the refinement search, or 'not_executed'."""
        return self.refinement.get("status", "not_executed")

    @property
    def refinement_stop_reason(self):
        """Reason the refinement search ended, or 'not_requested'."""
        return self.refinement.get("stop_reason", "not_requested")

    def as_dict(self) -> dict[str, Any]:
        """Return the public fields as a shallow mapping; arrays are not copied."""
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if not field.name.startswith("_")
        }

    @property
    def coef(self) -> np.ndarray:
        """Posterior mean effects ∑_ℓ α_ℓj μ_ℓj / d_j on the input predictor scale."""
        return (
            np.sum(self.alpha * self.mu, axis=0)[: len(self.variant_ids)]
            / self.X_column_scale_factors[: len(self.variant_ids)]
        )
