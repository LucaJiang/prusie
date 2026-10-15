"""Result contract shared with consumers without requiring this package."""

from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np


@dataclass
class SusieResult:
    """SuSiE posterior; component arrays are (L, p [+ null]) float64.

    ``mu2`` is the conditional second moment. All indices are zero-based.
    ``mu`` is on the fitted predictor scale; use ``coef`` for effects
    on the original predictor scale. The null column is excluded from PIP.
    ``sets`` retains original component indices, even when nonconsecutive.
    ``sets.coverage`` is the posterior mass of each returned set in that
    component; ``sets.requested_coverage`` is the selection threshold.
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
        """Describe stored dimensions and status without recomputing statistics."""
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
        """Inner solver status of the returned state, independent of refinement."""
        return self.converged

    @property
    def refinement_status(self):
        return self.refinement.get("status", "not_executed")

    @property
    def refinement_stop_reason(self):
        return self.refinement.get("stop_reason", "not_requested")

    def as_dict(self) -> dict[str, Any]:
        """Return a shallow mapping, preserving arrays without copying."""
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if not field.name.startswith("_")
        }

    @property
    def coef(self) -> np.ndarray:
        """Posterior mean coefficients on the original predictor scale."""
        return (
            np.sum(self.alpha * self.mu, axis=0)[: len(self.variant_ids)]
            / self.X_column_scale_factors[: len(self.variant_ids)]
        )
