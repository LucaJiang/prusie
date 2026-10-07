"""Result contract shared with consumers without requiring this package."""

from dataclasses import dataclass, fields
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

    def as_dict(self) -> dict[str, Any]:
        """Return a shallow mapping, preserving arrays without copying."""
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @property
    def coef(self) -> np.ndarray:
        """Posterior mean coefficients on the original predictor scale."""
        return (
            np.sum(self.alpha * self.mu, axis=0)[: len(self.variant_ids)]
            / self.X_column_scale_factors[: len(self.variant_ids)]
        )
