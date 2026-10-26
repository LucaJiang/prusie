"""Typed result containers returned by the SuSiE fitting functions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._types import CredibleSetAudit, FitParameters, InputChecks, RetainedInputs


@dataclass(frozen=True)
class InputQuality:
    """Input checks performed before fitting.

    Attributes
    ----------
    predictor_count : int
        Number of input predictor columns, before a null column is added.
    matrix_symmetric : bool
        Whether the working covariance is symmetric after preprocessing.
    psd_checked : bool
        Whether positive semidefiniteness was checked by eigendecomposition.
    missing_summary_values : str
        Policy for missing summary associations: ``'zero'`` or ``'reject'``.
    positional_variant_ids : bool
        Whether variant identifiers were generated from column positions.
    minimum_eigenvalue : float or None
        Smallest eigenvalue, when explicitly checked.
    projection_squared_fraction : float or None
        Fraction of association squared norm outside the covariance span.
    """

    predictor_count: int
    matrix_symmetric: bool = True
    psd_checked: bool = False
    missing_summary_values: str = "zero"
    positional_variant_ids: bool = True
    minimum_eigenvalue: float | None = None
    projection_squared_fraction: float | None = None


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
    params : FitParameters
        Resolved fitting options and normalized prior weights.
    backend_version : str
        Native backend and reference identifiers.
    warnings : list of str
        Messages also emitted as ``RuntimeWarning``.
    qc : InputChecks
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
    cs_audit : list of CredibleSetAudit
        Decisions taken while constructing each candidate credible set.
    data_signature : str or None
        SHA-256 digest of the owned input snapshot, used to validate warm
        starts. None for ordinary fits without retained inputs; missing
        signatures never establish that inputs match.
    input_quality : InputQuality
        Typed description of input checks. ``qc`` remains available for
        backwards-compatible access to the historical mapping.
    fitted : ndarray, shape (n,), or None
        Original-scale fitted responses for individual-level data.
    univariate_zscore : ndarray, shape (p,), or None
        Marginal regression z-scores when requested for individual-level data.
    slot_weights : ndarray, shape (L,), or None
        Posterior activity probabilities when a slot prior was fitted.
    lbf_grid : ndarray, shape (L, q, k), or None
        Per-component, per-variant log Bayes factors for each mixture variance.
    rv : ndarray, shape (L,), or None
        NIG residual-variance posterior means for each single effect.
    marginal_loglik : ndarray, shape (L,), or None
        Integrated NIG single-effect marginal log likelihoods.
    R_finite_diagnostics : dict
        Finite-reference LD and mismatch reliability summaries, including
        component and credible-set Bayes-factor attenuation when available.
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
    params: FitParameters
    backend_version: str
    warnings: list[str]
    qc: InputChecks
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
    cs_audit: list[CredibleSetAudit] = field(default_factory=list)
    data_signature: str | None = None
    input_quality: InputQuality | None = None
    fitted: NDArray[np.float64] | None = None
    univariate_zscore: NDArray[np.float64] | None = None
    slot_weights: NDArray[np.float64] | None = None
    lbf_grid: NDArray[np.float64] | None = None
    rv: NDArray[np.float64] | None = None
    marginal_loglik: NDArray[np.float64] | None = None
    R_finite_diagnostics: dict[str, Any] = field(default_factory=dict)
    _prepared: RetainedInputs | None = field(default=None, repr=False, compare=False)
    _input_quality_export: dict[str, Any] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _input_quality_snapshot: InputQuality | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Populate typed QC information for callers constructing legacy results."""
        if isinstance(self.input_quality, dict):
            self.input_quality = InputQuality(**self.input_quality)
        if self.input_quality is None:
            self.input_quality = InputQuality(
                predictor_count=int(self.qc.get("input_columns", len(self.variant_ids))),
                matrix_symmetric=bool(self.qc.get("matrix_symmetric", True)),
                psd_checked=bool(self.qc.get("psd_checked", False)),
                missing_summary_values=(
                    "reject" if self.qc.get("input_missing_policy") == "reject" else "zero"
                ),
                positional_variant_ids=bool(
                    self.qc.get(
                        "variant_ids_are_positional", self.qc.get("variant_ids_positional", True)
                    )
                ),
                minimum_eigenvalue=self.qc.get("minimum_eigenvalue"),
                projection_squared_fraction=self.qc.get("projection_squared_fraction"),
            )

    def __repr__(self) -> str:
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
    def ibss_converged(self) -> bool:
        """Convergence of the returned IBSS fit, independent of any refinement."""
        return self.converged

    @property
    def refinement_status(self) -> str:
        """Outcome of the refinement search, or 'not_executed'."""
        return self.refinement.get("status", "not_executed")

    @property
    def refinement_stop_reason(self) -> str:
        """Reason the refinement search ended, or 'not_requested'."""
        return self.refinement.get("stop_reason", "not_requested")

    def as_dict(self) -> dict[str, Any]:
        """Return public fields as a mapping; arrays remain shared, typed QC is serialized."""
        if self._input_quality_snapshot is not self.input_quality:
            self._input_quality_export = (
                None if self.input_quality is None else asdict(self.input_quality)
            )
            self._input_quality_snapshot = self.input_quality
        return {
            field.name: (
                self._input_quality_export
                if field.name == "input_quality" and self.input_quality is not None
                else getattr(self, field.name)
            )
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
