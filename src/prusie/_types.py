"""Stable schemas for fitting settings, input checks and credible-set decisions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Protocol, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .control import SlotPrior

_RidgeParameters = TypedDict("_RidgeParameters", {"lambda": float}, total=False)


class FitParameters(_RidgeParameters, total=False):
    """Resolved settings recorded with a fit; existing dictionary access is supported."""

    interface: str
    method: str
    n: float
    yty: float
    L: int
    scaled_prior_variance: float | ArrayLike
    prior_variance: float
    residual_variance: float | None
    estimate_residual_variance: bool
    estimate_prior_variance: bool
    estimate_prior_method: str
    estimate_residual_method: str
    prior_weights: list[float]
    null_weight: float
    standardize: bool
    max_iter: int
    n_threads: int
    overwrite_ld: bool
    tol: float
    coverage: float | None
    min_abs_corr: float | None
    median_abs_corr: float | None
    cs_extension_corr: float | None
    n_purity: int | str | None
    prior_tol: float
    check_null_threshold: float
    check_input: bool
    r_tol: float
    check_prior: bool
    s_init: str | None
    model_init: Mapping[str, Any] | None
    refine: bool
    retain_data: bool
    track_fit: bool | str
    convergence: dict[str, Any]
    convergence_resolution: dict[str, Any]
    convergence_method: str
    residual_variance_lowerbound: float
    residual_variance_upperbound: float | None
    prior_variance_grid: ArrayLike | None
    mixture_weights: ArrayLike | None
    alpha0: float
    beta0: float
    L_greedy: int | None
    greedy_lbf_cutoff: float
    greedy_history: list[dict[str, Any]]
    slot_prior: SlotPrior | Mapping[str, Any] | None
    verbose: bool
    init_only: bool
    input_kind: str
    input_n: float | None
    input_var_y: float | None
    effect_scale: str
    z_method: str
    z_ld_weight: float
    R_finite: bool | float | str | None
    R_mismatch: str | None
    control: dict[str, Any]
    initial_panel: int
    single_panel_elbo: list[float]
    omega: ArrayLike
    intercept: bool
    na_rm: bool
    hard_exclusion: bool
    tol_resolution: str
    effective_L: int


class InputChecks(TypedDict, total=False):
    """Input-QC metadata, including keys accepted when loading older results."""

    matrix_symmetric: bool
    psd_checked: bool
    input_columns: int
    original_input_columns: int
    input_missing_policy: Literal["reject", "zero", "susieR_zero_z_and_Xty"]
    variant_ids_are_positional: bool
    minimum_eigenvalue: float
    projection_squared_fraction: float
    ld_overwrite: dict[str, Any]
    p: int
    variant_ids_positional: bool


class CredibleSetAudit(TypedDict, total=False):
    """Candidate membership and the checks deciding whether a CS is retained."""

    component_id: int
    candidate_members: list[int]
    extended_members: list[int]
    reasons: list[str]
    stages: list[str]
    requested_coverage: float
    actual_coverage: float
    min_abs_corr: float | None
    active_threshold: float
    purity_evaluation: str
    purity: dict[str, float]


class RetainedInputs(Protocol):
    """Owned working inputs required for continuation and objective evaluation."""

    matrix: NDArray[np.float64]
    response: NDArray[np.float64]
    yty: float
    n: float
    options: Mapping[str, Any]
    cs_matrix: NDArray[np.float64]
    diagonal: NDArray[np.float64]
    scales: NDArray[np.float64]
    means: NDArray[np.float64]
    y_mean: float
    matrix_scale: float
    variant_ids: NDArray[np.str_]
    alleles: Mapping[str, ArrayLike]

    @property
    def signature(self) -> str:
        """Identity of the owned inputs and variant order."""
        ...
