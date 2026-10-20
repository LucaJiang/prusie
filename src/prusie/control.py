"""Typed fitting controls and slot-activity prior specifications.

Defaults and validation follow susieR 0.16.6 (BSD-3-Clause).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike


def _positive(value: float, name: str) -> float:
    if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


@dataclass(frozen=True)
class SlotPrior(Mapping[str, Any]):
    """Validated prior for posterior single-effect activity probabilities.

    Construct with :func:`slot_prior_betabinom` or :func:`slot_prior_poisson`.
    Mapping access is provided for serialization and native fitting.
    """

    family: Literal["betabinom", "poisson"]
    a_beta: float = 1.0
    b_beta: float = 2.0
    C: float = 1.0
    nu: float = 8.0
    update_schedule: Literal["sequential", "batch"] = "sequential"
    c_hat_init: tuple[float, ...] | None = None
    skip_threshold_multiplier: float = 0.0

    def __getitem__(self, key: str) -> Any:
        return asdict(self)[key]

    def __iter__(self) -> Iterator[str]:
        return iter(asdict(self))

    def __len__(self) -> int:
        return len(asdict(self))


def _slot_initial(value: ArrayLike | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if (
        array.ndim != 1
        or not array.size
        or not np.isfinite(array).all()
        or np.any((array < 0) | (array > 1))
    ):
        raise ValueError("c_hat_init must be a finite probability vector")
    return tuple(array.tolist())


def slot_prior_betabinom(
    *,
    a_beta: float = 1.0,
    b_beta: float = 2.0,
    c_hat_init: ArrayLike | None = None,
    skip_threshold_multiplier: float = 0.0,
) -> SlotPrior:
    """Construct a Beta-Binomial single-effect activity prior.

    Parameters
    ----------
    a_beta, b_beta : float
        Positive Beta shape parameters; defaults give prior mean activity 1/3.
    c_hat_init : array_like, optional
        Initial probability for each fitted effect.
    skip_threshold_multiplier : float, default 0
        Nonnegative multiplier for the adaptive inactive-slot skip threshold.

    Returns
    -------
    SlotPrior
        A validated sequential Beta-Binomial specification.
    """
    if not np.isfinite(skip_threshold_multiplier) or skip_threshold_multiplier < 0:
        raise ValueError("skip_threshold_multiplier must be finite and nonnegative")
    return SlotPrior(
        family="betabinom",
        a_beta=_positive(a_beta, "a_beta"),
        b_beta=_positive(b_beta, "b_beta"),
        c_hat_init=_slot_initial(c_hat_init),
        skip_threshold_multiplier=float(skip_threshold_multiplier),
    )


def slot_prior_poisson(
    C: float,
    *,
    nu: float = 8.0,
    update_schedule: Literal["sequential", "batch"] = "sequential",
    c_hat_init: ArrayLike | None = None,
    skip_threshold_multiplier: float = 0.0,
) -> SlotPrior:
    """Construct a Gamma-Poisson single-effect activity prior.

    Parameters
    ----------
    C : float
        Expected positive number of causal variables per region.
    nu : float, default 8
        Positive Gamma shape controlling overdispersion.
    update_schedule : {'sequential', 'batch'}, default 'sequential'
        Update the Gamma state after each effect or after each sweep.
    c_hat_init : array_like, optional
        Initial effect activity probabilities.
    skip_threshold_multiplier : float, default 0
        Nonnegative multiplier for adaptive inactive-slot skipping.

    Returns
    -------
    SlotPrior
        A validated Gamma-Poisson specification.
    """
    if update_schedule not in ("sequential", "batch"):
        raise ValueError("update_schedule must be 'sequential' or 'batch'")
    if not np.isfinite(skip_threshold_multiplier) or skip_threshold_multiplier < 0:
        raise ValueError("skip_threshold_multiplier must be finite and nonnegative")
    return SlotPrior(
        family="poisson",
        C=_positive(C, "C"),
        nu=_positive(nu, "nu"),
        update_schedule=update_schedule,
        c_hat_init=_slot_initial(c_hat_init),
        skip_threshold_multiplier=float(skip_threshold_multiplier),
    )


@dataclass(frozen=True)
class RSSControl(Mapping[str, Any]):
    """Validated RSS mismatch estimation and input-check controls.

    Mapping access is supported to keep control objects easy to serialize.
    """

    mismatch_estimator: Literal["mle", "map"] = "mle"
    mixture_reference_p: float = 5e-8
    qc_eigen_tol_rel: float = 1e-3
    qc_eigen_tol_abs: float = 0.0
    artifact_threshold: float = 0.1
    sensitivity_threshold: float = 30.0
    r_tol: float = 1e-8
    check_input: bool = False
    check_prior: bool = True

    def __getitem__(self, key: str) -> Any:
        return asdict(self)[key]

    def __iter__(self) -> Iterator[str]:
        return iter(asdict(self))

    def __len__(self) -> int:
        return len(asdict(self))


def susie_rss_control(
    *,
    mismatch_estimator: Literal["mle", "map"] = "mle",
    mixture_reference_p: float = 5e-8,
    qc_eigen_tol_rel: float = 1e-3,
    qc_eigen_tol_abs: float = 0.0,
    artifact_threshold: float = 0.1,
    sensitivity_threshold: float = 30.0,
    r_tol: float = 1e-8,
    check_input: bool = False,
    check_prior: bool = True,
) -> RSSControl:
    """Construct RSS mismatch and numerical input-check controls.

    Parameters
    ----------
    mismatch_estimator : {'mle', 'map'}, default 'mle'
        Region-level mismatch-variance estimator.
    mixture_reference_p : float, default 5e-8
        Reference two-sided P-value calibrating mismatch-mixture prior odds.
    qc_eigen_tol_rel, qc_eigen_tol_abs : float
        Nonnegative low-eigenvalue diagnostic cutoffs.
    artifact_threshold : float, default 0.1
        Residual artifact score threshold in [0, 1].
    sensitivity_threshold : float, default 30
        Nonnegative credible-set Bayes-factor attenuation threshold.
    r_tol : float, default 1e-8
        Nonnegative positive-semidefinite check tolerance.
    check_input, check_prior : bool
        Enable input eigenspace and estimated-prior checks.

    Returns
    -------
    RSSControl
        Validated controls for :func:`susie_rss`.
    """
    if mismatch_estimator not in ("mle", "map"):
        raise ValueError("mismatch_estimator must be 'mle' or 'map'")
    for name, value in (
        ("qc_eigen_tol_rel", qc_eigen_tol_rel),
        ("qc_eigen_tol_abs", qc_eigen_tol_abs),
        ("sensitivity_threshold", sensitivity_threshold),
        ("r_tol", r_tol),
    ):
        if not np.isscalar(value) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if (
        not np.isscalar(mixture_reference_p)
        or not np.isfinite(mixture_reference_p)
        or not 0 < mixture_reference_p <= 1
    ):
        raise ValueError("mixture_reference_p must be finite and in (0, 1]")
    if (
        not np.isscalar(artifact_threshold)
        or not np.isfinite(artifact_threshold)
        or not 0 <= artifact_threshold <= 1
    ):
        raise ValueError("artifact_threshold must be finite and in [0, 1]")
    if not isinstance(check_input, (bool, np.bool_)) or not isinstance(
        check_prior, (bool, np.bool_)
    ):
        raise TypeError("check_input and check_prior must be booleans")
    return RSSControl(
        mismatch_estimator=mismatch_estimator,
        mixture_reference_p=float(mixture_reference_p),
        qc_eigen_tol_rel=float(qc_eigen_tol_rel),
        qc_eigen_tol_abs=float(qc_eigen_tol_abs),
        artifact_threshold=float(artifact_threshold),
        sensitivity_threshold=float(sensitivity_threshold),
        r_tol=float(r_tol),
        check_input=bool(check_input),
        check_prior=bool(check_prior),
    )
