"""Finite-reference reliability summaries ported from susieR (BSD-3-Clause)."""

from __future__ import annotations

from typing import Any

import numpy as np


def finite_reference_diagnostics(
    correlation: np.ndarray,
    reference_n: float,
    *,
    bias: float,
    inflation: np.ndarray | None,
    residual: np.ndarray,
    control: Any,
    mismatch: str,
) -> dict[str, Any]:
    p = len(correlation)
    frobenius = float(np.sum(correlation**2))
    debiased = (
        (reference_n * frobenius - p**2) / (reference_n + 1)
        if np.isfinite(reference_n)
        else frobenius
    )
    effective_rank = p**2 / max(debiased, 1)
    diagnostics = dict(
        B=reference_n,
        p=p,
        R_frob_sq_debiased=debiased,
        effective_rank=effective_rank,
        r_over_B=effective_rank / reference_n,
        Rhat_diag_deviation=np.abs(np.diag(correlation) - 1),
        R_mismatch=mismatch,
    )
    if inflation is not None:
        diagnostics["per_variable_penalty"] = inflation - 1
    if mismatch != "none":
        diagnostics.update(
            lambda_bias=bias,
            B_corrected=1 / (1 / reference_n + bias)
            if bias > 0 or np.isfinite(reference_n)
            else np.inf,
        )
        values, vectors = np.linalg.eigh(correlation)
        cutoff = max(control.qc_eigen_tol_abs, control.qc_eigen_tol_rel * float(np.max(values)))
        low = values <= cutoff
        residual_norm = float(residual @ residual)
        evaluable = bool(np.any(low) and residual_norm > 1e-12)
        fraction = (
            float(np.clip(np.sum((vectors[:, low].T @ residual) ** 2) / residual_norm, 0, 1))
            if evaluable
            else 0.0
        )
        flag = evaluable and fraction > control.artifact_threshold
        diagnostics.update(
            Q_art=fraction,
            artifact_evaluable=evaluable,
            artifact_flag=flag,
            low_eigen_count=int(low.sum()),
            low_eigen_fraction=float(low.mean()),
            eig_delta=cutoff,
            mode_label="warning" if flag else "normal",
            R_mismatch_method=control.mismatch_estimator,
        )
    return diagnostics


def summarize_attenuation(fit: Any, attenuation: np.ndarray, threshold: float) -> None:
    """Attach per-variant and per-CS evidence attenuation without changing the fit."""

    def label(value: float) -> str | None:
        if not np.isfinite(value):
            return None
        return (
            "stable"
            if value < np.log(3)
            else "mildly_sensitive"
            if value < np.log(20)
            else "sensitive"
            if value < np.log(150)
            else "highly_sensitive"
        )

    maximum = np.full(attenuation.shape[1], np.nan)
    component = np.full(attenuation.shape[1], -1, dtype=int)
    for variant, column in enumerate(attenuation.T):
        if np.isfinite(column).any():
            component[variant] = int(np.nanargmax(column))
            maximum[variant] = float(column[component[variant]])
    records: dict[str, dict[str, Any]] = {}
    for members, effect in zip(fit.sets.get("cs", []), fit.sets.get("cs_index", []), strict=True):
        values = attenuation[effect, members]
        if np.isfinite(values).any():
            top = int(np.nanargmax(values))
            records[f"L{effect + 1}"] = dict(
                max=float(np.nanmax(values)),
                weighted=float(np.nansum(fit.alpha[effect, members] * values)),
                top_variable=int(members[top]),
                top_attenuation=float(values[top]),
                label=label(float(np.nanmax(values))),
            )
        else:
            records[f"L{effect + 1}"] = dict(
                max=np.nan, weighted=0.0, top_variable=None, top_attenuation=np.nan, label=None
            )
    diagnostics = fit.R_finite_diagnostics
    diagnostics["bf_attenuation"] = dict(
        variant_max=maximum,
        variant_component=component,
        variant_label=[label(value) for value in maximum],
        cs_max={key: value["max"] for key, value in records.items()},
        cs_weighted={key: value["weighted"] for key, value in records.items()},
        cs_label={key: value["label"] for key, value in records.items()},
        cs_top_variable={key: value["top_variable"] for key, value in records.items()},
        cs_top_attenuation={key: value["top_attenuation"] for key, value in records.items()},
        threshold=threshold,
    )
    diagnostics["R_sensitivity_flag"] = any(value["max"] >= threshold for value in records.values())
    diagnostics["R_reliability_flag"] = bool(
        diagnostics.get("artifact_flag", False) or diagnostics["R_sensitivity_flag"]
    )
