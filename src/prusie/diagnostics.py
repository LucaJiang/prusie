"""Stopping rules and convergence summaries for Gaussian IBSS."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .result import SusieResult


@dataclass(frozen=True)
class ConvergenceConfig:
    """Stopping rule for iterative Bayesian stepwise selection (IBSS).

    A rule is first evaluated after the second sweep; successful and cycle
    stops also require at least ``min_iter`` sweeps.

    Parameters
    ----------
    criterion : {'elbo', 'alpha_pip', 'joint'}, default 'elbo'
        'elbo' stops when 0 ≤ ΔELBO < tol, with tol the fitting tolerance;
        a decrease does not satisfy the rule. 'alpha_pip' stops when both
        max_ℓj |Δα_ℓj| < alpha_tol and max_j |ΔPIP_j| < pip_tol. 'joint'
        requires both conditions.
    alpha_tol : float, default 1e-4
        Tolerance for the maximum absolute change in α between sweeps.
    pip_tol : float, default 1e-4
        Tolerance for the maximum absolute change in PIP between sweeps.
    min_iter : int, default 2
        Earliest sweep at which a successful or cycle stop may occur.
    cycle_lag : int, default 5
        Largest lag (1 to 5) at which repeated α/PIP states are recorded.

    Notes
    -----
    In 'elbo' mode with a finite objective, repeated states at lags 2 to
    ``cycle_lag`` are recorded as diagnostics only. In 'alpha_pip' and
    'joint' modes they may stop the fit with ``converged=False``, returning
    the current posterior. Every nonfinite objective (NaN or either infinity)
    stops the fit as a numerical failure in every mode, even before
    ``min_iter``. The first finite objective, which follows the initial −∞
    value, is never used for a convergence check.
    """

    criterion: str = "elbo"
    alpha_tol: float = 1e-4
    pip_tol: float = 1e-4
    min_iter: int = 2
    cycle_lag: int = 5

    def options(self) -> dict[str, str | float | int]:
        """Validate the controls and return them as native fitting options."""
        if self.criterion not in ("elbo", "alpha_pip", "joint"):
            raise ValueError("criterion must be elbo, alpha_pip, or joint")
        if any(not math.isfinite(x) or x < 0 for x in (self.alpha_tol, self.pip_tol)):
            raise ValueError("alpha_tol and pip_tol must be finite and nonnegative")
        if (
            isinstance(self.min_iter, bool)
            or int(self.min_iter) != self.min_iter
            or self.min_iter < 2
        ):
            raise ValueError("min_iter must be an integer >= 2")
        if (
            isinstance(self.cycle_lag, bool)
            or int(self.cycle_lag) != self.cycle_lag
            or not 1 <= self.cycle_lag <= 5
        ):
            raise ValueError("cycle_lag must be an integer in [1, 5]")
        return dict(
            criterion=str(self.criterion),
            alpha_tol=float(self.alpha_tol),
            pip_tol=float(self.pip_tol),
            min_iter=int(self.min_iter),
            cycle_lag=int(self.cycle_lag),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ConvergenceConfig:
        """Restore and validate controls saved in a result or cache record."""
        config = cls(**value)
        config.options()
        return config


def resolve_convergence(
    fit: SusieResult | None = None, override: ConvergenceConfig | None = None
) -> tuple[ConvergenceConfig, dict[str, Any]]:
    """Return the requested stopping controls and a record of their source.

    Controls are taken from ``override``, else from ``fit.params``, else from
    the convergence record or retained options of results that lack
    ``params['convergence']``. If none is available, the defaults are used and
    labelled as such. Only requested controls are resolved; a runtime NaN
    fallback is never inherited.
    """
    previous = None
    source = "default" if fit is None else "legacy_default"
    if fit is not None:
        saved = getattr(fit, "params", {}).get("convergence")
        if saved is not None:
            previous = ConvergenceConfig.from_dict(saved)
            source = "inherited_params"
        else:
            diagnostic = getattr(fit, "convergence", {}) or {}
            fields = dict(
                criterion="requested_criterion",
                alpha_tol="alpha_tol",
                pip_tol="pip_tol",
                min_iter="min_iter",
                cycle_lag="max_cycle_lag",
            )
            if all(key in diagnostic for key in fields.values()):
                previous = ConvergenceConfig.from_dict(
                    {k: diagnostic[v] for k, v in fields.items()}
                )
                source = "recovered_legacy_diagnostics"
            else:
                prepared = getattr(fit, "_prepared", None)
                options = prepared.options if prepared is not None else {}
                if all(key in options for key in fields):
                    previous = ConvergenceConfig.from_dict({key: options[key] for key in fields})
                    source = "recovered_prepared_options"
    if override is not None:
        if not isinstance(override, ConvergenceConfig):
            raise TypeError("convergence must be a ConvergenceConfig")
        override.options()
        return override, dict(
            source="explicit_override",
            previous_requested=previous.options() if previous is not None else None,
        )
    control = previous or ConvergenceConfig()
    provenance = dict(source=source)
    if source == "legacy_default":
        provenance["compatibility_note"] = (
            "Original requested controls were not recorded; current default "
            "ELBO controls are used. No inheritance can be established."
        )
    return control, provenance


def nonconvergence_message(native: Mapping[str, Any]) -> str:
    """Warning text for a fit that did not satisfy its stopping rule."""
    return (
        f"IBSS did not converge after {native['niter']} iterations "
        f"(stop_reason={native['diagnostics']['stop_reason']}); "
        "check summary/LD consistency"
    )


def tracking_level(value: bool | str) -> int:
    """Map ``track_fit`` to the native level: 0 off, 1 compact, 2 dense."""
    if value is False or value is None:
        return 0
    if value is True or value == "compact":
        return 1
    if value == "dense":
        return 2
    raise ValueError("track_fit must be False, True, 'compact', or 'dense'")


def convergence_summary(
    native: Mapping[str, Any], options: Mapping[str, Any], control: ConvergenceConfig
) -> dict[str, Any]:
    """Combine native stopping diagnostics with the resolved controls."""
    d = dict(native["diagnostics"])
    trace = d.pop("trace")
    d.update(
        requested_criterion=control.criterion,
        requested_config=control.options(),
        tol=options["tol"],
        alpha_tol=control.alpha_tol,
        pip_tol=control.pip_tol,
        min_iter=control.min_iter,
        max_cycle_lag=control.cycle_lag,
        niter=native["niter"],
        ibss_converged=native["converged"],
        phase="post_sweep_pre_sigma2_update",
        posterior_phase="post_loop_post_sigma2_update_and_trim",
        alpha_distance_includes_null=True,
        pip_excludes_null=True,
        online_component_entropy_includes_null=True,
        online_top_variant_uses_internal_indices=True,
        cs_activity_threshold=1e-9,
        pip_activity_threshold=options["prior_tol"],
        bbar_scale="standardized working predictor scale",
        V_fixed=not options["estimate_prior_variance"],
        sigma2_fixed=not options["estimate_residual_variance"],
        relative_elbo_formula="delta_elbo / max(1, abs(previous_elbo))",
        elbo_valid=all(math.isfinite(x) for x in native["elbo"]),
        warnings=list(native["warnings"]),
        trace_available=bool(trace),
    )
    return d, trace


def diagnose_convergence(fit: SusieResult) -> dict[str, Any]:
    """Return a copy of the convergence record of a fit.

    Parameters
    ----------
    fit : SusieResult
        A fitted result.

    Returns
    -------
    dict
        Requested and effective stopping rule, stop reason, last distances
        and whether a trace is available. Per-sweep metrics are stored
        separately in ``fit.trace``. If no convergence record exists,
        ``stop_reason`` is 'not_available'; metrics are not reconstructed.
    """
    import copy

    if not getattr(fit, "convergence", None):
        return dict(
            stop_reason="not_available",
            trace_available=False,
            warnings=["Exact online convergence metrics were not recorded."],
        )
    return copy.deepcopy(fit.convergence)
