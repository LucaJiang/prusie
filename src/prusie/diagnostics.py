"""Stopping rules and convergence summaries for Gaussian IBSS."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from numbers import Integral, Real
from statistics import median
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
    the current posterior. In the native Gaussian loop, the default NaN-ELBO
    fallback uses the susieR-compatible PIP rule with fitting ``tol`` for both
    posterior distances; its cycle stop averages the repeated alpha states
    and counts as convergence. Other 'elbo' fallback configurations and
    'alpha_pip' can proceed with NaN using their configured posterior
    tolerances, whereas 'joint' fails. A computed infinite objective raises
    ``ValueError`` in that native loop. Dynamic RSS models have a separate
    posterior-rule branch for nonfinite objectives. The first finite objective,
    which follows the initial −∞ value, is never used for an ELBO check.
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
    source = "default" if fit is None else "default_unrecorded"
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
                source = "recovered_from_diagnostics"
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
    if source == "default_unrecorded":
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
        null_index=options.get("null_index"),
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


def _finite_metric(value: Any) -> float | None:
    if isinstance(value, Real) and math.isfinite(value):
        return float(value)
    return None


def _tail_stability(
    rows: list[Mapping[str, Any]], ids: Mapping[int, str], null_index: int | None, threshold: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read component movement and the recorded sweep-to-sweep PIP distances."""
    unchanged, maximum, moving = None, None, []
    distances = [_finite_metric(row.get("max_abs_delta_pip")) for row in rows]
    # The initial sweep has no predecessor and hence no recorded distance.
    if rows and rows[0].get("iteration") == 1 and rows[0].get("max_abs_delta_pip") is None:
        distances = distances[1:]
    if distances and all(x is not None and x >= 0 for x in distances):
        maximum = max(distances)
    if len(rows) >= 2 and all(row.get("active") is not None for row in rows):
        sizes = [len(row["active"]) for row in rows]
        if len(set(sizes)) == 1:
            unchanged = True
            for component in range(sizes[0]):
                active = [row["active"][component] for row in rows]
                if any(value is None for value in active):
                    if unchanged is not False:
                        unchanged = None
                    continue
                if not all(active):
                    continue
                tops = [row.get("top_variant", []) for row in rows]
                tops = [
                    value[component] if value is not None and len(value) > component else None
                    for value in tops
                ]
                changes = None
                if any(value is None for value in tops):
                    if unchanged is not False:
                        unchanged = None
                else:
                    changes = sum(a != b for a, b in pairwise(tops))
                if changes:
                    unchanged = False
                entropy = [row.get("entropy", []) for row in (rows[0], rows[-1])]
                entropy = [
                    _finite_metric(value[component])
                    if value is not None and len(value) > component
                    else None
                    for value in entropy
                ]
                change = entropy[1] - entropy[0] if None not in entropy else None
                if changes or (change is not None and abs(change) > 1e-6):
                    index = tops[-1]
                    variant = None
                    if index is not None and index != null_index:
                        index -= int(null_index is not None and index > null_index)
                        variant = ids.get(index)
                    moving.append(
                        dict(
                            component_id=component,
                            top_variant_changes=changes,
                            last_top_variant=variant,
                            entropy_change=change,
                        )
                    )
    stable = None
    if unchanged is False or (maximum is not None and maximum >= threshold):
        stable = False
    elif unchanged is True and maximum is not None:
        stable = True
    moving.sort(
        key=lambda row: (
            -(row["top_variant_changes"] or 0),
            -abs(row["entropy_change"] or 0.0),
            row["component_id"],
        )
    )
    return dict(
        top_variants_unchanged=unchanged,
        max_abs_delta_pip_tail=maximum,
        stable=stable,
        pip_stability=threshold,
    ), moving


def assess_convergence(
    fit_or_summary: SusieResult | Mapping[str, Any], *, tail: int = 5, pip_stability: float = 1e-3
) -> dict[str, Any]:
    """Describe the end of a fit using its recorded convergence diagnostics.

    Read ``status`` to decide whether to continue IBSS, inspect changing
    components, or check input and LD consistency. The input may be a
    :class:`SusieResult` or the dictionary returned by :func:`prusie.summarize`.
    No fitting is performed and the fit's ``converged`` flag is unchanged.

    ``tail`` selects at most that many recorded sweeps and must be an integer
    of at least two. ``pip_stability`` must lie in (0, 1); it is a descriptive
    display threshold, not a fitted stopping rule or a correctness tolerance.
    PIP stability uses the largest recorded between-sweep PIP distance in the
    tail, including the distance from the sweep immediately preceding it.
    Missing distances leave stability unevaluated unless movement is known.

    Component comparisons use original component IDs and require activity in
    every selected sweep. ``entropy_change`` is the last entropy minus the
    first; ``last_top_variant`` is a variant ID, or None for the null column
    or an unavailable ID. Older summaries without ``null_index`` use the
    public fitter's convention of appending the null column after all variants.
    The geometric ELBO extrapolation describes the recorded tail only.
    """
    if isinstance(tail, bool) or not isinstance(tail, Integral) or tail < 2:
        raise ValueError("tail must be an integer >= 2")
    if (
        isinstance(pip_stability, bool)
        or not isinstance(pip_stability, Real)
        or not math.isfinite(pip_stability)
        or not 0 < pip_stability < 1
    ):
        raise ValueError("pip_stability must lie in (0, 1)")
    if isinstance(fit_or_summary, SusieResult):
        fit = fit_or_summary
        convergence = fit.convergence or {}
        trace = fit.trace or []
        niter, tol = fit.niter, convergence.get("tol", fit.params.get("tol"))
        inner = convergence.get("inner_ibss_converged", fit.converged)
        ids = dict(enumerate(map(str, fit.variant_ids)))
        null_index = fit.null_index
    elif isinstance(fit_or_summary, Mapping):
        summary = fit_or_summary
        convergence = summary.get("convergence") or {}
        trace = summary.get("trace") or []
        niter = convergence.get("niter", summary.get("niter"))
        tol = convergence.get("tol")
        inner = convergence.get(
            "inner_ibss_converged", convergence.get("ibss_converged", summary.get("converged"))
        )
        ids = {
            row["index"]: str(row["variant_id"])
            for row in summary.get("variants", [])
            if "index" in row and "variant_id" in row
        }
        null_index = summary.get("null_index", convergence.get("null_index"))
    else:
        raise TypeError("fit_or_summary must be a SusieResult or a summary mapping")

    rows = list(trace[-int(tail) :])
    stop_reason = convergence.get("stop_reason", "not_available")
    last = (trace[-1] if trace else convergence.get("last")) or {}
    averaged = "alpha_pip_cycle_averaged" in (last.get("events") or [])
    events = [set(row.get("events") or []) for row in rows]
    decreases = sum("elbo_decreased" in row for row in events)
    nonfinite = sum(
        "nonfinite_elbo" in event
        or (isinstance(row.get("elbo"), Real) and not math.isfinite(row["elbo"]))
        for row, event in zip(rows, events, strict=True)
    )
    deltas = [_finite_metric(row.get("delta_elbo")) for row in rows]
    if rows and rows[0].get("iteration") == 1 and rows[0].get("delta_elbo") is None:
        deltas = deltas[1:]
    sign_changes, previous = 0, None
    for value in deltas:
        if value is None:
            previous = None
        elif value != 0:
            sign = value > 0
            sign_changes += previous is not None and previous != sign
            previous = sign

    trend = dict(ratio=None, extrapolated_remaining_gain=None, estimated_extra_sweeps=None)
    if len(deltas) >= 2 and all(value is not None and value > 0 for value in deltas):
        ratios = [b / a for a, b in pairwise(deltas)]
        if all(0 < value < 1 for value in ratios):
            ratio, delta = median(ratios), deltas[-1]
            trend["ratio"] = ratio
            gain = delta * ratio / (1 - ratio)
            trend["extrapolated_remaining_gain"] = gain if math.isfinite(gain) else None
            tolerance = _finite_metric(tol)
            if tolerance is not None and tolerance > 0:
                trend["estimated_extra_sweeps"] = (
                    max(0, math.ceil((math.log(tolerance) - math.log(delta)) / math.log(ratio)))
                    if delta > tolerance
                    else 0
                )
    if inner or stop_reason == "criterion_satisfied" or averaged:
        status = "converged"
    elif not rows:
        status = "not_assessable"
    elif any("numerical_failure" in event for event in events):
        status = "numerical_failure"
    elif decreases:
        status = "elbo_decreasing"
    elif sign_changes >= 2 or any(
        row.get("cycle_lag") is not None or any("cycle" in event for event in event_set)
        for row, event_set in zip(rows, events, strict=True)
    ):
        status = "oscillating"
    elif trend["ratio"] is not None:
        status = "slow_monotone"
    else:
        status = "unclassified"
    messages = {
        "converged": (
            "The inner IBSS stopping rule was satisfied; review posterior stability "
            "alongside the credible sets."
        ),
        "not_assessable": (
            "No sweep trace was recorded; use track_fit=True on a future fit "
            "to assess its convergence pattern."
        ),
        "numerical_failure": (
            "The trace records a numerical failure; check summary statistics "
            "and LD consistency before using this fit."
        ),
        "elbo_decreasing": (
            "The ELBO decreased in the selected tail; inspect the trace "
            "and check summary statistics and LD consistency."
        ),
        "oscillating": (
            "The selected tail shows repeated or oscillating states; inspect "
            "the moving components and LD consistency."
        ),
        "slow_monotone": (
            "The ELBO gains are shrinking steadily; consider more iterations "
            "and inspect posterior stability."
        ),
        "unclassified": (
            "The recorded tail has no clear convergence pattern; inspect "
            "the trace and moving components."
        ),
    }
    stability, moving = _tail_stability(rows, ids, null_index, float(pip_stability))
    return dict(
        schema_version="prusie.convergence_assessment.v1",
        status=status,
        stop_reason=stop_reason,
        niter=niter,
        tol=tol,
        inner_ibss_converged=inner,
        refinement_converged=convergence.get("refinement_converged"),
        trace_available=bool(rows),
        tail=len(rows),
        elbo_tail=trend,
        inference_stability=stability,
        moving_components=moving,
        elbo_decrease_count=decreases,
        nonfinite_elbo_count=nonfinite,
        message=messages[status],
    )
