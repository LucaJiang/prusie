"""Convergence plotting helpers."""

from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .._style import COLORS, format_number
from ..result import SusieResult
from ._common import _axes, _bottom_labels, _local_style, _missing, _number


def _plot_convergence_stage(
    fit_or_summary,
    *,
    axes=None,
    variance_components=None,
    layout="full",
    relative_elbo=False,
    color=None,
):
    getter = (
        fit_or_summary.get
        if isinstance(fit_or_summary, dict)
        else lambda k, d=None: getattr(fit_or_summary, k, d)
    )
    trace = getter("trace", []) or []
    summary = getter("convergence", {}) or {}
    count = {"full": 6, "compact": 4, "variance": 2}[layout]
    figure, axs = _axes(axes, count, figsize=(8.5, 2.1 * count + 0.6))
    color = color or COLORS["before"]
    iterations = np.asarray([row.get("iteration", i + 1) for i, row in enumerate(trace)])
    criterion = summary.get(
        "effective_criterion", summary.get("requested_criterion", "not recorded")
    )
    requested = summary.get("requested_criterion", "not recorded")
    keys = (
        ()
        if layout == "variance"
        else ("elbo", "delta_elbo", "max_abs_delta_alpha", "max_abs_delta_pip")
    )
    metric_labels = (
        "ELBO − first recorded ELBO" if relative_elbo else "Logged ELBO",
        "Signed ΔELBO",
        "Max |Δalpha|",
        "Max |ΔPIP|",
    )
    origin = None
    if not trace:
        for ax in axs:
            _missing(ax, "Trace not recorded; history cannot be reconstructed")
    # Variance axes are handled separately; visit only the available metric axes here.
    for ax, key, label in zip(axs, keys, metric_labels, strict=False):
        values = np.asarray([_number(row.get(key)) for row in trace])
        if key == "elbo" and relative_elbo:
            finite = values[np.isfinite(values)]
            origin = float(finite[0]) if len(finite) else None
            if origin is not None:
                values = values - origin
        ax.plot(iterations, values, marker="o", color=color)
        ax.set_ylabel(label)
        ax.grid(axis="y")
        if key == "delta_elbo":
            ax.axhline(0, color=COLORS["muted"], lw=0.7)
            ax.set_yscale("symlog", linthresh=1e-8)
        tolerance = summary.get(
            {
                "delta_elbo": "elbo_tol",
                "max_abs_delta_alpha": "alpha_tol",
                "max_abs_delta_pip": "pip_tol",
            }.get(key, "")
        )
        if tolerance is None:
            tolerance = summary.get("tolerances", {}).get(
                {
                    "delta_elbo": "elbo",
                    "max_abs_delta_alpha": "alpha",
                    "max_abs_delta_pip": "pip",
                }.get(key, "")
            )
        if tolerance is None and key == "delta_elbo":
            tolerance = summary.get("tol")
        if criterion == "pip" and key in ("max_abs_delta_alpha", "max_abs_delta_pip"):
            tolerance = summary.get("tol")
        if tolerance is not None:
            selected = (
                criterion == "joint"
                or (key == "delta_elbo" and criterion == "elbo")
                or (
                    key != "delta_elbo"
                    and criterion in ("pip", "alpha_pip", "alpha_pip_nonfinite_fallback")
                )
            )
            if layout == "compact":
                threshold_label = (
                    ("Stopping threshold" if selected else "Diagnostic only")
                    + ": "
                    + format_number(tolerance)
                )
            else:
                threshold_label = (
                    "Selected stopping threshold"
                    if selected
                    else "Diagnostic threshold (not selected stopping criterion)"
                ) + f": {tolerance:.5g}"
            ax.axhline(
                tolerance,
                ls="--" if selected else ":",
                color=COLORS["text"] if selected else COLORS["muted"],
                label=threshold_label,
            )
            ax.legend(fontsize=9, loc="best")
        if key == "delta_elbo":
            from matplotlib.ticker import SymmetricalLogLocator

            finite = values[np.isfinite(values)]
            extent = np.append(finite, tolerance) if tolerance is not None else finite
            if len(extent):
                low, high = min(0.0, float(np.min(extent))), max(0.0, float(np.max(extent)))
                if low == high:
                    low, high = -1e-8, 1e-8
                ax.set_ylim(low * 1.15, high * 1.15)
            locator = SymmetricalLogLocator(base=10, linthresh=1e-8)
            locator.set_params(numticks=6)
            ax.yaxis.set_major_locator(locator)
    v_rows = [row.get("V") for row in trace]
    shown_components, zero_components, omitted_components = [], [], []
    variance_note = ""
    if layout in ("full", "variance"):
        va, sa = axs[-2:]
        if trace and all(v is not None for v in v_rows):
            width = max(len(v) for v in v_rows)
            complete = np.asarray(
                [[v[i] if len(v) > i else np.nan for i in range(width)] for v in v_rows]
            )
            zero_components = [i for i in range(width) if np.all(complete[:, i] == 0)]
            chosen = (
                [i for i in range(width) if i not in zero_components]
                if variance_components is None
                else list(variance_components)
            )
            if any(i not in range(width) for i in chosen):
                raise ValueError("Unknown variance component ID")
            chosen.sort(
                key=lambda i: (-float(np.nanmax(complete[:, i]) - np.nanmin(complete[:, i])), i)
            )
            limit = 6 if layout == "variance" and variance_components is None else len(chosen)
            shown_components, omitted_components = chosen[:limit], chosen[limit:]
            for i in shown_components:
                va.plot(
                    iterations,
                    complete[:, i],
                    color=color,
                    alpha=max(0.45, 1.0 - 0.1 * (i % 5)),
                    linestyle=("-", "--", "-.", ":")[i % 4],
                    label=f"Component {i}",
                )
            if shown_components:
                va.legend(fontsize=9, ncol=2 if len(shown_components) > 3 else 1, loc="best")
            notes = []
            if zero_components:
                notes.append(f"{len(zero_components)} always-zero components")
            if omitted_components:
                notes.append(f"{len(omitted_components)} further components in complete trace")
            if notes:
                variance_note = "; ".join(notes)
                va.set_title(variance_note, fontsize=9)
        elif trace:
            _missing(va, "Component variance history not recorded")
        va.set_ylabel("Working effect variance V" + (" (fixed)" if summary.get("V_fixed") else ""))
        sa.plot(iterations, [_number(row.get("sigma2")) for row in trace], color=color, marker="o")
        sa.set_ylabel("Residual variance σ²" + (" (fixed)" if summary.get("sigma2_fixed") else ""))
        va.grid(axis="y")
        sa.grid(axis="y")
    events = [
        {"iteration": int(iteration), "events": row.get("events")}
        for iteration, row in zip(iterations, trace, strict=True)
        if row.get("events")
    ]
    for ax in axs:
        if len(iterations):
            from matplotlib.ticker import MaxNLocator

            ax.axvline(
                iterations[-1],
                color=COLORS["muted"],
                ls=":",
                lw=0.8,
                label="Actual stopping iteration",
            )
            # Missing first-sweep changes must not give each metric a different
            # x scale when upper tick labels are intentionally hidden.
            span = max(float(np.ptp(iterations)), 1.0)
            ax.set_xlim(
                float(np.min(iterations)) - 0.04 * span, float(np.max(iterations)) + 0.04 * span
            )
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("IBSS iteration (recorded phase only)")
    if layout != "full":
        _bottom_labels(axs, "IBSS iteration")
    phases = sorted({str(row.get("phase", "not recorded")) for row in trace})
    axs[0].set_title(
        (
            f"Requested: {requested}"
            f" · effective: {criterion}"
            f"\n"
            f"Stop: {summary.get('stop_reason', 'not assessed')}"
        )
        + (f" at iteration {iterations[-1]}" if len(iterations) else "")
        + (
            "\n" + textwrap.fill(variance_note, 58)
            if layout == "variance" and variance_note
            else ""
        ),
        fontsize=10,
    )
    return (
        figure,
        axs,
        {
            "trace": trace,
            "convergence": summary,
            "events": events,
            "missing_trace": not bool(trace),
            "phases": phases,
            "variance_components_shown": shown_components,
            "zero_variance_components": zero_components,
            "omitted_variance_components": omitted_components,
            "finalization_separate_from_iterations": summary.get("finalization"),
            "layout": layout,
            "relative_elbo": relative_elbo,
            "elbo_origin": origin,
        },
    )


@_local_style
def plot_convergence(
    fit_or_summary: SusieResult | Mapping[str, Any],
    *,
    after: SusieResult | Mapping[str, Any] | None = None,
    axes: Any = None,
    variance_components: Sequence[int] | None = None,
    layout: str = "full",
    relative_elbo: bool = False,
) -> tuple[Any, Any, dict[str, Any]]:
    """Plot convergence metrics of IBSS against the sweep number.

    Parameters
    ----------
    fit_or_summary : SusieResult or dict
        Fit with ``track_fit`` enabled, or its summary.
    after : SusieResult or dict, optional
        A second fit, for example after continuation or refinement, plotted
        on its own sweep axis.
    axes : sequence of matplotlib Axes, optional
        Existing axes.
    variance_components : sequence of int, optional
        Effects whose prior variance is shown in the 'variance' layout; by
        default up to six effects with changing or nonzero V.
    layout : {'full', 'compact', 'variance'}, default 'full'
        'full': six panels per fit; 'compact': ELBO, ΔELBO, max |Δα| and
        max |ΔPIP|; 'variance': V and σ².
    relative_elbo : bool, default False
        Plot the ELBO relative to its first finite value; returned trace
        values are unchanged.
    style : dict or False, optional
        Matplotlib rc settings applied to this figure only.

    Returns
    -------
    figure : matplotlib Figure
    axes : ndarray of Axes
    plotted_data : dict
        Plotted traces and the requested and effective stopping rules.
    """
    if layout not in ("full", "compact", "variance"):
        raise ValueError("layout must be 'full', 'compact' or 'variance'")
    count = {"full": 6, "compact": 4, "variance": 2}[layout]
    if after is None:
        out = _plot_convergence_stage(
            fit_or_summary,
            axes=axes,
            variance_components=variance_components,
            layout=layout,
            relative_elbo=relative_elbo,
        )
        out[0].suptitle("Variance details" if layout == "variance" else "Fit convergence")
        return out
    if axes is None:
        import matplotlib.pyplot as plt

        figure, grid = plt.subplots(
            count, 2, figsize=(10, 2.1 * count + 1.1), constrained_layout=True
        )
        axs = grid.T.reshape(-1)
    else:
        figure, axs = _axes(axes, count * 2)
    refinement = (
        after.get("refinement", {}) if isinstance(after, dict) else getattr(after, "refinement", {})
    )
    accepted = any(row.get("accepted") for row in refinement.get("history", []))
    after_label = (
        "Accepted after"
        if accepted
        else "After search (unchanged)"
        if refinement.get("history")
        else "After"
    )
    stages = {}
    for label, value, stage_axes, color in (
        ("Before", fit_or_summary, axs[:count], COLORS["before"]),
        (after_label, after, axs[count:], COLORS["after"]),
    ):
        _, _, data = _plot_convergence_stage(
            value,
            axes=stage_axes,
            variance_components=variance_components,
            layout=layout,
            relative_elbo=relative_elbo,
            color=color,
        )
        stages[label] = data
        stage_axes[0].set_title(label + "\n" + stage_axes[0].get_title(), fontsize=10)
        if layout == "full":
            for ax in stage_axes:
                ax.set_xlabel(f"{label}: IBSS iteration (separate iteration axis)")
        else:
            stage_axes[-1].set_xlabel(f"{label}: IBSS iteration")
    figure.suptitle(
        ("Variance details · colors identify fit stage; component IDs are local")
        if layout == "variance"
        else "Before and returned-after convergence"
    )
    return (
        figure,
        axs,
        {
            "stages": stages,
            "independent_iteration_axes": True,
            "refinement_candidate_accepted": accepted,
            "after_label": after_label,
            "layout": layout,
        },
    )
