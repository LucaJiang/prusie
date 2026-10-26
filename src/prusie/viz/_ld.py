"""Ld plotting helpers."""

from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .._style import COLORS
from ._common import _axes, _layout, _local_style, _missing, _number, _table


@_local_style
def plot_ld_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    axes: Any = None,
    table_page: int = 0,
    table_page_size: int | None = None,
    layout: str = "full",
    auxiliary: str = "residual",
    label_variants: Sequence[str] | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Plot the LD diagnostics of :func:`prusie.diagnose_ld`.

    Parameters
    ----------
    diagnostics : dict
        Output of :func:`prusie.diagnose_ld` or :func:`prusie.kriging_rss`.
    axes : sequence of matplotlib Axes, optional
        Existing axes.
    table_page, table_page_size : int, optional
        Page of the table of flagged and unassessed variants.
    layout : {'full', 'compact'}, default 'full'
        'compact' omits the table and returns its rows as data.
    auxiliary : {'residual', 'log_lr'}, default 'residual'
        Second panel: standardized conditional residuals or flip log
        likelihood ratios.
    label_variants : sequence of str, optional
        Variants to annotate; defaults to up to three flagged variants or
        variants with the largest residuals.
    style : dict or False, optional
        Matplotlib rc settings applied to this figure only.

    Returns
    -------
    figure : matplotlib Figure
    axes : ndarray of Axes
    plotted_data : dict
        Plotted values, the status of each variant and the reason each label
        was chosen.

    Notes
    -----
    Assessed and flagged, assessed and not flagged, conditional-only and
    unassessed variants are shown separately. A large residual alone is not
    labelled as a flagged allele flip.
    """
    compact = _layout(layout)
    if auxiliary not in ("residual", "log_lr"):
        raise ValueError("auxiliary must be 'residual' or 'log_lr'")
    from ..ld import diagnostic_counts, diagnostic_input_missing, diagnostic_status

    rows = diagnostics.get("variants", [])
    states = np.asarray([diagnostic_status(row) for row in rows])
    details = [row for row, state in zip(rows, states, strict=True) if state != "not_flagged"]
    shown_count = min(len(details), table_page_size) if table_page_size else len(details)
    figure, axs = _axes(
        axes,
        2 if compact else 3,
        figsize=(8.5, 6.3) if compact else (11, 5.6 + max(2.6, 0.32 * shown_count)),
    )
    expected = np.asarray([_number(row.get("expected_z")) for row in rows])
    # Saved records may contain missing z-scores replaced by zero; these are
    # not observations.
    missing = np.asarray([diagnostic_input_missing(row) for row in rows], dtype=bool)
    observed = np.asarray([_number(row.get("observed_z")) for row in rows])
    residual = np.asarray([_number(row.get("standardized_residual")) for row in rows])
    observed[missing], residual[missing] = np.nan, np.nan
    log_lr = np.asarray([_number(row.get("log_likelihood_ratio")) for row in rows])
    log_lr[missing | np.isin(states, ["conditional_only", "not_assessed"])] = np.nan
    auxiliary_values = residual if auxiliary == "residual" else log_lr
    styles = (
        ("not_flagged", "o", "Assessed, not flagged"),
        ("flagged", "x", "Assessed, flagged"),
        ("conditional_only", "^", "Conditional only; flip not assessed"),
        ("not_assessed", "s", "Not assessed / missing input"),
    )
    for state, marker, label in styles:
        mask = states == state
        if np.any(mask):
            finite_pair = mask & np.isfinite(expected) & np.isfinite(observed)
            finite_residual = mask & np.isfinite(auxiliary_values)
            color = COLORS[
                {
                    "not_flagged": "not_flagged",
                    "flagged": "flagged",
                    "conditional_only": "conditional",
                    "not_assessed": "missing",
                }[state]
            ]
            # Empty scatter still records the unavailable class in the legend.
            axs[0].scatter(
                expected[finite_pair],
                observed[finite_pair],
                marker=marker,
                label=f"{label} ({int(mask.sum())})",
                color=color,
                s=24,
            )
            axs[1].scatter(
                np.flatnonzero(finite_residual),
                auxiliary_values[finite_residual],
                marker=marker,
                label=label,
                color=color,
                s=24,
            )
    finite = np.concatenate((expected[np.isfinite(expected)], observed[np.isfinite(observed)]))
    if len(finite):
        axs[0].plot([min(finite), max(finite)], [min(finite), max(finite)], ls="--", color="0.5")
    axs[0].set_xlabel("Expected z (diagnostic model)")
    axs[0].set_ylabel("Observed z")
    axs[0].set_title("Diagnostic scale; missing observations omitted", fontsize=9)
    if rows:
        axs[0].legend(fontsize=9)
    axs[1].set_xlabel("Variant index (zero-based)")
    if rows:
        from matplotlib.ticker import MaxNLocator

        axs[1].set_xlim(-0.5, len(rows) - 0.5)
        axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].set_ylabel(
        "Standardized observed residual" if auxiliary == "residual" else "Flip log likelihood ratio"
    )
    axs[1].axhline(0, color=COLORS["muted"], lw=0.7)
    if not compact:
        axs[2].axis("off")
    counts = diagnostic_counts(diagnostics)
    summary = (
        f"Assessed: {counts['assessed']}"
        f"; flagged: {counts['flagged']}"
        f"; "
        f"conditional only: {counts['conditional_only']}"
        f"; unassessed: {counts['unassessed']}"
    )
    (axs[0] if compact else axs[2]).set_title(summary, fontsize=9)
    if details and not compact:
        _table(
            axs[2],
            [
                [
                    r.get("variant_id"),
                    r.get("standardized_residual") if not diagnostic_input_missing(r) else None,
                    r.get("log_likelihood_ratio"),
                    diagnostic_status(r),
                    r.get("unavailable_reason"),
                ]
                for r in details
            ],
            labels=[
                "Variant",
                "Observed residual",
                "Log likelihood ratio",
                "Assessment",
                "Unavailable reason",
            ],
            page=table_page,
            page_size=table_page_size,
        )
    elif not compact:
        _missing(
            axs[2],
            "No flags among assessed variants"
            if counts["assessed"]
            else "LD diagnostics not assessed",
        )
    thresholds = diagnostics.get("thresholds", {})
    mixture = diagnostics.get("mixture", {})
    rule = (
        f"Flag requires log LR > {thresholds.get('flip_log_lr', 'not recorded')}"
        f" AND "
        f"|adjusted z| > {thresholds.get('flip_abs_z', 'not recorded')}"
    )
    axs[1].set_title(rule + "\nMixture: " + str(mixture.get("status", "not recorded")), fontsize=9)
    if auxiliary == "log_lr" and thresholds.get("flip_log_lr") is not None:
        axs[1].axhline(thresholds["flip_log_lr"], color=COLORS["flagged"], ls="--")
    ids = [row.get("variant_id", str(i)) for i, row in enumerate(rows)]
    if label_variants is None:
        chosen = sorted(
            (i for i in range(len(rows)) if np.isfinite(residual[i])),
            key=lambda i: (states[i] != "flagged", -abs(residual[i]), str(ids[i])),
        )[:3]
    else:
        if any(value not in ids for value in label_variants):
            raise ValueError("An annotated LD variant ID is absent")
        chosen = [ids.index(value) for value in dict.fromkeys(label_variants)]
    annotations = []
    for offset, i in enumerate(chosen):
        reason = (
            "explicit selection"
            if label_variants is not None
            else "allele-flip flag"
            if states[i] == "flagged"
            else "extreme residual; not a flip flag"
        )
        annotations.append(dict(variant_id=ids[i], reason=reason, assessment=states[i]))
        if np.isfinite(auxiliary_values[i]):
            axs[1].annotate(
                textwrap.fill(str(ids[i]), 24),
                (i, auxiliary_values[i]),
                xytext=(4, 8 + 9 * (offset % 2)),
                textcoords="offset points",
                fontsize=9,
                arrowprops=dict(arrowstyle="-", color=COLORS["muted"], lw=0.6),
            )
    for ax in axs[:2]:
        ax.grid(alpha=0.6)
        ax.margins(y=0.18)
    if compact:
        axs[1].set_xlabel("Variant index · labels: flags first, then extreme residuals (not flags)")
        if not counts["assessed"]:
            status = mixture.get("reason") or (
                "missing inputs" if counts["unassessed"] else "not assessed"
            )
            axs[0].set_title(
                summary
                + "\nFlip assessment unavailable: "
                + textwrap.fill(str(status).replace("_", " "), 80),
                fontsize=9,
            )
    figure.suptitle("LD compatibility diagnostics")
    return (
        figure,
        axs,
        {
            "variants": rows,
            "flagged_or_unassessed": details,
            "assessment_states": states.tolist(),
            "observed_z": observed,
            "expected_z": expected,
            "standardized_residual": residual,
            "status_counts": counts,
            "assessment_summary": summary,
            "thresholds": thresholds,
            "mixture": mixture,
            "regularization": diagnostics.get("regularization"),
            "method": diagnostics.get("method"),
            "layout": layout,
            "auxiliary": auxiliary,
            "log_likelihood_ratio": log_lr,
            "annotations": annotations,
            "flag_rule": rule,
        },
    )
