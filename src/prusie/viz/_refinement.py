"""Refinement plotting helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .._style import COLORS, format_number
from ..result import SusieResult
from ._common import (
    _axes,
    _display_labels,
    _layout,
    _local_style,
    _missing,
    _number,
    _table,
    _variant_ticks,
)


def _formal_sets(value):
    if isinstance(value, dict):
        return value.get("credible_sets", [])
    ids = list(value.variant_ids)
    return [
        {"component_id": int(c), "members": [str(ids[int(j)]) for j in m]}
        for c, m in zip(value.sets.get("cs_index", []), value.sets.get("cs", []), strict=True)
    ]


@_local_style
def plot_refinement(
    before: SusieResult | Mapping[str, Any],
    after: SusieResult | Mapping[str, Any] | None,
    *,
    comparison: Mapping[str, Any] | None = None,
    axes: Any = None,
    table_page: int = 0,
    table_page_size: int | None = None,
    layout: str = "full",
    candidates_only: bool | None = None,
    variant_ids: Sequence[str] | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Compare PIPs before and after refinement and show the search path.

    Parameters
    ----------
    before, after : SusieResult or dict
        Fits (or summaries) before and after :func:`prusie.refine_fit`.
    comparison : dict, optional
        Output of :func:`prusie.compare_fits` for the two fits.
    axes : sequence of matplotlib Axes, optional
        Existing axes.
    table_page, table_page_size : int, optional
        Page of the credible-set table to draw.
    layout : {'full', 'compact'}, default 'full'
        'compact' uses three panels without a table.
    candidates_only : bool, optional
        Show only the credible-set members of either fit, the ten leading
        PIPs and variants with PIP ≥ 0.1. Defaults to True for the compact
        layout and False otherwise.
    variant_ids : sequence of str, optional
        Variants to show together with the credible-set members.
    style : dict or False, optional
        Matplotlib rc settings applied to this figure only.

    Returns
    -------
    figure : matplotlib Figure
    axes : ndarray of Axes
    plotted_data : dict
        PIPs of both fits, ELBO gains from refits started at excluded-variant
        solutions, the accepted ELBO path and the credible sets.
    """
    compact = _layout(layout)
    from ..comparison import comparison_relations

    def values(value):
        if isinstance(value, dict):
            return {v["variant_id"]: v["pip"] for v in value["variants"]}
        return dict(zip(value.variant_ids.tolist(), value.pip.tolist(), strict=True))

    a, b = values(before), values(after)
    ids = list(dict.fromkeys([*a, *b]))
    av, bv = [a.get(v, np.nan) for v in ids], [b.get(v, np.nan) for v in ids]
    refinement = (
        after.get("refinement", {}) if isinstance(after, dict) else getattr(after, "refinement", {})
    )
    history = refinement.get("history", [])
    objectives = [_number(row.get("candidate_objective", row.get("objective"))) for row in history]
    improvements = [_number(row.get("improvement")) for row in history]
    before_cs, after_cs = _formal_sets(before), _formal_sets(after)
    candidates_only = compact if candidates_only is None else bool(candidates_only)
    cs_ids = {v for cs in before_cs + after_cs for v in cs["members"]}
    if variant_ids is not None:
        if any(v not in ids for v in variant_ids):
            raise ValueError("A selected refinement variant ID is absent")
        selected_ids = set(variant_ids) | cs_ids
    elif candidates_only:
        top = sorted(ids, key=lambda v: (-max(a.get(v, 0), b.get(v, 0)), str(v)))[:10]
        selected_ids = set(top) | cs_ids | {v for v in ids if max(a.get(v, 0), b.get(v, 0)) >= 0.1}
    else:
        selected_ids = set(ids)
    shown_ids = (
        [
            v
            for v in sorted(ids, key=lambda v: (-max(a.get(v, 0), b.get(v, 0)), str(v)))
            if v in selected_ids
        ]
        if candidates_only or variant_ids is not None
        else ids
    )
    relations = comparison_relations(comparison, baseline_only=True) if comparison else []
    table_rows, other_attempts = [], []
    for i, row in sorted(
        enumerate(history), key=lambda item: (not item[1].get("accepted", False), item[0])
    ):
        (table_rows if row.get("accepted") else other_attempts).append(
            [
                f"Attempt {i + 1}, round {row.get('round', '?')}",
                "",
                "",
                "exclude " + ", ".join(map(str, row.get("excluded_ids", []))),
                ("ACCEPTED; " if row.get("accepted") else "") + str(row.get("reason", "attempted")),
            ]
        )
    before_members = {tuple(sorted(cs["members"])) for cs in before_cs}
    after_members = {tuple(sorted(cs["members"])) for cs in after_cs}
    cs_rows = []
    for stage, sets, opposite in (
        ("Before CS", before_cs, after_members),
        ("After CS", after_cs, before_members),
    ):
        for cs in sets:
            changed = tuple(sorted(cs["members"])) not in opposite
            cs_rows.append(
                (
                    not changed,
                    [
                        stage,
                        str(cs["component_id"]) if stage == "Before CS" else "",
                        str(cs["component_id"]) if stage == "After CS" else "",
                        ", ".join(map(str, cs["members"])),
                        "changed membership" if changed else "same member set (matching separate)",
                    ],
                )
            )
    table_rows.extend(row for _, row in sorted(cs_rows, key=lambda item: item[0]))
    for relation in relations:
        table_rows.append(
            [
                "Signal relation",
                str(relation.get("component_a", "")),
                str(relation.get("component_b", "")),
                "overlap="
                + (
                    f"{relation['posterior_overlap']:.5g}"
                    if relation.get("posterior_overlap") is not None
                    else "not assessed"
                ),
                relation.get("status", "not assessed"),
            ]
        )
    table_rows.extend(other_attempts)
    shown_count = min(len(table_rows), table_page_size) if table_page_size else len(table_rows)
    count = 3 if compact or (axes is not None and np.asarray(axes, dtype=object).size == 3) else 4
    legacy_three = count == 3 and not compact
    table_height = max(2.6, 0.38 * shown_count)
    figure, axs = _axes(
        axes,
        count,
        figsize=(8.5, 7.8) if compact else (11, 2.6 * (count - 1) + table_height),
        height_ratios=None if compact else [2.6] * (count - 1) + [table_height],
    )
    x = np.arange(len(shown_ids))
    axs[0].plot(
        x - 0.08, [a.get(v, np.nan) for v in shown_ids], "o", label="Before", color=COLORS["before"]
    )
    axs[0].plot(
        x + 0.08, [b.get(v, np.nan) for v in shown_ids], "x", label="After", color=COLORS["after"]
    )
    axs[0].set_ylabel("Marginal PIP")
    _variant_ticks(axs[0], shown_ids, priority_indices=range(min(3, len(shown_ids))))
    axs[0].set_xlabel(
        f"{'Main candidates' if candidates_only else 'Variants'} ({len(shown_ids)} of {len(ids)})"
    )
    axs[0].legend()
    candidate_states = []
    for row, delta in zip(history, improvements, strict=True):
        stages = row.get("stages", {})
        if row.get("accepted"):
            state = "Accepted"
        elif any(
            stage.get("ibss_converged") is False for stage in stages.values()
        ) or "not_converged" in str(row.get("reason")):
            state = "Not converged"
        elif not np.isfinite(delta):
            state = "Failed / unavailable"
        elif delta <= refinement.get("config", {}).get("absolute_tol", 0):
            state = "Below acceptance threshold"
        else:
            state = "Valid, not selected"
        candidate_states.append(state)
    markers = {
        "Accepted": ("o", COLORS["accepted"]),
        "Not converged": ("X", COLORS["flagged"]),
        "Failed / unavailable": ("s", COLORS["missing"]),
        "Below acceptance threshold": ("x", COLORS["rejected"]),
        "Valid, not selected": ("^", COLORS["conditional"]),
    }
    for state, (marker, color) in markers.items():
        indices = [i for i, value in enumerate(candidate_states) if value == state]
        if indices:
            numeric = [i for i in indices if np.isfinite(improvements[i])]
            missing = [i for i in indices if not np.isfinite(improvements[i])]
            if numeric:
                axs[1].scatter(
                    np.asarray(numeric) + 1,
                    np.asarray(improvements)[numeric],
                    marker=marker,
                    color=color,
                    label=state,
                )
            if missing:
                axs[1].scatter(
                    np.asarray(missing) + 1,
                    np.full(len(missing), 0.03),
                    marker=marker,
                    color=color,
                    transform=axs[1].get_xaxis_transform(),
                    label=state + " (no objective)",
                )
    axs[1].axhline(0, color=COLORS["muted"], lw=0.7)
    threshold = refinement.get("config", {}).get("absolute_tol")
    if threshold is not None:
        axs[1].axhline(
            threshold,
            color=COLORS["muted"],
            ls="--",
            label="Accept above " + format_number(threshold),
        )
    axs[1].set_xlabel("Refit attempt (round shown below)")
    axs[1].set_ylabel("ELBO gain over current best")
    if history:
        axs[1].legend(fontsize=9, ncol=2 if len(set(candidate_states)) > 2 else 1)
        axs[1].set_xticks(
            np.arange(len(history)) + 1,
            [f"{i + 1}\nR{row.get('round', '?')}" for i, row in enumerate(history)],
        )
    incumbent_path, candidate_records = [], []
    for i, row in enumerate(history):
        objective, delta = objectives[i], improvements[i]
        incumbent = _number(row.get("incumbent_objective"))
        if not np.isfinite(incumbent) and np.isfinite(objective) and np.isfinite(delta):
            incumbent = objective - delta
        round_id = row.get("round", 1)
        candidate_records.append(
            {
                "attempt": i + 1,
                "round": round_id,
                "incumbent_objective": incumbent,
                "B_objective": objective,
                "improvement": delta,
                "accepted": bool(row.get("accepted")),
                "display_status": candidate_states[i],
            }
        )
        if np.isfinite(incumbent) and not any(v["round"] == round_id for v in incumbent_path):
            incumbent_path.append(
                {"round": round_id, "objective": incumbent, "stage": "round incumbent"}
            )
        if row.get("accepted") and np.isfinite(objective):
            incumbent_path.append(
                {"round": round_id, "objective": objective, "stage": "accepted B candidate"}
            )
    if not legacy_three and incumbent_path:
        axs[2].plot(
            range(len(incumbent_path)),
            [row["objective"] for row in incumbent_path],
            "o-",
            color=COLORS["after"],
        )
        axs[2].set_xticks(
            range(len(incumbent_path)),
            [
                f"Round {row['round']}\n"
                + ("Accepted refit" if row["stage"] == "accepted B candidate" else "Current best")
                for row in incumbent_path
            ],
            fontsize=9,
        )
    elif not legacy_three:
        _missing(axs[2], "Accepted incumbent trajectory not recorded")
    if not legacy_three:
        axs[2].set_ylabel("Full-model ELBO")
    best = refinement.get("best_objective")
    axs[1 if legacy_three else 2].set_title(
        "Final best: "
        + (format_number(best, precision=7) if best is not None else "not recorded")
        + ("; incumbent path in returned data (3-axis layout)" if legacy_three else ""),
        fontsize=10,
    )
    if not history:
        _missing(axs[1], (f"No candidate history: {refinement.get('stop_reason', 'not recorded')}"))
    if table_rows and not compact:
        _table(
            axs[-1],
            table_rows,
            ["Record", "Before component", "After component", "Members / overlap", "Status"],
            page=table_page,
            page_size=table_page_size,
        )
    elif not compact:
        _missing(axs[-1], "No formal CS or signal relationships supplied")
    if comparison is None and not compact:
        axs[-1].set_title(
            ("Matching not assessed; equal component IDs do not establish a match"), fontsize=9
        )
    if not compact:
        axs[-1].axis("off")
    for ax in axs[: 3 if not legacy_three else 2]:
        ax.grid(axis="y")
    figure.suptitle(
        f"Refinement · {sum(bool(row.get('accepted')) for row in history)}"
        f" accepted / {len(history)}"
        f" attempts"
    )
    return (
        figure,
        axs,
        {
            "variant_ids": ids,
            "before_pip": av,
            "after_pip": bv,
            "history": history,
            "candidate_objectives": objectives,
            "improvements": improvements,
            "signal_relations": relations,
            "before_cs": before_cs,
            "after_cs": after_cs,
            "table_rows": table_rows,
            "candidate_records": candidate_records,
            "incumbent_path": incumbent_path,
            "all_component_relations": comparison.get("component_relations", [])
            if comparison
            else [],
            "table_page": table_page,
            "table_page_size": table_page_size,
            "layout": layout,
            "displayed_variant_ids": shown_ids,
            "candidates_only": candidates_only,
            "variant_label_map": dict(zip(shown_ids, _display_labels(shown_ids), strict=True)),
            "candidate_selection": (
                "explicit variant IDs + all formal CS members"
                if variant_ids is not None
                else "all formal CS members + top ten PIPs + PIP ≥ 0.1"
                if candidates_only
                else "full region"
            ),
            "candidate_statuses": candidate_states,
            "objective_definition": refinement.get("objective_definition") or "not recorded",
        },
    )
