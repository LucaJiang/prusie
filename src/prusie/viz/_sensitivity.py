"""Sensitivity plotting helpers."""

from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .._style import COLORS, annotation_color, format_number, format_pip_cell
from ._common import (
    _axes,
    _bottom_labels,
    _component_color,
    _display_labels,
    _layout,
    _local_style,
    _missing,
    _number,
    _table,
    _variant_ticks,
)
from ._locus import _select_component


@_local_style
def plot_sensitivity(
    comparison: Mapping[str, Any] | None,
    *,
    candidates_only: bool | None = True,
    axes: Any = None,
    layer: str = "signals",
    baseline_only: bool = True,
    table_page: int = 0,
    table_page_size: int | None = None,
    layout: str = "full",
    display_names: Mapping[str, str] | None = None,
    cmap: str | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Plot a heatmap of candidate PIPs across the fits of a comparison.

    Parameters
    ----------
    comparison : dict
        Output of :func:`prusie.compare_fits`, or the ``comparison`` entry of
        :func:`prusie.run_sensitivity`.
    candidates_only : bool, default True
        Show the candidate variants of the comparison; False shows all.
    axes : sequence of matplotlib Axes, optional
        Existing axes.
    layer : {'signals', 'components'}, default 'signals'
        'signals' shows retained credible sets and posterior-overlap
        relations to unresolved effects; 'components' includes all active
        or CS-retained effects.
    baseline_only : bool, default True
        Show only relations to the baseline.
    table_page, table_page_size : int, optional
        Page of the relation table to draw.
    layout : {'full', 'compact'}, default 'full'
        'compact' shows every candidate with a short caption.
    display_names : mapping of str to str, optional
        Display labels of the settings; stored names are unchanged.
    cmap : str or Colormap, optional
        Colormap of the PIPs, on a fixed linear [0, 1] scale; defaults to Blues.
    style : dict or False, optional
        Matplotlib rc settings applied to this figure only.

    Returns
    -------
    figure : matplotlib Figure
    axes : ndarray of Axes
    plotted_data : dict
        PIP matrix, cell display text and colors, the largest PIP change over
        all compared variants, relations and the candidate rule.

    Notes
    -----
    Variants absent from a fit are shown as missing (NA), not as zero PIP.
    The baseline is the first column.
    """
    import matplotlib.pyplot as plt

    from ..comparison import comparison_relations

    compact = _layout(layout)
    display_names = dict(display_names or {})
    raw_settings = list(comparison.get("settings", []))
    if set(display_names) - set(raw_settings):
        raise ValueError("display_names contains unknown sensitivity settings")
    baseline = comparison.get("baseline")
    settings = (
        [baseline] + [s for s in raw_settings if s != baseline]
        if baseline in raw_settings
        else raw_settings
    )
    labels = [str(display_names.get(s, s)) for s in settings]
    relations = comparison_relations(comparison, layer=layer, baseline_only=baseline_only)
    relations = sorted(relations, key=lambda row: row.get("status") == "matched")
    all_rows = comparison.get("variants", [])
    rows = [r for r in all_rows if r.get("candidate", True)] if candidates_only else list(all_rows)
    if compact:
        # Retain every selected candidate, ordering by maximum observed PIP.
        rows = sorted(
            rows,
            key=lambda r: (
                -max((_number(v) for v in r["pips"].values() if v is not None), default=0.0),
                str(r["variant_id"]),
            ),
        )
    matrix = np.asarray(
        [[_number(row["pips"].get(s)) for s in settings] for row in rows], dtype=float
    )
    from matplotlib.colors import Normalize

    tick_labels = [
        "\n".join(textwrap.fill(line, 12 if compact else 16) for line in label.splitlines())
        for label in labels
    ]
    line_count = max((label.count("\n") + 1 for label in tick_labels), default=1)
    row_labels = _display_labels([r["variant_id"] for r in rows])
    label_width = min(2.1, 0.065 * max((len(label) for label in row_labels), default=1))
    heat_height = max(0.95, 0.30 * len(rows)) if compact else max(3.2, 0.26 * len(rows))
    shown_count = min(len(relations), table_page_size) if table_page_size else len(relations)
    detail_height = (
        (0.45 if np.any(~np.isfinite(matrix)) else 0.24)
        if compact
        else max(2.6, 0.32 * shown_count)
    )
    width = (
        max(3.1 if len(settings) <= 2 else 4.8, 0.78 * len(settings) + label_width + 1.4)
        if compact
        else max(8.5, min(12.0, 5.5 + 0.9 * len(settings)))
    )
    height = (
        heat_height + detail_height + 0.85 + 0.14 * line_count
        if compact
        else heat_height + detail_height
    )
    figure, axs = _axes(
        axes, 2, figsize=(width, height), height_ratios=[heat_height, detail_height]
    )
    annotated = bool(len(rows) <= 18 and len(settings) <= 7 and matrix.size <= 100)
    text_matrix = [[format_pip_cell(value) for value in row] for row in matrix]
    text_colors = []
    color_map = plt.get_cmap(cmap or COLORS["pip_cmap"]).copy()
    color_map.set_bad(COLORS["missing"])
    norm = Normalize(0, 1)
    if rows and settings:
        obj = axs[0].imshow(
            np.ma.masked_invalid(matrix),
            cmap=color_map,
            norm=norm,
            aspect="auto",
            interpolation="none",
        )
        # Limit strip-like cells without requiring square PIP cells. User-supplied
        # axes retain their own size; the data and labels remain identical.
        if compact and axes is None:
            axs[0].set_box_aspect(heat_height / max(0.78 * len(settings), 0.78))
        if compact:
            cax = axs[0].inset_axes([1.04, 0.07, 0.035, 0.86])
            colorbar = figure.colorbar(obj, cax=cax, ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
        else:
            colorbar = figure.colorbar(
                obj,
                ax=axs[0],
                fraction=0.035,
                pad=0.035,
                shrink=0.86,
                ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
            )
        colorbar.ax.set_title("PIP", fontsize=10, pad=5)
        colorbar.ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
        axs[0].set_xticks(
            range(len(settings)),
            tick_labels,
            rotation=0 if len(settings) <= 6 else 35,
            ha="center" if len(settings) <= 6 else "right",
        )
        # Every selected candidate is labelled; no twelve-row thinning here.
        axs[0].set_yticks(range(len(rows)), row_labels)
        row_axis_label = "Candidate variant" if candidates_only else "Variant"
        if compact and label_width > 1.2:
            axs[0].text(
                -0.035,
                1.02,
                row_axis_label,
                ha="right",
                va="bottom",
                transform=axs[0].transAxes,
                fontsize=10,
            )
        else:
            axs[0].set_ylabel(row_axis_label, labelpad=1)
        axs[0].tick_params(axis="both", length=0, pad=5)
        for spine in axs[0].spines.values():
            spine.set_visible(False)
        if matrix.size <= 250:
            axs[0].set_xticks(np.arange(len(settings) + 1) - 0.5, minor=True)
            axs[0].set_yticks(np.arange(len(rows) + 1) - 0.5, minor=True)
            axs[0].grid(which="minor", color="white", linewidth=0.4)
            axs[0].tick_params(which="minor", bottom=False, left=False)
        text_colors = [
            [
                annotation_color(
                    color_map(norm(value)) if np.isfinite(value) else color_map.get_bad(),
                    background=axs[0].get_facecolor(),
                )
                for value in row
            ]
            for row in matrix
        ]
        if annotated:
            for i in range(len(rows)):
                for j in range(len(settings)):
                    axs[0].text(
                        j,
                        i,
                        text_matrix[i][j],
                        ha="center",
                        va="center",
                        fontsize=9,
                        color=text_colors[i][j],
                    )
    else:
        _missing(
            axs[0],
            "No variants in the declared comparison support"
            if settings
            else "No sensitivity comparison supplied",
        )
    axs[1].axis("off")
    counts = {
        status: sum(r.get("status") == status for r in relations)
        for status in sorted({str(r.get("status", "not assessed")) for r in relations})
    }
    shown_ids = {r["variant_id"] for r in rows}
    finite_changes = [
        r for r in all_rows if np.isfinite(_number(r.get("max_abs_delta_from_baseline")))
    ]
    maximum = (
        max(finite_changes, key=lambda r: r["max_abs_delta_from_baseline"])
        if finite_changes
        else None
    )
    outside_maximum = maximum if maximum and maximum["variant_id"] not in shown_ids else None
    scope = comparison.get("support", {}).get("pairwise_metrics", "not recorded")
    count_text = (
        ", ".join(f"{count} {status.replace('_', ' ')}" for status, count in counts.items())
        or "No relationships assessed"
    )
    max_text = (
        (
            "Largest baseline PIP change: "
            + str(maximum["variant_id"])
            + " = "
            + format_number(maximum["max_abs_delta_from_baseline"])
        )
        if maximum
        else "Baseline PIP changes unavailable"
    )
    if outside_maximum:
        max_text += " (outside displayed candidates; see complete variant records)"
    baseline_changes = []
    base = baseline if baseline in settings else (settings[0] if settings else None)
    for setting in settings:
        if setting == base:
            continue
        available = [
            (
                abs(_number(r["pips"].get(setting)) - _number(r["pips"].get(base))),
                str(r["variant_id"]),
            )
            for r in all_rows
            if np.isfinite(_number(r["pips"].get(setting)))
            and np.isfinite(_number(r["pips"].get(base)))
        ]
        largest = (
            sorted(available, key=lambda pair: (-pair[0], pair[1]))[0]
            if available
            else (None, None)
        )
        baseline_changes.append(
            dict(
                setting=setting,
                baseline=base,
                max_abs_delta_pip=largest[0],
                variant_id=largest[1],
                outside_displayed_candidates=largest[1] not in shown_ids
                if largest[1] is not None
                else None,
                shared_variant_count=len(available),
                support_scope=scope,
            )
        )
    per_setting = []
    for change in baseline_changes:
        label = str(display_names.get(change["setting"], change["setting"]))
        if change["max_abs_delta_pip"] == 0:
            per_setting.append(
                f"Baseline → {label}: all compared variants unchanged (max |ΔPIP| = 0)"
            )
            continue
        line = f"Baseline → {label}: max |ΔPIP| = {format_number(change['max_abs_delta_pip'])}"
        if change["variant_id"] is not None:
            line += f" at {change['variant_id']}"
        if change["outside_displayed_candidates"]:
            line += " (outside candidates; full variant table)"
        per_setting.append(line)
    caption_details = [
        f"{layer.title()} relationships: {count_text}",
        *per_setting,
        (f"{comparison.get('mode', 'unspecified')} mode; metrics: {scope.replace('_', ' ')}"),
        (
            "Missing variants are NA, distinct from zero PIP. Full "
            "settings and records are exported."
        ),
    ]
    if compact:
        candidate_word = "candidate" if len(rows) == 1 else "candidates"
        variant_word = "variant" if len(all_rows) == 1 else "variants"
        caption = (
            (f"{len(rows)} selected {candidate_word} / {len(all_rows)} compared {variant_word}")
            if candidates_only
            else f"{len(rows)} compared {variant_word}"
        )
        axs[1].text(
            0.5, 0.95, caption, ha="center", va="top", transform=axs[1].transAxes, fontsize=9
        )
        if np.any(~np.isfinite(matrix)):
            from matplotlib.patches import Patch

            axs[1].legend(
                handles=[Patch(facecolor=COLORS["missing"], label="NA: variant not included")],
                loc="lower center",
                fontsize=9,
                handlelength=1.0,
                borderaxespad=0.0,
            )
    elif relations:
        _table(
            axs[1],
            [
                [
                    r.get("setting_a", ""),
                    r.get("setting_b", ""),
                    r.get("component_a", ""),
                    r.get("component_b", ""),
                    r.get("posterior_overlap", "not assessed"),
                    r.get("status", "not assessed"),
                ]
                for r in relations
            ],
            labels=[
                "Setting A",
                "Setting B",
                "Component A",
                "Component B",
                "Posterior overlap",
                "Relation",
            ],
            page=table_page,
            page_size=table_page_size,
        )
    else:
        _missing(axs[1], "No signal relationships assessed")
    if not compact:
        axs[1].set_title(
            f"{layer.title()} layer · "
            + ("baseline versus each setting" if baseline_only else "all setting pairs"),
            fontsize=10,
        )
    figure.suptitle("PIP sensitivity", fontsize=12)
    return (
        figure,
        axs,
        {
            "variant_ids": [r["variant_id"] for r in rows],
            "settings": settings,
            "pip_matrix": matrix,
            "signal_relations": relations,
            "missing_is_zero": False,
            "candidates_only": candidates_only,
            "all_variant_records": all_rows,
            "layer": layer,
            "baseline_only": baseline_only,
            "all_component_relations": comparison.get("component_relations", []),
            "table_page": table_page,
            "table_page_size": table_page_size,
            "layout": layout,
            "display_names": dict(zip(settings, labels, strict=True)),
            "cell_annotations": annotated,
            "display_text": text_matrix,
            "annotation_colors": text_colors,
            "caption_details": caption_details,
            "display_policy": {
                "cmap": color_map.name,
                "normalization": [0, 1],
                "scale": "linear",
                "annotation_rule": "at most 18 rows, 7 settings and 100 cells",
                "annotation_format": (
                    "exact 0/1; <0.01 or >0.99 for interior tails; "
                    "otherwise two decimals; NA for missing"
                ),
                "row_labels": "all selected candidates; long IDs uniquely abbreviated",
                "rendered_setting_labels": dict(zip(settings, tick_labels, strict=True)),
            },
            "relation_counts": counts,
            "baseline_changes": baseline_changes,
            "maximum_baseline_change": maximum,
            "maximum_change_outside_candidates": outside_maximum,
            "mode": comparison.get("mode"),
            "support": comparison.get("support", {}),
            "fit_settings": comparison.get("fit_settings", {}),
            "source_comparison": comparison,
            "candidate_selection": comparison.get("rules", {}).get(
                "candidates", "saved candidate flags"
            ),
            "variant_label_map": dict(
                zip(
                    [r["variant_id"] for r in rows],
                    _display_labels([r["variant_id"] for r in rows]),
                    strict=True,
                )
            ),
        },
    )


@_local_style
def plot_prior_influence(
    summary: Mapping[str, Any],
    component_id: int | None = None,
    *,
    variant_id: str | None = None,
    variant_ids: Sequence[str] | None = None,
    axes: Any = None,
    layout: str = "full",
    max_variants: int | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Plot prior weights, conditional log Bayes factors and α for one single effect.

    Parameters
    ----------
    summary : dict
        Output of :func:`prusie.summarize`.
    component_id : int, optional
        Effect to plot; alternatively selected through ``variant_id``.
    variant_id : str, optional
        Variant used to select the effect.
    variant_ids : sequence of str, optional
        Variants to show; in the compact layout, credible-set members are
        added.
    axes : sequence of matplotlib Axes, optional
        Existing axes.
    layout : {'full', 'compact'}, default 'full'
        'compact' ranks variants by α and shows credible-set members and the
        ten leading variants; 'full' shows the whole table in its source
        order unless ``variant_ids`` or ``max_variants`` is given.
    max_variants : int, optional
        Number of leading variants by α to show together with the
        credible-set members.
    style : dict or False, optional
        Matplotlib rc settings applied to this figure only.

    Returns
    -------
    figure : matplotlib Figure
    axes : ndarray of Axes
    plotted_data : dict
        The plotted rows of the prior-influence table, unchanged.
    """
    compact = _layout(layout)
    if max_variants is not None and (
        isinstance(max_variants, bool) or not isinstance(max_variants, int) or max_variants < 1
    ):
        raise ValueError("max_variants must be a positive integer or None")
    if component_id is None or summary.get("components"):
        component_id = _select_component(summary, component_id, variant_id)["component_id"]
    elif variant_id is not None:
        raise ValueError("Variant-based selection requires component posterior records")
    all_rows = [r for r in summary.get("prior_influence", []) if r["component_id"] == component_id]
    ids = {r["variant_id"] for r in all_rows}
    cs = next(
        (c for c in summary.get("credible_sets", []) if c["component_id"] == component_id), None
    )
    cs_members = set(cs["members"]) if cs else set()
    chosen = None if variant_ids is None else set(variant_ids)
    if chosen is not None and not chosen.issubset(ids):
        raise ValueError("Selected prior-influence variant ID is absent")
    ranked = sorted(all_rows, key=lambda r: (-_number(r.get("alpha")), str(r["variant_id"])))
    if chosen is not None:
        rule = "explicit variant IDs" + (" + all formal CS members" if compact else "")
        if compact:
            chosen |= cs_members
    elif compact or max_variants is not None:
        limit = max_variants or 10
        chosen = {r["variant_id"] for r in ranked[:limit]} | cs_members
        if variant_id is not None:
            chosen.add(variant_id)
        rule = f"all formal CS members + top {limit} alpha candidates + explicitly selected variant"
    else:
        rule = "full component table in source order"
    rows = [
        r for r in (ranked if compact else all_rows) if chosen is None or r["variant_id"] in chosen
    ]
    figure, axs = _axes(axes, 3, figsize=(8.5, 6.6) if compact else None)
    displayed_ids = [r["variant_id"] for r in rows]
    for ax, key, label in zip(
        axs,
        ("prior_weight", "conditional_log_bf", "alpha"),
        ("Variant prior", "Conditional log BF", "Component alpha"),
        strict=True,
    ):
        ax.bar(
            range(len(rows)),
            [_number(row.get(key)) for row in rows],
            color=[
                _component_color(component_id)
                if r["variant_id"] in cs_members
                else COLORS["missing"]
                for r in rows
            ],
            edgecolor=COLORS["text"],
            linewidth=0.35,
        )
        ax.set_ylabel(label)
        _variant_ticks(ax, displayed_ids, priority_indices=range(min(3, len(rows))))
        ax.set_xlabel(
            "Variant order (source table)"
            if chosen is None
            else "Selected variants; full table exported"
        )
        ax.grid(axis="y")
        if not rows:
            _missing(ax, "Prior influence not assessed")
    if rows and any("effective_prior_weight" in row for row in rows):
        axs[0].containers[0].set_label("Supplied prior")
        axs[0].plot(
            range(len(rows)),
            [_number(row.get("effective_prior_weight")) for row in rows],
            "s",
            mfc="none",
            color=COLORS["text"],
            label="Effective prior",
        )
        axs[0].legend(fontsize=9, ncol=2, loc="upper center")
        prior_values = [
            _number(r.get(key)) for r in rows for key in ("prior_weight", "effective_prior_weight")
        ]
        finite_priors = [v for v in prior_values if np.isfinite(v)]
        if finite_priors and max(finite_priors) > 0:
            axs[0].set_ylim(0.0, max(finite_priors) * 1.45)
    if compact:
        _bottom_labels(
            axs,
            (f"Alpha-ranked candidates ({len(rows)} of {len(all_rows)}; all CS members retained)"),
        )
        axs[0].set_title(
            ("Component-conditional evidence · colored bars: CS members; gray: other candidates"),
            fontsize=9,
        )
    figure.suptitle(f"Prior influence · component {component_id}")
    return (
        figure,
        axs,
        {
            "component_id": component_id,
            "prior_influence": rows,
            "all_prior_influence": all_rows,
            "selected_variant": variant_id,
            "selected_variant_ids": None if chosen is None else sorted(chosen),
            "displayed_variant_ids": displayed_ids,
            "candidate_selection": rule,
            "layout": layout,
            "variant_label_map": dict(
                zip(displayed_ids, _display_labels(displayed_ids), strict=True)
            ),
        },
    )
