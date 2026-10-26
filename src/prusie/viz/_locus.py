"""Locus plotting helpers."""

from __future__ import annotations

import re
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .._style import COLORS, annotation_color, format_ld_cell, format_number
from ._common import (
    _axes,
    _bottom_labels,
    _chromosome,
    _component_color,
    _display_labels,
    _layout,
    _ld,
    _local_style,
    _missing,
    _number,
    _table,
    _variant_ticks,
)


def read_gene_track(
    path: str | Path,
    *,
    genome_build: str,
    source: str,
    transcript_ids: Sequence[str] | None = None,
    chromosome: str | None = None,
    start: int | None = None,
    end: int | None = None,
) -> dict[str, Any]:
    """Read transcript intervals from a local GTF file for :func:`plot_locus`.

    Parameters
    ----------
    path : str or Path
        GTF file.
    genome_build : str
        Genome build of the coordinates, recorded with the track.
    source : str
        Annotation source, recorded with the track.
    transcript_ids : iterable of str, optional
        Transcripts to keep; all are kept by default.
    chromosome : str, optional
        Chromosome to keep; required when ``start`` or ``end`` is given.
    start, end : int, optional
        One-based closed interval; records overlapping it are kept.

    Returns
    -------
    dict
        Interval records with chromosome, one-based closed start and end,
        strand, label and transcript ID, together with the build, source,
        filter and transcript selection.

    Notes
    -----
    Gene records are used only when no transcript records pass the filters;
    no transcript is selected implicitly. A table of the same form can be
    passed to :func:`plot_locus` directly.
    """
    if (start is not None or end is not None) and chromosome is None:
        raise ValueError("Locus interval filtering requires chromosome")
    if start is not None and (int(start) != start or start < 1):
        raise ValueError("Locus start must be a positive integer")
    if end is not None and (int(end) != end or end < 1 or (start is not None and end < start)):
        raise ValueError("Locus end must be an integer at least start")
    rows, genes = [], []
    chosen = None if transcript_ids is None else set(transcript_ids)
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                raise ValueError("GTF records must have nine tab-separated fields")
            if fields[2] not in ("gene", "transcript"):
                continue
            attr = dict(re.findall(r'(\w+) "([^\"]*)"', fields[8]))
            identifier = attr.get("transcript_id", attr.get("gene_id"))
            if chosen is not None and identifier not in chosen:
                continue
            left, right = int(fields[3]), int(fields[4])
            if left < 1 or right < left:
                raise ValueError("Invalid GTF interval")
            if chromosome is not None and _chromosome(fields[0]) != _chromosome(chromosome):
                continue
            if (start is not None and right < start) or (end is not None and left > end):
                continue
            row = {
                "chromosome": fields[0],
                "start": left,
                "end": right,
                "strand": fields[6],
                "label": attr.get("gene_name", identifier),
                "transcript_id": identifier,
            }
            (rows if fields[2] == "transcript" else genes).append(row)
    return {
        "records": rows or genes,
        "genome_build": genome_build,
        "source": source,
        "locus_filter": {"chromosome": chromosome, "start": start, "end": end},
        "coordinate_convention": "1-based closed",
        "transcript_selection": "all supplied transcripts" if chosen is None else sorted(chosen),
    }


def _locus_references(summary, ids, reference_variant):
    """Select references from component alpha without using marginal PIP for a CS."""
    if reference_variant is not None:
        if reference_variant not in ids:
            raise ValueError("LD reference variant is absent")
        return [
            dict(
                component_id=None,
                top_variant_id=reference_variant,
                top_index=ids.index(reference_variant),
                rule="explicit reference variant",
            )
        ]
    sets = sorted(summary.get("credible_sets", []), key=lambda cs: int(cs["component_id"]))
    if not sets:
        rows = summary["variants"]
        index = min(range(len(ids)), key=lambda j: (-_number(rows[j].get("pip")), str(ids[j])))
        return [
            dict(
                component_id=None,
                top_variant_id=ids[index],
                top_index=index,
                rule="no retained CS; highest marginal PIP; lexical ID breaks ties",
            )
        ]
    components = {c["component_id"]: c for c in summary.get("components", [])}
    references = []
    for cs in sets:
        members = cs["members"]
        if any(v not in ids for v in members):
            raise ValueError("A credible-set member is absent from the variant order")
        component = components.get(cs["component_id"], {})
        alpha = dict(zip(component.get("ranking", []), component.get("alpha", []), strict=True))
        if not members or any(v not in alpha or not np.isfinite(alpha[v]) for v in members):
            raise ValueError("CS top selection requires finite component alpha for every member")
        # Match summarize's representative rule, including exact ties.
        top = min(members, key=lambda v: (-float(alpha[v]), str(v)))
        references.append(
            dict(
                component_id=cs["component_id"],
                cs_members=list(members),
                top_variant_id=top,
                top_index=ids.index(top),
                top_alpha=float(alpha[top]),
                tied_variant_ids=sorted(v for v in members if alpha[v] == alpha[top]),
                rule="highest component alpha among CS members; exact ties use lexical variant ID",
            )
        )
    return references


@_local_style
def plot_locus(
    summary: Mapping[str, Any],
    *,
    R: ArrayLike | None = None,
    reference_variant: str | int | None = None,
    genes: Mapping[str, Any] | None = None,
    axes: Any = None,
    layout: str = "full",
    position_unit: str = "auto",
    label_variants: Sequence[str] | None = None,
    color_by_ld: bool = False,
) -> tuple[Any, Any, dict[str, Any]]:
    """Plot association, PIP, credible-set and LD tracks for a region.

    Parameters
    ----------
    summary : dict
        Output of :func:`prusie.summarize`.
    R : array_like, optional
        Signed LD correlation matrix in the variant order of ``summary``.
        Each LD track shows r² with the top variant of one retained CS.
    reference_variant : str, optional
        Override the per-CS tracks with a single reference. By default, the
        top of each CS maximizes its own component alpha among CS members;
        exact ties use lexical variant ID order, as in ``summarize``.
        With no retained CS, use the highest marginal PIP (same tie rule).
    genes : dict, optional
        Gene track from :func:`read_gene_track`, or a table of the same form.
    axes : sequence of matplotlib Axes, optional
        Existing axes, one per track: association, PIP, CS, then one per LD
        reference (one placeholder if R is missing), then an optional gene track.
    layout : {'full', 'compact'}, default 'full'
        'compact' reduces track spacing; every variant is still plotted.
    position_unit : {'auto', 'bp', 'kb', 'Mb'}, default 'auto'
        Unit of the position axis. Without positions, the variant index is
        used.
    label_variants : sequence of str, optional
        Variants to annotate; defaults to the three highest nonzero PIPs.
    color_by_ld : bool, default False
        Color association points by r² with the first displayed reference;
        the colorbar identifies it. Supply ``reference_variant`` to choose it.
    style : dict or False, optional
        Matplotlib rc settings applied to this figure only; False uses the
        current settings.

    Returns
    -------
    figure : matplotlib Figure
    axes : ndarray of Axes
    plotted_data : dict
        Plotted values and selection rules. ``ld_references`` contains each
        component ID (original zero-based fit row), reference ID/index, alpha,
        tied IDs, signed r and r² in the full variant order. Missing R gives
        an empty list. Legacy ``ld_reference`` and ``r_squared`` refer to the
        first displayed series. Input LD and summary arrays are never modified.
    """
    compact = _layout(layout)
    rows = summary["variants"]
    ids = [row["variant_id"] for row in rows]
    positions = [row.get("position") for row in rows]
    physical = bool(rows) and all(v is not None and np.isfinite(float(v)) for v in positions)
    raw_x = np.asarray(positions if physical else np.arange(len(rows)), dtype=float)
    if position_unit not in ("auto", "bp", "kb", "Mb"):
        raise ValueError("position_unit must be auto, bp, kb or Mb")
    unit = (
        (
            "Mb"
            if physical and np.max(np.abs(raw_x)) >= 1e6
            else "kb"
            if physical and np.max(np.abs(raw_x)) >= 1e4
            else "bp"
        )
        if position_unit == "auto"
        else position_unit
    )
    scale = {"bp": 1.0, "kb": 1e3, "Mb": 1e6}[unit] if physical else 1.0
    x = raw_x / scale
    label = (
        f"Genomic position ({unit})"
        if physical
        else ("Variant index (zero-based; physical positions unavailable)")
    )
    if genes is not None and not physical:
        raise ValueError("Gene intervals require physical variant positions")
    coordinate_status = {"chromosome": "not_assessed", "genome_build": "not_assessed"}
    if genes is not None:
        required = (
            "records",
            "genome_build",
            "source",
            "coordinate_convention",
            "transcript_selection",
        )
        if not all(key in genes for key in required):
            raise ValueError(
                "Gene track requires build, source, transcript selection and coordinate convention"
            )
        if genes["coordinate_convention"] != "1-based closed":
            raise ValueError("Convert the gene table to 1-based closed coordinates explicitly")
        qc = summary.get("input_qc", {})
        study = qc.get("study_information", {})

        def known_build(x):
            return x not in (None, "not_assessed", "unknown", "")

        declared_builds = {
            value
            for value in (qc.get("genome_build"), study.get("genome_build"))
            if known_build(value)
        }
        if len(declared_builds) > 1:
            raise ValueError("Variant genome builds differ between metadata sources")
        build = next(iter(declared_builds), None)
        row_builds = {
            row.get("genome_build", row.get("metadata", {}).get("genome_build")) for row in rows
        }
        row_builds = {v for v in row_builds if known_build(v)}
        if len(row_builds) > 1 or (known_build(build) and row_builds and row_builds != {build}):
            raise ValueError("Variant genome builds differ")
        if not known_build(build) and row_builds:
            build = next(iter(row_builds))
        if (
            known_build(build)
            and known_build(genes["genome_build"])
            and build != genes["genome_build"]
        ):
            raise ValueError("Gene and variant genome builds differ")
        coordinate_status["genome_build"] = (
            "matched"
            if known_build(build) and known_build(genes["genome_build"])
            else "not_assessed"
        )
        chromosomes = {_chromosome(row.get("chromosome")) for row in genes["records"]} - {None}
        declared_chromosome = _chromosome(
            genes.get("chromosome", genes.get("locus_filter", {}).get("chromosome"))
        )
        if declared_chromosome:
            if chromosomes and chromosomes != {declared_chromosome}:
                raise ValueError("Gene records differ from the declared locus chromosome")
            chromosomes = {declared_chromosome}
        variant_chromosomes = {
            _chromosome(
                row.get(
                    "chromosome",
                    row.get(
                        "chrom",
                        row.get("metadata", {}).get(
                            "chromosome", row.get("metadata", {}).get("chrom")
                        ),
                    ),
                )
            )
            for row in rows
        } - {None}
        declared_variant_chromosomes = {
            _chromosome(qc.get("chromosome", qc.get("chrom"))),
            _chromosome(study.get("chromosome", study.get("chrom"))),
        } - {None}
        if len(declared_variant_chromosomes) > 1 or (
            variant_chromosomes
            and declared_variant_chromosomes
            and variant_chromosomes != declared_variant_chromosomes
        ):
            raise ValueError("Variant chromosomes differ between metadata sources")
        if not variant_chromosomes:
            variant_chromosomes = declared_variant_chromosomes
        if len(chromosomes) > 1 or len(variant_chromosomes) > 1:
            raise ValueError("A locus gene track and variants must each use one chromosome")
        if chromosomes and variant_chromosomes and chromosomes != variant_chromosomes:
            raise ValueError("Gene and variant chromosomes differ")
        coordinate_status["chromosome"] = (
            "matched" if chromosomes and variant_chromosomes else "not_assessed"
        )
    references = (
        _locus_references(summary, ids, reference_variant) if R is not None and rows else []
    )
    ld_count = max(1, len(references))
    count = 3 + ld_count + int(genes is not None)
    figure, axs = _axes(
        axes,
        count,
        figsize=(8.5, 1.55 * count + 0.7) if compact else None,
        height_ratios=([1.4, 1.4, 0.8] + [1.0] * ld_count + ([0.8] if genes is not None else []))
        if compact
        else None,
    )
    association = np.asarray([_number(row.get("neg_log10_p")) for row in rows])
    pip = np.asarray([_number(row.get("pip")) for row in rows])
    points_association = axs[0].scatter(
        x, association, color=COLORS["muted"], s=16 if len(rows) > 200 else 22
    )
    axs[0].set_ylabel("−log₁₀ P")
    if not np.any(np.isfinite(association)):
        _missing(axs[0], "Association statistics not supplied")
    axs[1].scatter(x, pip, color=COLORS["text"], s=16 if len(rows) > 200 else 22)
    axs[1].set_ylabel("Marginal PIP")
    axs[1].set_ylim(-0.03, 1.03)
    memberships = []
    credible_sets = sorted(summary.get("credible_sets", []), key=lambda cs: int(cs["component_id"]))
    for i, cs in enumerate(credible_sets):
        member_indices = [ids.index(v) for v in cs["members"] if v in ids]
        axs[2].scatter(
            x[member_indices],
            np.full(len(member_indices), i),
            color=_component_color(cs["component_id"]),
            marker=("o", "s", "^", "D")[i % 4],
            label=f"CS component {cs['component_id']}",
        )
        memberships.extend(
            {"variant_id": ids[j], "component_id": cs["component_id"]} for j in member_indices
        )
    axs[2].set_ylabel("Formal CS")
    _variant_ticks(
        axs[2],
        [f"Component {cs['component_id']}" for cs in credible_sets],
        axis="y",
    )
    if not memberships:
        _missing(axs[2], "No retained credible sets; consult CS audit")
    ld_values = None
    if R is not None and rows:
        matrix = _ld(R, len(rows))
        for ax, reference in zip(axs[3 : 3 + ld_count], references, strict=True):
            reference_index = reference["top_index"]
            signed_r = matrix[reference_index].copy()
            values = signed_r**2
            reference.update(signed_r=signed_r, r_squared=values, reference_marker_r_squared=1.0)
            points = ax.scatter(
                x, values, c=values, cmap=COLORS["ld_r2_cmap"], vmin=0, vmax=1, s=18
            )
            component = reference["component_id"]
            accent = _component_color(component) if component is not None else COLORS["text"]
            ax.scatter(
                [x[reference_index]],
                [1.0],
                marker="D",
                s=44,
                facecolors="none",
                edgecolors=accent,
                linewidths=1.5,
            )
            suffix = (
                f"top of CS component {component} (zero-based)"
                if component is not None
                else reference["rule"]
            )
            ax.set_title(
                f"LD r² with {reference['top_variant_id']} · {suffix}", fontsize=10, color=accent
            )
        reference_variant = references[0]["top_variant_id"]
        ld_values = references[0]["r_squared"]
        figure.colorbar(points, ax=list(axs[3 : 3 + ld_count]), label="LD r²", ticks=[0, 0.5, 1])
        if color_by_ld:
            points_association.set_array(ld_values)
            points_association.set_cmap(COLORS["ld_r2_cmap"])
            points_association.set_clim(0, 1)
            figure.colorbar(points_association, ax=axs[0], label=f"LD r² with {reference_variant}")
    else:
        _missing(axs[3], "LD not supplied" if R is None else "No variants to assess")
    for ax in axs[3 : 3 + ld_count]:
        ax.set_ylabel("LD r²")
        ax.set_ylim(-0.03, 1.03)
    if genes is not None:
        for i, gene in enumerate(genes["records"]):
            axs[-1].plot([gene["start"] / scale, gene["end"] / scale], [i, i], color=COLORS["text"])
            axs[-1].text(
                gene["start"] / scale, i, str(gene.get("label", "")), va="bottom", fontsize=9
            )
        axs[-1].set_ylabel("Transcripts")
        axs[-1].set_title(
            (
                f"{genes['genome_build'] or 'unknown build'}"
                f"; {genes['coordinate_convention']}"
                f"; {genes['source']}"
                f"\n"
                f"Chromosome agreement: {coordinate_status['chromosome']}"
                f"; build agreement: {coordinate_status['genome_build']}"
            ),
            fontsize=9,
        )
    for ax in axs:
        if len(x) > 1:
            span = max(float(np.ptp(x)), 1.0 / scale)
            ax.set_xlim(float(min(x)) - 0.02 * span, float(max(x)) + 0.02 * span)
    _bottom_labels(axs, label)
    if label_variants is None:
        labels = [
            ids[i]
            for i in sorted(
                range(len(ids)), key=lambda i: (-np.nan_to_num(pip[i], nan=-1), ids[i])
            )[:3]
            if np.isfinite(pip[i]) and pip[i] > 0
        ]
        label_rule = "three highest nonzero marginal PIPs; lexical ID tie-break"
    else:
        labels = list(dict.fromkeys(label_variants))
        if any(v not in ids for v in labels):
            raise ValueError("An annotated variant ID is absent")
        label_rule = "explicit variant IDs"
    for offset, variant in enumerate(sorted(labels, key=lambda v: x[ids.index(v)])):
        i = ids.index(variant)
        axs[1].annotate(
            textwrap.fill(str(variant), 24),
            (x[i], pip[i]),
            xytext=(4, 9 + 11 * (offset % 2)),
            textcoords="offset points",
            fontsize=9,
            color=COLORS["text"],
            arrowprops=dict(arrowstyle="-", color=COLORS["muted"], lw=0.6),
        )
    for ax in [axs[0], axs[1], *axs[3 : 3 + ld_count]]:
        ax.grid(axis="y")
    return (
        figure,
        axs,
        {
            "variant_ids": ids,
            "x": x,
            "raw_positions": raw_x,
            "physical_positions": physical,
            "neg_log10_p": association,
            "pip": pip,
            "cs_memberships": memberships,
            "ld_reference": reference_variant if references else None,
            "r_squared": ld_values,
            "ld_references": references,
            "ld_status": (
                "not_supplied"
                if R is None
                else "no_variants"
                if not rows
                else "explicit_reference"
                if reference_variant is not None
                and references[0]["rule"] == "explicit reference variant"
                else "per_cs"
                if credible_sets
                else "global_pip_fallback"
            ),
            "ld_reference_rule": references[0]["rule"] if references else None,
            "component_id_convention": "original zero-based fit row",
            "genes": genes,
            "coordinate_status": coordinate_status,
            "position_unit": unit if physical else "index",
            "labelled_variants": labels,
            "label_selection": label_rule,
            "color_by_ld": color_by_ld,
            "layout": layout,
        },
    )


def _select_component(summary, component_id, variant_id):
    components = summary.get("components", [])
    ids = {row["variant_id"] for row in summary.get("variants", [])}
    if variant_id is not None and variant_id not in ids:
        raise ValueError("Selected variant ID is absent")
    if component_id is not None:
        candidates = [c for c in components if c["component_id"] == component_id]
    else:
        retained = {cs["component_id"] for cs in summary.get("credible_sets", [])}
        candidates = [c for c in components if c["component_id"] in retained] or list(components)
    if not candidates:
        raise ValueError("Unknown component ID or no component available")

    def support(component):
        ranking, alpha = component.get("ranking", []), component.get("alpha", [])
        if variant_id is not None:
            return float(alpha[ranking.index(variant_id)]) if variant_id in ranking else -np.inf
        pips = {row["variant_id"]: _number(row.get("pip")) for row in summary.get("variants", [])}
        return max(
            (pips.get(v, 0.0) * float(a) for v, a in zip(ranking, alpha, strict=True)), default=0.0
        )

    return max(candidates, key=lambda c: (support(c), -int(c["component_id"])))


@_local_style
def plot_signal(
    summary: Mapping[str, Any],
    component_id: int | None = None,
    *,
    variant_id: str | None = None,
    R: ArrayLike | None = None,
    axes: Any = None,
    table_page: int = 0,
    table_page_size: int | None = None,
    max_variants: int | None = None,
    layout: str = "full",
    ld_measure: str = "r",
    cmap: str | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Plot the posterior of one single effect, its cumulative mass and LD within its CS.

    Parameters
    ----------
    summary : dict
        Output of :func:`prusie.summarize`.
    component_id : int, optional
        Effect to plot. If omitted, selection prefers effects with a credible
        set; ``variant_id`` then selects the effect with the highest alpha
        for that variant among those candidates.
    variant_id : str, optional
        Variant used to select the effect; it is always shown.
    R : array_like, optional
        Signed LD correlation matrix in the variant order of ``summary``.
    axes : sequence of matplotlib Axes, optional
        Existing axes.
    table_page, table_page_size : int, optional
        Page of the proxy table to draw.
    max_variants : int, optional
        Number of leading variants shown; CS members and ``variant_id`` are
        always included. Cumulative probabilities use the full ranking.
    layout : {'full', 'compact'}, default 'full'
        'compact' shows CS members and ten leading variants and returns the
        proxies as data instead of drawing a table.
    ld_measure : {'r', 'r2'}, default 'r'
        Display signed r or r²; ``within_cs_r`` in the returned data is
        always signed.
    cmap : str or Colormap, optional
        Colormap of the LD matrix; defaults to RdBu_r for r and Reds for r².
    style : dict or False, optional
        Matplotlib rc settings applied to this figure only.

    Returns
    -------
    figure : matplotlib Figure
    axes : ndarray of Axes
    plotted_data : dict
        The complete posterior ranking of the effect, the plotted subset, the
        LD matrix and its display text, and the proxies.
    """
    compact = _layout(layout)
    if ld_measure not in ("r", "r2"):
        raise ValueError("ld_measure must be 'r' or 'r2'")
    component = _select_component(summary, component_id, variant_id)
    component_id = component["component_id"]
    cs_size = max(
        (
            len(c["members"])
            for c in summary.get("credible_sets", [])
            if c["component_id"] == component_id
        ),
        default=0,
    )
    large_cs = cs_size > 12
    ld_height = 0.65 if cs_size == 1 else 1.55
    figure, axs = _axes(
        axes,
        3 if compact else 4,
        figsize=(9.0, 9.0)
        if compact and large_cs
        else (8.0, 4.85 + ld_height)
        if compact
        else None,
        height_ratios=([1.6, 1.8, 3.3] if large_cs else [1.6, 1.6, ld_height]) if compact else None,
    )
    ranking, alpha = component["ranking"], np.asarray(component["alpha"])
    cs = next(
        (c for c in summary.get("credible_sets", []) if c["component_id"] == component_id), None
    )
    if max_variants is not None and (
        isinstance(max_variants, bool) or not isinstance(max_variants, int) or max_variants < 1
    ):
        raise ValueError("max_variants must be a positive integer or None")
    limit = 10 if compact and max_variants is None else max_variants
    selected = set(range(min(limit or len(ranking), len(ranking))))
    selected.update(
        i
        for i, value in enumerate(ranking)
        if value in (cs["members"] if cs else []) or value == variant_id
    )
    selected = np.asarray(sorted(selected), dtype=int)
    labels = [ranking[i] for i in selected]
    rank = np.arange(len(selected))
    color = _component_color(component_id)
    cs_members = set(cs["members"]) if cs else set()
    axs[0].bar(
        rank,
        alpha[selected],
        color=[color if v in cs_members else COLORS["missing"] for v in labels],
        edgecolor=COLORS["text"],
        linewidth=0.35,
    )
    axs[0].set_ylabel("Component alpha")
    alpha_max = float(np.max(alpha[selected])) if len(selected) else 0.0
    axs[0].set_ylim(0, alpha_max * 1.15 if alpha_max > 0 else 1.0)
    axs[1].plot(
        rank, np.asarray(component["cumulative_posterior"])[selected], marker=".", color=color
    )
    axs[1].set_ylabel("Cumulative posterior")
    axs[1].set_ylim(max(0, min(axs[1].get_ylim()[0], 0.95)), 1.01)
    if cs:
        axs[1].axhline(
            cs["requested_coverage"],
            ls="--",
            color=COLORS["muted"],
            label=(f"CS target {format_number(cs['requested_coverage'], probability=True)}"),
        )
        axs[1].legend(loc="lower right")
    selected_mass = float(np.sum(alpha[selected]))
    axs[1].set_title(
        (
            f"Full-posterior cumulative mass; selected "
            f"α mass = {format_number(selected_mass, probability=True)}"
        ),
        fontsize=10,
    )
    for ax in axs[:2]:
        _variant_ticks(ax, labels, priority_indices=range(min(3, len(labels))))
        if len(selected):
            ax.set_xlim(-0.5, len(selected) - 0.5)
        ax.grid(axis="y")
    axs[0].set_title("Colored bars: formal CS members · gray: other candidates", fontsize=9)
    if compact:
        _bottom_labels(
            axs[:2],
            (
                f"Alpha-ranked candidates ({len(selected)}"
                f" of {len(ranking)}"
                f"; all CS members retained)"
            )
            + (
                "\nLong IDs abbreviated; complete IDs in plotted data"
                if any(len(str(v)) > 30 for v in labels)
                else ""
            ),
        )
    else:
        for ax in axs[:2]:
            ax.set_xlabel("Alpha-ranked candidates (complete ranks retained in returned data)")
    within = None
    ld_text, ld_text_colors, ld_cmap = [], [], None
    if R is not None and cs:
        ids = [v["variant_id"] for v in summary["variants"]]
        indices = [ids.index(v) for v in cs["members"]]
        within = _ld(R, len(ids))[np.ix_(indices, indices)]
        shown_ld = within**2 if ld_measure == "r2" else within.copy()
        # Mathematical self-correlation is 1; retain unrounded input in within_cs_r.
        np.fill_diagonal(shown_ld, 1.0)
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        ld_cmap = plt.get_cmap(
            cmap or COLORS["ld_r2_cmap" if ld_measure == "r2" else "ld_signed_cmap"]
        ).copy()
        obj = axs[2].imshow(
            shown_ld,
            cmap=ld_cmap,
            norm=Normalize(0 if ld_measure == "r2" else -1, 1),
            aspect="equal",
            interpolation="none",
        )
        axs[2].set_box_aspect(1)
        _variant_ticks(axs[2], cs["members"])
        _variant_ticks(axs[2], cs["members"], axis="y")
        colorbar_label = "LD r²" if ld_measure == "r2" else "Signed r"
        if compact:
            cax = axs[2].inset_axes([1.07, 0.0, 0.045, 1.0])
            figure.colorbar(
                obj,
                cax=cax,
                label=colorbar_label,
                ticks=[0.0, 0.5, 1.0] if ld_measure == "r2" else [-1.0, 0.0, 1.0],
            )
        else:
            figure.colorbar(obj, ax=axs[2], panchor=False, label=colorbar_label)
        ld_text = [[format_ld_cell(value) for value in row] for row in shown_ld]
        ld_text_colors = [
            [
                annotation_color(obj.cmap(obj.norm(value)), background=axs[2].get_facecolor())
                for value in row
            ]
            for row in shown_ld
        ]
        if len(indices) <= 6:
            for i in range(len(indices)):
                for j in range(len(indices)):
                    axs[2].text(
                        j,
                        i,
                        ld_text[i][j],
                        ha="center",
                        va="center",
                        color=ld_text_colors[i][j],
                        fontsize=9,
                    )
    else:
        _missing(axs[2], "Within-CS LD not assessed" if cs else "No retained CS for this component")
    proxies = [p for p in summary.get("proxies", []) if p["component_id"] == component_id]
    if not compact:
        axs[3].axis("off")
    if proxies and not compact:
        _table(
            axs[3],
            [[p["variant_id"], p["cs_variant_id"], f"{p['signed_r']:.6g}"] for p in proxies],
            labels=["Proxy outside CS", "Direct CS partner", "Signed r"],
            page=table_page,
            page_size=table_page_size,
        )
    elif not compact:
        _missing(axs[3], "No direct proxies reported; no transitive expansion")
    if compact:
        axs[2].set_title(
            ("Singleton CS · " if cs_size == 1 else "")
            + (f"{len(proxies)} direct proxies" if proxies else "No direct proxies"),
            fontsize=10,
        )
    figure.suptitle(f"Signal detail · component {component_id}")
    return (
        figure,
        axs,
        {
            "component": component,
            "credible_set": cs,
            "within_cs_r": within,
            "proxies": proxies,
            "selected_variant": variant_id,
            "all_components": summary.get("components", []),
            "displayed_variant_ids": labels,
            "displayed_full_rank_indices": selected,
            "variant_label_map": dict(zip(labels, _display_labels(labels), strict=True)),
            "max_variants": max_variants,
            "cumulative_posterior_renormalized": False,
            "selected_posterior_mass": selected_mass,
            "ld_measure": ld_measure,
            "displayed_ld": shown_ld if within is not None else None,
            "ld_display_text": ld_text,
            "ld_annotation_colors": ld_text_colors,
            "ld_display_policy": {
                "self_correlation": "diagonal displayed as 1; raw values retained in within_cs_r",
                "cmap": ld_cmap.name if ld_cmap is not None else None,
                "normalization": [0 if ld_measure == "r2" else -1, 1],
                "scale": "linear",
                "cell_annotations": bool(within is not None and len(within) <= 6),
                "annotation_rule": (
                    "at most 6 CS members; four decimal places with exact-endpoint protection"
                ),
                "variant_ids": list(cs["members"]) if cs else [],
            },
            "selection_rule": (
                "formal CS members + leading full-posterior ranks + explicit variant"
            ),
            "layout": layout,
        },
    )
