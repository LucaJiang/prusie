"""Seven deterministic analysis views with optional, lazily imported Matplotlib.

Every view returns ``(figure, axes, plotted_data)`` and accepts existing axes.
The functions never mutate input evidence or global plotting configuration.
Regional views thin axis labels while retaining observations. Compact layouts
select explicit candidate subsets and return complete source records. Full-layout
tables retain prioritized records; explicit pagination is available.
"""
from __future__ import annotations

import functools
from pathlib import Path
import re
import textwrap

import numpy as np
from ._style import COLORS, PLOT_STYLE, format_number


def _local_style(function):
    @functools.wraps(function)
    def decorated(*args, style=None, **kwargs):
        import matplotlib.pyplot as plt
        local = {} if style is False else dict(PLOT_STYLE)
        if style is not None and style is not False:
            local.update(style)
        with plt.rc_context(local):
            return function(*args, **kwargs)
    return decorated


def _axes(axes, count, *, figsize=None, height_ratios=None):
    import matplotlib.pyplot as plt
    if axes is None:
        figure, supplied = plt.subplots(count, 1, squeeze=False,
                                       figsize=figsize or (9, 2.6 * count), constrained_layout=True,
                                       gridspec_kw={'height_ratios': height_ratios} if height_ratios else None)
        return figure, supplied[:, 0]
    supplied = np.asarray(axes, dtype=object).reshape(-1)
    if len(supplied) != count:
        raise ValueError(f"This view requires {count} axes")
    if len({id(ax.figure) for ax in supplied}) != 1:
        raise ValueError("All axes must belong to one figure")
    return supplied[0].figure, supplied


def _missing(ax, message):
    ax.text(.5, .5, textwrap.fill(message, 70), ha="center", va="center", transform=ax.transAxes,
            color=COLORS['muted'], fontsize=10)


def _layout(value):
    if value not in ('full', 'compact'):
        raise ValueError("layout must be 'full' or 'compact'")
    return value == 'compact'


def _component_color(component_id):
    return COLORS['cs'][int(component_id) % len(COLORS['cs'])]


def _bottom_labels(axes, label):
    for ax in axes[:-1]:
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelbottom=False)
    axes[-1].set_xlabel(label)


def _table(ax, rows, labels, *, page=0, page_size=None):
    """Render an explicit page, without a hidden twelve-row truncation."""
    if not isinstance(page, int) or page < 0:
        raise ValueError("table_page must be a nonnegative integer")
    if page_size is not None and (not isinstance(page_size, int) or page_size < 1):
        raise ValueError("table_page_size must be a positive integer or None")
    if page_size is None and page:
        raise ValueError("table_page requires table_page_size")
    start = page * page_size if page_size else 0
    selected = rows[start:start + page_size] if page_size else rows
    if rows and not selected:
        raise ValueError("table_page is outside the available records")
    def display(value):
        if isinstance(value, (float, np.floating)):
            return format_number(value)
        text = 'NA' if value is None else str(value)
        text = text if len(text) <= 48 else text[:45] + "…"
        return '\n'.join(textwrap.wrap(text, width=24 if len(labels) >= 5 else 36))
    shown = [[display(v) for v in row] for row in selected]
    table = ax.table(cellText=shown, colLabels=labels, cellLoc='left', bbox=[0, .14, 1, .84])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    # Multi-line cells need proportionate row heights; wrapping alone would
    # move text into the next row. Keep the complete table within its own axes.
    line_counts = [1] + [max(text.count('\n') + 1 for text in row) for row in shown]
    for (row, _), cell in table.get_celld().items():
        cell.set_height(line_counts[row] / sum(line_counts))
    if page_size and len(rows) > page_size:
        ax.text(.5, .03, f"Rows {start + 1}–{start + len(selected)} of {len(rows)}; "
                "select table_page to view others; complete records in returned data.",
                ha="center", va="bottom", transform=ax.transAxes, fontsize=9)
    return table


def _number(value):
    if isinstance(value, dict) and "nonfinite" in value:
        return float(value["nonfinite"])
    return np.nan if value is None else float(value)


def _display_labels(labels):
    """Readable middle elision; extend prefixes if shortening creates a collision."""
    full = [str(label) for label in labels]
    shown = [value if len(value) <= 30 else value[:15] + '…' + value[-10:] for value in full]
    for i, value in enumerate(full):
        if len(value) > 30 and shown.count(shown[i]) > 1:
            prefix = 16
            while prefix < len(value) - 10:
                candidate = value[:prefix] + '…' + value[-10:]
                if candidate not in shown:
                    shown[i] = candidate
                    break
                prefix += 1
            else:
                shown[i] = value
    return shown


def _variant_ticks(ax, labels, *, axis="x", priority_indices=()):
    """Thin labels only, deterministically; never thin observations."""
    limit = 6 if any(len(str(label)) > 30 for label in labels) else 12
    priority = list(dict.fromkeys(int(i) for i in priority_indices if 0 <= i < len(labels)))[:limit]
    other = np.linspace(0, len(labels)-1, min(limit - len(priority), len(labels)), dtype=int) if labels else []
    indices = list(range(len(labels))) if len(labels) <= limit else sorted(set(priority) | set(other))
    display = _display_labels(labels)
    selected = [display[i] for i in indices]
    if axis == "x":
        ax.set_xticks(indices, selected, rotation=60, ha="right")
    else:
        ax.set_yticks(indices, selected)
    return list(indices)


def _ld(R, p):
    matrix = np.asarray(R, dtype=float)
    if matrix.shape != (p, p) or not np.all(np.isfinite(matrix)):
        raise ValueError("R must be a finite matrix aligned to the variant table")
    return matrix


def _chromosome(value):
    if value is None or str(value).lower() in ("", "unknown", "not_assessed"):
        return None
    return re.sub(r"^chr", "", str(value), flags=re.IGNORECASE)


def read_gene_track(path, *, genome_build, source, transcript_ids=None,
                    chromosome=None, start=None, end=None):
    """Read a local GTF into explicit 1-based closed transcript intervals.

    All transcripts are retained unless IDs are supplied. Gene-level GTF rows
    are used only when no transcript rows exist; no transcript is chosen silently.
    A user table can instead be passed directly to :func:`plot_locus`.
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
            row = {"chromosome": fields[0], "start": left, "end": right,
                   "strand": fields[6], "label": attr.get("gene_name", identifier),
                   "transcript_id": identifier}
            (rows if fields[2] == "transcript" else genes).append(row)
    return {"records": rows or genes, "genome_build": genome_build, "source": source,
            "locus_filter": {"chromosome": chromosome, "start": start, "end": end},
            "coordinate_convention": "1-based closed", "transcript_selection":
            "all supplied transcripts" if chosen is None else sorted(chosen)}


@_local_style
def plot_locus(summary, *, R=None, reference_variant=None, genes=None, axes=None,
               layout='full', position_unit='auto', label_variants=None, color_by_ld=False):
    """Aligned association, PIP, CS and LD tracks with optional gene intervals.

    ``layout='compact'`` reduces track spacing, retaining every variant. Physical
    positions use explicit bp/kb/Mb units; absent positions use variant index.
    ``label_variants`` selects annotations (default: three highest nonzero PIPs).
    ``color_by_ld=True`` colors association points by the same reference r² as
    the LD panel, while the separate CS track retains component identities.
    """
    compact = _layout(layout)
    rows = summary["variants"]
    ids = [row["variant_id"] for row in rows]
    positions = [row.get("position") for row in rows]
    physical = bool(rows) and all(v is not None and np.isfinite(float(v)) for v in positions)
    raw_x = np.asarray(positions if physical else np.arange(len(rows)), dtype=float)
    if position_unit not in ('auto', 'bp', 'kb', 'Mb'):
        raise ValueError("position_unit must be auto, bp, kb or Mb")
    unit = ('Mb' if physical and np.max(np.abs(raw_x)) >= 1e6 else
            'kb' if physical and np.max(np.abs(raw_x)) >= 1e4 else 'bp') if position_unit == 'auto' else position_unit
    scale = {'bp': 1., 'kb': 1e3, 'Mb': 1e6}[unit] if physical else 1.
    x = raw_x / scale
    label = f"Genomic position ({unit})" if physical else "Variant index (zero-based; physical positions unavailable)"
    if genes is not None and not physical:
        raise ValueError("Gene intervals require physical variant positions")
    coordinate_status = {"chromosome": "not_assessed", "genome_build": "not_assessed"}
    if genes is not None:
        required = ("records", "genome_build", "source", "coordinate_convention", "transcript_selection")
        if not all(key in genes for key in required):
            raise ValueError("Gene track requires build, source, transcript selection and coordinate convention")
        if genes["coordinate_convention"] != "1-based closed":
            raise ValueError("Convert the gene table to 1-based closed coordinates explicitly")
        qc = summary.get("input_qc", {})
        study = qc.get("study_information", {})
        known_build = lambda x: x not in (None, "not_assessed", "unknown", "")
        declared_builds = {value for value in (qc.get("genome_build"), study.get("genome_build")) if known_build(value)}
        if len(declared_builds) > 1:
            raise ValueError("Variant genome builds differ between metadata sources")
        build = next(iter(declared_builds), None)
        row_builds = {row.get("genome_build", row.get("metadata", {}).get("genome_build")) for row in rows}
        row_builds = {v for v in row_builds if known_build(v)}
        if len(row_builds) > 1 or (known_build(build) and row_builds and row_builds != {build}):
            raise ValueError("Variant genome builds differ")
        if not known_build(build) and row_builds:
            build = next(iter(row_builds))
        if known_build(build) and known_build(genes["genome_build"]) and build != genes["genome_build"]:
            raise ValueError("Gene and variant genome builds differ")
        coordinate_status["genome_build"] = "matched" if known_build(build) and known_build(genes["genome_build"]) else "not_assessed"
        chromosomes = {_chromosome(row.get("chromosome")) for row in genes["records"]} - {None}
        declared_chromosome = _chromosome(genes.get("chromosome", genes.get("locus_filter", {}).get("chromosome")))
        if declared_chromosome:
            if chromosomes and chromosomes != {declared_chromosome}:
                raise ValueError("Gene records differ from the declared locus chromosome")
            chromosomes = {declared_chromosome}
        variant_chromosomes = {_chromosome(row.get("chromosome", row.get("chrom", row.get("metadata", {}).get(
            "chromosome", row.get("metadata", {}).get("chrom"))))) for row in rows} - {None}
        declared_variant_chromosomes = {_chromosome(qc.get("chromosome", qc.get("chrom"))),
                                        _chromosome(study.get("chromosome", study.get("chrom")))} - {None}
        if len(declared_variant_chromosomes) > 1 or (variant_chromosomes and declared_variant_chromosomes
                                                    and variant_chromosomes != declared_variant_chromosomes):
            raise ValueError("Variant chromosomes differ between metadata sources")
        if not variant_chromosomes:
            variant_chromosomes = declared_variant_chromosomes
        if len(chromosomes) > 1 or len(variant_chromosomes) > 1:
            raise ValueError("A locus gene track and variants must each use one chromosome")
        if chromosomes and variant_chromosomes and chromosomes != variant_chromosomes:
            raise ValueError("Gene and variant chromosomes differ")
        coordinate_status["chromosome"] = "matched" if chromosomes and variant_chromosomes else "not_assessed"
    count = 5 if genes is not None else 4
    figure, axs = _axes(axes, count, figsize=(8.5, 1.55 * count + .7) if compact else None,
                        height_ratios=([1.4, 1.4, .8, 1.] + ([.8] if genes is not None else [])) if compact else None)
    association = np.asarray([_number(row.get("neg_log10_p")) for row in rows])
    pip = np.asarray([_number(row.get("pip")) for row in rows])
    points_association = axs[0].scatter(x, association, color=COLORS['muted'], s=16 if len(rows) > 200 else 22)
    axs[0].set_ylabel("−log₁₀ P")
    if not np.any(np.isfinite(association)):
        _missing(axs[0], "Association statistics not supplied")
    axs[1].scatter(x, pip, color=COLORS['text'], s=16 if len(rows) > 200 else 22)
    axs[1].set_ylabel("Marginal PIP")
    axs[1].set_ylim(-.03, 1.03)
    memberships = []
    for i, cs in enumerate(summary.get("credible_sets", [])):
        member_indices = [ids.index(v) for v in cs["members"] if v in ids]
        axs[2].scatter(x[member_indices], np.full(len(member_indices), i),
                       color=_component_color(cs['component_id']), marker=('o', 's', '^', 'D')[i % 4],
                       label=f"CS component {cs['component_id']}")
        memberships.extend({"variant_id": ids[j], "component_id": cs["component_id"]} for j in member_indices)
    axs[2].set_ylabel("Formal CS")
    _variant_ticks(axs[2], [f"Component {cs['component_id']}" for cs in summary.get("credible_sets", [])], axis="y")
    if not memberships:
        _missing(axs[2], "No retained credible sets; consult CS audit")
    ld_values = None
    if R is not None and rows:
        matrix = _ld(R, len(rows))
        reference_was_selected = reference_variant is None
        if reference_was_selected:
            finite = np.where(np.isfinite(pip), pip, -np.inf)
            reference_variant = ids[int(np.argmax(finite))]
        if reference_variant not in ids:
            raise ValueError("LD reference variant is absent")
        ld_values = matrix[ids.index(reference_variant)] ** 2
        points = axs[3].scatter(x, ld_values, c=ld_values, cmap=COLORS['sequential'], vmin=0, vmax=1, s=18)
        figure.colorbar(points, ax=axs[3], label="LD r²")
        if color_by_ld:
            points_association.set_array(ld_values)
            points_association.set_cmap(COLORS['sequential'])
            points_association.set_clim(0, 1)
        reference_index = ids.index(reference_variant)
        axs[3].scatter([x[reference_index]], [ld_values[reference_index]], marker='D', s=44,
                       facecolors='none', edgecolors=COLORS['text'], linewidths=1.2)
        axs[3].set_title(f"LD reference: {reference_variant}" + (' (highest PIP)' if reference_was_selected else ''), fontsize=10)
    else:
        _missing(axs[3], "LD not assessed")
    axs[3].set_ylabel("LD r²")
    axs[3].set_ylim(-.03, 1.03)
    if genes is not None:
        for i, gene in enumerate(genes["records"]):
            axs[4].plot([gene["start"] / scale, gene["end"] / scale], [i, i], color=COLORS['text'])
            axs[4].text(gene["start"] / scale, i, str(gene.get("label", "")), va="bottom", fontsize=9)
        axs[4].set_ylabel("Transcripts")
        axs[4].set_title(f"{genes['genome_build'] or 'unknown build'}; {genes['coordinate_convention']}; {genes['source']}\n"
                         f"Chromosome agreement: {coordinate_status['chromosome']}; build agreement: {coordinate_status['genome_build']}", fontsize=9)
    for ax in axs:
        if len(x) > 1:
            span = max(float(np.ptp(x)), 1. / scale)
            ax.set_xlim(float(min(x)) - .02 * span, float(max(x)) + .02 * span)
    _bottom_labels(axs, label)
    if label_variants is None:
        labels = [ids[i] for i in sorted(range(len(ids)), key=lambda i: (-np.nan_to_num(pip[i], nan=-1), ids[i]))[:3]
                  if np.isfinite(pip[i]) and pip[i] > 0]
        label_rule = 'three highest nonzero marginal PIPs; lexical ID tie-break'
    else:
        labels = list(dict.fromkeys(label_variants))
        if any(v not in ids for v in labels):
            raise ValueError('An annotated variant ID is absent')
        label_rule = 'explicit variant IDs'
    for offset, variant in enumerate(sorted(labels, key=lambda v: x[ids.index(v)])):
        i = ids.index(variant)
        axs[1].annotate(textwrap.fill(str(variant), 24), (x[i], pip[i]),
                        xytext=(4, 9 + 11 * (offset % 2)), textcoords='offset points', fontsize=9,
                        color=COLORS['text'], arrowprops=dict(arrowstyle='-', color=COLORS['muted'], lw=.6))
    for ax in (axs[0], axs[1], axs[3]):
        ax.grid(axis='y')
    return figure, axs, {"variant_ids": ids, "x": x, "raw_positions": raw_x, "physical_positions": physical,
                         "neg_log10_p": association, "pip": pip, "cs_memberships": memberships,
                         "ld_reference": reference_variant, "r_squared": ld_values, "genes": genes,
                         "coordinate_status": coordinate_status, "position_unit": unit if physical else 'index',
                         "labelled_variants": labels, "label_selection": label_rule, "color_by_ld": color_by_ld,
                         "layout": layout}


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
        return max((pips.get(v, 0.) * float(a) for v, a in zip(ranking, alpha)), default=0.)
    return max(candidates, key=lambda c: (support(c), -int(c["component_id"])))


@_local_style
def plot_signal(summary, component_id=None, *, variant_id=None, R=None, axes=None,
                table_page=0, table_page_size=None, max_variants=None, layout='full', ld_measure='r'):
    """Ranked alpha and cumulative mass, within-CS signed LD, direct proxies.

    An explicit ``max_variants`` focuses on the leading ranks while always
    including formal CS members and an explicitly selected variant. Cumulative
    probabilities retain the complete ranking, without renormalizing a subset.
    The returned component always contains the complete posterior ranking.
    Compact layout uses CS members plus ten leading candidates by default and
    keeps proxies in returned data instead of a figure table. ``ld_measure='r2'``
    is an explicit display option; ``within_cs_r`` always retains signed values.
    """
    compact = _layout(layout)
    if ld_measure not in ('r', 'r2'):
        raise ValueError("ld_measure must be 'r' or 'r2'")
    component = _select_component(summary, component_id, variant_id)
    component_id = component["component_id"]
    large_cs = max((len(c['members']) for c in summary.get('credible_sets', []) if c['component_id'] == component_id), default=0) > 12
    figure, axs = _axes(axes, 3 if compact else 4,
                        figsize=(9., 9.) if compact and large_cs else (8.5, 7.) if compact else None,
                        height_ratios=[1.6, 1.8, 3.3] if compact and large_cs else None)
    ranking, alpha = component["ranking"], np.asarray(component["alpha"])
    cs = next((c for c in summary.get("credible_sets", []) if c["component_id"] == component_id), None)
    if max_variants is not None and (isinstance(max_variants, bool) or not isinstance(max_variants, int) or max_variants < 1):
        raise ValueError('max_variants must be a positive integer or None')
    limit = 10 if compact and max_variants is None else max_variants
    selected = set(range(min(limit or len(ranking), len(ranking))))
    selected.update(i for i, value in enumerate(ranking) if value in (cs['members'] if cs else []) or value == variant_id)
    selected = np.asarray(sorted(selected), dtype=int)
    labels = [ranking[i] for i in selected]
    rank = np.arange(len(selected))
    color = _component_color(component_id)
    cs_members = set(cs['members']) if cs else set()
    axs[0].bar(rank, alpha[selected], color=[color if v in cs_members else COLORS['missing'] for v in labels],
               edgecolor=COLORS['text'], linewidth=.35)
    axs[0].set_ylabel("Component alpha")
    alpha_max = float(np.max(alpha[selected])) if len(selected) else 0.
    axs[0].set_ylim(0, alpha_max * 1.15 if alpha_max > 0 else 1.)
    axs[1].plot(rank, np.asarray(component["cumulative_posterior"])[selected], marker=".", color=color)
    axs[1].set_ylabel("Cumulative posterior")
    if cs:
        axs[1].axhline(cs["requested_coverage"], ls="--", color=COLORS['muted'],
                       label=f"CS target {format_number(cs['requested_coverage'], probability=True)}")
        axs[1].legend(loc='lower right')
    selected_mass = float(np.sum(alpha[selected]))
    axs[1].set_title(f"Full-posterior cumulative mass; selected α mass = {format_number(selected_mass, probability=True)}", fontsize=10)
    for ax in axs[:2]:
        _variant_ticks(ax, labels, priority_indices=range(min(3, len(labels))))
        if len(selected):
            ax.set_xlim(-.5, len(selected) - .5)
        ax.grid(axis='y')
    axs[0].set_title('Colored bars: formal CS members · gray: other candidates', fontsize=9)
    if compact:
        _bottom_labels(axs[:2], f"Alpha-ranked candidates ({len(selected)} of {len(ranking)}; all CS members retained)" +
                       ('\nLong IDs abbreviated; complete IDs in plotted data' if any(len(str(v)) > 30 for v in labels) else ''))
    else:
        for ax in axs[:2]:
            ax.set_xlabel("Alpha-ranked candidates (complete ranks retained in returned data)")
    within = None
    if R is not None and cs:
        ids = [v["variant_id"] for v in summary["variants"]]
        indices = [ids.index(v) for v in cs["members"]]
        within = _ld(R, len(ids))[np.ix_(indices, indices)]
        shown_ld = within ** 2 if ld_measure == 'r2' else within
        obj = axs[2].imshow(shown_ld, cmap=COLORS['sequential'] if ld_measure == 'r2' else COLORS['diverging'],
                            vmin=0 if ld_measure == 'r2' else -1, vmax=1)
        _variant_ticks(axs[2], cs["members"])
        _variant_ticks(axs[2], cs["members"], axis="y")
        colorbar_label = "LD r² within CS" if ld_measure == 'r2' else "Signed r within formal CS"
        if compact:
            cax = axs[2].inset_axes([1.07, 0., .045, 1.])
            figure.colorbar(obj, cax=cax, label=colorbar_label)
        else:
            figure.colorbar(obj, ax=axs[2], panchor=False, label=colorbar_label)
        if len(indices) <= 6:
            for i in range(len(indices)):
                for j in range(len(indices)):
                    axs[2].text(j, i, format_number(shown_ld[i, j], precision=3), ha='center', va='center',
                                color='white' if (shown_ld[i, j] < .45 if ld_measure == 'r2' else abs(shown_ld[i, j]) > .6) else COLORS['text'], fontsize=9)
    else:
        _missing(axs[2], "Within-CS LD not assessed" if cs else "No retained CS for this component")
    proxies = [p for p in summary.get("proxies", []) if p["component_id"] == component_id]
    if not compact:
        axs[3].axis("off")
    if proxies and not compact:
        _table(axs[3], [[p["variant_id"], p["cs_variant_id"], f"{p['signed_r']:.6g}"] for p in proxies],
                     labels=["Proxy outside CS", "Direct CS partner", "Signed r"],
                     page=table_page, page_size=table_page_size)
    elif not compact:
        _missing(axs[3], "No direct proxies reported; no transitive expansion")
    if compact:
        axs[2].set_title(f"{len(proxies)} direct CS-outside proxies" if proxies else "No direct CS-outside proxies", fontsize=10)
    figure.suptitle(f"Signal detail · component {component_id}")
    return figure, axs, {"component": component, "credible_set": cs, "within_cs_r": within, "proxies": proxies,
                         "selected_variant": variant_id, "all_components": summary.get("components", []),
                         "displayed_variant_ids": labels, "displayed_full_rank_indices": selected,
                         "variant_label_map": dict(zip(labels, _display_labels(labels))),
                         "max_variants": max_variants, "cumulative_posterior_renormalized": False,
                         "selected_posterior_mass": selected_mass, "ld_measure": ld_measure,
                         "displayed_ld": within ** 2 if within is not None and ld_measure == 'r2' else within,
                         "selection_rule": 'formal CS members + leading full-posterior ranks + explicit variant', "layout": layout}


def _plot_convergence_stage(fit_or_summary, *, axes=None, variance_components=None,
                            layout='full', relative_elbo=False, color=None):
    getter = fit_or_summary.get if isinstance(fit_or_summary, dict) else lambda k, d=None: getattr(fit_or_summary, k, d)
    trace = getter("trace", []) or []
    summary = getter("convergence", {}) or {}
    count = {'full': 6, 'compact': 4, 'variance': 2}[layout]
    figure, axs = _axes(axes, count, figsize=(8.5, 2.1 * count + .6))
    color = color or COLORS['before']
    iterations = np.asarray([row.get("iteration", i + 1) for i, row in enumerate(trace)])
    criterion = summary.get('effective_criterion', summary.get('requested_criterion', 'not recorded'))
    requested = summary.get('requested_criterion', 'not recorded')
    keys = () if layout == 'variance' else ('elbo', 'delta_elbo', 'max_abs_delta_alpha', 'max_abs_delta_pip')
    metric_labels = ('ELBO − first recorded ELBO' if relative_elbo else 'Logged ELBO',
                     'Signed ΔELBO', 'Max |Δalpha|', 'Max |ΔPIP|')
    origin = None
    if not trace:
        for ax in axs:
            _missing(ax, 'Trace not recorded; history cannot be reconstructed')
    for ax, key, label in zip(axs, keys, metric_labels):
        values = np.asarray([_number(row.get(key)) for row in trace])
        if key == 'elbo' and relative_elbo:
            finite = values[np.isfinite(values)]
            origin = float(finite[0]) if len(finite) else None
            if origin is not None:
                values = values - origin
        ax.plot(iterations, values, marker='o', color=color)
        ax.set_ylabel(label)
        ax.grid(axis='y')
        if key == 'delta_elbo':
            ax.axhline(0, color=COLORS['muted'], lw=.7)
            ax.set_yscale('symlog', linthresh=1e-8)
        tolerance = summary.get({'delta_elbo': 'elbo_tol', 'max_abs_delta_alpha': 'alpha_tol',
                                 'max_abs_delta_pip': 'pip_tol'}.get(key, ''))
        if tolerance is None:
            tolerance = summary.get('tolerances', {}).get({'delta_elbo': 'elbo',
                'max_abs_delta_alpha': 'alpha', 'max_abs_delta_pip': 'pip'}.get(key, ''))
        if tolerance is None and key == 'delta_elbo':
            tolerance = summary.get('tol')
        if tolerance is not None:
            selected = criterion == 'joint' or (key == 'delta_elbo' and criterion == 'elbo') or (
                key != 'delta_elbo' and criterion in ('alpha_pip', 'alpha_pip_nonfinite_fallback'))
            if layout == 'compact':
                threshold_label = ('Stopping threshold' if selected else 'Diagnostic only') + ': ' + format_number(tolerance)
            else:
                threshold_label = ('Selected stopping threshold' if selected else
                    'Diagnostic threshold (not selected stopping criterion)') + f': {tolerance:.5g}'
            ax.axhline(tolerance, ls='--' if selected else ':', color=COLORS['text'] if selected else COLORS['muted'],
                       label=threshold_label)
            ax.legend(fontsize=9, loc='best')
        if key == 'delta_elbo':
            from matplotlib.ticker import SymmetricalLogLocator
            finite = values[np.isfinite(values)]
            extent = np.append(finite, tolerance) if tolerance is not None else finite
            if len(extent):
                low, high = min(0., float(np.min(extent))), max(0., float(np.max(extent)))
                if low == high:
                    low, high = -1e-8, 1e-8
                ax.set_ylim(low * 1.15, high * 1.15)
            locator = SymmetricalLogLocator(base=10, linthresh=1e-8)
            locator.set_params(numticks=6)
            ax.yaxis.set_major_locator(locator)
    v_rows = [row.get('V') for row in trace]
    shown_components, zero_components, omitted_components = [], [], []
    variance_note = ''
    if layout in ('full', 'variance'):
        va, sa = axs[-2:]
        if trace and all(v is not None for v in v_rows):
            width = max(len(v) for v in v_rows)
            complete = np.asarray([[v[i] if len(v) > i else np.nan for i in range(width)] for v in v_rows])
            zero_components = [i for i in range(width) if np.all(complete[:, i] == 0)]
            chosen = [i for i in range(width) if i not in zero_components] if variance_components is None else list(variance_components)
            if any(i not in range(width) for i in chosen):
                raise ValueError('Unknown variance component ID')
            chosen.sort(key=lambda i: (-float(np.nanmax(complete[:, i]) - np.nanmin(complete[:, i])), i))
            limit = 6 if layout == 'variance' and variance_components is None else len(chosen)
            shown_components, omitted_components = chosen[:limit], chosen[limit:]
            for i in shown_components:
                va.plot(iterations, complete[:, i], color=color, alpha=max(.45, 1. - .1 * (i % 5)),
                        linestyle=('-', '--', '-.', ':')[i % 4], label=f'Component {i}')
            if shown_components:
                va.legend(fontsize=9, ncol=2 if len(shown_components) > 3 else 1, loc='best')
            notes = []
            if zero_components:
                notes.append(f'{len(zero_components)} always-zero components')
            if omitted_components:
                notes.append(f'{len(omitted_components)} further components in complete trace')
            if notes:
                variance_note = '; '.join(notes)
                va.set_title(variance_note, fontsize=9)
        elif trace:
            _missing(va, 'Component variance history not recorded')
        va.set_ylabel('Working effect variance V' + (' (fixed)' if summary.get('V_fixed') else ''))
        sa.plot(iterations, [_number(row.get('sigma2')) for row in trace], color=color, marker='o')
        sa.set_ylabel('Residual variance σ²' + (' (fixed)' if summary.get('sigma2_fixed') else ''))
        va.grid(axis='y'); sa.grid(axis='y')
    events = [{'iteration': int(iteration), 'events': row.get('events')} for iteration, row in zip(iterations, trace)
              if row.get('events')]
    for ax in axs:
        if len(iterations):
            from matplotlib.ticker import MaxNLocator
            ax.axvline(iterations[-1], color=COLORS['muted'], ls=':', lw=.8, label='Actual stopping iteration')
            # Missing first-sweep changes must not give each metric a different
            # x scale when upper tick labels are intentionally hidden.
            span = max(float(np.ptp(iterations)), 1.)
            ax.set_xlim(float(np.min(iterations)) - .04 * span, float(np.max(iterations)) + .04 * span)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel('IBSS iteration (recorded phase only)')
    if layout != 'full':
        _bottom_labels(axs, 'IBSS iteration')
    phases = sorted({str(row.get('phase', 'not recorded')) for row in trace})
    axs[0].set_title(f'Requested: {requested} · effective: {criterion}\n'
                     f"Stop: {summary.get('stop_reason', 'not assessed')}" +
                     (f' at iteration {iterations[-1]}' if len(iterations) else '') +
                     ('\n' + textwrap.fill(variance_note, 58) if layout == 'variance' and variance_note else ''), fontsize=10)
    return figure, axs, {'trace': trace, 'convergence': summary, 'events': events,
        'missing_trace': not bool(trace), 'phases': phases, 'variance_components_shown': shown_components,
        'zero_variance_components': zero_components, 'omitted_variance_components': omitted_components,
        'finalization_separate_from_iterations': summary.get('finalization'), 'layout': layout,
        'relative_elbo': relative_elbo, 'elbo_origin': origin}


@_local_style
def plot_convergence(fit_or_summary, *, after=None, axes=None, variance_components=None,
                     layout='full', relative_elbo=False):
    """Independent solver trajectories with requested/effective stopping rules.

    ``full`` preserves six panels per fit. ``compact`` shows the four solver
    metrics; ``variance`` shows V and sigma² separately, selecting at most six
    changing/nonzero components unless explicit ``variance_components`` are
    supplied. ``relative_elbo=True`` subtracts each trace's first finite ELBO
    and labels that transformation; exported trace values remain unchanged.
    """
    if layout not in ('full', 'compact', 'variance'):
        raise ValueError("layout must be 'full', 'compact' or 'variance'")
    count = {'full': 6, 'compact': 4, 'variance': 2}[layout]
    if after is None:
        out = _plot_convergence_stage(fit_or_summary, axes=axes, variance_components=variance_components,
                                      layout=layout, relative_elbo=relative_elbo)
        out[0].suptitle('Variance details' if layout == 'variance' else 'Fit convergence')
        return out
    if axes is None:
        import matplotlib.pyplot as plt
        figure, grid = plt.subplots(count, 2, figsize=(10, 2.1 * count + 1.1), constrained_layout=True)
        axs = grid.T.reshape(-1)
    else:
        figure, axs = _axes(axes, count * 2)
    refinement = after.get('refinement', {}) if isinstance(after, dict) else getattr(after, 'refinement', {})
    accepted = any(row.get('accepted') for row in refinement.get('history', []))
    after_label = 'Accepted after' if accepted else 'After search (unchanged)' if refinement.get('history') else 'After'
    stages = {}
    for label, value, stage_axes, color in (('Before', fit_or_summary, axs[:count], COLORS['before']),
            (after_label, after, axs[count:], COLORS['after'])):
        _, _, data = _plot_convergence_stage(value, axes=stage_axes, variance_components=variance_components,
                                              layout=layout, relative_elbo=relative_elbo, color=color)
        stages[label] = data
        stage_axes[0].set_title(label + '\n' + stage_axes[0].get_title(), fontsize=10)
        if layout == 'full':
            for ax in stage_axes:
                ax.set_xlabel(f'{label}: IBSS iteration (separate iteration axis)')
        else:
            stage_axes[-1].set_xlabel(f'{label}: IBSS iteration')
    figure.suptitle('Variance details · colors identify fit stage; component IDs are local' if layout == 'variance'
                    else 'Before and returned-after convergence')
    return figure, axs, {'stages': stages, 'independent_iteration_axes': True,
                         'refinement_candidate_accepted': accepted, 'after_label': after_label, 'layout': layout}


def _formal_sets(value):
    if isinstance(value, dict):
        return value.get("credible_sets", [])
    ids = list(value.variant_ids)
    return [{"component_id": int(c), "members": [str(ids[int(j)]) for j in m]}
            for c, m in zip(value.sets.get("cs_index", []), value.sets.get("cs", []))]


@_local_style
def plot_refinement(before, after, *, comparison=None, axes=None,
                    table_page=0, table_page_size=None, layout='full', candidates_only=None,
                    variant_ids=None):
    """Compare PIPs, B-stage candidates and the accepted incumbent trajectory.

    Compact layout uses three panels without a figure table and selects all
    before/after CS members, ten leading PIPs and PIP >= .1 candidates. Set
    ``candidates_only=False`` for the region; explicit ``variant_ids`` augment
    formal CS members. Complete PIPs, stage history and CS tables are returned.
    """
    compact = _layout(layout)
    from .comparison import comparison_relations
    def values(value):
        if isinstance(value, dict):
            return {v["variant_id"]: v["pip"] for v in value["variants"]}
        return dict(zip(value.variant_ids.tolist(), value.pip.tolist()))
    a, b = values(before), values(after)
    ids = list(dict.fromkeys([*a, *b]))
    av, bv = [a.get(v, np.nan) for v in ids], [b.get(v, np.nan) for v in ids]
    refinement = after.get("refinement", {}) if isinstance(after, dict) else getattr(after, "refinement", {})
    history = refinement.get("history", [])
    objectives = [_number(row.get("candidate_objective", row.get("objective"))) for row in history]
    improvements = [_number(row.get("improvement")) for row in history]
    before_cs, after_cs = _formal_sets(before), _formal_sets(after)
    candidates_only = compact if candidates_only is None else bool(candidates_only)
    cs_ids = {v for cs in before_cs + after_cs for v in cs['members']}
    if variant_ids is not None:
        if any(v not in ids for v in variant_ids):
            raise ValueError('A selected refinement variant ID is absent')
        selected_ids = set(variant_ids) | cs_ids
    elif candidates_only:
        top = sorted(ids, key=lambda v: (-max(a.get(v, 0), b.get(v, 0)), str(v)))[:10]
        selected_ids = set(top) | cs_ids | {v for v in ids if max(a.get(v, 0), b.get(v, 0)) >= .1}
    else:
        selected_ids = set(ids)
    shown_ids = ([v for v in sorted(ids, key=lambda v: (-max(a.get(v, 0), b.get(v, 0)), str(v))) if v in selected_ids]
                 if candidates_only or variant_ids is not None else ids)
    relations = comparison_relations(comparison, baseline_only=True) if comparison else []
    table_rows, other_attempts = [], []
    for i, row in sorted(enumerate(history), key=lambda item: (not item[1].get("accepted", False), item[0])):
        (table_rows if row.get("accepted") else other_attempts).append([f"Attempt {i + 1}, round {row.get('round', '?')}", "", "",
                           "exclude " + ", ".join(map(str, row.get("excluded_ids", []))),
                           ("ACCEPTED; " if row.get("accepted") else "") + str(row.get("reason", "attempted"))])
    before_members = {tuple(sorted(cs["members"])) for cs in before_cs}
    after_members = {tuple(sorted(cs["members"])) for cs in after_cs}
    cs_rows = []
    for stage, sets, opposite in (("Before CS", before_cs, after_members), ("After CS", after_cs, before_members)):
        for cs in sets:
            changed = tuple(sorted(cs["members"])) not in opposite
            cs_rows.append((not changed, [stage, str(cs["component_id"]) if stage == "Before CS" else "",
                str(cs["component_id"]) if stage == "After CS" else "", ", ".join(map(str, cs["members"])),
                "changed membership" if changed else "same member set (matching separate)"]))
    table_rows.extend(row for _, row in sorted(cs_rows, key=lambda item: item[0]))
    for relation in relations:
        table_rows.append(["Signal relation", str(relation.get("component_a", "")),
                           str(relation.get("component_b", "")),
                           "overlap=" + (f"{relation['posterior_overlap']:.5g}" if relation.get("posterior_overlap") is not None else "not assessed"),
                           relation.get("status", "not assessed")])
    table_rows.extend(other_attempts)
    shown_count = min(len(table_rows), table_page_size) if table_page_size else len(table_rows)
    count = 3 if compact or (axes is not None and np.asarray(axes, dtype=object).size == 3) else 4
    legacy_three = count == 3 and not compact
    table_height = max(2.6, .38 * shown_count)
    figure, axs = _axes(axes, count, figsize=(8.5, 7.8) if compact else (11, 2.6 * (count-1) + table_height),
                        height_ratios=None if compact else [2.6] * (count-1) + [table_height])
    x = np.arange(len(shown_ids))
    axs[0].plot(x - .08, [a.get(v, np.nan) for v in shown_ids], "o", label="Before", color=COLORS['before'])
    axs[0].plot(x + .08, [b.get(v, np.nan) for v in shown_ids], "x", label="After", color=COLORS['after'])
    axs[0].set_ylabel("Marginal PIP")
    _variant_ticks(axs[0], shown_ids, priority_indices=range(min(3, len(shown_ids))))
    axs[0].set_xlabel(f"{'Main candidates' if candidates_only else 'Variants'} ({len(shown_ids)} of {len(ids)})")
    axs[0].legend()
    candidate_states = []
    for row, delta in zip(history, improvements):
        stages = row.get('stages', {})
        if row.get('accepted'):
            state = 'Accepted'
        elif any(stage.get('ibss_converged') is False for stage in stages.values()) or 'not_converged' in str(row.get('reason')):
            state = 'Not converged'
        elif not np.isfinite(delta):
            state = 'Failed / unavailable'
        elif delta <= refinement.get('config', {}).get('absolute_tol', 0):
            state = 'Below acceptance threshold'
        else:
            state = 'Valid, not selected'
        candidate_states.append(state)
    markers = {'Accepted': ('o', COLORS['accepted']), 'Not converged': ('X', COLORS['flagged']),
               'Failed / unavailable': ('s', COLORS['missing']), 'Below acceptance threshold': ('x', COLORS['rejected']),
               'Valid, not selected': ('^', COLORS['conditional'])}
    for state, (marker, color) in markers.items():
        indices = [i for i, value in enumerate(candidate_states) if value == state]
        if indices:
            numeric = [i for i in indices if np.isfinite(improvements[i])]
            missing = [i for i in indices if not np.isfinite(improvements[i])]
            if numeric:
                axs[1].scatter(np.asarray(numeric) + 1, np.asarray(improvements)[numeric], marker=marker, color=color, label=state)
            if missing:
                axs[1].scatter(np.asarray(missing) + 1, np.full(len(missing), .03), marker=marker, color=color,
                               transform=axs[1].get_xaxis_transform(), label=state + ' (no objective)')
    axs[1].axhline(0, color=COLORS['muted'], lw=.7)
    threshold = refinement.get('config', {}).get('absolute_tol')
    if threshold is not None:
        axs[1].axhline(threshold, color=COLORS['muted'], ls='--', label='Accept above ' + format_number(threshold))
    axs[1].set_xlabel("B-stage attempt (round shown below)")
    axs[1].set_ylabel("B objective − round incumbent")
    if history:
        axs[1].legend(fontsize=9, ncol=2 if len(set(candidate_states)) > 2 else 1)
        axs[1].set_xticks(np.arange(len(history)) + 1, [f"{i + 1}\nR{row.get('round', '?')}" for i, row in enumerate(history)])
    incumbent_path, candidate_records = [], []
    for i, row in enumerate(history):
        objective, delta = objectives[i], improvements[i]
        incumbent = _number(row.get("incumbent_objective"))
        if not np.isfinite(incumbent) and np.isfinite(objective) and np.isfinite(delta):
            incumbent = objective - delta
        round_id = row.get("round", 1)
        candidate_records.append({"attempt": i + 1, "round": round_id, "incumbent_objective": incumbent,
                                  "B_objective": objective, "improvement": delta, "accepted": bool(row.get("accepted")),
                                  "display_status": candidate_states[i]})
        if np.isfinite(incumbent) and not any(v["round"] == round_id for v in incumbent_path):
            incumbent_path.append({"round": round_id, "objective": incumbent, "stage": "round incumbent"})
        if row.get("accepted") and np.isfinite(objective):
            incumbent_path.append({"round": round_id, "objective": objective, "stage": "accepted B candidate"})
    if not legacy_three and incumbent_path:
        axs[2].plot(range(len(incumbent_path)), [row["objective"] for row in incumbent_path], "o-", color=COLORS['after'])
        axs[2].set_xticks(range(len(incumbent_path)), [f"Round {row['round']}\n{row['stage']}" for row in incumbent_path], fontsize=9)
    elif not legacy_three:
        _missing(axs[2], "Accepted incumbent trajectory not recorded")
    if not legacy_three:
        axs[2].set_ylabel("Post-trim original-model ELBO")
    best = refinement.get("best_objective")
    axs[1 if legacy_three else 2].set_title("Final best: " + (format_number(best, precision=7) if best is not None else "not recorded") +
        ("; incumbent path in returned data (3-axis layout)" if legacy_three else ""), fontsize=10)
    if not history:
        _missing(axs[1], f"No candidate history: {refinement.get('stop_reason', 'not recorded')}")
    if table_rows and not compact:
        _table(axs[-1], table_rows, ["Record", "Before component", "After component", "Members / overlap", "Status"],
               page=table_page, page_size=table_page_size)
    elif not compact:
        _missing(axs[-1], "No formal CS or signal relationships supplied")
    if comparison is None and not compact:
        axs[-1].set_title("Matching not assessed; equal component IDs do not establish a match", fontsize=9)
    if not compact:
        axs[-1].axis("off")
    for ax in axs[:3 if not legacy_three else 2]:
        ax.grid(axis='y')
    figure.suptitle(f"Refinement · {sum(bool(row.get('accepted')) for row in history)} accepted / {len(history)} attempts")
    return figure, axs, {"variant_ids": ids, "before_pip": av, "after_pip": bv,
                         "history": history, "candidate_objectives": objectives,
                         "improvements": improvements, "signal_relations": relations,
                         "before_cs": before_cs, "after_cs": after_cs, "table_rows": table_rows,
                         "candidate_records": candidate_records, "incumbent_path": incumbent_path,
                         "all_component_relations": comparison.get("component_relations", []) if comparison else [],
                         "table_page": table_page, "table_page_size": table_page_size, "layout": layout,
                         "displayed_variant_ids": shown_ids, "candidates_only": candidates_only,
                         "variant_label_map": dict(zip(shown_ids, _display_labels(shown_ids))),
                         "candidate_selection": ('explicit variant IDs + all formal CS members' if variant_ids is not None else
                            'all formal CS members + top ten PIPs + PIP ≥ 0.1' if candidates_only else 'full region'),
                         "candidate_statuses": candidate_states, "objective_definition": 'original-prior post-trim ELBO'}


@_local_style
def plot_ld_diagnostics(diagnostics, *, axes=None, table_page=0, table_page_size=None,
                        layout='full', auxiliary='residual', label_variants=None):
    """Keep completed flip assessments distinct from conditional or missing data.

    Compact layout omits the figure table, preserving it in returned records.
    The auxiliary panel is ``residual`` or ``log_lr``. Default annotations show
    up to three flagged/extreme-residual variants; their selection reasons are
    returned and a large residual is never relabeled as a flip flag.
    """
    compact = _layout(layout)
    if auxiliary not in ('residual', 'log_lr'):
        raise ValueError("auxiliary must be 'residual' or 'log_lr'")
    from .ld import diagnostic_status, diagnostic_counts, diagnostic_input_missing
    rows = diagnostics.get("variants", [])
    states = np.asarray([diagnostic_status(row) for row in rows])
    details = [row for row, state in zip(rows, states) if state != "not_flagged"]
    shown_count = min(len(details), table_page_size) if table_page_size else len(details)
    figure, axs = _axes(axes, 2 if compact else 3,
                        figsize=(8.5, 6.3) if compact else (11, 5.6 + max(2.6, .32 * shown_count)))
    expected = np.asarray([_number(row.get("expected_z")) for row in rows])
    # Read old persisted data safely: upstream zero replacements are never observations.
    missing = np.asarray([diagnostic_input_missing(row) for row in rows], dtype=bool)
    observed = np.asarray([_number(row.get("observed_z")) for row in rows])
    residual = np.asarray([_number(row.get("standardized_residual")) for row in rows])
    observed[missing], residual[missing] = np.nan, np.nan
    log_lr = np.asarray([_number(row.get('log_likelihood_ratio')) for row in rows])
    log_lr[missing | np.isin(states, ['conditional_only', 'not_assessed'])] = np.nan
    auxiliary_values = residual if auxiliary == 'residual' else log_lr
    styles = (("not_flagged", "o", "Assessed, not flagged"), ("flagged", "x", "Assessed, flagged"),
              ("conditional_only", "^", "Conditional only; flip not assessed"),
              ("not_assessed", "s", "Not assessed / missing input"))
    for state, marker, label in styles:
        mask = states == state
        if np.any(mask):
            finite_pair = mask & np.isfinite(expected) & np.isfinite(observed)
            finite_residual = mask & np.isfinite(auxiliary_values)
            color = COLORS[{'not_flagged': 'not_flagged', 'flagged': 'flagged',
                            'conditional_only': 'conditional', 'not_assessed': 'missing'}[state]]
            # Empty scatter still records the unavailable class in the legend.
            axs[0].scatter(expected[finite_pair], observed[finite_pair], marker=marker,
                           label=f"{label} ({int(mask.sum())})", color=color, s=24)
            axs[1].scatter(np.flatnonzero(finite_residual), auxiliary_values[finite_residual], marker=marker,
                           label=label, color=color, s=24)
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
        axs[1].set_xlim(-.5, len(rows) - .5)
        axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].set_ylabel("Standardized observed residual" if auxiliary == 'residual' else 'Flip log likelihood ratio')
    axs[1].axhline(0, color=COLORS['muted'], lw=.7)
    if not compact:
        axs[2].axis("off")
    counts = diagnostic_counts(diagnostics)
    summary = (f"Assessed: {counts['assessed']}; flagged: {counts['flagged']}; "
               f"conditional only: {counts['conditional_only']}; unassessed: {counts['unassessed']}")
    (axs[0] if compact else axs[2]).set_title(summary, fontsize=9)
    if details and not compact:
        _table(axs[2], [[r.get("variant_id"), r.get("standardized_residual") if not diagnostic_input_missing(r) else None,
                        r.get("log_likelihood_ratio"), diagnostic_status(r), r.get("unavailable_reason")]
                       for r in details],
               labels=["Variant", "Observed residual", "Log likelihood ratio", "Assessment", "Unavailable reason"],
               page=table_page, page_size=table_page_size)
    elif not compact:
        _missing(axs[2], "No flags among assessed variants" if counts["assessed"] else "LD diagnostics not assessed")
    thresholds = diagnostics.get("thresholds", {})
    mixture = diagnostics.get("mixture", {})
    rule = (f"Flag requires log LR > {thresholds.get('flip_log_lr', 'not recorded')} AND "
            f"|adjusted z| > {thresholds.get('flip_abs_z', 'not recorded')}")
    axs[1].set_title(rule + '\nMixture: ' + str(mixture.get('status', 'not recorded')),
                     fontsize=9)
    if auxiliary == 'log_lr' and thresholds.get('flip_log_lr') is not None:
        axs[1].axhline(thresholds['flip_log_lr'], color=COLORS['flagged'], ls='--')
    ids = [row.get('variant_id', str(i)) for i, row in enumerate(rows)]
    if label_variants is None:
        chosen = sorted((i for i in range(len(rows)) if np.isfinite(residual[i])),
                        key=lambda i: (states[i] != 'flagged', -abs(residual[i]), str(ids[i])))[:3]
    else:
        if any(value not in ids for value in label_variants):
            raise ValueError('An annotated LD variant ID is absent')
        chosen = [ids.index(value) for value in dict.fromkeys(label_variants)]
    annotations = []
    for offset, i in enumerate(chosen):
        reason = ('explicit selection' if label_variants is not None else
                  'allele-flip flag' if states[i] == 'flagged' else 'extreme residual; not a flip flag')
        annotations.append(dict(variant_id=ids[i], reason=reason, assessment=states[i]))
        if np.isfinite(auxiliary_values[i]):
            axs[1].annotate(textwrap.fill(str(ids[i]), 24), (i, auxiliary_values[i]),
                            xytext=(4, 8 + 9 * (offset % 2)), textcoords='offset points', fontsize=9,
                            arrowprops=dict(arrowstyle='-', color=COLORS['muted'], lw=.6))
    for ax in axs[:2]:
        ax.grid(alpha=.6)
        ax.margins(y=.18)
    if compact:
        axs[1].set_xlabel('Variant index · labels: flags first, then extreme residuals (not flags)')
        if not counts['assessed']:
            status = mixture.get('reason') or ('missing inputs' if counts['unassessed'] else 'not assessed')
            axs[0].set_title(summary + '\nFlip assessment unavailable: ' + textwrap.fill(str(status).replace('_', ' '), 80), fontsize=9)
    figure.suptitle('LD compatibility diagnostics')
    return figure, axs, {"variants": rows, "flagged_or_unassessed": details, "assessment_states": states.tolist(),
                         "observed_z": observed, "expected_z": expected, "standardized_residual": residual,
                         "status_counts": counts, "assessment_summary": summary,
                         "thresholds": thresholds, "mixture": mixture,
                         "regularization": diagnostics.get("regularization"), "method": diagnostics.get("method"),
                         "layout": layout, "auxiliary": auxiliary, "log_likelihood_ratio": log_lr,
                         "annotations": annotations, "flag_rule": rule}


@_local_style
def plot_sensitivity(comparison, *, candidates_only=True, axes=None, layer="signals",
                     baseline_only=True, table_page=0, table_page_size=None,
                     layout='full', display_names=None):
    """PIP heatmap with missing variants masked, baseline first and fixed [0, 1].

    Compact layout replaces the complete relationship table with short counts.
    ``display_names`` maps actual setting keys to readable labels; raw keys,
    full parameters, support rules and source comparison remain in plotted data.
    The default candidate policy is recorded by :func:`compare_fits`.
    """
    import matplotlib.pyplot as plt
    from .comparison import comparison_relations
    compact = _layout(layout)
    display_names = dict(display_names or {})
    raw_settings = list(comparison.get('settings', []))
    if set(display_names) - set(raw_settings):
        raise ValueError('display_names contains unknown sensitivity settings')
    baseline = comparison.get('baseline')
    settings = ([baseline] + [s for s in raw_settings if s != baseline]
                if baseline in raw_settings else raw_settings)
    labels = [str(display_names.get(s, s)) for s in settings]
    relations = comparison_relations(comparison, layer=layer, baseline_only=baseline_only)
    relations = sorted(relations, key=lambda row: row.get('status') == 'matched')
    all_rows = comparison.get('variants', [])
    rows = [r for r in all_rows if r.get('candidate', True)] if candidates_only else list(all_rows)
    if compact:
        # Retain every selected candidate, ordering by maximum observed PIP.
        rows = sorted(rows, key=lambda r: (-max((_number(v) for v in r['pips'].values() if v is not None), default=0.), str(r['variant_id'])))
    matrix = np.asarray([[_number(row['pips'].get(s)) for s in settings] for row in rows], dtype=float)
    heat_height = max(2.8, min(9., .27 * len(rows) + .7)) if compact else 3.2
    shown_count = min(len(relations), table_page_size) if table_page_size else len(relations)
    detail_height = (1.65 + .28 * max(0, len(settings) - 2)) if compact else max(2.6, .32 * shown_count)
    figure, axs = _axes(axes, 2, figsize=(max(8.5, min(12., 5.5 + .9 * len(settings))), heat_height + detail_height),
                        height_ratios=[heat_height, detail_height])
    annotated = bool(len(rows) <= 18 and len(settings) <= 7)
    if rows and settings:
        cmap = plt.get_cmap(COLORS['sequential']).copy()
        cmap.set_bad(COLORS['missing'])
        obj = axs[0].imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=1, aspect='auto')
        figure.colorbar(obj, ax=axs[0], label='Marginal PIP · gray = variant not included')
        tick_labels = [textwrap.fill(s, 24) for s in labels]
        axs[0].set_xticks(range(len(settings)), tick_labels, rotation=0 if len(settings) <= 4 else 35,
                          ha='center' if len(settings) <= 4 else 'right')
        _variant_ticks(axs[0], [r['variant_id'] for r in rows], axis='y')
        axs[0].set_ylabel('Candidate variants' if candidates_only else 'Variants')
        if annotated:
            for i in range(len(rows)):
                for j in range(len(settings)):
                    value = matrix[i, j]
                    label = 'NA' if not np.isfinite(value) else format_number(value, probability=True, precision=3)
                    axs[0].text(j, i, label, ha='center', va='center', fontsize=9,
                                color='white' if np.isfinite(value) and value < .45 else COLORS['text'])
    else:
        _missing(axs[0], 'No variants in the declared comparison support' if settings else 'No sensitivity comparison supplied')
    axs[1].axis('off')
    counts = {status: sum(r.get('status') == status for r in relations)
              for status in sorted({str(r.get('status', 'not assessed')) for r in relations})}
    shown_ids = {r['variant_id'] for r in rows}
    finite_changes = [r for r in all_rows if np.isfinite(_number(r.get('max_abs_delta_from_baseline')))]
    maximum = max(finite_changes, key=lambda r: r['max_abs_delta_from_baseline']) if finite_changes else None
    outside_maximum = maximum if maximum and maximum['variant_id'] not in shown_ids else None
    scope = comparison.get('support', {}).get('pairwise_metrics', 'not recorded')
    count_text = ', '.join(f'{count} {status.replace("_", " ")}' for status, count in counts.items()) or 'No relationships assessed'
    max_text = ('Largest baseline PIP change: ' + str(maximum['variant_id']) + ' = ' +
                format_number(maximum['max_abs_delta_from_baseline'])) if maximum else 'Baseline PIP changes unavailable'
    if outside_maximum:
        max_text += ' (outside displayed candidates; see complete variant records)'
    baseline_changes = []
    base = baseline if baseline in settings else (settings[0] if settings else None)
    for setting in settings:
        if setting == base:
            continue
        available = [(abs(_number(r['pips'].get(setting)) - _number(r['pips'].get(base))), str(r['variant_id']))
                     for r in all_rows if np.isfinite(_number(r['pips'].get(setting))) and np.isfinite(_number(r['pips'].get(base)))]
        largest = sorted(available, key=lambda pair: (-pair[0], pair[1]))[0] if available else (None, None)
        baseline_changes.append(dict(setting=setting, baseline=base, max_abs_delta_pip=largest[0],
            variant_id=largest[1], outside_displayed_candidates=largest[1] not in shown_ids if largest[1] is not None else None,
            shared_variant_count=len(available), support_scope=scope))
    per_setting = []
    for change in baseline_changes:
        label = str(display_names.get(change['setting'], change['setting']))
        line = f'Baseline → {label}: max |ΔPIP| = {format_number(change["max_abs_delta_pip"])}'
        if change['variant_id'] is not None:
            line += f' at {change["variant_id"]}'
        if change['outside_displayed_candidates']:
            line += ' (outside candidates; full variant table)'
        per_setting.append(line)
    if compact:
        change_text = '\n'.join(per_setting) if per_setting else max_text
        summary = (f'{layer.title()} relationships: {count_text}.\n{change_text}.\n'
                   f'{comparison.get("mode", "unspecified")} mode; metrics: {scope.replace("_", " ")}. '
                   'Missing variants are NA, distinct from zero PIP.\nComplete settings, relationships and variant records are exported.')
        axs[1].text(0., .95, '\n'.join(textwrap.fill(line, 105) for line in summary.splitlines()),
                    va='top', transform=axs[1].transAxes, fontsize=9, linespacing=1.5)
    elif relations:
        _table(axs[1], [[r.get('setting_a', ''), r.get('setting_b', ''), r.get('component_a', ''),
                        r.get('component_b', ''), r.get('posterior_overlap', 'not assessed'), r.get('status', 'not assessed')]
                       for r in relations],
               labels=['Setting A', 'Setting B', 'Component A', 'Component B', 'Posterior overlap', 'Relation'],
               page=table_page, page_size=table_page_size)
    else:
        _missing(axs[1], 'No signal relationships assessed')
    axs[1].set_title(f'{layer.title()} layer · ' + ('baseline versus each setting' if baseline_only else 'all setting pairs'), fontsize=10)
    figure.suptitle('Sensitivity across explicit settings')
    return figure, axs, {'variant_ids': [r['variant_id'] for r in rows], 'settings': settings,
        'pip_matrix': matrix, 'signal_relations': relations, 'missing_is_zero': False,
        'candidates_only': candidates_only, 'all_variant_records': all_rows,
        'layer': layer, 'baseline_only': baseline_only, 'all_component_relations': comparison.get('component_relations', []),
        'table_page': table_page, 'table_page_size': table_page_size, 'layout': layout,
        'display_names': dict(zip(settings, labels)), 'cell_annotations': annotated,
        'relation_counts': counts, 'baseline_changes': baseline_changes, 'maximum_baseline_change': maximum, 'maximum_change_outside_candidates': outside_maximum,
        'mode': comparison.get('mode'), 'support': comparison.get('support', {}),
        'fit_settings': comparison.get('fit_settings', {}), 'source_comparison': comparison,
        'candidate_selection': comparison.get('rules', {}).get('candidates', 'saved candidate flags'),
        'variant_label_map': dict(zip([r['variant_id'] for r in rows], _display_labels([r['variant_id'] for r in rows])))}


@_local_style
def plot_prior_influence(summary, component_id=None, *, variant_id=None, variant_ids=None,
                         axes=None, layout='full', max_variants=None):
    """Aligned prior, conditional log BF and alpha, with unchanged source values.

    Compact layout ranks candidates by alpha and includes every formal CS member
    plus ten leading candidates by default. Explicit ``variant_ids`` also retain
    all CS members in compact layout. Full layout preserves source-table order.
    ``max_variants`` changes the leading-candidate count; CS members are retained.
    """
    compact = _layout(layout)
    if max_variants is not None and (isinstance(max_variants, bool) or not isinstance(max_variants, int) or max_variants < 1):
        raise ValueError('max_variants must be a positive integer or None')
    if component_id is None or summary.get('components'):
        component_id = _select_component(summary, component_id, variant_id)['component_id']
    elif variant_id is not None:
        raise ValueError('Variant-based selection requires component posterior records')
    all_rows = [r for r in summary.get('prior_influence', []) if r['component_id'] == component_id]
    ids = {r['variant_id'] for r in all_rows}
    cs = next((c for c in summary.get('credible_sets', []) if c['component_id'] == component_id), None)
    cs_members = set(cs['members']) if cs else set()
    chosen = None if variant_ids is None else set(variant_ids)
    if chosen is not None and not chosen.issubset(ids):
        raise ValueError('Selected prior-influence variant ID is absent')
    ranked = sorted(all_rows, key=lambda r: (-_number(r.get('alpha')), str(r['variant_id'])))
    if chosen is not None:
        rule = 'explicit variant IDs' + (' + all formal CS members' if compact else '')
        if compact:
            chosen |= cs_members
    elif compact or max_variants is not None:
        limit = max_variants or 10
        chosen = {r['variant_id'] for r in ranked[:limit]} | cs_members
        if variant_id is not None:
            chosen.add(variant_id)
        rule = f'all formal CS members + top {limit} alpha candidates + explicitly selected variant'
    else:
        rule = 'full component table in source order'
    rows = [r for r in (ranked if compact else all_rows) if chosen is None or r['variant_id'] in chosen]
    figure, axs = _axes(axes, 3, figsize=(8.5, 6.6) if compact else None)
    displayed_ids = [r['variant_id'] for r in rows]
    for ax, key, label in zip(axs, ('prior_weight', 'conditional_log_bf', 'alpha'),
                            ('Variant prior', 'Conditional log BF', 'Component alpha')):
        ax.bar(range(len(rows)), [_number(row.get(key)) for row in rows],
               color=[_component_color(component_id) if r['variant_id'] in cs_members else COLORS['missing'] for r in rows],
               edgecolor=COLORS['text'], linewidth=.35)
        ax.set_ylabel(label)
        _variant_ticks(ax, displayed_ids, priority_indices=range(min(3, len(rows))))
        ax.set_xlabel('Variant order (source table)' if chosen is None else 'Selected variants; full table exported')
        ax.grid(axis='y')
        if not rows:
            _missing(ax, 'Prior influence not assessed')
    if rows and any('effective_prior_weight' in row for row in rows):
        axs[0].containers[0].set_label('Supplied prior')
        axs[0].plot(range(len(rows)), [_number(row.get('effective_prior_weight')) for row in rows],
                    's', mfc='none', color=COLORS['text'], label='Effective prior')
        axs[0].legend(fontsize=9, ncol=2, loc='upper center')
        prior_values = [_number(r.get(key)) for r in rows for key in ('prior_weight', 'effective_prior_weight')]
        finite_priors = [v for v in prior_values if np.isfinite(v)]
        if finite_priors and max(finite_priors) > 0:
            axs[0].set_ylim(0., max(finite_priors) * 1.45)
    if compact:
        _bottom_labels(axs, f'Alpha-ranked candidates ({len(rows)} of {len(all_rows)}; all CS members retained)')
        axs[0].set_title('Component-conditional evidence · colored bars: CS members; gray: other candidates', fontsize=9)
    figure.suptitle(f'Prior influence · component {component_id}')
    return figure, axs, {'component_id': component_id, 'prior_influence': rows,
        'all_prior_influence': all_rows, 'selected_variant': variant_id,
        'selected_variant_ids': None if chosen is None else sorted(chosen),
        'displayed_variant_ids': displayed_ids, 'candidate_selection': rule, 'layout': layout,
        'variant_label_map': dict(zip(displayed_ids, _display_labels(displayed_ids)))}
