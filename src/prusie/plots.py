"""Seven deterministic analysis views with optional, lazily imported Matplotlib.

Every view returns ``(figure, axes, plotted_data)`` and accepts existing axes.
The functions never mutate input evidence or global plotting configuration.
At most twelve evenly spaced variant labels are drawn per axis. This keeps
regional views readable while retaining every observation in the plot and data.
Tables show all prioritized records by default; explicit pagination is available.
"""
from __future__ import annotations

import functools
from pathlib import Path
import re
import textwrap

import numpy as np


def _local_style(function):
    @functools.wraps(function)
    def decorated(*args, style=None, **kwargs):
        import matplotlib.pyplot as plt
        with plt.rc_context(style or {}):
            return function(*args, **kwargs)
    return decorated


def _axes(axes, count, *, figsize=None):
    import matplotlib.pyplot as plt
    if axes is None:
        figure, supplied = plt.subplots(count, 1, squeeze=False,
                                       figsize=figsize or (9, 2.6 * count), constrained_layout=True)
        return figure, supplied[:, 0]
    supplied = np.asarray(axes, dtype=object).reshape(-1)
    if len(supplied) != count:
        raise ValueError(f"This view requires {count} axes")
    if len({id(ax.figure) for ax in supplied}) != 1:
        raise ValueError("All axes must belong to one figure")
    return supplied[0].figure, supplied


def _missing(ax, message):
    ax.text(.5, .5, message, ha="center", va="center", transform=ax.transAxes)


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
            return f"{value:.5g}"
        text = 'NA' if value is None else str(value)
        text = text if len(text) <= 48 else text[:45] + "…"
        return '\n'.join(textwrap.wrap(text, width=24 if len(labels) >= 5 else 36))
    shown = [[display(v) for v in row] for row in selected]
    table = ax.table(cellText=shown, colLabels=labels, cellLoc='left', bbox=[0, .14, 1, .84])
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    if page_size and len(rows) > page_size:
        ax.text(.5, .03, f"Rows {start + 1}–{start + len(selected)} of {len(rows)}; "
                "select table_page to view others; complete records in returned data.",
                ha="center", va="bottom", transform=ax.transAxes, fontsize="small")
    return table


def _number(value):
    if isinstance(value, dict) and "nonfinite" in value:
        return float(value["nonfinite"])
    return np.nan if value is None else float(value)


def _variant_ticks(ax, labels, *, axis="x", priority_indices=()):
    """Thin labels only, deterministically; never thin observations."""
    priority = list(dict.fromkeys(int(i) for i in priority_indices if 0 <= i < len(labels)))[:12]
    other = np.linspace(0, len(labels)-1, min(12 - len(priority), len(labels)), dtype=int) if labels else []
    indices = sorted(set(priority) | set(other))
    selected = [str(labels[i]) for i in indices]
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
def plot_locus(summary, *, R=None, reference_variant=None, genes=None, axes=None):
    """Association, marginal PIP, multi-membership CS and reference LD panels."""
    rows = summary["variants"]
    ids = [row["variant_id"] for row in rows]
    positions = [row.get("position") for row in rows]
    physical = bool(rows) and all(v is not None and np.isfinite(float(v)) for v in positions)
    x = np.asarray(positions if physical else np.arange(len(rows)), dtype=float)
    label = "Genomic position (bp)" if physical else "Variant index (zero-based; physical positions unavailable)"
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
    figure, axs = _axes(axes, 5 if genes is not None else 4)
    association = np.asarray([_number(row.get("neg_log10_p")) for row in rows])
    pip = np.asarray([_number(row.get("pip")) for row in rows])
    axs[0].scatter(x, association, color="black", s=20)
    axs[0].set_ylabel("−log₁₀ P")
    if not np.any(np.isfinite(association)):
        _missing(axs[0], "Association statistics not supplied")
    axs[1].scatter(x, pip, color="black", s=20)
    axs[1].set_ylabel("Marginal PIP")
    axs[1].set_ylim(-.03, 1.03)
    memberships = []
    for i, cs in enumerate(summary.get("credible_sets", [])):
        member_indices = [ids.index(v) for v in cs["members"] if v in ids]
        axs[2].scatter(x[member_indices], np.full(len(member_indices), i),
                       label=f"CS component {cs['component_id']}")
        memberships.extend({"variant_id": ids[j], "component_id": cs["component_id"]} for j in member_indices)
    axs[2].set_ylabel("Formal CS")
    _variant_ticks(axs[2], [f"Component {cs['component_id']}" for cs in summary.get("credible_sets", [])], axis="y")
    if memberships:
        axs[2].legend(fontsize="small")
    else:
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
        points = axs[3].scatter(x, ld_values, c=ld_values, cmap="viridis", vmin=0, vmax=1)
        figure.colorbar(points, ax=axs[3], label="LD r²")
        axs[3].set_title(f"LD reference: {reference_variant} (highest PIP; table order breaks ties)"
                          if reference_was_selected else f"LD reference: {reference_variant}")
    else:
        _missing(axs[3], "LD not assessed")
    axs[3].set_ylabel("LD r²")
    axs[3].set_ylim(-.03, 1.03)
    if genes is not None:
        for i, gene in enumerate(genes["records"]):
            axs[4].plot([gene["start"], gene["end"]], [i, i], color="black")
            axs[4].text(gene["start"], i, str(gene.get("label", "")), va="bottom")
        axs[4].set_ylabel("Transcripts")
        axs[4].set_title(f"{genes['genome_build'] or 'unknown build'}; {genes['coordinate_convention']}; {genes['source']}\n"
                         f"Chromosome agreement: {coordinate_status['chromosome']}; build agreement: {coordinate_status['genome_build']}", fontsize="small")
    for ax in axs:
        ax.set_xlabel(label)
        if len(x) > 1:
            span = max(float(np.ptp(x)), 1.)
            ax.set_xlim(float(min(x)) - .02 * span, float(max(x)) + .02 * span)
    return figure, axs, {"variant_ids": ids, "x": x, "physical_positions": physical,
                         "neg_log10_p": association, "pip": pip, "cs_memberships": memberships,
                         "ld_reference": reference_variant, "r_squared": ld_values, "genes": genes,
                         "coordinate_status": coordinate_status}


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
                table_page=0, table_page_size=None, max_variants=None):
    """Ranked alpha and cumulative mass, within-CS signed LD, direct proxies.

    An explicit ``max_variants`` focuses on the leading ranks while always
    including formal CS members and an explicitly selected variant. Cumulative
    probabilities retain the complete ranking, without renormalizing a subset.
    The returned component always contains the complete posterior ranking.
    """
    component = _select_component(summary, component_id, variant_id)
    component_id = component["component_id"]
    figure, axs = _axes(axes, 4)
    ranking, alpha = component["ranking"], np.asarray(component["alpha"])
    cs = next((c for c in summary.get("credible_sets", []) if c["component_id"] == component_id), None)
    if max_variants is not None and (isinstance(max_variants, bool) or not isinstance(max_variants, int) or max_variants < 1):
        raise ValueError('max_variants must be a positive integer or None')
    selected = set(range(min(max_variants or len(ranking), len(ranking))))
    selected.update(i for i, value in enumerate(ranking) if value in (cs['members'] if cs else []) or value == variant_id)
    selected = np.asarray(sorted(selected), dtype=int)
    labels = [ranking[i] for i in selected]
    rank = np.arange(len(selected))
    axs[0].bar(rank, alpha[selected], color="0.3")
    axs[0].set_ylabel("Component alpha")
    axs[1].plot(rank, np.asarray(component["cumulative_posterior"])[selected], marker=".", color="black")
    axs[1].set_ylabel("Cumulative posterior")
    if cs:
        axs[1].axhline(cs["requested_coverage"], ls="--", color="black", label="Requested CS coverage")
        axs[1].legend()
    for ax in axs[:2]:
        _variant_ticks(ax, labels, priority_indices=range(min(3, len(labels))))
        ax.set_xlabel("Variants in full-alpha rank order" +
                      (f" ({len(selected)}/{len(ranking)} displayed; complete posterior exported)" if len(selected) < len(ranking) else " (selected ID labels)"))
    within = None
    if R is not None and cs:
        ids = [v["variant_id"] for v in summary["variants"]]
        indices = [ids.index(v) for v in cs["members"]]
        within = _ld(R, len(ids))[np.ix_(indices, indices)]
        obj = axs[2].imshow(within, cmap="coolwarm", vmin=-1, vmax=1)
        _variant_ticks(axs[2], cs["members"])
        _variant_ticks(axs[2], cs["members"], axis="y")
        figure.colorbar(obj, ax=axs[2], label="Signed r within formal CS")
    else:
        _missing(axs[2], "Within-CS LD not assessed" if cs else "No retained CS for this component")
    proxies = [p for p in summary.get("proxies", []) if p["component_id"] == component_id]
    axs[3].axis("off")
    if proxies:
        _table(axs[3], [[p["variant_id"], p["cs_variant_id"], f"{p['signed_r']:.6g}"] for p in proxies],
                     labels=["Proxy outside CS", "Direct CS partner", "Signed r"],
                     page=table_page, page_size=table_page_size)
    else:
        _missing(axs[3], "No direct proxies reported; no transitive expansion")
    figure.suptitle(f"Component {component_id}; null mass = {component.get('null_mass', 'not assessed')}")
    return figure, axs, {"component": component, "credible_set": cs, "within_cs_r": within, "proxies": proxies,
                         "selected_variant": variant_id, "all_components": summary.get("components", []),
                         "displayed_variant_ids": labels, "displayed_full_rank_indices": selected,
                         "max_variants": max_variants, "cumulative_posterior_renormalized": False}


def _plot_convergence_stage(fit_or_summary, *, axes=None, variance_components=None):
    """Exact recorded online metrics; signed ELBO increments stay signed."""
    getter = fit_or_summary.get if isinstance(fit_or_summary, dict) else lambda k, d=None: getattr(fit_or_summary, k, d)
    trace = getter("trace", []) or []
    summary = getter("convergence", {}) or {}
    figure, axs = _axes(axes, 6)
    keys = ("elbo", "delta_elbo", "max_abs_delta_alpha", "max_abs_delta_pip")
    iterations = np.asarray([row.get("iteration", i + 1) for i, row in enumerate(trace)])
    if not trace:
        for ax in axs:
            _missing(ax, "Trace not recorded; history cannot be reconstructed")
    for ax, key in zip(axs[:4], keys):
        values = np.asarray([_number(row.get(key)) for row in trace])
        ax.plot(iterations, values, marker=".", color="black")
        ax.set_ylabel(key.replace("max_abs_", "").replace("_", " "))
        if key == "delta_elbo":
            ax.axhline(0, color="0.5", lw=.7)
            ax.set_yscale("symlog", linthresh=1e-8)
        tolerance = summary.get({"delta_elbo": "elbo_tol", "max_abs_delta_alpha": "alpha_tol",
                                 "max_abs_delta_pip": "pip_tol"}.get(key, ""))
        tolerances = summary.get("tolerances", {})
        if tolerance is None:
            tolerance = tolerances.get({"delta_elbo": "elbo", "max_abs_delta_alpha": "alpha",
                                        "max_abs_delta_pip": "pip"}.get(key, ""))
        if tolerance is None and key == "delta_elbo":
            tolerance = summary.get("tol")
        if tolerance is not None:
            criterion = summary.get("effective_criterion", summary.get("requested_criterion"))
            selected = criterion == "joint" or (key == "delta_elbo" and criterion == "elbo") or (
                key != "delta_elbo" and criterion in ("alpha_pip", "alpha_pip_nonfinite_fallback"))
            label = "Selected stopping threshold" if selected else "Diagnostic threshold (not selected stopping criterion)"
            label += f": {tolerance:.5g}"
            ax.axhline(tolerance, ls="--", color="0.5", label=label)
            ax.legend(fontsize="small")
    v_rows = [row.get("V") for row in trace]
    shown_components, zero_components = [], []
    if trace and all(v is not None for v in v_rows):
        width = max(len(v) for v in v_rows)
        complete = np.asarray([[v[i] if len(v) > i else np.nan for i in range(width)] for v in v_rows])
        zero_components = [i for i in range(width) if np.all(complete[:, i] == 0)]
        shown_components = [i for i in range(width) if i not in zero_components] if variance_components is None else list(variance_components)
        if any(i not in range(width) for i in shown_components):
            raise ValueError("Unknown variance component ID")
        shown_components.sort(key=lambda i: (-float(np.nanmax(complete[:, i]) - np.nanmin(complete[:, i])), i))
        for i in shown_components:
            axs[4].plot(iterations, [v[i] if len(v) > i else np.nan for v in v_rows], label=f"V component {i}")
        if shown_components:
            axs[4].legend(fontsize="small")
        if zero_components:
            axs[4].text(.01, .98, f"{len(zero_components)} identically zero components; IDs in returned data.",
                        transform=axs[4].transAxes, va="top", fontsize="x-small")
    elif trace:
        _missing(axs[4], "Component variance history not recorded")
    axs[4].set_ylabel("Effect variance V" + (" (fixed)" if summary.get("V_fixed") else ""))
    axs[5].plot(iterations, [_number(row.get("sigma2")) for row in trace], color="black")
    axs[5].set_ylabel("Residual variance σ²" + (" (fixed)" if summary.get("sigma2_fixed") else ""))
    events = []
    for iteration, row in zip(iterations, trace):
        event = row.get("events", [])
        if event:
            events.append({"iteration": int(iteration), "events": event})
            axs[0].axvline(iteration, color="0.6", ls=":")
            axs[0].annotate(str(event), (iteration, _number(row.get("elbo"))), fontsize="x-small")
    for ax in axs:
        ax.set_xlabel("IBSS iteration (recorded phase only)")
        if len(iterations):
            ax.axvline(iterations[-1], color="0.7", ls=":", label="Actual stopping iteration")
    phases = sorted({str(row.get("phase", "not recorded")) for row in trace})
    figure.suptitle(f"Stop: {summary.get('stop_reason', 'not assessed')}; phases: {', '.join(phases) or 'unavailable'}")
    if summary.get("finalization"):
        axs[-1].text(1., -.4, "Finalization is recorded separately; not an IBSS iteration.",
                     ha="right", transform=axs[-1].transAxes, fontsize="small")
    return figure, axs, {"trace": trace, "convergence": summary, "events": events,
                         "missing_trace": not bool(trace), "phases": phases,
                         "variance_components_shown": shown_components, "zero_variance_components": zero_components,
                         "finalization_separate_from_iterations": summary.get("finalization")}


@_local_style
def plot_convergence(fit_or_summary, *, after=None, axes=None, variance_components=None):
    """Recorded online metrics, optionally comparing independent before/after fits.

    ``after`` creates two separate six-panel columns. Iteration numbering and
    selected stopping thresholds remain local to each fit. All variance records
    remain available in returned traces; ``variance_components`` selects IDs to
    display (default: nonzero/changed components, ordered by variation).
    """
    if after is None:
        return _plot_convergence_stage(fit_or_summary, axes=axes, variance_components=variance_components)
    if axes is None:
        import matplotlib.pyplot as plt
        figure, grid = plt.subplots(6, 2, figsize=(16, 15), constrained_layout=True)
        axs = grid.T.reshape(-1)
    else:
        figure, axs = _axes(axes, 12)
    stages = {}
    refinement = after.get('refinement', {}) if isinstance(after, dict) else getattr(after, 'refinement', {})
    accepted = any(row.get('accepted') for row in refinement.get('history', []))
    after_label = 'Accepted after' if accepted else 'After search (unchanged)' if refinement.get('history') else 'After'
    for label, value, stage_axes in (("Before", fit_or_summary, axs[:6]), (after_label, after, axs[6:])):
        _, _, data = _plot_convergence_stage(value, axes=stage_axes, variance_components=variance_components)
        stages[label] = data
        summary = data["convergence"]
        criterion = summary.get("effective_criterion", summary.get("requested_criterion", "not recorded"))
        stage_axes[0].set_title(f"{label}: {summary.get('stop_reason', 'not assessed')}\nStopping rule: {criterion}", fontsize="small")
        for ax in stage_axes:
            ax.set_xlabel(f"{label}: IBSS iteration (separate iteration axis)")
    figure.suptitle("Before and returned-after convergence: separate IBSS panels")
    return figure, axs, {"stages": stages, "independent_iteration_axes": True,
                         "refinement_candidate_accepted": accepted, "after_label": after_label}


def _formal_sets(value):
    if isinstance(value, dict):
        return value.get("credible_sets", [])
    ids = list(value.variant_ids)
    return [{"component_id": int(c), "members": [str(ids[int(j)]) for j in m]}
            for c, m in zip(value.sets.get("cs_index", []), value.sets.get("cs", []))]


@_local_style
def plot_refinement(before, after, *, comparison=None, axes=None,
                    table_page=0, table_page_size=None):
    """Separate candidate improvements from the accepted incumbent trajectory."""
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
                           "overlap=" + str(relation.get("posterior_overlap", "not assessed")),
                           relation.get("status", "not assessed")])
    table_rows.extend(other_attempts)
    shown_count = min(len(table_rows), table_page_size) if table_page_size else len(table_rows)
    count = 3 if axes is not None and np.asarray(axes, dtype=object).size == 3 else 4
    figure, axs = _axes(axes, count, figsize=(11, 8 + max(2.6, .28 * shown_count)))
    x = np.arange(len(ids))
    axs[0].plot(x, av, "o-", label="Before")
    axs[0].plot(x, bv, "x-", label="After")
    axs[0].set_ylabel("Marginal PIP")
    _variant_ticks(axs[0], ids)
    axs[0].set_xlabel("Variant order (selected ID labels; all data shown)")
    axs[0].legend()
    for accepted, marker in ((False, "x"), (True, "o")):
        indices = [i for i, row in enumerate(history) if bool(row.get("accepted")) == accepted]
        if indices:
            axs[1].scatter(np.asarray(indices) + 1, np.asarray(improvements)[indices], marker=marker,
                           color="black", label="Accepted best candidate" if accepted else "Not accepted")
    axs[1].axhline(0, color="0.5")
    axs[1].set_xlabel("Candidate attempt (B stage; separate round incumbents)")
    axs[1].set_ylabel("B objective − round incumbent")
    if history:
        axs[1].legend(fontsize="small")
    incumbent_path, candidate_records = [], []
    for i, row in enumerate(history):
        objective, delta = objectives[i], improvements[i]
        incumbent = _number(row.get("incumbent_objective"))
        if not np.isfinite(incumbent) and np.isfinite(objective) and np.isfinite(delta):
            incumbent = objective - delta
        round_id = row.get("round", 1)
        candidate_records.append({"attempt": i + 1, "round": round_id, "incumbent_objective": incumbent,
                                  "B_objective": objective, "improvement": delta, "accepted": bool(row.get("accepted"))})
        if np.isfinite(incumbent) and not any(v["round"] == round_id for v in incumbent_path):
            incumbent_path.append({"round": round_id, "objective": incumbent, "stage": "round incumbent"})
        if row.get("accepted") and np.isfinite(objective):
            incumbent_path.append({"round": round_id, "objective": objective, "stage": "accepted B candidate"})
        if np.isfinite(delta):
            axs[1].annotate(f"round {round_id}", (i + 1, delta), fontsize="x-small", xytext=(3, 3), textcoords="offset points")
    if count == 4 and incumbent_path:
        axs[2].plot(range(len(incumbent_path)), [row["objective"] for row in incumbent_path], "o-", color="black")
        axs[2].set_xticks(range(len(incumbent_path)), [f"Round {row['round']}\n{row['stage']}" for row in incumbent_path], fontsize="small")
    elif count == 4:
        _missing(axs[2], "Accepted incumbent trajectory not recorded")
    if count == 4:
        axs[2].set_ylabel("Post-trim original-model ELBO")
    best = refinement.get("best_objective")
    axs[count - 2].set_title("Final best: " + (f"{best:.12g}" if best is not None else "not recorded") +
        ("; incumbent path in returned data (3-axis layout)" if count == 3 else ""), fontsize="small")
    if not history:
        _missing(axs[1], f"No candidate history: {refinement.get('stop_reason', 'not recorded')}")
    if table_rows:
        _table(axs[-1], table_rows, ["Record", "Before component", "After component", "Members / overlap", "Status"],
               page=table_page, page_size=table_page_size)
    else:
        _missing(axs[-1], "No formal CS or signal relationships supplied")
    if comparison is None:
        axs[-1].set_title("Matching not assessed; equal component IDs do not establish a match", fontsize="small")
    axs[-1].axis("off")
    figure.suptitle(f"Refinement: {refinement.get('status', 'not assessed')}; rejected candidates do not change the incumbent")
    return figure, axs, {"variant_ids": ids, "before_pip": av, "after_pip": bv,
                         "history": history, "candidate_objectives": objectives,
                         "improvements": improvements, "signal_relations": relations,
                         "before_cs": before_cs, "after_cs": after_cs, "table_rows": table_rows,
                         "candidate_records": candidate_records, "incumbent_path": incumbent_path,
                         "all_component_relations": comparison.get("component_relations", []) if comparison else [],
                         "table_page": table_page, "table_page_size": table_page_size}


@_local_style
def plot_ld_diagnostics(diagnostics, *, axes=None, table_page=0, table_page_size=None):
    """Conditional information and completed flip assessments remain distinct."""
    from .ld import diagnostic_status, diagnostic_counts, diagnostic_input_missing
    rows = diagnostics.get("variants", [])
    states = np.asarray([diagnostic_status(row) for row in rows])
    details = [row for row, state in zip(rows, states) if state != "not_flagged"]
    shown_count = min(len(details), table_page_size) if table_page_size else len(details)
    figure, axs = _axes(axes, 3, figsize=(11, 5.6 + max(2.6, .26 * shown_count)))
    expected = np.asarray([_number(row.get("expected_z")) for row in rows])
    # Read old persisted data safely: upstream zero replacements are never observations.
    missing = np.asarray([diagnostic_input_missing(row) for row in rows], dtype=bool)
    observed = np.asarray([_number(row.get("observed_z")) for row in rows])
    residual = np.asarray([_number(row.get("standardized_residual")) for row in rows])
    observed[missing], residual[missing] = np.nan, np.nan
    styles = (("not_flagged", "o", "Assessed, not flagged"), ("flagged", "x", "Assessed, flagged"),
              ("conditional_only", "^", "Conditional only; flip not assessed"),
              ("not_assessed", "s", "Not assessed / missing input"))
    for state, marker, label in styles:
        mask = states == state
        if np.any(mask):
            finite_pair = mask & np.isfinite(expected) & np.isfinite(observed)
            finite_residual = mask & np.isfinite(residual)
            # Empty scatter still records the unavailable class in the legend.
            axs[0].scatter(expected[finite_pair], observed[finite_pair], marker=marker,
                           label=f"{label} ({int(mask.sum())})", color="black")
            axs[1].scatter(np.flatnonzero(finite_residual), residual[finite_residual], marker=marker,
                           label=label, color="black")
    finite = np.concatenate((expected[np.isfinite(expected)], observed[np.isfinite(observed)]))
    if len(finite):
        axs[0].plot([min(finite), max(finite)], [min(finite), max(finite)], ls="--", color="0.5")
    axs[0].set_xlabel("Expected z (diagnostic model)")
    axs[0].set_ylabel("Observed z")
    axs[0].set_title("Diagnostic scale; missing observations omitted", fontsize="small")
    if rows:
        axs[0].legend(fontsize="small")
    axs[1].set_xlabel("Variant index (zero-based)")
    axs[1].set_ylabel("Standardized observed residual")
    axs[1].axhline(0, color="0.5")
    axs[2].axis("off")
    counts = diagnostic_counts(diagnostics)
    summary = (f"Assessed: {counts['assessed']}; flagged: {counts['flagged']}; "
               f"conditional only: {counts['conditional_only']}; unassessed: {counts['unassessed']}")
    axs[2].set_title(summary, fontsize="small")
    if details:
        _table(axs[2], [[r.get("variant_id"), r.get("standardized_residual") if not diagnostic_input_missing(r) else None,
                        r.get("log_likelihood_ratio"), diagnostic_status(r), r.get("unavailable_reason")]
                       for r in details],
               labels=["Variant", "Observed residual", "Log likelihood ratio", "Assessment", "Unavailable reason"],
               page=table_page, page_size=table_page_size)
    else:
        _missing(axs[2], "No flags among assessed variants" if counts["assessed"] else "LD diagnostics not assessed")
    thresholds = diagnostics.get("thresholds", {})
    mixture = diagnostics.get("mixture", {})
    axs[1].set_title(f"Flip thresholds: log LR > {thresholds.get('flip_log_lr', 'not recorded')}, "
                    f"|z| > {thresholds.get('flip_abs_z', 'not recorded')}; mixture: {mixture.get('status', 'not recorded')}", fontsize="small")
    figure.suptitle(str(diagnostics.get("method", "LD diagnostic method not recorded")))
    return figure, axs, {"variants": rows, "flagged_or_unassessed": details, "assessment_states": states.tolist(),
                         "observed_z": observed, "expected_z": expected, "standardized_residual": residual,
                         "status_counts": counts, "assessment_summary": summary,
                         "thresholds": thresholds, "mixture": mixture,
                         "regularization": diagnostics.get("regularization"), "method": diagnostics.get("method")}


@_local_style
def plot_sensitivity(comparison, *, candidates_only=True, axes=None, layer="signals",
                     baseline_only=True, table_page=0, table_page_size=None):
    """Candidate × setting PIP heatmap; missing inputs remain masked, not zero."""
    import matplotlib.pyplot as plt
    from .comparison import comparison_relations
    relations = comparison_relations(comparison, layer=layer, baseline_only=baseline_only)
    # Show changed/unresolved relationships before stable matched rows.
    relations = sorted(relations, key=lambda row: row.get("status") == "matched")
    shown_count = min(len(relations), table_page_size) if table_page_size else len(relations)
    figure, axs = _axes(axes, 2, figsize=(11, 3.2 + max(2.6, .26 * shown_count)))
    all_rows, settings = comparison.get("variants", []), comparison.get("settings", [])
    rows = [row for row in all_rows if row.get("candidate", True)] if candidates_only else all_rows
    matrix = np.asarray([[_number(row["pips"].get(s)) for s in settings] for row in rows], dtype=float)
    if len(rows) and settings:
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#cccccc")
        obj = axs[0].imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        figure.colorbar(obj, ax=axs[0], label="Marginal PIP (gray = missing variant)")
        _variant_ticks(axs[0], settings)
        _variant_ticks(axs[0], [row["variant_id"] for row in rows], axis="y")
        axs[0].set_ylabel("Candidate variants" if candidates_only else "Variants (selected ID labels)")
    else:
        _missing(axs[0], "No sensitivity comparison supplied")
    axs[1].axis("off")
    if relations:
        _table(axs[1], [[r.get("setting_a", ""), r.get("setting_b", ""),
                               r.get("component_a", ""), r.get("component_b", ""),
                               r.get("posterior_overlap", "not assessed"), r.get("status", "not assessed")]
                              for r in relations],
                     labels=["Setting A", "Setting B", "Component A", "Component B", "Posterior overlap", "Relation"],
                     page=table_page, page_size=table_page_size)
    else:
        _missing(axs[1], "No signal relationships assessed")
    axs[1].set_title(f"{layer.title()} layer; " + ("baseline versus each setting" if baseline_only else "all setting pairs"), fontsize="small")
    return figure, axs, {"variant_ids": [r["variant_id"] for r in rows], "settings": settings,
                         "pip_matrix": matrix, "signal_relations": relations, "missing_is_zero": False,
                         "candidates_only": candidates_only, "all_variant_records": all_rows,
                         "layer": layer, "baseline_only": baseline_only,
                         "all_component_relations": comparison.get("component_relations", []),
                         "table_page": table_page, "table_page_size": table_page_size}


@_local_style
def plot_prior_influence(summary, component_id=None, *, variant_id=None, variant_ids=None, axes=None):
    """Aligned prior, conditional component log BF and alpha panels."""
    if component_id is None or summary.get("components"):
        component_id = _select_component(summary, component_id, variant_id)["component_id"]
    elif variant_id is not None:
        raise ValueError("Variant-based selection requires component posterior records")
    all_rows = [r for r in summary.get("prior_influence", []) if r["component_id"] == component_id]
    chosen = None if variant_ids is None else set(variant_ids)
    if chosen is not None and not chosen.issubset({r["variant_id"] for r in all_rows}):
        raise ValueError("Selected prior-influence variant ID is absent")
    rows = [r for r in all_rows if chosen is None or r["variant_id"] in chosen]
    figure, axs = _axes(axes, 3)
    ids = [r["variant_id"] for r in rows]
    for ax, key, label in zip(axs, ("prior_weight", "conditional_log_bf", "alpha"),
                              ("Variant prior weight", "Conditional component log BF", "Component alpha")):
        ax.bar(range(len(rows)), [_number(row.get(key)) for row in rows], color="0.3")
        ax.set_ylabel(label)
        _variant_ticks(ax, ids)
        ax.set_xlabel("Variant order (selected ID labels; all data shown)")
        if not rows:
            _missing(ax, "Prior influence not assessed")
    if rows and any("effective_prior_weight" in row for row in rows):
        axs[0].containers[0].set_label("Raw supplied prior")
        axs[0].plot(range(len(rows)), [_number(row.get("effective_prior_weight")) for row in rows],
                    "s", mfc="none", color="black", label="Effective SER prior (recorded floor)")
        axs[0].legend(fontsize="small")
    figure.suptitle(f"Component {component_id}: conditional BF depends on other components and fitting settings")
    return figure, axs, {"component_id": component_id, "prior_influence": rows,
                         "all_prior_influence": all_rows, "selected_variant": variant_id,
                         "selected_variant_ids": None if chosen is None else sorted(chosen)}
