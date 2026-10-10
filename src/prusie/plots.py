"""Seven deterministic analysis views with optional, lazily imported Matplotlib.

Every view returns ``(figure, axes, plotted_data)`` and accepts existing axes.
The functions never mutate input evidence or global plotting configuration.
At most twelve evenly spaced variant labels are drawn per axis. This keeps
regional views readable while retaining every observation in the plot and data.
Tables show at most twelve rows; complete records remain in returned data.
"""
from __future__ import annotations

import functools
from pathlib import Path
import re

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


def _table(ax, rows, labels):
    """Bound the rendered table, retaining complete records in the view data."""
    def display(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.5g}"
        text = str(value)
        return text if len(text) <= 48 else text[:45] + "…"
    shown = [[display(v) for v in row] for row in rows[:12]]
    table = ax.table(cellText=shown, colLabels=labels, bbox=[0, .14, 1, .84])
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    if len(rows) > 12:
        ax.text(.5, .03, f"First 12 of {len(rows)} rows; complete records in returned data/report.",
                ha="center", va="bottom", transform=ax.transAxes, fontsize="small")
    return table


def _number(value):
    return np.nan if value is None else float(value)


def _variant_ticks(ax, labels, *, axis="x"):
    """Thin labels only, deterministically; never thin observations."""
    indices = np.unique(np.linspace(0, len(labels)-1, min(12, len(labels)), dtype=int)) if labels else []
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


def read_gene_track(path, *, genome_build, source, transcript_ids=None):
    """Read a local GTF into explicit 1-based closed transcript intervals.

    All transcripts are retained unless IDs are supplied. Gene-level GTF rows
    are used only when no transcript rows exist; no transcript is chosen silently.
    A user table can instead be passed directly to :func:`plot_locus`.
    """
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
            start, end = int(fields[3]), int(fields[4])
            if start < 1 or end < start:
                raise ValueError("Invalid GTF interval")
            row = {"chromosome": fields[0], "start": start, "end": end,
                   "strand": fields[6], "label": attr.get("gene_name", identifier),
                   "transcript_id": identifier}
            (rows if fields[2] == "transcript" else genes).append(row)
    return {"records": rows or genes, "genome_build": genome_build, "source": source,
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
    if genes is not None:
        required = ("records", "genome_build", "source", "coordinate_convention", "transcript_selection")
        if not all(key in genes for key in required):
            raise ValueError("Gene track requires build, source, transcript selection and coordinate convention")
        if genes["coordinate_convention"] != "1-based closed":
            raise ValueError("Convert the gene table to 1-based closed coordinates explicitly")
        qc = summary.get("input_qc", {})
        build = qc.get("genome_build")
        if build not in (None, "not_assessed", genes["genome_build"]):
            raise ValueError("Gene and variant genome builds differ")
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
        chromosomes = {str(row["chromosome"]) for row in genes["records"] if "chromosome" in row}
        if len(chromosomes) > 1:
            raise ValueError("A locus gene track must use one chromosome")
        for i, gene in enumerate(genes["records"]):
            axs[4].plot([gene["start"], gene["end"]], [i, i], color="black")
            axs[4].text(gene["start"], i, str(gene.get("label", "")), va="bottom")
        axs[4].set_ylabel("Transcripts")
        axs[4].set_title(f"{genes['genome_build']}; {genes['coordinate_convention']}; {genes['source']}")
    for ax in axs:
        ax.set_xlabel(label)
        if len(x) > 1:
            span = max(float(np.ptp(x)), 1.)
            ax.set_xlim(float(min(x)) - .02 * span, float(max(x)) + .02 * span)
    return figure, axs, {"variant_ids": ids, "x": x, "physical_positions": physical,
                         "neg_log10_p": association, "pip": pip, "cs_memberships": memberships,
                         "ld_reference": reference_variant, "r_squared": ld_values, "genes": genes}


@_local_style
def plot_signal(summary, component_id, *, R=None, axes=None):
    """Ranked alpha and cumulative mass, within-CS signed LD, direct proxies."""
    component = next((c for c in summary["components"] if c["component_id"] == component_id), None)
    if component is None:
        raise ValueError("Unknown component ID")
    figure, axs = _axes(axes, 4)
    ranking, alpha = component["ranking"], np.asarray(component["alpha"])
    rank = np.arange(len(ranking))
    axs[0].bar(rank, alpha, color="0.3")
    axs[0].set_ylabel("Component alpha")
    axs[1].plot(rank, component["cumulative_posterior"], marker=".", color="black")
    axs[1].set_ylabel("Cumulative posterior")
    cs = next((c for c in summary.get("credible_sets", []) if c["component_id"] == component_id), None)
    if cs:
        axs[1].axhline(cs["requested_coverage"], ls="--", color="black", label="Requested CS coverage")
        axs[1].legend()
    for ax in axs[:2]:
        _variant_ticks(ax, ranking)
        ax.set_xlabel("Variants ranked by component alpha (selected ID labels)")
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
                     labels=["Proxy outside CS", "Direct CS partner", "Signed r"])
    else:
        _missing(axs[3], "No direct proxies reported; no transitive expansion")
    figure.suptitle(f"Component {component_id}; null mass = {component.get('null_mass', 'not assessed')}")
    return figure, axs, {"component": component, "credible_set": cs, "within_cs_r": within, "proxies": proxies}


@_local_style
def plot_convergence(fit_or_summary, *, axes=None):
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
                key != "delta_elbo" and criterion == "alpha_pip")
            label = "Selected stopping threshold" if selected else "Diagnostic threshold (not selected stopping criterion)"
            ax.axhline(tolerance, ls="--", color="0.5", label=label)
            ax.legend(fontsize="small")
    v_rows = [row.get("V") for row in trace]
    if trace and all(v is not None for v in v_rows):
        width = max(len(v) for v in v_rows)
        for i in range(width):
            axs[4].plot(iterations, [v[i] if len(v) > i else np.nan for v in v_rows], label=f"V component {i}")
        axs[4].legend(fontsize="small")
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
    phases = sorted({str(row.get("phase", "not recorded")) for row in trace})
    figure.suptitle(f"Stop: {summary.get('stop_reason', 'not assessed')}; phases: {', '.join(phases) or 'unavailable'}")
    if summary.get("finalization"):
        axs[-1].text(1., -.4, "Finalization is recorded separately; not an IBSS iteration.",
                     ha="right", transform=axs[-1].transAxes, fontsize="small")
    return figure, axs, {"trace": trace, "convergence": summary, "events": events,
                         "missing_trace": not bool(trace), "phases": phases,
                         "finalization_separate_from_iterations": summary.get("finalization")}


@_local_style
def plot_refinement(before, after, *, comparison=None, axes=None):
    """PIP before/after, candidate objectives and explicitly matched CS changes."""
    figure, axs = _axes(axes, 3)
    def values(value):
        if isinstance(value, dict):
            return {v["variant_id"]: v["pip"] for v in value["variants"]}
        return dict(zip(value.variant_ids.tolist(), value.pip.tolist()))
    a, b = values(before), values(after)
    ids = list(dict.fromkeys([*a, *b]))
    x = np.arange(len(ids))
    av, bv = [a.get(v, np.nan) for v in ids], [b.get(v, np.nan) for v in ids]
    axs[0].plot(x, av, "o-", label="Before")
    axs[0].plot(x, bv, "x-", label="After")
    axs[0].set_ylabel("Marginal PIP")
    _variant_ticks(axs[0], ids)
    axs[0].set_xlabel("Variant order (selected ID labels; all data shown)")
    axs[0].legend()
    refinement = after.get("refinement", {}) if isinstance(after, dict) else getattr(after, "refinement", {})
    history = refinement.get("history", [])
    objectives = [_number(row.get("candidate_objective", row.get("objective"))) for row in history]
    improvements = [_number(row.get("improvement")) for row in history]
    axs[1].plot(np.arange(len(history)), improvements, "o", color="black")
    axs[1].axhline(0, color="0.5")
    axs[1].set_xlabel("Candidate attempt (stage B only)")
    axs[1].set_ylabel("Original-model ELBO improvement")
    for i, row in enumerate(history):
        if np.isfinite(improvements[i]):
            axs[1].annotate(("accepted" if row.get("accepted") else str(row.get("reason", "attempted"))).replace("_", " "),
                            (i, improvements[i]), fontsize="x-small", rotation=20,
                            ha="right" if i >= len(history) / 2 else "left")
    if not history:
        _missing(axs[1], f"No candidate history: {refinement.get('stop_reason', 'not recorded')}")
    before_cs = before.get("credible_sets", []) if isinstance(before, dict) else before.sets
    after_cs = after.get("credible_sets", []) if isinstance(after, dict) else after.sets
    if comparison is None:
        relations = []
    else:
        relations = comparison.get("signal_relations", [])
    table_rows = []
    for stage, sets in (("Before CS", before_cs), ("After CS", after_cs)):
        if isinstance(sets, dict):
            sets = [{"component_id": int(c), "members": list(m)} for c, m in zip(sets.get("cs_index", []), sets.get("cs", []))]
        for cs in sets:
            table_rows.append([stage, str(cs["component_id"]), "", ", ".join(map(str, cs["members"])), "retained"])
    for relation in relations:
        table_rows.append(["Signal relation", str(relation.get("component_a", "")),
                           str(relation.get("component_b", "")),
                           "overlap=" + str(relation.get("posterior_overlap", "not assessed")),
                           relation.get("status", "not assessed")])
    if table_rows:
        _table(axs[2], table_rows, ["Record", "Before component", "After component", "Members / overlap", "Status"])
    else:
        _missing(axs[2], "No formal CS or signal relationships supplied")
    if comparison is None:
        axs[2].set_title("Matching not assessed; equal component IDs do not establish a match", fontsize="small")
    axs[2].axis("off")
    figure.suptitle(f"Refinement: {refinement.get('status', 'not assessed')}")
    return figure, axs, {"variant_ids": ids, "before_pip": av, "after_pip": bv,
                         "history": history, "candidate_objectives": objectives,
                         "improvements": improvements, "signal_relations": relations,
                         "before_cs": before_cs, "after_cs": after_cs}


@_local_style
def plot_ld_diagnostics(diagnostics, *, axes=None):
    """Observed/expected adjusted z, standardized residuals and flagged details."""
    figure, axs = _axes(axes, 3)
    rows = diagnostics.get("variants", [])
    expected = np.asarray([_number(row.get("expected_z")) for row in rows])
    observed = np.asarray([_number(row.get("observed_z")) for row in rows])
    residual = np.asarray([_number(row.get("standardized_residual")) for row in rows])
    flag = np.asarray([bool(row.get("flag", False)) for row in rows])
    for mask, marker, label in ((~flag & np.isfinite(residual), "o", "Evaluated, not flagged"), (flag, "x", "Flagged")):
        axs[0].scatter(expected[mask], observed[mask], marker=marker, label=label, color="black")
        axs[1].scatter(np.flatnonzero(mask), residual[mask], marker=marker, label=label, color="black")
    finite = np.concatenate((expected[np.isfinite(expected)], observed[np.isfinite(observed)]))
    if len(finite):
        axs[0].plot([min(finite), max(finite)], [min(finite), max(finite)], ls="--", color="0.5")
    axs[0].set_xlabel("Expected z (diagnostic model)")
    axs[0].set_ylabel("Observed z (diagnostic scale)")
    axs[0].legend()
    axs[1].set_xlabel("Variant index (zero-based)")
    axs[1].set_ylabel("Standardized residual")
    axs[1].axhline(0, color="0.5")
    axs[2].axis("off")
    flagged = [row for row in rows if row.get("flag") or not np.isfinite(_number(row.get("standardized_residual")))]
    if flagged:
        _table(axs[2], [[r.get("variant_id"), r.get("standardized_residual"),
                               r.get("log_likelihood_ratio"), r.get("reason", r.get("status", ""))] for r in flagged],
                     labels=["Variant", "Residual", "Log likelihood ratio", "Status"])
    else:
        _missing(axs[2], "No flagged variants" if rows else "LD diagnostics not assessed")
    figure.suptitle(str(diagnostics.get("method", "LD diagnostic method not recorded")))
    return figure, axs, {"variants": rows, "flagged_or_unassessed": flagged,
                         "regularization": diagnostics.get("regularization"), "method": diagnostics.get("method")}


@_local_style
def plot_sensitivity(comparison, *, candidates_only=False, axes=None):
    """Candidate × setting PIP heatmap; missing inputs remain masked, not zero."""
    import matplotlib.pyplot as plt
    figure, axs = _axes(axes, 2)
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
    relations = comparison.get("signal_relations", [])
    if relations:
        _table(axs[1], [[r.get("setting_a", ""), r.get("setting_b", ""),
                               r.get("component_a", ""), r.get("component_b", ""),
                               r.get("posterior_overlap", "not assessed"), r.get("status", "not assessed")]
                              for r in relations],
                     labels=["Setting A", "Setting B", "Component A", "Component B", "Posterior overlap", "Relation"])
    else:
        _missing(axs[1], "No signal relationships assessed")
    return figure, axs, {"variant_ids": [r["variant_id"] for r in rows], "settings": settings,
                         "pip_matrix": matrix, "signal_relations": relations, "missing_is_zero": False,
                         "candidates_only": candidates_only, "all_variant_records": all_rows}


@_local_style
def plot_prior_influence(summary, component_id, *, axes=None):
    """Aligned prior, conditional component log BF and alpha panels."""
    rows = [r for r in summary.get("prior_influence", []) if r["component_id"] == component_id]
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
    figure.suptitle("Conditional BF depends on other components and fitting settings")
    return figure, axs, {"component_id": component_id, "prior_influence": rows}
