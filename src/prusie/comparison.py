"""Comparison of fits across settings and execution of sensitivity plans.

:func:`compare_fits` compares existing fits without fitting. Only
:func:`run_sensitivity` fits models, and only for the baseline and the
single-factor changes listed in its plan; no grid is expanded.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import inspect
import itertools
import json
import math
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .analysis import _biological, _correlation, _ids, summarize
from .result import SusieResult


def _directions(a, b, shared):
    """Relative allele orientation (+1/-1) of shared variants, or None if unknown."""
    ia, ib = {x: i for i, x in enumerate(_ids(a))}, {x: i for i, x in enumerate(_ids(b))}
    ma, mb = a.variant_metadata, b.variant_metadata
    names = ("effect_allele", "other_allele")
    present = all(key in m for m in (ma, mb) for key in names)
    if not present:
        return {x: None for x in shared}, "not_assessed_missing_alleles"
    directions = {}
    for x in shared:
        pa = tuple(str(ma[k][ia[x]]).upper() for k in names)
        pb = tuple(str(mb[k][ib[x]]).upper() for k in names)
        if any(x in ("", "NONE", "NAN") for x in pa + pb):
            directions[x] = None
        elif pa == pb:
            directions[x] = 1
        elif pa == pb[::-1]:
            directions[x] = -1
        else:
            raise ValueError(f"incompatible allele coding for variant {x}; harmonize explicitly")
    return directions, "explicit_alleles_aligned"


def _signals(fit):
    """Records of active effects and of effects with a retained credible set."""
    ids, alpha = _ids(fit), _biological(fit, fit.alpha)
    formal = {
        int(i): {
            int(x) - int(fit.null_index is not None and x > fit.null_index)
            for x in members
            if x != fit.null_index
        }
        for i, members in zip(fit.sets["cs_index"], fit.sets["cs"], strict=True)
    }
    records = []
    for i, a in enumerate(alpha):
        active = bool(fit.V[i] > fit.params.get("prior_tol", 1e-9))
        retained = i in formal
        # The PIP activity threshold (prior_tol) and the fixed CS threshold can
        # differ, so a retained CS is kept even for an effect inactive for PIP.
        if not active and not retained:
            continue
        top = min(range(len(ids)), key=lambda j: (-float(a[j]), ids[j])) if ids else None
        positive = a[a > 0]
        mass = float(positive.sum())
        effective = None
        if mass > 0:
            weights = positive / mass
            effective = float(np.exp(-np.sum(weights * np.log(weights))))
        resolution = dict(
            effective_candidates=effective,
            effective_fraction=effective / len(ids) if effective is not None and ids else None,
            biological_mass=mass,
            cs_size=len(formal[i]) if retained else None,
            definition=("entropy effective candidates normalized over biological variants"),
        )
        record = dict(
            component_id=i,
            alpha={x: float(v) for x, v in zip(ids, a, strict=True)},
            cs={ids[j] for j in formal.get(i, set())},
            top=ids[top] if top is not None else None,
            V=float(fit.V[i]),
            active=active,
            cs_retained=retained,
            cs_size=resolution["cs_size"],
            purity=None,
            effective_candidates=effective,
            resolution=resolution,
            localization_status="formal_cs_retained" if retained else "unresolved_no_formal_cs",
            support_sources=["formal_cs"] if retained else ["active_component_only"],
        )
        if retained:
            k = list(fit.sets["cs_index"]).index(i)
            values = fit.sets.get("purity", {}).get("min_abs_corr", [])
            record["purity"] = float(values[k]) if k < len(values) else None
        records.append(record)
    return records


def _classify_signal_edges(edges):
    """Label edges between retained credible sets as matched, split, merge or ambiguous."""
    formal = [e for e in edges if e.get("cs_retained_a") and e.get("cs_retained_b")]
    for edge in edges:
        if not (edge.get("cs_retained_a") and edge.get("cs_retained_b")):
            edge["status"] = "unresolved"
            continue
        da = sum(e["component_a"] == edge["component_a"] for e in formal)
        db = sum(e["component_b"] == edge["component_b"] for e in formal)
        edge["status"] = (
            "ambiguous"
            if da > 1 and db > 1
            else "split"
            if da > 1
            else "merge"
            if db > 1
            else "matched"
        )
        edge["degree_a"], edge["degree_b"] = da, db
    return edges


def _unmatched(na, nb, record, side, shared, *, layer, support_scope="pairwise_intersection"):
    other = "b" if side == "a" else "a"
    return dict(
        setting_a=na,
        setting_b=nb,
        **{
            f"component_{side}": record["component_id"],
            f"component_{other}": None,
            f"cs_retained_{side}": record["cs_retained"],
            f"cs_retained_{other}": None,
            f"resolution_{side}": copy.deepcopy(record["resolution"]),
            f"resolution_{other}": None,
            f"cs_size_{side}": record["cs_size"],
            f"cs_size_{other}": None,
            f"representative_{side}": record["top"],
            f"representative_{other}": None,
        },
        status="unmatched" if layer == "signals" else "unmatched_component",
        side=side,
        matching_layer=layer,
        support_sources=list(record["support_sources"]),
        support_scope=support_scope,
        shared_variant_count=len(shared),
        unobserved_mass=sum(v for x, v in record["alpha"].items() if x not in shared),
    )


def _baseline_first(rows, baseline):
    """Orient relation rows so that the baseline setting is always side a."""
    out = []
    for original in rows:
        if baseline not in (original.get("setting_a"), original.get("setting_b")):
            continue
        row = copy.deepcopy(original)
        if row.get("setting_b") == baseline:
            row = {
                (
                    key[:-2] + "_b"
                    if key.endswith("_a")
                    else key[:-2] + "_a"
                    if key.endswith("_b")
                    else key
                ): value
                for key, value in row.items()
            }
            if row.get("status") in ("split", "merge"):
                row["status"] = {"split": "merge", "merge": "split"}[row["status"]]
            if row.get("side") in ("a", "b"):
                row["side"] = {"a": "b", "b": "a"}[row["side"]]
        out.append(row)
    return out


def comparison_relations(
    comparison: Mapping[str, Any], *, layer: str = "signals", baseline_only: bool = True
) -> list[dict[str, Any]]:
    """Return the relations of a comparison record, including version-1 records.

    Parameters
    ----------
    comparison : dict
        Output of :func:`compare_fits`, or a saved record of either schema.
    layer : {'signals', 'components'}, default 'signals'
        'signals' returns relations involving a retained credible set;
        'components' returns all relations between active effects.
    baseline_only : bool, default True
        Keep only relations to the baseline, oriented baseline first.

    Returns
    -------
    list of dict

    Notes
    -----
    Version-1 records store credible-set sizes only for connected effects.
    For these records, unmatched effects without a recorded size are kept in
    the component layer and are not treated as localized signals.
    """
    if layer not in ("signals", "components"):
        raise ValueError("layer must be signals or components")
    if "component_relations" in comparison:
        rows = copy.deepcopy(
            comparison.get("signal_relations" if layer == "signals" else "component_relations", [])
        )
    else:
        rows = copy.deepcopy(comparison.get("signal_relations", []))
        for row in rows:
            for side in ("a", "b"):
                row[f"cs_retained_{side}"] = bool(row.get(f"cs_size_{side}"))
            row["matching_layer"] = "components"
            row["legacy_support_status"] = (
                "inferred_from_recorded_cs_size; unmatched_support_unavailable"
            )
            sources = []
            if row.get("posterior_overlap", -1) >= comparison.get("thresholds", {}).get(
                "posterior_overlap", 0.5
            ):
                sources.append("posterior_overlap")
            if row.get("cs_overlap"):
                sources.append("formal_cs_overlap")
            if row.get("signed_r") is not None and abs(row["signed_r"]) >= comparison.get(
                "thresholds", {}
            ).get("abs_r", 0.9):
                sources.append("representative_ld")
            row["support_sources"] = sources
        if layer == "signals":
            recorded = rows
            rows = [
                row
                for row in recorded
                if (row["cs_retained_a"] and row["cs_retained_b"])
                or (
                    (row["cs_retained_a"] or row["cs_retained_b"])
                    and "posterior_overlap" in row["support_sources"]
                )
            ]
            for pair in {(r["setting_a"], r["setting_b"]) for r in rows}:
                _classify_signal_edges(
                    [r for r in rows if (r["setting_a"], r["setting_b"]) == pair]
                )
            for row in rows:
                row["matching_layer"] = "signals"
            # In a version-1 record, an LD-only edge to an effect without a CS
            # still shows that the CS on the other side exists, but not that it
            # is matched; keep that CS as an unmatched signal.
            for row in recorded:
                for side in ("a", "b"):
                    if not row[f"cs_retained_{side}"]:
                        continue
                    if any(
                        e["setting_a"] == row["setting_a"]
                        and e["setting_b"] == row["setting_b"]
                        and e.get(f"component_{side}") == row[f"component_{side}"]
                        for e in rows
                    ):
                        continue
                    other = "b" if side == "a" else "a"
                    rows.append(
                        dict(
                            setting_a=row["setting_a"],
                            setting_b=row["setting_b"],
                            **{
                                f"component_{side}": row[f"component_{side}"],
                                f"component_{other}": None,
                                f"cs_retained_{side}": True,
                                f"cs_retained_{other}": None,
                                f"cs_size_{side}": row[f"cs_size_{side}"],
                                f"cs_size_{other}": None,
                                f"representative_{side}": row.get(f"representative_{side}"),
                                f"representative_{other}": None,
                            },
                            status="unmatched",
                            side=side,
                            matching_layer="signals",
                            support_sources=["formal_cs"],
                            unobserved_mass=row.get(f"unobserved_mass_{side}"),
                            legacy_support_status="formal_CS_identity_recovered_from_recorded_cs_size",
                        )
                    )
        else:
            for row in rows:
                if not (row["cs_retained_a"] and row["cs_retained_b"]):
                    row["status"] = (
                        "component_overlap"
                        if "posterior_overlap" in row["support_sources"]
                        else "component_relation"
                    )
    return _baseline_first(rows, comparison.get("baseline")) if baseline_only else rows


def compare_fits(
    fits: Mapping[str, SusieResult],
    *,
    baseline: str | None = None,
    R: ArrayLike | Mapping[str, ArrayLike] | None = None,
    ld_reference: str | None = None,
    pip_threshold: float = 0.1,
    overlap_threshold: float = 0.5,
    ld_threshold: float = 0.9,
    mode: str = "union",
    top_candidates: int = 3,
) -> dict[str, Any]:
    """Compare existing fits by variant PIPs and by relations between their effects.

    Parameters
    ----------
    fits : mapping of str to SusieResult
        Named fits, aligned by variant ID.
    baseline : str, optional
        Name of the reference fit; defaults to the first.
    R : array_like, optional
        Signed LD correlation matrix in the variant order of ``ld_reference``,
        used for LD between the representative variants of two effects.
    ld_reference : str, optional
        Fit whose variant order matches ``R``; defaults to ``baseline``.
    pip_threshold : float, default 0.1
        Variants with PIP ≥ pip_threshold in any fit are candidates.
    overlap_threshold : float, default 0.5
        Minimum posterior overlap for a relation between two effects.
    ld_threshold : float, default 0.9
        Minimum |r| between representative variants for an LD relation.
    mode : {'union', 'common'}, default 'union'
        Variant support of the comparison; see Notes.
    top_candidates : int, default 3
        Number of leading variants per retained credible set (or of leading
        PIPs when a fit has no credible set) added to the candidates.

    Returns
    -------
    dict
        Per-variant PIPs, ranks and candidate sources; ``signal_relations``
        between retained credible sets and their posterior-overlap relations
        to unresolved effects; ``component_relations`` for all active or
        CS-retained effects; pairwise PIP summaries, allele alignment,
        thresholds, definitions and limitations.

    Notes
    -----
    Candidates comprise all credible-set members, the leading variants
    described by ``top_candidates`` and every variant crossing
    ``pip_threshold`` in any fit. In ``mode='union'``, all variants are
    listed, and each pair of fits is compared on the variants they share. In
    ``mode='common'``, ranks, candidate selection, PIP comparisons and
    posterior/CS overlap use the variants present in all fits, and an LD
    relation requires both representatives to be among them. Component
    records, formal credible sets and resolution summaries retain the
    original fit support.
    Posterior mass is never renormalized to the shared variants, and missing
    variants are reported as None rather than as zero PIP. Relations
    describe overlap of posterior support; they are not evidence that
    effects have split or merged, and LD alone does not promote an effect
    without a credible set to a signal.
    """
    if not fits or any(not isinstance(name, str) or not name for name in fits):
        raise ValueError("fits must be a nonempty mapping with nonempty string names")
    original_names = list(fits)
    baseline = original_names[0] if baseline is None else baseline
    if baseline not in fits:
        raise ValueError("baseline must name a supplied fit")
    names = [baseline] + [name for name in original_names if name != baseline]
    if mode not in ("union", "common"):
        raise ValueError("mode must be union or common")
    if not 0 <= pip_threshold <= 1 or not 0 <= overlap_threshold <= 1 or not 0 <= ld_threshold <= 1:
        raise ValueError("comparison thresholds must lie in [0, 1]")
    if (
        isinstance(top_candidates, bool)
        or not isinstance(top_candidates, (int, np.integer))
        or top_candidates < 0
    ):
        raise ValueError("top_candidates must be a nonnegative integer")
    ids = {name: _ids(fit) for name, fit in fits.items()}
    maps = {
        name: dict(zip(ids[name], map(float, fit.pip), strict=True)) for name, fit in fits.items()
    }
    signals = {name: _signals(fit) for name, fit in fits.items()}
    universe = sorted(
        set.union(*(set(v) for v in ids.values()))
        if mode == "union"
        else set.intersection(*(set(v) for v in ids.values()))
    )
    common_support = set(universe) if mode == "common" else None
    support_scope = "global_intersection" if mode == "common" else "pairwise_intersection"
    ranks = {
        name: {
            x: i + 1
            for i, x in enumerate(
                sorted(
                    universe if mode == "common" else ids[name], key=lambda x: (-maps[name][x], x)
                )
            )
        }
        for name in names
    }
    candidate_sources = {x: {} for x in universe}
    for name in names:
        formal = [s for s in signals[name] if s["cs_retained"]]
        cs_members = set().union(*(s["cs"] for s in formal))
        top_members = {
            x
            for s in formal
            for x in sorted(
                (y for y in s["alpha"] if common_support is None or y in common_support),
                key=lambda y: (-s["alpha"][y], y),
            )[:top_candidates]
            if s["alpha"][x] > 0
        }
        for x in universe:
            sources = []
            if x in maps[name] and maps[name][x] >= pip_threshold:
                sources.append("pip_threshold")
            if x in cs_members:
                sources.append("formal_cs_member")
            if x in top_members:
                sources.append("component_top_candidate")
            if (
                not formal
                and ranks[name].get(x, math.inf) <= top_candidates
                and maps[name].get(x, 0) > 0
            ):
                sources.append("regional_top_pip_without_cs")
            if sources:
                candidate_sources[x][name] = sources
    variants = []
    for x in universe:
        pips = {name: maps[name].get(x) for name in names}
        observed = [v for v in pips.values() if v is not None]
        differences = (
            [abs(v - pips[baseline]) for v in observed] if pips[baseline] is not None else []
        )
        threshold = [v >= pip_threshold for v in observed]
        variants.append(
            dict(
                variant_id=x,
                pips=pips,
                pip_min=min(observed),
                pip_max=max(observed),
                pip_range=max(observed) - min(observed),
                max_abs_delta_from_baseline=max(differences) if differences else None,
                ranks={name: ranks[name].get(x) for name in names},
                threshold_crossings=bool(any(threshold) and not all(threshold)),
                candidate=bool(candidate_sources[x]),
                candidate_sources=candidate_sources[x],
                missing_settings=[name for name in names if pips[name] is None],
            )
        )
    corr, ld_ids = None, {}
    if R is not None:
        ld_reference = baseline if ld_reference is None else ld_reference
        if ld_reference not in fits:
            raise ValueError("ld_reference must name a fit whose ID order matches R")
        corr = _correlation(R, len(ids[ld_reference]))
        ld_ids = {x: i for i, x in enumerate(ids[ld_reference])}
    relations, component_relations, alignments, pairwise, display = [], [], [], [], []
    for na, nb in itertools.combinations(names, 2):
        shared = common_support if common_support is not None else set(ids[na]) & set(ids[nb])
        ordered_shared = sorted(shared)
        directions, direction_status = _directions(fits[na], fits[nb], ordered_shared)
        alignments.append(
            dict(
                setting_a=na,
                setting_b=nb,
                shared_ids=ordered_shared,
                directions=directions,
                status=direction_status,
                support_scope=support_scope,
                omitted_a=sorted(set(ids[na]) - shared),
                omitted_b=sorted(set(ids[nb]) - shared),
            )
        )
        pa, pb = [maps[na][x] for x in ordered_shared], [maps[nb][x] for x in ordered_shared]
        correlation = (
            float(np.corrcoef(pa, pb)[0, 1])
            if len(shared) > 1 and np.std(pa) > 0 and np.std(pb) > 0
            else None
        )
        pairwise.append(
            dict(
                setting_a=na,
                setting_b=nb,
                shared_variant_count=len(shared),
                support_scope=support_scope,
                full_region_pip_correlation=correlation,
                max_abs_delta_pip=max(
                    (abs(a - b) for a, b in zip(pa, pb, strict=True)), default=None
                ),
                candidate_max_abs_delta_pip=max(
                    (
                        abs(maps[na][x] - maps[nb][x])
                        for x in ordered_shared
                        if candidate_sources[x]
                    ),
                    default=None,
                ),
            )
        )
        edges = []
        for a, b in itertools.product(signals[na], signals[nb]):
            overlap = math.fsum(min(a["alpha"][x], b["alpha"][x]) for x in ordered_shared)
            common_cs = sorted(a["cs"] & b["cs"] & shared)
            representatives_available = mode == "union" or (
                a["top"] in shared and b["top"] in shared
            )
            r = (
                float(corr[ld_ids[a["top"]], ld_ids[b["top"]]])
                if representatives_available
                and corr is not None
                and a["top"] in ld_ids
                and b["top"] in ld_ids
                else None
            )
            support = (
                ["posterior_overlap"]
                if overlap >= overlap_threshold and (mode == "union" or bool(shared))
                else []
            )
            support += ["formal_cs_overlap"] if common_cs else []
            support += ["representative_ld"] if r is not None and abs(r) >= ld_threshold else []
            if not support:
                continue
            edge = dict(
                setting_a=na,
                setting_b=nb,
                component_a=a["component_id"],
                component_b=b["component_id"],
                posterior_overlap=overlap,
                cs_overlap=common_cs,
                representative_a=a["top"],
                representative_b=b["top"],
                signed_r=r,
                signed_r_direction="as encoded in the explicitly named LD-reference fit",
                r_squared=r * r if r is not None else None,
                candidate_exchange_high_ld=bool(
                    a["top"] != b["top"] and r is not None and abs(r) >= ld_threshold
                ),
                unobserved_mass_a=math.fsum(v for x, v in a["alpha"].items() if x not in shared),
                unobserved_mass_b=math.fsum(v for x, v in b["alpha"].items() if x not in shared),
                cs_size_a=a["cs_size"],
                cs_size_b=b["cs_size"],
                min_abs_corr_a=a["purity"],
                min_abs_corr_b=b["purity"],
                effective_candidates_a=a["effective_candidates"],
                effective_candidates_b=b["effective_candidates"],
                V_a=a["V"],
                V_b=b["V"],
                cs_retained_a=a["cs_retained"],
                cs_retained_b=b["cs_retained"],
                resolution_a=copy.deepcopy(a["resolution"]),
                resolution_b=copy.deepcopy(b["resolution"]),
                support_sources=support,
                matching_layer="components",
                support_scope=support_scope,
                shared_variant_count=len(shared),
            )
            edges.append(edge)
        main = [
            copy.deepcopy(e)
            for e in edges
            if (e["cs_retained_a"] and e["cs_retained_b"])
            or (
                (e["cs_retained_a"] or e["cs_retained_b"])
                and "posterior_overlap" in e["support_sources"]
            )
        ]
        _classify_signal_edges(main)
        by_component = {(e["component_a"], e["component_b"]): e for e in main}
        for edge in main:
            edge["matching_layer"] = "signals"
        for edge in edges:
            projected = by_component.get((edge["component_a"], edge["component_b"]))
            edge["status"] = (
                projected["status"]
                if projected
                else "component_overlap"
                if "posterior_overlap" in edge["support_sources"]
                else "ld_proximity"
            )
        for side, records in (("a", signals[na]), ("b", signals[nb])):
            for record in records:
                if not any(e[f"component_{side}"] == record["component_id"] for e in edges):
                    component_relations.append(
                        _unmatched(
                            na,
                            nb,
                            record,
                            side,
                            shared,
                            layer="components",
                            support_scope=support_scope,
                        )
                    )
                if record["cs_retained"] and not any(
                    e[f"component_{side}"] == record["component_id"] for e in main
                ):
                    main.append(
                        _unmatched(
                            na,
                            nb,
                            record,
                            side,
                            shared,
                            layer="signals",
                            support_scope=support_scope,
                        )
                    )
        component_relations.extend(edges)
        relations.extend(main)
        used_a, used_b = set(), set()
        formal_edges = [e for e in main if e.get("cs_retained_a") and e.get("cs_retained_b")]
        for edge in sorted(
            formal_edges,
            key=lambda e: (
                -e["posterior_overlap"],
                -len(e["cs_overlap"]),
                e["representative_a"],
                e["representative_b"],
                e["component_a"],
                e["component_b"],
            ),
        ):
            if edge["component_a"] not in used_a and edge["component_b"] not in used_b:
                display.append(copy.deepcopy(edge))
                used_a.add(edge["component_a"])
                used_b.add(edge["component_b"])
    primary = _baseline_first(relations, baseline)
    return dict(
        schema_version="prusie.comparison.v2",
        settings=names,
        baseline=baseline,
        mode=mode,
        support=dict(
            variant_table="global_intersection" if mode == "common" else "union",
            variant_ranks="global_intersection" if mode == "common" else "original_fit",
            pairwise_metrics=support_scope,
            matching_overlap=support_scope,
            omitted_posterior_mass=support_scope,
            component_records="original_fit",
            representative_ld="original_representatives_within_global_intersection"
            if mode == "common"
            else "original_representatives_in_ld_reference",
            global_common_ids=universe if mode == "common" else None,
        ),
        fit_settings={name: copy.deepcopy(fits[name].params) for name in names},
        variants=variants,
        components={name: [dict(s, cs=sorted(s["cs"])) for s in signals[name]] for name in names},
        signal_relations=relations,
        baseline_signal_relations=primary,
        component_relations=component_relations,
        display_matches=display,
        baseline_display_matches=_baseline_first(display, baseline),
        alignments=alignments,
        pairwise=pairwise,
        baseline_pairwise=_baseline_first(pairwise, baseline),
        layer_summary=dict(
            settings={
                name: dict(
                    retained_cs=sum(s["cs_retained"] for s in signals[name]),
                    active_components=sum(s["active"] for s in signals[name]),
                    unretained_active_components=sum(
                        s["active"] and not s["cs_retained"] for s in signals[name]
                    ),
                )
                for name in names
            },
            baseline_relation_counts={
                status: sum(e["status"] == status for e in primary)
                for status in ("matched", "ambiguous", "split", "merge", "unresolved", "unmatched")
            },
        ),
        ld_reference=ld_reference if corr is not None else None,
        thresholds=dict(
            pip=pip_threshold,
            posterior_overlap=overlap_threshold,
            abs_r=ld_threshold,
            top_candidates=int(top_candidates),
        ),
        rules=dict(
            missing="None, never zero",
            posterior_overlap="unrenormalized shared-support sum(min(alpha_a,alpha_b))",
            common_support=(
                "Global intersection for candidates/ranks, all pair "
                "metrics and matching evidence; empty support cannot "
                "establish a match"
            ),
            omitted_mass=(
                "unobserved_mass is posterior mass outside the declared "
                "comparison support; it may include variants observed "
                "in an individual pair but excluded from the global "
                "intersection"
            ),
            source_component_metadata=(
                "CS sizes, purity, representatives and entropy "
                "resolution describe original fits, without redefining "
                "their formal CS"
            ),
            matching=(
                "CS-to-CS: overlap OR CS overlap OR representative LD; "
                "CS-to-unretained: posterior overlap only, unresolved"
            ),
            component_layer=(
                "All active components plus formal CS boundary cases; "
                "diffuse overlap is descriptive, not a confirmed "
                "split/merge"
            ),
            resolution=(
                "Continuous entropy effective-candidate count and "
                "fraction; no calibrated binary diffuse threshold"
            ),
            candidates=(
                "Union of formal CS members, requested top candidates and PIP-threshold candidates"
            ),
            display=(
                "Baseline first; greedy one-to-one formal-CS aid is not "
                "evidence of unique signal identity"
            ),
        ),
        limitations=[
            "PIP across settings is not averaged into a posterior.",
            ("CS count and matching relations do not identify an independent causal signal count."),
            (
                "Genotype correlation alone does not establish that two "
                "effects are conditionally indistinguishable."
            ),
        ],
    )


def _hash_value(h, value):
    """Feed a canonical, type-tagged encoding of a nested value to a hash."""
    if isinstance(value, np.ndarray):
        h.update(b"array" + str(value.shape).encode() + value.dtype.str.encode())
        if value.dtype.kind == "O":
            _hash_value(h, value.tolist())
        else:
            h.update(np.ascontiguousarray(value).tobytes())
    elif dataclasses.is_dataclass(value):
        # Private fields (retained data, caches) are excluded; inputs are hashed separately.
        _hash_value(
            h,
            {
                f.name: getattr(value, f.name)
                for f in dataclasses.fields(value)
                if not f.name.startswith("_")
            },
        )
    elif isinstance(value, dict):
        h.update(b"dict")
        for key in sorted(value):
            _hash_value(h, key)
            _hash_value(h, value[key])
    elif isinstance(value, (list, tuple)):
        h.update(b"list")
        for v in value:
            _hash_value(h, v)
    elif isinstance(value, np.generic):
        _hash_value(h, value.item())
    elif isinstance(value, (str, int, float, bool, type(None))):
        h.update(json.dumps(value, ensure_ascii=False, allow_nan=True).encode() + b";")
    else:
        raise TypeError(f"unsupported sensitivity cache value type {type(value).__name__}")


def sensitivity_key(
    inputs: Mapping[str, Any],
    settings: Mapping[str, Any],
    *,
    initialization: str = "cold",
    software_version: str | None = None,
) -> str:
    """Cache key for a sensitivity fit.

    The SHA-256 digest covers the inputs (including variant order and allele
    metadata), the settings, the initialization and the software version.
    """
    if software_version is None:
        try:
            software_version = version("prusie")
        except PackageNotFoundError:
            software_version = "uninstalled"
    h = hashlib.sha256()
    _hash_value(
        h,
        dict(
            schema="prusie.sensitivity.v1",
            inputs=inputs,
            settings=settings,
            initialization=initialization,
            software_version=software_version,
        ),
    )
    return h.hexdigest()


def _sensitivity_inputs(data, *, explicit_ids=False):
    """Validate RSS inputs of a sensitivity plan before any fit is run."""
    from ._validation import _metadata, _scalar, _validated_matrix, _vector

    if not isinstance(data, dict):
        raise ValueError("sensitivity inputs must be a mapping")
    if explicit_ids and ("variant_ids" not in data or data["variant_ids"] is None):
        raise ValueError("alternative inputs require explicit variant_ids")
    if data.get("z") is not None:
        if data.get("bhat") is not None or data.get("shat") is not None:
            raise ValueError("Provide either z or bhat/shat, not both")
        values = _vector(data["z"], "z")
    else:
        if data.get("bhat") is None or data.get("shat") is None:
            raise ValueError("Provide either z or both bhat and shat")
        values = _vector(data["bhat"], "bhat")
        se = data["shat"]
        if np.ndim(se) == 0:
            _scalar(se, "shat", strict=True)
        elif np.any(_vector(se, "shat", len(values)) <= 0):
            raise ValueError("shat must be finite and strictly positive")
    if data.get("n") is not None:
        _scalar(data["n"], "n", lower=1.0, strict=True)
    _validated_matrix(data.get("R"), "R", len(values), correlation=True)
    ids, metadata = _metadata(data.get("variant_ids"), data.get("variant_metadata"), len(values))
    return ids.tolist(), metadata


def _sensitivity_prior(weights, ids, null_weight, *, policy, source_ids=None):
    """Resolve prior weights as in the fitting functions.

    Raw weights are augmented with the null weight and normalized once,
    without normalizing the variant weights first.
    """
    from ._validation import _scalar, _vector

    null_weight = _scalar(null_weight, "null_weight")
    if null_weight >= 1:
        raise ValueError("null_weight must be < 1")
    raw = (
        np.full(len(ids), 1.0 / len(ids))
        if weights is None
        else _vector(weights, "prior_weights", len(ids)).copy()
    )
    # Same finite-positive-sum condition as the fitting functions, including overflow.
    with np.errstate(over="ignore", invalid="ignore"):
        total = raw.sum()
    if np.any(raw < 0) or not np.isfinite(total) or total <= 0:
        raise ValueError("prior_weights must be nonnegative with a finite positive sum")
    augmented = np.r_[raw * (1 - null_weight), null_weight] if null_weight else raw.copy()
    normalized = augmented / math.fsum(augmented)
    source_index = {x: i for i, x in enumerate(source_ids or ids)}
    return raw, dict(
        policy=policy,
        variant_ids=list(ids),
        source_indices=[source_index.get(x) for x in ids],
        raw_weights=raw.tolist(),
        normalized_weights=normalized.tolist(),
        null_weight=null_weight,
        effective_null_probability=float(normalized[-1]) if null_weight else 0.0,
        normalization=(
            "augment [(1-null_weight)*raw_weights, null_weight], "
            "then normalize once; no preliminary biological "
            "normalization"
        ),
    )


def _alternative_sensitivity_inputs(alt, settings, baseline_ids, baseline_metadata):
    """Resolve an alternative input set, mapping prior weights and metadata by variant ID."""
    if not isinstance(alt, dict):
        raise ValueError("alternative inputs must be a mapping")
    data = dict(alt)
    options = dict(settings)
    explicit_prior = "prior_weights" in data
    supplied_prior = data.pop("prior_weights", None)
    # An input mapping may restate null_weight but not change it, which would be
    # a second factor in a single-factor plan.
    if "null_weight" in data and not np.array_equal(
        data.pop("null_weight"), settings.get("null_weight", 0.0)
    ):
        raise ValueError("alternative inputs must retain baseline null_weight")
    if "s_init" in data:
        raise ValueError("alternative inputs are cold-started; s_init is not supported")
    if set(data) & set(options):
        raise ValueError(
            "alternative inputs and baseline_settings must not duplicate argument keys"
        )
    ids, _ = _sensitivity_inputs(data, explicit_ids=True)
    source_index = {x: i for i, x in enumerate(baseline_ids)}
    new_ids = [x for x in ids if x not in source_index]
    raw = settings["prior_weights"]
    if explicit_prior:
        weights = supplied_prior
        policy = "explicit_uniform" if supplied_prior is None else "explicit_alternative_weights"
    elif new_ids:
        if not np.all(raw == raw[0]):
            raise ValueError(
                "new variant IDs require explicit alternative "
                "prior_weights under a nonuniform baseline prior"
            )
        weights = np.full(len(ids), raw[0])
        policy = "extend_equal_existing_weights"
    else:
        weights = raw[[source_index[x] for x in ids]]
        policy = (
            "reindex_existing_weights"
            if len(ids) == len(baseline_ids)
            else "condition_on_retained_variants_and_null"
        )
    options["prior_weights"], prior_resolution = _sensitivity_prior(
        weights, ids, settings.get("null_weight", 0.0), policy=policy, source_ids=baseline_ids
    )
    if "variant_metadata" in data:
        metadata_policy = "explicit_alternative_columns"
    elif baseline_metadata:
        if new_ids:
            raise ValueError(
                "new variant IDs require explicit alternative "
                "variant_metadata when baseline columns are present"
            )
        data["variant_metadata"] = {
            key: values[[source_index[x] for x in ids]].copy()
            for key, values in baseline_metadata.items()
        }
        metadata_policy = "reindex_existing_columns"
    else:
        metadata_policy = "unavailable_no_baseline_columns"
    return (
        data,
        options,
        prior_resolution,
        dict(
            variant_ids=ids,
            new_variant_ids=new_ids,
            source_indices=[source_index.get(x) for x in ids],
            metadata_policy=metadata_policy,
            summary_statistics_and_LD=(
                "explicit_alternative_input_order; never inferred or reused as a warm state"
            ),
        ),
    )


def run_sensitivity(
    inputs: Mapping[str, Any],
    plan: Sequence[Mapping[str, Any]],
    *,
    baseline: SusieResult | None = None,
    baseline_settings: Mapping[str, Any] | None = None,
    cache: MutableMapping[str, Any] | None = None,
    fit_function: Callable[..., SusieResult] | None = None,
    software_version: str | None = None,
) -> dict[str, Any]:
    """Fit a baseline and a list of single-factor alternatives, then compare them.

    Parameters
    ----------
    inputs : dict
        Keyword arguments of the fitting function describing the data: ``z``
        or ``bhat``/``shat``, ``R``, optionally ``n``, ``variant_ids``,
        ``variant_metadata``, ``prior_weights`` and ``null_weight``.
    plan : sequence of dict
        Items with a unique ``name`` and exactly one of ``changes`` (one of L,
        prior_weights, scaled_prior_variance or prior_variance, and
        estimate_prior_variance, or refine), ``inputs`` (a complete
        alternative set of summary statistics and LD, for example another
        window or filter), ``report`` (coverage and/or min_abs_corr, applied
        to the baseline posterior without refitting) or ``continuation``
        (max_iter, tol, track_fit and/or convergence).
    baseline : SusieResult, optional
        Existing baseline fit; its variant IDs, alleles and recorded settings
        are checked against ``inputs`` and ``baseline_settings``.
    baseline_settings : dict, optional
        Model settings shared by all fits.
    cache : dict, optional
        Caller-owned cache of fits keyed by :func:`sensitivity_key`.
    fit_function : callable, default susie_rss
        Fitting function.
    software_version : str, optional
        Version string included in cache keys; defaults to the installed one.

    Returns
    -------
    dict
        ``fits`` (by name), ``summaries`` (for report items), per-fit
        ``records`` (cache use, initialization and resolved priors), the
        ``plan``, ``baseline_settings`` and the ``comparison`` of all fits
        produced by :func:`compare_fits`.

    Notes
    -----
    The complete plan is validated before any fit is run. Fits start from
    the prior (cold start) unless the item is a continuation. Changing the
    initial prior variance while it is estimated is recorded as sensitivity
    to initialization; a fixed prior variance requires
    ``estimate_prior_variance=False``. For alternative inputs, prior weights
    and metadata are mapped by variant ID. A subset conditions the original
    raw weights on the retained variants and the null column. New variants
    require explicit weights unless the baseline prior is uniform, in which
    case they receive the same raw weight, and explicit metadata when the
    baseline has metadata columns. Convergence controls are passed to the
    fitting function only when they were requested, by ``baseline_settings`` or
    by the supplied baseline; otherwise its default rule applies, as in a direct
    call. Cached fit objects are shared, so they
    must not be modified while the cache is in use. When a baseline fit is
    supplied, equality of its z-scores and LD with ``inputs`` cannot be
    verified and is reported as not assessed.
    """
    from .diagnostics import ConvergenceConfig, resolve_convergence
    from .rss import susie_rss

    fit_function = susie_rss if fit_function is None else fit_function
    if not isinstance(inputs, dict) or not isinstance(plan, (list, tuple)):
        raise TypeError("inputs must be a mapping and plan must be an explicit sequence")
    inputs = dict(inputs)
    settings = copy.deepcopy(baseline_settings or {})
    if set(inputs) & set(settings):
        raise ValueError("inputs and baseline_settings must not duplicate argument keys")
    if set(settings) & {"z", "R", "n", "bhat", "shat", "variant_ids"}:
        raise ValueError(
            "variant data, IDs and sample size belong in inputs, not baseline_settings"
        )
    if "variant_metadata" in settings:
        inputs["variant_metadata"] = settings.pop("variant_metadata")
    for key in ("prior_weights", "null_weight"):
        if key in inputs:
            settings[key] = inputs.pop(key)
    if "s_init" in inputs:
        raise ValueError("baseline sensitivity is cold-started; use an explicit continuation item")
    baseline_ids, baseline_metadata = _sensitivity_inputs(inputs)
    # Fitting functions take a ConvergenceConfig; saved settings store a dict.
    override = settings.get("convergence")
    if isinstance(override, dict):
        override = ConvergenceConfig.from_dict(override)
    control, control_resolution = resolve_convergence(baseline, override)
    baseline_validation = None
    if baseline is not None:
        if "variant_ids" not in inputs or [str(x) for x in inputs["variant_ids"]] != _ids(baseline):
            raise ValueError("supplied baseline needs identical explicit variant_ids and order")
        metadata = inputs.get("variant_metadata", {})
        for key in ("effect_allele", "other_allele"):
            expected, given = baseline.variant_metadata.get(key), metadata.get(key)
            if (expected is None) != (given is None) or (
                expected is not None and not np.array_equal(expected, given)
            ):
                raise ValueError("supplied baseline allele metadata differs")
        input_n = baseline.params.get("input_n", baseline.params.get("n"))
        if input_n != inputs.get("n"):
            raise ValueError("supplied baseline sample size differs from inputs")
        expected_kind = baseline.params.get("input_kind")
        if expected_kind is not None and expected_kind != ("z" if "z" in inputs else "bhat_shat"):
            raise ValueError("supplied baseline input scale/kind differs")
        for key, given in settings.items():
            if (
                key in baseline.params
                and key
                not in ("prior_weights", "s_init", "track_fit", "retain_data", "convergence")
                and not np.array_equal(given, baseline.params[key])
            ):
                raise ValueError(f"baseline_settings {key} differs from supplied baseline")
        # Start from the settings recorded in the baseline, so that a
        # single-factor change does not reset L or the variance options.
        allowed_settings = (
            set(inspect.signature(susie_rss).parameters)
            - set(inputs)
            - {"s_init", "variant_metadata", "variant_ids", "z", "R", "n", "bhat", "shat"}
        )
        recorded = {
            k: copy.deepcopy(v) for k, v in baseline.params.items() if k in allowed_settings
        }
        if "input_var_y" in baseline.params:
            recorded["var_y"] = baseline.params["input_var_y"]
        if baseline.params.get("effect_scale") == "noncentrality":
            recorded["scaled_prior_variance"] = 0.2
        # Recover the raw prior of the baseline from its normalized weights; the
        # null weight was appended before normalization.
        prior = np.asarray(baseline.params["prior_weights"], dtype=float)
        null_weight = float(baseline.params.get("null_weight", 0.0))
        biological = _biological(baseline, prior)
        if baseline.null_index is None:
            recorded["prior_weights"] = biological
        elif prior[baseline.null_index] > 0:
            recorded["prior_weights"] = (
                biological * (null_weight / prior[baseline.null_index]) / (1 - null_weight)
            )
        else:
            raise ValueError("cannot reconstruct supplied baseline null prior; provide a valid fit")
        if "prior_weights" in settings:
            supplied, _ = _sensitivity_prior(
                settings["prior_weights"],
                baseline_ids,
                null_weight,
                policy="explicit_baseline_weights",
            )
            if baseline.null_index is not None:
                supplied = np.insert(supplied * (1 - null_weight), baseline.null_index, null_weight)
            supplied = supplied / math.fsum(supplied)
            if not np.allclose(supplied, prior, atol=1e-15, rtol=0):
                raise ValueError("baseline_settings prior_weights differs from supplied baseline")
        recorded.update(settings)
        settings = recorded
        baseline_validation = dict(
            ids_order="verified",
            allele_coding="verified_as_provided",
            explicit_model_settings="verified",
            raw_input_identity="not_assessed_no_original_RSS_input_hash",
            warning=("Matching IDs/metadata/settings alone does not prove equal z or LD values."),
        )
        baseline_validation["convergence_resolution"] = copy.deepcopy(control_resolution)
    # Forward controls only when they were requested. Otherwise each fit uses the
    # fitting function's own default rule, as a direct call would (for example
    # susieR's PIP rule with R_finite or R_mismatch).
    forward_convergence = override is not None or (
        baseline is not None
        and baseline.params.get("convergence_resolution", {}).get("source") != "default"
    )
    if forward_convergence:
        settings["convergence"] = control
    else:
        settings.pop("convergence", None)
    if "s_init" in settings:
        raise ValueError("baseline sensitivity is cold-started; use an explicit continuation item")
    if set(inputs) & set(settings):
        raise ValueError("inputs and baseline_settings must not duplicate argument keys")
    settings["prior_weights"], baseline_prior = _sensitivity_prior(
        settings.get("prior_weights"),
        baseline_ids,
        settings.get("null_weight", 0.0),
        policy="reconstructed_baseline_weights" if baseline is not None else "baseline_weights",
    )
    plan = copy.deepcopy(plan)
    if any(not isinstance(item, dict) for item in plan):
        raise ValueError("each plan item must be a mapping")
    names = [item.get("name") for item in plan]
    if any(not isinstance(name, str) or not name or name == "baseline" for name in names) or len(
        set(names)
    ) != len(names):
        raise ValueError("plan names must be distinct nonempty strings other than baseline")
    # The whole plan is validated before any fit is run.
    allowed = {
        "L",
        "prior_weights",
        "scaled_prior_variance",
        "prior_variance",
        "estimate_prior_variance",
        "refine",
    }
    resolved = {}
    for item in plan:
        categories = set(item) & {"changes", "inputs", "report", "continuation"}
        if len(categories) != 1 or set(item) - {
            "name",
            "changes",
            "inputs",
            "report",
            "continuation",
        }:
            raise ValueError("each plan item must change exactly one factor category")
        if "changes" in item:
            changes = item["changes"]
            keys = set(changes)
            variance_pair = keys in (
                {"scaled_prior_variance", "estimate_prior_variance"},
                {"prior_variance", "estimate_prior_variance"},
            )
            if not keys or keys - allowed or (len(keys) > 1 and not variance_pair):
                raise ValueError("changes must describe a single supported factor")
            variance_key = "prior_variance" if inputs.get("n") is None else "scaled_prior_variance"
            wrong_key = (
                "scaled_prior_variance" if variance_key == "prior_variance" else "prior_variance"
            )
            if wrong_key in keys:
                raise ValueError(f"this RSS sample-size mode uses {variance_key}, not {wrong_key}")
            options = dict(settings, **changes)
            options["prior_weights"], prior_resolution = _sensitivity_prior(
                options.get("prior_weights"),
                baseline_ids,
                options.get("null_weight", 0.0),
                policy="explicit_model_prior_change"
                if "prior_weights" in changes
                else "unchanged_baseline_weights",
            )
            resolved[item["name"]] = (inputs, options, prior_resolution, None)
        if "report" in item and (
            not item["report"] or set(item["report"]) - {"coverage", "min_abs_corr"}
        ):
            raise ValueError("report changes only coverage/min_abs_corr")
        if "continuation" in item and set(item["continuation"]) - {
            "max_iter",
            "tol",
            "track_fit",
            "convergence",
        }:
            raise ValueError("continuation changes only max_iter/tol/track_fit/convergence")
        if "continuation" in item and "convergence" in item["continuation"]:
            given = item["continuation"]["convergence"]
            if isinstance(given, dict):
                given = ConvergenceConfig.from_dict(given)
                item["continuation"]["convergence"] = given
            resolve_convergence(baseline, given)  # Raises before any fit is run.
        if "inputs" in item:
            resolved[item["name"]] = _alternative_sensitivity_inputs(
                item["inputs"], settings, baseline_ids, baseline_metadata
            )
    cache = {} if cache is None else cache
    results, records, summaries = {}, [], {}
    retain = any("continuation" in item for item in plan)
    if retain:
        settings["retain_data"] = True
        for _, options, _, _ in resolved.values():
            options["retain_data"] = True

    def execute(
        label, data, options, initialization="cold", prior_resolution=None, input_resolution=None
    ):
        key = sensitivity_key(
            data, options, initialization=initialization, software_version=software_version
        )
        hit = key in cache
        if not hit:
            cache[key] = fit_function(**copy.deepcopy(data), **copy.deepcopy(options))
        records.append(
            dict(
                name=label,
                cache_key=key,
                cache_hit=hit,
                initialization=initialization,
                fit_performed=not hit,
                analysis="fit_change",
                requested_convergence=control.options(),
                convergence_resolution=copy.deepcopy(control_resolution),
                prior_resolution=copy.deepcopy(prior_resolution),
                input_resolution=copy.deepcopy(input_resolution),
            )
        )
        return cache[key]

    results["baseline"] = (
        baseline
        if baseline is not None
        else execute("baseline", inputs, settings, prior_resolution=baseline_prior)
    )
    if baseline is not None:
        original_control, original_resolution = resolve_convergence(baseline)
        records.append(
            dict(
                name="baseline",
                cache_hit=False,
                initialization="provided",
                fit_performed=False,
                analysis="existing_fit",
                cache_key=None,
                requested_convergence=original_control.options(),
                convergence_resolution=original_resolution,
                prior_resolution=copy.deepcopy(baseline_prior),
            )
        )
    for item in plan:
        name = item["name"]
        if "report" in item:
            if "R" not in inputs:
                raise ValueError("posterior report changes need explicit R")
            summaries[name] = summarize(results["baseline"], inputs["R"], **item["report"])
            # A shallow copy of the baseline with the re-extracted credible sets;
            # the posterior arrays are shared and the baseline is not modified.
            view = copy.copy(results["baseline"])
            rows = summaries[name]["credible_sets"]
            actual = np.array([row["actual_coverage"] for row in rows])
            view.params = dict(view.params, **item["report"])
            view.sets = dict(
                cs=[
                    np.array(
                        [
                            j + int(view.null_index is not None and j >= view.null_index)
                            for j in row["member_indices"]
                        ],
                        dtype=np.int64,
                    )
                    for row in rows
                ],
                cs_index=np.array([row["component_id"] for row in rows], dtype=np.int64),
                coverage=actual,
                actual_coverage=actual,
                requested_coverage=view.params.get("coverage", 0.95),
                purity={
                    key: np.array([row[key] for row in rows])
                    for key in ("min_abs_corr", "mean_abs_corr", "median_abs_corr")
                },
            )
            view.cs_audit = copy.deepcopy(summaries[name]["cs_audit"])
            results[name] = view
            records.append(
                dict(
                    name=name,
                    analysis="report_filter_only",
                    fit_performed=False,
                    initialization="same_posterior",
                    cache_hit=False,
                    cache_key=None,
                    requested_convergence=resolve_convergence(view)[0].options(),
                    convergence_resolution=dict(source="unchanged_report_only"),
                )
            )
        elif "continuation" in item:
            from .refinement import continue_fit

            continuation = dict(item["continuation"])
            if "convergence" not in continuation and override is not None:
                continuation["convergence"] = control
            continuation_control, continuation_resolution = resolve_convergence(
                results["baseline"], continuation.get("convergence")
            )
            key = sensitivity_key(
                inputs,
                dict(settings, continuation=continuation, initial_state=results["baseline"]),
                initialization="continuation",
                software_version=software_version,
            )
            hit = key in cache
            if not hit:
                cache[key] = continue_fit(results["baseline"], **continuation)
            results[name] = cache[key]
            records.append(
                dict(
                    name=name,
                    analysis="explicit_continuation",
                    fit_performed=not hit,
                    initialization="baseline_posterior",
                    cache_hit=hit,
                    cache_key=key,
                    requested_convergence=continuation_control.options(),
                    convergence_resolution=continuation_resolution,
                )
            )
        else:
            data, options, prior_resolution, input_resolution = resolved[name]
            results[name] = execute(
                name,
                data,
                options,
                prior_resolution=prior_resolution,
                input_resolution=input_resolution,
            )
            changes = item.get("changes", {})
            if {"scaled_prior_variance", "prior_variance", "estimate_prior_variance"} & set(
                changes
            ):
                records[-1]["analysis"] = (
                    "computational_initialization_sensitivity"
                    if options.get("estimate_prior_variance", True)
                    else "fixed_effect_variance_prior"
                )
            elif "inputs" in item:
                records[-1]["analysis"] = "explicit_LD_or_window_or_filter_change"
    saved_plan = copy.deepcopy(plan)
    for item in saved_plan:
        if isinstance(item.get("continuation", {}).get("convergence"), ConvergenceConfig):
            item["continuation"]["convergence"] = item["continuation"]["convergence"].options()
    return dict(
        schema_version="prusie.sensitivity.v2",
        fits=results,
        summaries=summaries,
        records=records,
        plan=saved_plan,
        baseline_settings=dict(
            settings, **({"convergence": control.options()} if forward_convergence else {})
        ),
        convergence_resolution=copy.deepcopy(control_resolution),
        baseline_validation=baseline_validation,
        comparison=compare_fits(results, R=inputs.get("R"), ld_reference="baseline"),
        boundary_note=(
            "Larger windows require explicitly supplied summary "
            "statistics and LD; no variants are invented."
        ),
    )
