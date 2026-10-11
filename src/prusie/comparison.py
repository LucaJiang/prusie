"""Explicit sensitivity plans and posterior signal comparisons.

Comparison never fits a model. Only ``run_sensitivity`` executes a user-supplied
list of baseline plus one-factor configurations; it never expands a grid.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
import itertools
import inspect
import math
import numpy as np
from .analysis import _biological, _correlation, _ids, summarize


def _directions(a, b, shared):
    """Align explicit alleles, retaining an unassessed state when absent."""
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
        if any(x in ("", "NONE", "NAN") for x in pa+pb):
            directions[x] = None
        elif pa == pb:
            directions[x] = 1
        elif pa == pb[::-1]:
            directions[x] = -1
        else:
            raise ValueError(f"incompatible allele coding for variant {x}; harmonize explicitly")
    return directions, "explicit_alleles_aligned"


def _signals(fit):
    """Retain active components and formal CS support without conflating them."""
    ids, alpha = _ids(fit), _biological(fit, fit.alpha)
    formal = {int(i): {int(x)-int(fit.null_index is not None and x > fit.null_index)
                      for x in members if x != fit.null_index}
              for i, members in zip(fit.sets["cs_index"], fit.sets["cs"])}
    records = []
    for i, a in enumerate(alpha):
        active = bool(fit.V[i] > fit.params.get("prior_tol", 1e-9))
        retained = i in formal
        # The inherited CS and PIP activity thresholds can differ. Preserve a
        # formal CS even at that boundary; label both facts separately.
        if not active and not retained:
            continue
        top = min(range(len(ids)), key=lambda j: (-float(a[j]), ids[j])) if ids else None
        positive = a[a > 0]
        mass = float(positive.sum())
        effective = None
        if mass > 0:
            weights = positive/mass
            effective = float(np.exp(-np.sum(weights*np.log(weights))))
        resolution = dict(effective_candidates=effective,
                          effective_fraction=effective/len(ids) if effective is not None and ids else None,
                          biological_mass=mass, cs_size=len(formal[i]) if retained else None,
                          definition="entropy effective candidates normalized over biological variants")
        record = dict(component_id=i, alpha={x: float(v) for x, v in zip(ids, a)},
                      cs={ids[j] for j in formal.get(i, set())},
                      top=ids[top] if top is not None else None,
                      V=float(fit.V[i]), active=active, cs_retained=retained,
                      cs_size=resolution["cs_size"], purity=None,
                      effective_candidates=effective, resolution=resolution,
                      localization_status="formal_cs_retained" if retained else "unresolved_no_formal_cs",
                      support_sources=["formal_cs"] if retained else ["active_component_only"])
        if retained:
            k = list(fit.sets["cs_index"]).index(i)
            values = fit.sets.get("purity", {}).get("min_abs_corr", [])
            record["purity"] = float(values[k]) if k < len(values) else None
        records.append(record)
    return records


def _classify_signal_edges(edges):
    """Classify only the CS-supported subgraph, independently of diffuse edges."""
    formal = [e for e in edges if e.get("cs_retained_a") and e.get("cs_retained_b")]
    for edge in edges:
        if not (edge.get("cs_retained_a") and edge.get("cs_retained_b")):
            edge["status"] = "unresolved"
            continue
        da = sum(e["component_a"] == edge["component_a"] for e in formal)
        db = sum(e["component_b"] == edge["component_b"] for e in formal)
        edge["status"] = "ambiguous" if da > 1 and db > 1 else "split" if da > 1 else "merge" if db > 1 else "matched"
        edge["degree_a"], edge["degree_b"] = da, db
    return edges


def _unmatched(na, nb, record, side, shared, *, layer):
    other = "b" if side == "a" else "a"
    return dict(setting_a=na, setting_b=nb,
        **{f"component_{side}": record["component_id"], f"component_{other}": None,
           f"cs_retained_{side}": record["cs_retained"], f"cs_retained_{other}": None,
           f"resolution_{side}": copy.deepcopy(record["resolution"]), f"resolution_{other}": None,
           f"cs_size_{side}": record["cs_size"], f"cs_size_{other}": None,
           f"representative_{side}": record["top"], f"representative_{other}": None},
        status="unmatched" if layer == "signals" else "unmatched_component", side=side,
        matching_layer=layer, support_sources=list(record["support_sources"]),
        unobserved_mass=sum(v for x, v in record["alpha"].items() if x not in shared))


def _baseline_first(rows, baseline):
    """Orient baseline comparisons consistently even for legacy saved results."""
    out = []
    for original in rows:
        if baseline not in (original.get("setting_a"), original.get("setting_b")):
            continue
        row = copy.deepcopy(original)
        if row.get("setting_b") == baseline:
            row = {(key[:-2]+"_b" if key.endswith("_a") else key[:-2]+"_a" if key.endswith("_b") else key): value
                   for key, value in row.items()}
            if row.get("status") in ("split", "merge"):
                row["status"] = {"split": "merge", "merge": "split"}[row["status"]]
            if row.get("side") in ("a", "b"):
                row["side"] = {"a": "b", "b": "a"}[row["side"]]
        out.append(row)
    return out


def comparison_relations(comparison, *, layer="signals", baseline_only=True):
    """Read current or saved v1 relations without treating diffuse edges as signals.

    Legacy v1 records have CS sizes for connected components but not for every
    unmatched component. Unavailable legacy support is retained in the complete
    component view, and is never guessed to be a localized signal.
    """
    if layer not in ("signals", "components"):
        raise ValueError("layer must be signals or components")
    if "component_relations" in comparison:
        rows = copy.deepcopy(comparison.get("signal_relations" if layer == "signals" else "component_relations", []))
    else:
        rows = copy.deepcopy(comparison.get("signal_relations", []))
        for row in rows:
            for side in ("a", "b"):
                row[f"cs_retained_{side}"] = bool(row.get(f"cs_size_{side}"))
            row["matching_layer"] = "components"
            row["legacy_support_status"] = "inferred_from_recorded_cs_size; unmatched_support_unavailable"
            sources = []
            if row.get("posterior_overlap", -1) >= comparison.get("thresholds", {}).get("posterior_overlap", .5):
                sources.append("posterior_overlap")
            if row.get("cs_overlap"):
                sources.append("formal_cs_overlap")
            if row.get("signed_r") is not None and abs(row["signed_r"]) >= comparison.get("thresholds", {}).get("abs_r", .9):
                sources.append("representative_ld")
            row["support_sources"] = sources
        if layer == "signals":
            recorded = rows
            rows = [row for row in recorded if (row["cs_retained_a"] and row["cs_retained_b"])
                    or ((row["cs_retained_a"] or row["cs_retained_b"])
                        and "posterior_overlap" in row["support_sources"])]
            for pair in {(r["setting_a"], r["setting_b"]) for r in rows}:
                _classify_signal_edges([r for r in rows if (r["setting_a"], r["setting_b"]) == pair])
            for row in rows:
                row["matching_layer"] = "signals"
            # A legacy LD-only edge to an unretained component does establish
            # the formal endpoint's existence, even though it does not establish
            # a localized match. Keep that endpoint as explicitly unmatched.
            for row in recorded:
                for side in ("a", "b"):
                    if not row[f"cs_retained_{side}"]:
                        continue
                    if any(e["setting_a"] == row["setting_a"] and e["setting_b"] == row["setting_b"]
                           and e.get(f"component_{side}") == row[f"component_{side}"] for e in rows):
                        continue
                    other = "b" if side == "a" else "a"
                    rows.append(dict(setting_a=row["setting_a"], setting_b=row["setting_b"],
                        **{f"component_{side}": row[f"component_{side}"], f"component_{other}": None,
                           f"cs_retained_{side}": True, f"cs_retained_{other}": None,
                           f"cs_size_{side}": row[f"cs_size_{side}"], f"cs_size_{other}": None,
                           f"representative_{side}": row.get(f"representative_{side}"),
                           f"representative_{other}": None},
                        status="unmatched", side=side, matching_layer="signals",
                        support_sources=["formal_cs"],
                        unobserved_mass=row.get(f"unobserved_mass_{side}"),
                        legacy_support_status="formal_CS_identity_recovered_from_recorded_cs_size"))
        else:
            for row in rows:
                if not (row["cs_retained_a"] and row["cs_retained_b"]):
                    row["status"] = "component_overlap" if "posterior_overlap" in row["support_sources"] else "component_relation"
    return _baseline_first(rows, comparison.get("baseline")) if baseline_only else rows


def compare_fits(fits, *, baseline=None, R=None, ld_reference=None,
                 pip_threshold=.1, overlap_threshold=.5, ld_threshold=.9, mode="union",
                 top_candidates=3):
    """Compare existing fits, separating formal-CS signals from all components.

    ``signal_relations`` contains the formal-CS graph and posterior-overlap
    relations to unresolved components. Its degrees use only CS-to-CS edges.
    ``component_relations`` preserves every active-component relation, including
    diffuse overlap. Relations describe posterior evidence, not confirmed causal
    split/merge events. Representative LD alone cannot promote a component with
    no retained CS into the user signal view. Missing variants remain None.

    Candidate variants include all retained CS members, up to ``top_candidates``
    posterior-ranked variants per retained component (top regional PIPs when no
    CS was retained), and every variant crossing ``pip_threshold`` in any fit.
    """
    if not fits or any(not isinstance(name, str) or not name for name in fits):
        raise ValueError("fits must be a nonempty mapping with nonempty string names")
    original_names = list(fits)
    baseline = original_names[0] if baseline is None else baseline
    if baseline not in fits:
        raise ValueError("baseline must name a supplied fit")
    names = [baseline]+[name for name in original_names if name != baseline]
    if mode not in ("union", "common"):
        raise ValueError("mode must be union or common")
    if not 0 <= pip_threshold <= 1 or not 0 <= overlap_threshold <= 1 or not 0 <= ld_threshold <= 1:
        raise ValueError("comparison thresholds must lie in [0, 1]")
    if isinstance(top_candidates, bool) or not isinstance(top_candidates, (int, np.integer)) or top_candidates < 0:
        raise ValueError("top_candidates must be a nonnegative integer")
    ids = {name: _ids(fit) for name, fit in fits.items()}
    maps = {name: dict(zip(ids[name], map(float, fit.pip))) for name, fit in fits.items()}
    signals = {name: _signals(fit) for name, fit in fits.items()}
    universe = sorted(set.union(*(set(v) for v in ids.values())) if mode == "union" else set.intersection(*(set(v) for v in ids.values())))
    ranks = {name: {x: i+1 for i, x in enumerate(sorted(ids[name], key=lambda x: (-maps[name][x], x)))} for name in names}
    candidate_sources = {x: {} for x in universe}
    for name in names:
        formal = [s for s in signals[name] if s["cs_retained"]]
        cs_members = set().union(*(s["cs"] for s in formal))
        top_members = {x for s in formal
                       for x in sorted(s["alpha"], key=lambda y: (-s["alpha"][y], y))[:top_candidates]
                       if s["alpha"][x] > 0}
        for x in universe:
            sources = []
            if x in maps[name] and maps[name][x] >= pip_threshold:
                sources.append("pip_threshold")
            if x in cs_members:
                sources.append("formal_cs_member")
            if x in top_members:
                sources.append("component_top_candidate")
            if not formal and ranks[name].get(x, math.inf) <= top_candidates and maps[name].get(x, 0) > 0:
                sources.append("regional_top_pip_without_cs")
            if sources:
                candidate_sources[x][name] = sources
    variants = []
    for x in universe:
        pips = {name: maps[name].get(x) for name in names}
        observed = [v for v in pips.values() if v is not None]
        differences = [abs(v-pips[baseline]) for v in observed] if pips[baseline] is not None else []
        threshold = [v >= pip_threshold for v in observed]
        variants.append(dict(variant_id=x, pips=pips, pip_min=min(observed), pip_max=max(observed),
            pip_range=max(observed)-min(observed), max_abs_delta_from_baseline=max(differences) if differences else None,
            ranks={name: ranks[name].get(x) for name in names},
            threshold_crossings=bool(any(threshold) and not all(threshold)),
            candidate=bool(candidate_sources[x]), candidate_sources=candidate_sources[x],
            missing_settings=[name for name in names if pips[name] is None]))
    corr, ld_ids = None, {}
    if R is not None:
        ld_reference = baseline if ld_reference is None else ld_reference
        if ld_reference not in fits:
            raise ValueError("ld_reference must name a fit whose ID order matches R")
        corr = _correlation(R, len(ids[ld_reference]))
        ld_ids = {x: i for i, x in enumerate(ids[ld_reference])}
    relations, component_relations, alignments, pairwise, display = [], [], [], [], []
    for na, nb in itertools.combinations(names, 2):
        shared = set(ids[na]) & set(ids[nb])
        ordered_shared = sorted(shared)
        directions, direction_status = _directions(fits[na], fits[nb], ordered_shared)
        alignments.append(dict(setting_a=na, setting_b=nb, shared_ids=ordered_shared, directions=directions,
            status=direction_status, omitted_a=sorted(set(ids[na])-shared), omitted_b=sorted(set(ids[nb])-shared)))
        pa, pb = [maps[na][x] for x in ordered_shared], [maps[nb][x] for x in ordered_shared]
        correlation = float(np.corrcoef(pa, pb)[0, 1]) if len(shared) > 1 and np.std(pa) > 0 and np.std(pb) > 0 else None
        pairwise.append(dict(setting_a=na, setting_b=nb, shared_variant_count=len(shared),
            full_region_pip_correlation=correlation,
            max_abs_delta_pip=max((abs(a-b) for a, b in zip(pa, pb)), default=None),
            candidate_max_abs_delta_pip=max((abs(maps[na][x]-maps[nb][x]) for x in ordered_shared
                if candidate_sources[x]), default=None)))
        edges = []
        for a, b in itertools.product(signals[na], signals[nb]):
            overlap = math.fsum(min(a["alpha"][x], b["alpha"][x]) for x in ordered_shared)
            common_cs = sorted(a["cs"] & b["cs"])
            r = float(corr[ld_ids[a["top"]], ld_ids[b["top"]]]) if corr is not None and a["top"] in ld_ids and b["top"] in ld_ids else None
            support = (["posterior_overlap"] if overlap >= overlap_threshold else [])
            support += ["formal_cs_overlap"] if common_cs else []
            support += ["representative_ld"] if r is not None and abs(r) >= ld_threshold else []
            if not support:
                continue
            edge = dict(setting_a=na, setting_b=nb, component_a=a["component_id"], component_b=b["component_id"],
                posterior_overlap=overlap, cs_overlap=common_cs, representative_a=a["top"], representative_b=b["top"],
                signed_r=r, signed_r_direction="as encoded in the explicitly named LD-reference fit",
                r_squared=r*r if r is not None else None,
                candidate_exchange_high_ld=bool(a["top"] != b["top"] and r is not None and abs(r) >= ld_threshold),
                unobserved_mass_a=math.fsum(v for x, v in a["alpha"].items() if x not in shared),
                unobserved_mass_b=math.fsum(v for x, v in b["alpha"].items() if x not in shared),
                cs_size_a=a["cs_size"], cs_size_b=b["cs_size"], min_abs_corr_a=a["purity"], min_abs_corr_b=b["purity"],
                effective_candidates_a=a["effective_candidates"], effective_candidates_b=b["effective_candidates"],
                V_a=a["V"], V_b=b["V"], cs_retained_a=a["cs_retained"], cs_retained_b=b["cs_retained"],
                resolution_a=copy.deepcopy(a["resolution"]), resolution_b=copy.deepcopy(b["resolution"]),
                support_sources=support, matching_layer="components")
            edges.append(edge)
        main = [copy.deepcopy(e) for e in edges if (e["cs_retained_a"] and e["cs_retained_b"])
                or ((e["cs_retained_a"] or e["cs_retained_b"]) and "posterior_overlap" in e["support_sources"])]
        _classify_signal_edges(main)
        by_component = {(e["component_a"], e["component_b"]): e for e in main}
        for edge in main:
            edge["matching_layer"] = "signals"
        for edge in edges:
            projected = by_component.get((edge["component_a"], edge["component_b"]))
            edge["status"] = projected["status"] if projected else "component_overlap" if "posterior_overlap" in edge["support_sources"] else "ld_proximity"
        for side, records in (("a", signals[na]), ("b", signals[nb])):
            for record in records:
                if not any(e[f"component_{side}"] == record["component_id"] for e in edges):
                    component_relations.append(_unmatched(na, nb, record, side, shared, layer="components"))
                if record["cs_retained"] and not any(e[f"component_{side}"] == record["component_id"] for e in main):
                    main.append(_unmatched(na, nb, record, side, shared, layer="signals"))
        component_relations.extend(edges)
        relations.extend(main)
        used_a, used_b = set(), set()
        formal_edges = [e for e in main if e.get("cs_retained_a") and e.get("cs_retained_b")]
        for edge in sorted(formal_edges, key=lambda e: (-e["posterior_overlap"], -len(e["cs_overlap"]), e["representative_a"], e["representative_b"], e["component_a"], e["component_b"])):
            if edge["component_a"] not in used_a and edge["component_b"] not in used_b:
                display.append(copy.deepcopy(edge))
                used_a.add(edge["component_a"]); used_b.add(edge["component_b"])
    primary = _baseline_first(relations, baseline)
    return dict(schema_version="prusie.comparison.v2", settings=names, baseline=baseline, mode=mode,
        fit_settings={name: copy.deepcopy(fits[name].params) for name in names}, variants=variants,
        components={name: [dict(s, cs=sorted(s["cs"])) for s in signals[name]] for name in names},
        signal_relations=relations, baseline_signal_relations=primary,
        component_relations=component_relations, display_matches=display,
        baseline_display_matches=_baseline_first(display, baseline), alignments=alignments, pairwise=pairwise,
        baseline_pairwise=_baseline_first(pairwise, baseline),
        layer_summary=dict(settings={name: dict(retained_cs=sum(s["cs_retained"] for s in signals[name]),
                active_components=sum(s["active"] for s in signals[name]),
                unretained_active_components=sum(s["active"] and not s["cs_retained"] for s in signals[name])) for name in names},
            baseline_relation_counts={status: sum(e["status"] == status for e in primary)
                                      for status in ("matched", "ambiguous", "split", "merge", "unresolved", "unmatched")}),
        ld_reference=ld_reference if corr is not None else None,
        thresholds=dict(pip=pip_threshold, posterior_overlap=overlap_threshold, abs_r=ld_threshold,
                        top_candidates=int(top_candidates)),
        rules=dict(missing="None, never zero", posterior_overlap="unrenormalized shared-support sum(min(alpha_a,alpha_b))",
            matching="CS-to-CS: overlap OR CS overlap OR representative LD; CS-to-unretained: posterior overlap only, unresolved",
            component_layer="All active components plus formal CS boundary cases; diffuse overlap is descriptive, not a confirmed split/merge",
            resolution="Continuous entropy effective-candidate count and fraction; no calibrated binary diffuse threshold",
            candidates="Union of formal CS members, requested top candidates and PIP-threshold candidates",
            display="Baseline first; greedy one-to-one formal-CS aid is not evidence of unique signal identity"),
        limitations=["PIP across settings is not averaged into a posterior.",
                     "CS count and matching relations do not identify an independent causal signal count.",
                     "Genotype correlation alone does not establish that two effects are conditionally indistinguishable."])


def _hash_value(h, value):
    if isinstance(value, np.ndarray):
        h.update(b"array"+str(value.shape).encode()+value.dtype.str.encode())
        if value.dtype.kind == "O":
            _hash_value(h, value.tolist())
        else:
            h.update(np.ascontiguousarray(value).tobytes())
    elif dataclasses.is_dataclass(value):
        # No private retained data/cache fields; actual inputs are independently hashed.
        _hash_value(h, {f.name:getattr(value,f.name) for f in dataclasses.fields(value) if not f.name.startswith("_")})
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
    elif isinstance(value, (str,int,float,bool,type(None))):
        h.update(json.dumps(value, ensure_ascii=False, allow_nan=True).encode()+b";")
    else:
        raise TypeError(f"unsupported sensitivity cache value type {type(value).__name__}")


def sensitivity_key(inputs, settings, *, initialization="cold", software_version=None):
    """Content key including order, allele metadata, settings and initialization."""
    if software_version is None:
        try:
            software_version = version("prusie")
        except PackageNotFoundError:
            software_version = "uninstalled"
    h = hashlib.sha256()
    _hash_value(h, dict(schema="prusie.sensitivity.v1", inputs=inputs, settings=settings,
                        initialization=initialization, software_version=software_version))
    return h.hexdigest()


def run_sensitivity(inputs, plan, *, baseline=None, baseline_settings=None, cache=None,
                    fit_function=None, software_version=None):
    """Execute a finite, explicit baseline plus one-factor plan.

    Each plan item has ``name`` and one of: ``changes`` (L, prior_weights,
    scaled_prior_variance/prior_variance, estimate_prior_variance, refine), ``inputs`` (a
    complete explicit alternative LD/window/filter input mapping), ``report``
    (coverage/purity; no fit), or ``continuation`` (max_iter/tol only). Fixed
    effect variance requires estimate_prior_variance=False in the same change
    or baseline. With estimation enabled, variance changes are labelled
    computational initialization sensitivity. Cold starts are the default.
    Caching is explicit caller-owned memory and content-addressed; cached fit
    objects remain mutable and must not be edited while the cache is reused.
    A supplied baseline is checked for IDs, allele coding and model settings;
    raw-input identity is reported as unassessed without an original RSS hash.
    """
    from .api import susie_rss
    from .diagnostics import ConvergenceConfig, resolve_convergence
    fit_function = susie_rss if fit_function is None else fit_function
    if not isinstance(inputs, dict) or not isinstance(plan, (list, tuple)):
        raise TypeError("inputs must be a mapping and plan must be an explicit sequence")
    settings = copy.deepcopy(baseline_settings or {})
    # Public fits accept a config object; saved sensitivity settings use a
    # stable dictionary. Restore that dictionary only at this API boundary.
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
            if (expected is None) != (given is None) or (expected is not None and not np.array_equal(expected, given)):
                raise ValueError("supplied baseline allele metadata differs")
        input_n = baseline.params.get("input_n", baseline.params.get("n"))
        if input_n != inputs.get("n"):
            raise ValueError("supplied baseline sample size differs from inputs")
        expected_kind = baseline.params.get("input_kind")
        if expected_kind is not None and expected_kind != ("z" if "z" in inputs else "bhat_shat"):
            raise ValueError("supplied baseline input scale/kind differs")
        for key, given in settings.items():
            if key in baseline.params and key not in ("prior_weights", "s_init", "track_fit", "retain_data", "convergence"):
                if not np.array_equal(given, baseline.params[key]):
                    raise ValueError(f"baseline_settings {key} differs from supplied baseline")
        # Reuse recorded model settings, so a one-factor change does not silently
        # reset L or variance strategy to public defaults.
        allowed_settings = set(inspect.signature(susie_rss).parameters)-set(inputs)-{"s_init","variant_metadata","variant_ids","z","R","n","bhat","shat"}
        recorded = {k:copy.deepcopy(v) for k,v in baseline.params.items() if k in allowed_settings}
        if "input_var_y" in baseline.params:
            recorded["var_y"] = baseline.params["input_var_y"]
        if baseline.params.get("effect_scale") == "noncentrality":
            recorded["scaled_prior_variance"] = .2
        # Reconstruct the original model's normalized prior, including the
        # historical null augmentation-before-normalization convention.
        prior = np.asarray(baseline.params["prior_weights"], dtype=float)
        null_weight = float(baseline.params.get("null_weight", 0.))
        biological = _biological(baseline, prior)
        if baseline.null_index is None:
            recorded["prior_weights"] = biological
        elif prior[baseline.null_index] > 0:
            recorded["prior_weights"] = biological*(null_weight/prior[baseline.null_index])/(1-null_weight)
        else:
            raise ValueError("cannot reconstruct supplied baseline null prior; provide a valid fit")
        if "prior_weights" in settings:
            supplied = np.asarray(settings["prior_weights"],dtype=float)
            if supplied.shape != biological.shape:
                raise ValueError("baseline_settings prior_weights has wrong shape")
            if baseline.null_index is not None:
                supplied = np.insert(supplied*(1-null_weight),baseline.null_index,null_weight)
            supplied = supplied/math.fsum(supplied)
            if not np.allclose(supplied,prior,atol=1e-15,rtol=0):
                raise ValueError("baseline_settings prior_weights differs from supplied baseline")
        recorded.update(settings)
        settings = recorded
        baseline_validation = dict(ids_order="verified", allele_coding="verified_as_provided",
            explicit_model_settings="verified", raw_input_identity="not_assessed_no_original_RSS_input_hash",
            warning="Matching IDs/metadata/settings alone does not prove equal z or LD values.")
        baseline_validation["convergence_resolution"] = copy.deepcopy(control_resolution)
    settings["convergence"] = control
    if "s_init" in settings:
        raise ValueError("baseline sensitivity is cold-started; use an explicit continuation item")
    if set(inputs) & set(settings):
        raise ValueError("inputs and baseline_settings must not duplicate argument keys")
    plan = copy.deepcopy(plan)
    names = [item.get("name") for item in plan]
    if any(not isinstance(name,str) or not name or name == "baseline" for name in names) or len(set(names)) != len(names):
        raise ValueError("plan names must be distinct nonempty strings other than baseline")
    # Validate the entire plan before doing any work.
    allowed = {"L", "prior_weights", "scaled_prior_variance", "prior_variance", "estimate_prior_variance", "refine"}
    for item in plan:
        categories = set(item) & {"changes", "inputs", "report", "continuation"}
        if len(categories) != 1 or set(item)-{"name", "changes", "inputs", "report", "continuation"}:
            raise ValueError("each plan item must change exactly one factor category")
        if "changes" in item:
            changes = item["changes"]
            keys = set(changes)
            variance_pair = keys in ({"scaled_prior_variance", "estimate_prior_variance"},
                                     {"prior_variance", "estimate_prior_variance"})
            if not keys or keys-allowed or (len(keys) > 1 and not variance_pair):
                raise ValueError("changes must describe a single supported factor")
            variance_key = "prior_variance" if inputs.get("n") is None else "scaled_prior_variance"
            wrong_key = "scaled_prior_variance" if variance_key == "prior_variance" else "prior_variance"
            if wrong_key in keys:
                raise ValueError(f"this RSS sample-size mode uses {variance_key}, not {wrong_key}")
        if "report" in item and (not item["report"] or set(item["report"])-{"coverage", "min_abs_corr"}):
            raise ValueError("report changes only coverage/min_abs_corr")
        if "continuation" in item and set(item["continuation"])-{"max_iter", "tol", "track_fit", "convergence"}:
            raise ValueError("continuation changes only max_iter/tol/track_fit/convergence")
        if "continuation" in item and "convergence" in item["continuation"]:
            given = item["continuation"]["convergence"]
            if isinstance(given, dict):
                given = ConvergenceConfig.from_dict(given)
                item["continuation"]["convergence"] = given
            resolve_convergence(baseline, given)  # Validate before any fit runs.
        if "inputs" in item:
            alt = item["inputs"]
            if not isinstance(alt,dict) or "variant_ids" not in alt or "R" not in alt or not ("z" in alt or {"bhat","shat"} <= set(alt)):
                raise ValueError("alternative inputs require explicit variant_ids, LD and summary statistics")
            if set(alt) & set(settings):
                raise ValueError("alternative inputs and baseline_settings must not duplicate argument keys")
    cache = {} if cache is None else cache
    results, records, summaries = {}, [], {}
    retain = any("continuation" in item for item in plan)
    if retain:
        settings["retain_data"] = True
    def execute(label, data, options, initialization="cold"):
        key = sensitivity_key(data, options, initialization=initialization, software_version=software_version)
        hit = key in cache
        if not hit:
            cache[key] = fit_function(**copy.deepcopy(data), **copy.deepcopy(options))
        records.append(dict(name=label, cache_key=key, cache_hit=hit, initialization=initialization,
                            fit_performed=not hit, analysis="fit_change",
                            requested_convergence=options["convergence"].options(),
                            convergence_resolution=copy.deepcopy(control_resolution)))
        return cache[key]
    results["baseline"] = baseline if baseline is not None else execute("baseline", inputs, settings)
    if baseline is not None:
        original_control, original_resolution = resolve_convergence(baseline)
        records.append(dict(name="baseline", cache_hit=False, initialization="provided", fit_performed=False,
                            analysis="existing_fit", cache_key=None,
                            requested_convergence=original_control.options(),
                            convergence_resolution=original_resolution))
    for item in plan:
        name = item["name"]
        if "report" in item:
            if "R" not in inputs:
                raise ValueError("posterior report changes need explicit R")
            summaries[name] = summarize(results["baseline"], inputs["R"], **item["report"])
            # A new result view has identical posterior arrays and updated CS
            # display selection. The supplied fit/its sets remain unchanged,
            # and comparison sees the actual changed CS rather than stale sets.
            view = copy.copy(results["baseline"])
            rows = summaries[name]["credible_sets"]
            actual = np.array([row["actual_coverage"] for row in rows])
            view.params = dict(view.params, **item["report"])
            view.sets = dict(cs=[np.array([j+int(view.null_index is not None and j >= view.null_index)
                                            for j in row["member_indices"]], dtype=np.int64) for row in rows],
                cs_index=np.array([row["component_id"] for row in rows], dtype=np.int64),
                coverage=actual, actual_coverage=actual,
                requested_coverage=view.params.get("coverage", .95),
                purity={key:np.array([row[key] for row in rows]) for key in
                        ("min_abs_corr", "mean_abs_corr", "median_abs_corr")})
            view.cs_audit = copy.deepcopy(summaries[name]["cs_audit"])
            results[name] = view
            records.append(dict(name=name, analysis="report_filter_only", fit_performed=False,
                                initialization="same_posterior", cache_hit=False, cache_key=None,
                                requested_convergence=resolve_convergence(view)[0].options(),
                                convergence_resolution=dict(source="unchanged_report_only")))
        elif "continuation" in item:
            from .refinement import continue_fit
            continuation = dict(item["continuation"])
            if "convergence" not in continuation and override is not None:
                continuation["convergence"] = control
            continuation_control, continuation_resolution = resolve_convergence(
                results["baseline"], continuation.get("convergence"))
            key = sensitivity_key(inputs, dict(settings, continuation=continuation,
                                   initial_state=results["baseline"]), initialization="continuation",
                                   software_version=software_version)
            hit = key in cache
            if not hit:
                cache[key] = continue_fit(results["baseline"], **continuation)
            results[name] = cache[key]
            records.append(dict(name=name, analysis="explicit_continuation", fit_performed=not hit,
                                initialization="baseline_posterior", cache_hit=hit, cache_key=key,
                                requested_convergence=continuation_control.options(),
                                convergence_resolution=continuation_resolution))
        else:
            options = dict(settings, **item.get("changes", {}))
            data = item.get("inputs", inputs)
            results[name] = execute(name, data, options)
            changes = item.get("changes", {})
            if {"scaled_prior_variance", "prior_variance", "estimate_prior_variance"} & set(changes):
                records[-1]["analysis"] = "computational_initialization_sensitivity" if options.get("estimate_prior_variance", True) else "fixed_effect_variance_prior"
            elif "inputs" in item:
                records[-1]["analysis"] = "explicit_LD_or_window_or_filter_change"
    saved_plan = copy.deepcopy(plan)
    for item in saved_plan:
        if isinstance(item.get("continuation", {}).get("convergence"), ConvergenceConfig):
            item["continuation"]["convergence"] = item["continuation"]["convergence"].options()
    return dict(schema_version="prusie.sensitivity.v2", fits=results, summaries=summaries,
                records=records, plan=saved_plan,
                baseline_settings=dict(settings, convergence=control.options()),
                convergence_resolution=copy.deepcopy(control_resolution),
                baseline_validation=baseline_validation,
                comparison=compare_fits(results, R=inputs.get("R"), ld_reference="baseline"),
                boundary_note="Larger windows require explicitly supplied summary statistics and LD; no variants are invented.")
