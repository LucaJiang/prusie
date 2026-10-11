"""Offline evidence reports and lossless, pickle-free persistence.

JSON stores structure and type tags; NPZ stores exact NumPy bytes. CSV is a
human-readable export, not the lossless interchange format. Nothing here refits
a model, loads a remote resource, or invokes an AI provider.
"""
from __future__ import annotations

import base64
import csv
import dataclasses
import hashlib
import html
import json
import math
import struct
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCHEMA_VERSION = "prusie.analysis.v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_analysis(value: Any, directory: str | Path) -> dict:
    """Save nested evidence or a SusieResult, retaining IDs, dtypes and NaNs.

    Object arrays are encoded recursively and never pickled. Non-string mapping
    keys and tuples retain their types. Only recognized result dataclasses are
    reconstructed on load; arbitrary Python objects are deliberately rejected.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {}

    def encode(item):
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                if item.dtype != np.dtype(object):
                    raise TypeError("Structured object arrays are not supported")
                return {"type": "object_array", "shape": list(item.shape),
                        "values": [encode(v) for v in item.flat]}
            key = f"array_{len(arrays):06d}"
            arrays[key] = item
            return {"type": "array", "key": key}
        if isinstance(item, np.generic):
            return {"type": "numpy_scalar", "value": encode(np.asarray(item))}
        if dataclasses.is_dataclass(item):
            from .result import SusieResult
            if not isinstance(item, SusieResult):
                raise TypeError("Only SusieResult dataclasses are supported")
            return {"type": "SusieResult", "value": encode(item.as_dict())}
        if isinstance(item, Mapping):
            return {"type": "mapping", "items": [[encode(k), encode(v)]
                                                   for k, v in item.items()]}
        if isinstance(item, (tuple, list)):
            return {"type": "tuple" if isinstance(item, tuple) else "list",
                    "items": [encode(v) for v in item]}
        if isinstance(item, bytes):
            return {"type": "bytes", "value": base64.b64encode(item).decode("ascii")}
        if isinstance(item, float):
            # Preserve the complete binary64 representation, including NaN payloads.
            return {"type": "float", "value": struct.pack(">d", item).hex()}
        if isinstance(item, complex):
            return {"type": "complex", "real": encode(item.real), "imag": encode(item.imag)}
        if item is None or isinstance(item, (str, bool, int)):
            return {"type": "scalar", "value": item}
        raise TypeError(f"Unsupported evidence value: {type(item).__name__}")

    tree = encode(value)
    np.savez_compressed(directory / "arrays.npz", **arrays)
    document = {"schema_version": SCHEMA_VERSION, "arrays_sha256": _sha(directory / "arrays.npz"),
                "value": tree}
    (directory / "analysis.json").write_text(json.dumps(document, ensure_ascii=False, allow_nan=False,
                                                      indent=2), encoding="utf-8")
    return {name: {"path": str((directory / name).resolve()), "sha256": _sha(directory / name)}
            for name in ("analysis.json", "arrays.npz")}


def load_analysis(directory: str | Path) -> Any:
    """Read :func:`save_analysis` output, checking NPZ integrity without pickle."""
    directory = Path(directory)
    document = json.loads((directory / "analysis.json").read_text(encoding="utf-8"))
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported analysis schema")
    if _sha(directory / "arrays.npz") != document["arrays_sha256"]:
        raise ValueError("Array archive checksum mismatch")
    with np.load(directory / "arrays.npz", allow_pickle=False) as archive:
        def decode(item):
            kind = item["type"]
            if kind == "array":
                return archive[item["key"]].copy()
            if kind == "numpy_scalar":
                return decode(item["value"])[()]
            if kind == "object_array":
                values = [decode(v) for v in item["values"]]
                array = np.empty(len(values), dtype=object)
                for i, value in enumerate(values):
                    array[i] = value
                return array.reshape(item["shape"])
            if kind == "mapping":
                return {decode(k): decode(v) for k, v in item["items"]}
            if kind in ("list", "tuple"):
                values = [decode(v) for v in item["items"]]
                return tuple(values) if kind == "tuple" else values
            if kind == "float":
                return struct.unpack(">d", bytes.fromhex(item["value"]))[0]
            if kind == "complex":
                return complex(decode(item["real"]), decode(item["imag"]))
            if kind == "bytes":
                return base64.b64decode(item["value"], validate=True)
            if kind == "scalar":
                return item["value"]
            if kind == "SusieResult":
                from .result import SusieResult
                return SusieResult(**decode(item["value"]))
            raise ValueError(f"Unsupported value tag: {kind}")
        return decode(document["value"])


def _plain(value):
    """Readable JSON form; only persistence above promises exact type retention."""
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


def evidence_bundle(summary: Mapping, *, ld=None, comparison=None, provenance=None,
                    annotations=None) -> dict:
    """Build scalar facts with stable JSON-pointer-derived evidence IDs.

    Only these explicit analysis products enter the bundle. Dense fit histories,
    effect moments and raw model matrices are not collected from results.
    Caller-supplied provenance and annotations are included: review them and the
    returned bundle before consenting to transmission to an external provider.
    """
    allowed = ("variants", "components", "credible_sets", "cs_audit", "proxies",
               "cs_relations", "prior_influence", "input_qc", "convergence", "refinement", "warnings")
    selected = {k: _plain(summary[k]) for k in allowed if k in summary and k != "refinement"}
    if "refinement" in summary:
        refinement = summary["refinement"]
        selected["refinement"] = {k: _plain(refinement[k]) for k in
                                  ("status", "stop_reason", "total_fits", "best_objective", "config",
                                   "requested_convergence", "convergence_resolution", "objective_definition")
                                  if k in refinement}
        if "history" in refinement:
            selected["refinement"]["history"] = []
            for attempt in refinement["history"]:
                row = {k: _plain(v) for k, v in attempt.items() if k in
                       ("round", "component", "excluded_indices", "excluded_ids", "accepted",
                        "reason", "objective", "improvement", "relative_improvement", "incumbent_objective")}
                if "stages" in attempt:
                    row["stages"] = {phase: {k: _plain(v) for k, v in stage.items() if k in
                                            ("niter", "stop_reason", "ibss_converged", "seconds",
                                             "logged_elbo", "objective", "excluded_max_alpha", "error",
                                             "convergence", "convergence_config", "solver_settings",
                                             "convergence_source", "requested_convergence", "effective_criterion",
                                             "tol", "max_iter", "track_fit")}
                                     for phase, stage in attempt["stages"].items()}
                selected["refinement"]["history"].append(row)
        for stage in ("before", "after"):
            if stage in refinement:
                selected["refinement"][stage] = {k: _plain(v) for k, v in refinement[stage].items()
                                                  if k in ("pip", "cs", "cs_index", "ibss_converged",
                                                           "elbo", "logged_elbo", "niter", "convergence")}
    data = {"summary": selected,
            "ld_diagnostics": _plain(ld), "comparison": _plain(comparison),
            "provenance": _plain(provenance or {}), "annotations": _plain(annotations or {})}
    facts = []

    def visit(value, path):
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], path + "/" + key.replace("~", "~0").replace("/", "~1"))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}/{index}")
        else:
            facts.append({"id": "ev_" + hashlib.sha256(path.encode()).hexdigest()[:20],
                          "path": path, "value": value})
    visit(data, "")
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False,
                           separators=(",", ":"))
    return {"schema_version": "prusie.evidence.v1", "data": data, "facts": facts,
            "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "transmission_scope": "Selected deterministic result tables, diagnostics and comparisons; supplied provenance and annotations are included and require privacy review. Dense fit histories, effect moments and raw fit matrices are not collected from results."}


def export_tables(summary: Mapping, directory: str | Path) -> list[str]:
    """Export list-of-record tables; nested cells contain readable JSON."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, rows in summary.items():
        if not isinstance(rows, list) or not rows or not all(isinstance(r, Mapping) for r in rows):
            continue
        if not str(name).replace("_", "").isalnum():
            raise ValueError("Unsafe table name")
        path = directory / f"{name}.csv"
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: (json.dumps(_plain(v), ensure_ascii=False, allow_nan=False)
                                     if isinstance(v, (Mapping, list, tuple, np.ndarray)) else v)
                                 for k, v in row.items()})
        paths.append(str(path.resolve()))
    return paths


_METHODS = """PIP is marginal inclusion probability across active components; alpha is a
component probability. Credible sets use the recorded posterior coverage and LD purity
filters. Effective candidate counts summarize posterior concentration, not causal counts.
ELBO stopping, posterior stability and refinement improvement answer different questions.
Refinement compares restored-prior candidates against the same original model objective.
An unsuccessful attempted search does not establish global optimality. LD diagnostics assess
summary-statistics compatibility under their recorded model and do not automatically
correct alleles or filter variants. Missing or unassessed information is not a passing check.
Sensitivity results remain distinct posteriors and are never averaged. Genotype correlation
between representative variants does not by itself establish whether effects are conditionally
distinguishable in the fitted model. Neither CS count nor component matching confirms the
number of causal variants."""


def _records(value):
    return value if isinstance(value, (list, tuple)) else []


def _select(record, keys):
    return {key: _plain(record[key]) for key in keys if key in record}


def analysis_summary(summary: Mapping, *, ld=None, comparison=None,
                     candidate_pip=.1, top_candidates=10) -> dict:
    """Create a deterministic reading entry point without fitting or model calls.

    All formal CS members, the top ``top_candidates`` variants by PIP, and every
    variant with PIP >= ``candidate_pip`` are included. This is a display policy,
    not a new statistical threshold. Full-precision records remain in the
    evidence bundle; this compact view links them by table names, IDs and fields.
    Legacy comparison records are read through the comparison-layer helper.
    """
    if not 0 <= candidate_pip <= 1:
        raise ValueError("candidate_pip must lie in [0, 1]")
    if isinstance(top_candidates, bool) or not isinstance(top_candidates, int) or top_candidates < 0:
        raise ValueError("top_candidates must be a nonnegative integer")
    variants = _records(summary.get("variants"))
    components = {row.get("component_id"): row for row in _records(summary.get("components"))}
    sets = _records(summary.get("credible_sets"))
    memberships = {}
    cs_rows = []
    for row in sets:
        for variant in row.get("members", []):
            memberships.setdefault(str(variant), []).append(row.get("component_id"))
        cs = _select(row, ("component_id", "members", "size", "representative", "requested_coverage",
                           "actual_coverage", "min_abs_corr", "mean_abs_corr", "median_abs_corr"))
        cs.setdefault("size", len(row.get("members", [])))
        cs["effective_candidates"] = components.get(row.get("component_id"), {}).get("effective_candidates")
        cs_rows.append(cs)

    def pip_value(row):
        value = row.get("pip")
        return float(value) if isinstance(value, (int, float, np.number)) and np.isfinite(value) else -math.inf

    ordered = sorted(variants, key=lambda row: (-pip_value(row), str(row.get("variant_id", ""))))
    top = {str(row.get("variant_id")) for row in ordered[:top_candidates]}
    candidates = []
    for row in ordered:
        variant = str(row.get("variant_id"))
        reasons = (["formal_CS_member"] if variant in memberships else [])
        reasons += ["top_PIP"] if variant in top else []
        reasons += ["PIP_threshold"] if pip_value(row) >= candidate_pip else []
        if reasons:
            candidates.append(dict(variant_id=variant, pip=_plain(row.get("pip")), rank=row.get("rank"),
                                   cs_components=memberships.get(variant, []), selection_sources=reasons))
    control = summary.get("convergence") or {}
    last = control.get("last") or {}
    fit_state = _select(control, ("requested_criterion", "effective_criterion", "criterion", "niter",
                                 "stop_reason", "tol", "alpha_tol", "pip_tol", "min_iter", "cycle_lag", "max_cycle_lag",
                                 "requested_config", "configuration_source", "config_resolution", "fallback_iterations",
                                 "elbo_decrease_count", "nonfinite_elbo_count", "finalization"))
    fit_state.setdefault("stop_reason", "not_assessed")
    fit_state.setdefault("niter", last.get("iteration"))
    fit_state["last"] = _select(last, ("iteration", "phase", "elbo", "delta_elbo", "relative_delta_elbo",
                                       "max_abs_delta_alpha", "max_abs_delta_pip", "max_abs_delta_bbar",
                                       "sigma2", "delta_sigma2", "events"))
    fit_state["trajectory_available"] = bool(summary.get("trace"))
    fit_state["pending"] = ("No stopping rule was recorded." if not control else
                            "The selected inner stopping condition was satisfied; this does not establish a global optimum."
                            if control.get("stop_reason") == "criterion_satisfied" else
                            "Review the recorded stop reason; inner convergence is not established by this summary.")
    refinement = summary.get("refinement") or {}
    history = _records(refinement.get("history"))
    before, after = refinement.get("before") or {}, refinement.get("after") or {}
    before_objective = before.get("elbo")
    after_objective = refinement.get("best_objective", after.get("elbo"))
    improvement = None
    if all(isinstance(value, (float, int, np.number)) and np.isfinite(value)
           for value in (before_objective, after_objective)):
        improvement = float(after_objective - before_objective)
    search = dict(status=refinement.get("status", "not_requested"),
                  stop_reason=refinement.get("stop_reason", "not_assessed"),
                  attempts=len(history), accepted_attempts=sum(bool(row.get("accepted")) for row in history),
                  inner_fits=refinement.get("total_fits"),
                  rounds=len({row.get("round") for row in history if row.get("round") is not None}),
                  before_objective=before_objective, after_objective=after_objective,
                  improvement=improvement,
                  objective_definition="uniform original-model objective recomputed from the complete post-trim state",
                  before_logged_elbo=before.get("logged_elbo"), after_logged_elbo=after.get("logged_elbo"),
                  config=_plain(refinement.get("config", {})))
    if ld is not None:
        from .ld import diagnostic_counts
        ld_state = dict(status="recorded", **diagnostic_counts(ld),
                        mixture=_select(ld.get("mixture") or {}, ("status", "reason", "optimizer", "converged", "kkt_max_violation")),
                        thresholds=_plain(ld.get("thresholds", {})), regularization=ld.get("regularization"),
                        method=ld.get("method"), warnings=_plain(ld.get("warnings", [])))
    else:
        ld_state = dict(status="not_assessed", total=0, assessed=0, flagged=0, not_flagged=0,
                        conditional_only=0, unassessed=0, mixture={}, thresholds={})
    sensitivity = dict(status="not_assessed", baseline=None, settings=[], relation_counts={}, pairs=[])
    if comparison is not None:
        from .comparison import comparison_relations
        relations = comparison_relations(comparison, baseline_only=True)
        counts = {}
        for row in relations:
            state = row.get("status", "unknown")
            counts[state] = counts.get(state, 0) + 1
        baseline = comparison.get("baseline", next(iter(comparison.get("settings", [])), None))
        pairs = comparison.get("baseline_pairwise")
        if pairs is None:
            pairs = [row for row in comparison.get("pairwise", [])
                     if baseline in (row.get("setting_a"), row.get("setting_b"))]
        sensitivity = dict(status="recorded", baseline=baseline, settings=comparison.get("settings", []),
                           relation_counts=counts, pairs=_plain(pairs), layer="CS-supported signals",
                           layer_summary=_plain(comparison.get("layer_summary", {})),
                           legacy_format=comparison.get("schema_version") != "prusie.comparison.v2")
    return _plain(dict(schema_version="prusie.analysis_summary.v1",
                       evidence_entries={"main_results": "/summary/credible_sets", "variants": "/summary/variants",
                                         "fit": "/summary/convergence", "refinement": "/summary/refinement",
                                         "sensitivity": "/comparison", "ld": "/ld_diagnostics"},
                       results=dict(retained_cs=len(sets), variant_count=len(variants),
                                    active_components=sum(bool(row.get("active")) for row in components.values()),
                                    credible_sets=cs_rows, candidates=candidates,
                                    candidate_policy=dict(all_CS_members=True, pip_threshold=candidate_pip,
                                                          top_candidates=top_candidates),
                                    causal_count="not_identified_by_CS_count"),
                       fit=fit_state, refinement=search, ld=ld_state, sensitivity=sensitivity))


def _display(value):
    """Human display only: machine exports retain their full precision."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return "NA"
    if isinstance(value, float):
        return format(value, ".6g") if math.isfinite(value) else str(value)
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, Mapping):
        return "; ".join(f"{key}: {_display(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple, np.ndarray)):
        return ", ".join(_display(item) for item in value)
    return str(value)


def write_report(summary: Mapping, directory: str | Path, *, figures=None, ld=None,
                 comparison=None, provenance=None, annotations=None, ai=None,
                 title="prusie analysis") -> dict:
    """Write an offline report with a compact deterministic summary and exports.

    ``figures`` maps labels to Matplotlib Figures or plot return tuples. Embedded
    PNGs work offline; relative links open complete local JSON, NPZ, CSV and figure
    exports. Copy the directory to retain these links. All scalar evidence facts
    remain in ``evidence.json`` rather than being duplicated into the HTML DOM.
    No fit, network request or model invocation is performed here.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    required = ("data_checksums", "variant_harmonization", "parameters", "initial_state",
                "reference_sources", "software_versions", "random_seed", "elapsed_seconds", "skipped")
    recorded = {key: "not_assessed" for key in required}
    recorded.update(provenance or {})
    evidence = evidence_bundle(summary, ld=ld, comparison=comparison,
                               provenance=recorded, annotations=annotations)
    compact = analysis_summary(summary, ld=ld, comparison=comparison)
    compact["evidence_sha256"] = evidence["sha256"]
    compact["files"] = {"evidence": "evidence.json", "lossless_structure": "analysis.json",
                        "lossless_arrays": "arrays.npz", "tables": "tables/"}
    artifacts = save_analysis({"summary": summary, "ld": ld, "comparison": comparison,
                               "provenance": recorded, "annotations": annotations, "ai": ai}, directory)
    for name, value in (("evidence", evidence), ("analysis_summary", compact)):
        (directory / f"{name}.json").write_text(
            json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    tables = export_tables(summary, directory / "tables")
    sections = [f"<h1>{html.escape(str(title))}</h1>",
                "<p>Deterministic statistical summary. Optional AI text is separate and requires scientific review.</p>",
                '<h2 id="main-results">Main results and resolution</h2>',
                f'<p>{len(compact["results"]["credible_sets"])} retained credible sets among '
                f'{compact["results"]["variant_count"]} analyzed variants. '
                'A retained CS is a posterior candidate set, not a confirmed causal effect count.</p>']

    def table(rows, columns=None):
        rows = list(rows)
        if not rows:
            sections.append("<p>No records.</p>")
            return
        columns = columns or list(dict.fromkeys(key for row in rows for key in row))
        sections.append('<div class="table-scroll"><table><thead><tr>' + ''.join(
            '<th>' + html.escape(str(key).replace('_', ' ')) + '</th>' for key in columns) + '</tr></thead><tbody>')
        for row in rows:
            sections.append('<tr>' + ''.join('<td>' + html.escape(_display(row.get(key))) + '</td>'
                                            for key in columns) + '</tr>')
        sections.append('</tbody></table></div>')

    def details(label, value, columns=None):
        sections.append('<details><summary>' + html.escape(label) + '</summary>')
        if isinstance(value, (list, tuple)) and all(isinstance(row, Mapping) for row in value):
            table(value, columns)
        else:
            sections.append('<pre>' + html.escape(json.dumps(_plain(value), ensure_ascii=False,
                                                              allow_nan=False, indent=2)) + '</pre>')
        sections.append('</details>')

    table(compact['results']['credible_sets'])
    sections.append('<h3>Main candidates</h3><p>All formal CS members, the top ten variants by PIP, '
                    'and all PIP ≥ 0.1 candidates are included. These are display selections; '
                    'missing or excluded variants are not assigned zero PIP. Display numbers are rounded '
                    'to six significant digits; exports retain full precision.</p>')
    table(compact['results']['candidates'])
    sections.append('<h2 id="fit-status">Fit, stopping and refinement</h2>')
    control = compact['fit']
    sections.append('<p>' + html.escape(control['pending']) + '</p>')
    table([_select(control, ('requested_criterion', 'effective_criterion', 'niter', 'stop_reason',
                            'tol', 'alpha_tol', 'pip_tol', 'min_iter', 'cycle_lag', 'max_cycle_lag',
                            'elbo_decrease_count', 'nonfinite_elbo_count'))])
    if control['last']:
        table([control['last']])
    if not control['trajectory_available']:
        sections.append('<p>No iteration trajectory is available in this summary; final diagnostics alone '
                        'cannot reconstruct the path taken by IBSS.</p>')
    search = compact['refinement']
    sections.append('<p>Refinement: ' + html.escape(_display(search['status'])) + '. '
                    + str(search['attempts']) + ' exclusion attempts; ' + str(search['accepted_attempts'])
                    + ' accepted; ' + html.escape(_display(search['inner_fits'])) + ' inner fits. '
                    'The initial stopping condition and finding a better solution concern different questions.</p>')
    if search['status'] != 'not_requested':
        table([_select(search, ('stop_reason', 'before_objective', 'after_objective', 'improvement',
                               'before_logged_elbo', 'after_logged_elbo'))])
        sections.append('<p>Improvement uses the uniform original-model objective recomputed from the '
                        'complete post-trim state. Logged pre-finalization ELBO is shown separately. '
                        'A negative rejected candidate does not mean the accepted solution regressed.</p>')
        attempts = []
        for row in (summary.get('refinement') or {}).get('history', []):
            attempt = _select(row, ('round', 'component', 'excluded_ids', 'accepted', 'reason', 'incumbent_objective', 'objective', 'improvement'))
            for phase, stage in row.get('stages', {}).items():
                attempt[f'{phase}_stop'] = stage.get('stop_reason')
                attempt[f'{phase}_iterations'] = stage.get('niter')
            attempts.append(attempt)
        accepted = [row for row in attempts if row.get('accepted')]
        if accepted:
            sections.append('<h3>Accepted attempts</h3>')
            table(accepted)
        details('Complete refinement attempts and stage stops', attempts)
    sections.append('<h2 id="figures">Figures and before/after evidence</h2>')
    figure_paths = []
    for index, (name, value) in enumerate((figures or {}).items()):
        figure = value[0] if isinstance(value, tuple) else value
        stem = f'figure_{index + 1}'
        # Fixed SVG IDs and omitted creation time make identical input figures
        # reproducible without changing global Matplotlib configuration.
        import matplotlib as mpl
        with mpl.rc_context({'svg.hashsalt': 'prusie-offline-report-v1'}):
            for suffix in ('svg', 'png'):
                path = directory / f'{stem}.{suffix}'
                metadata = {'Date': None} if suffix == 'svg' else {'Software': 'prusie'}
                figure.savefig(path, bbox_inches='tight', metadata=metadata)
                figure_paths.append(str(path.resolve()))
        encoded = base64.b64encode((directory / f'{stem}.png').read_bytes()).decode('ascii')
        sections.append(f'<figure><img alt="{html.escape(str(name), quote=True)}" '
                        f'src="data:image/png;base64,{encoded}"><figcaption>{html.escape(str(name))}'
                        f' · <a href="{stem}.png">PNG</a> · <a href="{stem}.svg">SVG</a>'
                        '</figcaption></figure>')
    if not figure_paths:
        sections.append('<p>No figures were supplied.</p>')
    sections.append('<h2 id="sensitivity-ld">Sensitivity and LD diagnostics</h2>')
    sensitivity = compact['sensitivity']
    if comparison is not None:
        sections.append('<p>Main comparisons use CS-supported signals against baseline '
                        + html.escape(_display(sensitivity['baseline']))
                        + '. Active diffuse components remain in the complete component view; '
                        'posterior overlap alone does not confirm a localized split or merge.</p>')
        table([sensitivity['relation_counts']])
        table(sensitivity['pairs'])
        from .comparison import comparison_relations
        table(comparison_relations(comparison, baseline_only=True))
    else:
        sections.append('<p>Sensitivity was not assessed.</p>')
    counts = compact['ld']
    if ld is not None:
        sections.append('<p>LD flip assessment: ' + str(counts['assessed']) + ' assessed ('
                        + str(counts['flagged']) + ' flagged; ' + str(counts['not_flagged'])
                        + ' assessed and not flagged); ' + str(counts['conditional_only'])
                        + ' conditional-only; ' + str(counts['unassessed']) + ' unassessed. '
                        'A finite conditional residual is not a completed flip assessment. '
                        'Missing-z working zeros are not observed z statistics.</p>')
        table([_select(counts, ('method', 'regularization', 'thresholds', 'mixture'))])
        from .ld import diagnostic_input_missing, diagnostic_status
        needs_review = []
        for row in ld.get('variants', []):
            if diagnostic_status(row) == 'not_flagged':
                continue
            displayed = dict(row, assessment=diagnostic_status(row))
            if diagnostic_input_missing(row):
                # Old files can retain upstream zero-replacement statistics.
                # Preserve them in evidence, never label them observed in HTML.
                displayed.update(observed_z=None, standardized_residual=None)
            needs_review.append(displayed)
        if needs_review:
            details('Flagged, conditional-only and unassessed variants', needs_review,
                    ('variant_id', 'assessment', 'observed_z', 'expected_z', 'standardized_residual',
                     'log_likelihood_ratio', 'flag', 'unavailable_reason'))
        elif counts['assessed']:
            sections.append('<p>No variants were flagged among completed assessments. '
                            'This does not establish that the LD reference is correct.</p>')
        else:
            sections.append('<p>No completed flip assessment is available.</p>')
    else:
        sections.append('<p>LD diagnostics were not assessed.</p>')
    sections.append('<h2 id="details">Complete records and provenance</h2>')
    details('All components, including inactive and diffuse components', summary.get('components', []),
            ('component_id', 'active', 'cs_retained', 'V', 'top_variant', 'top_alpha',
             'effective_candidates', 'biological_mass', 'null_mass', 'component_log_bf'))
    sections.append('<p>Complete component rankings and alpha arrays are in '
                    '<a href="evidence.json">evidence.json</a> under data.summary.components.</p>')
    for label, name in (('CS audit', 'cs_audit'), ('CS-outside direct proxies', 'proxies'),
                        ('Between-CS genotype correlations', 'cs_relations')):
        details(label, summary.get(name, []))
    if comparison is not None:
        details('All setting pairs: CS-supported relations', comparison_relations(comparison, baseline_only=False))
        details('All active-component relations, including diffuse overlap',
                comparison_relations(comparison, layer='components', baseline_only=False))
    details('Provenance', recorded)
    details('Input QC and warnings', {'input_qc': summary.get('input_qc', {}),
                                     'warnings': summary.get('warnings', [])})
    sections.append('<h2>Methods and interpretation</h2><p>' + html.escape(_METHODS) + '</p>')
    if ai is not None:
        details('Optional AI interpretation — separate from computed evidence', ai)
    sections.append('<h2 id="exports">Machine-readable evidence references</h2>'
                    '<p>Start with <a href="analysis_summary.json">analysis_summary.json</a>; '
                    'query <a href="evidence.json">evidence.json</a> using data paths or stable fact IDs. '
                    'All ' + str(len(evidence['facts'])) + ' scalar facts remain in that file and are not '
                    'duplicated into this page. No browser scripts or remote resources are required.</p>'
                    '<p>Lossless saved structure: <a href="analysis.json">analysis.json</a> with '
                    '<a href="arrays.npz">arrays.npz</a>. Open index.html locally and keep its directory '
                    'together to retain export links. GitHub file pages do not execute this report.</p>')
    sections.append('<ul>' + ''.join('<li><a href="tables/' + html.escape(Path(path).name, quote=True)
                                    + '">' + html.escape(Path(path).name) + '</a></li>' for path in tables) + '</ul>')
    content = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
               '<meta name="viewport" content="width=device-width, initial-scale=1">'
               '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
               'img-src data:; style-src \'unsafe-inline\'">'
               f'<title>{html.escape(str(title))}</title><style>body{{max-width:1100px;margin:auto;'
               'padding:1.5rem;font-family:system-ui,sans-serif;line-height:1.5}pre{white-space:'
               'pre-wrap;overflow-wrap:anywhere;background:#f4f4f4;padding:1rem}img{max-width:100%}'
               'figure{margin:2rem 0}.table-scroll{overflow:auto}table{border-collapse:collapse}'
               'th,td{border:1px solid #ccc;padding:.4rem;text-align:left;vertical-align:top}'
               'details{margin:1rem 0}summary{cursor:pointer;font-weight:600}'
               '</style></head><body>' + '\n'.join(sections) + '</body></html>')
    path = directory / 'index.html'
    path.write_text(content, encoding='utf-8')
    return {'html': str(path.resolve()), 'persistence': artifacts, 'tables': tables,
            'figures': figure_paths, 'evidence_sha256': evidence['sha256'],
            'analysis_summary': str((directory / 'analysis_summary.json').resolve()),
            'evidence': str((directory / 'evidence.json').resolve())}
