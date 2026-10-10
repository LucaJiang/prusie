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
                                  ("status", "stop_reason", "total_fits", "best_objective", "config")
                                  if k in refinement}
        if "history" in refinement:
            selected["refinement"]["history"] = []
            for attempt in refinement["history"]:
                row = {k: _plain(v) for k, v in attempt.items() if k in
                       ("round", "component", "excluded_indices", "excluded_ids", "accepted",
                        "reason", "objective", "improvement", "relative_improvement")}
                if "stages" in attempt:
                    row["stages"] = {phase: {k: _plain(v) for k, v in stage.items() if k in
                                            ("niter", "stop_reason", "ibss_converged", "seconds",
                                             "logged_elbo", "objective", "excluded_max_alpha", "error")}
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
Sensitivity results remain distinct posteriors and are never averaged."""


def write_report(summary: Mapping, directory: str | Path, *, figures=None, ld=None,
                 comparison=None, provenance=None, annotations=None, ai=None,
                 title="prusie analysis") -> dict:
    """Write a self-contained offline HTML report, evidence, tables and NPZ.

    ``figures`` maps names to Matplotlib Figure objects or plot return tuples.
    SVG/PNG files are exported separately; embedded PNGs keep HTML fully offline.
    Provenance is user supplied; missing provenance fields are explicitly unknown.
    AI text is a separate optional section and never replaces deterministic facts.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    required = ("data_checksums", "variant_harmonization", "parameters", "initial_state",
                "reference_sources", "software_versions", "random_seed", "elapsed_seconds", "skipped")
    recorded = {k: "not_assessed" for k in required}
    recorded.update(provenance or {})
    evidence = evidence_bundle(summary, ld=ld, comparison=comparison,
                               provenance=recorded, annotations=annotations)
    artifacts = save_analysis({"summary": summary, "ld": ld, "comparison": comparison,
                               "provenance": recorded, "annotations": annotations, "ai": ai}, directory)
    (directory / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, allow_nan=False,
                                                      indent=2), encoding="utf-8")
    tables = export_tables(summary, directory / "tables")
    sections = [f"<h1>{html.escape(title)}</h1>", "<h2>Methods and interpretation</h2>",
                f"<p>{html.escape(_METHODS)}</p>"]

    def show(label, value):
        sections.append(f"<h2>{html.escape(label)}</h2>")
        readable = json.dumps(_plain(value), ensure_ascii=False, allow_nan=False, indent=2)
        collapsed = len(readable) > 6000 or label == "Machine-readable evidence references"
        if collapsed:
            sections.append('<details><summary>Show complete recorded data</summary>')
        if isinstance(value, list) and value and all(isinstance(row, Mapping) for row in value):
            columns = list(dict.fromkeys(k for row in value for k in row))
            sections.append('<div class="table-scroll"><table><thead><tr>' + "".join(
                "<th>" + html.escape(str(k)) + "</th>" for k in columns) + "</tr></thead><tbody>")
            for row in value:
                cells = [html.escape(json.dumps(_plain(row.get(k)), ensure_ascii=False, allow_nan=False))
                         for k in columns]
                sections.append("<tr>" + "".join("<td>" + cell + "</td>" for cell in cells) + "</tr>")
            sections.append("</tbody></table></div>")
        else:
            sections.append("<pre>" + html.escape(readable) + "</pre>")
        if collapsed:
            sections.append("</details>")
    show("Provenance", recorded)
    for name, value in summary.items():
        show(name.replace("_", " ").title(), value)
    if ld is not None:
        show("LD diagnostics and numerical treatment", ld)
    if comparison is not None:
        show("Sensitivity and signal comparison", comparison)
    figure_paths = []
    for index, (name, value) in enumerate((figures or {}).items()):
        figure = value[0] if isinstance(value, tuple) else value
        stem = f"figure_{index + 1}"
        for suffix in ("svg", "png"):
            path = directory / f"{stem}.{suffix}"
            figure.savefig(path, bbox_inches="tight")
            figure_paths.append(str(path.resolve()))
        encoded = base64.b64encode((directory / f"{stem}.png").read_bytes()).decode()
        sections.append(f'<figure><img alt="{html.escape(str(name), quote=True)}" '
                        f'src="data:image/png;base64,{encoded}"><figcaption>{html.escape(str(name))}'
                        "</figcaption></figure>")
    show("Machine-readable evidence references", evidence["facts"])
    if ai is not None:
        show("Optional AI interpretation — separate from computed evidence", ai)
    content = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
               '<meta name="viewport" content="width=device-width, initial-scale=1">'
               '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
               "img-src data:; style-src 'unsafe-inline'\">"
               f"<title>{html.escape(title)}</title><style>body{{max-width:1100px;margin:auto;"
               "padding:1.5rem;font-family:system-ui,sans-serif;line-height:1.5}pre{white-space:"
               "pre-wrap;overflow-wrap:anywhere;background:#f4f4f4;padding:1rem}img{max-width:100%}"
               "figure{margin:2rem 0}.table-scroll{overflow:auto}table{border-collapse:collapse}"
               "th,td{border:1px solid #ccc;padding:.4rem;text-align:left;vertical-align:top}"
               "</style></head><body>" + "\n".join(sections) + "</body></html>")
    path = directory / "index.html"
    path.write_text(content, encoding="utf-8")
    return {"html": str(path.resolve()), "persistence": artifacts, "tables": tables,
            "figures": figure_paths, "evidence_sha256": evidence["sha256"]}
