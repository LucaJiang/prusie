"""Offline reports and lossless, pickle-free storage of analysis results.

JSON stores the structure and type tags, and NPZ stores the exact NumPy
array bytes. CSV files provide a readable tabular export.
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
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "prusie.analysis.v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_analysis(value: Any, directory: str | Path) -> dict:
    """Save a SusieResult or nested analysis record without pickle.

    Parameters
    ----------
    value : SusieResult or nested mapping/sequence
        Value to save. Arrays keep their dtype, NaN payloads are preserved,
        object arrays are encoded element by element, and tuples and
        non-string mapping keys keep their types.
    directory : str or Path
        Output directory; ``analysis.json`` and ``arrays.npz`` are written.

    Returns
    -------
    dict
        Path and SHA-256 digest of each written file.

    Raises
    ------
    TypeError
        A value of an unsupported type, including dataclasses other than
        SusieResult.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {}

    def encode(item):
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                if item.dtype != np.dtype(object):
                    raise TypeError("Structured object arrays are not supported")
                return {
                    "type": "object_array",
                    "shape": list(item.shape),
                    "values": [encode(v) for v in item.flat],
                }
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
            return {"type": "mapping", "items": [[encode(k), encode(v)] for k, v in item.items()]}
        if isinstance(item, (tuple, list)):
            return {
                "type": "tuple" if isinstance(item, tuple) else "list",
                "items": [encode(v) for v in item],
            }
        if isinstance(item, bytes):
            return {"type": "bytes", "value": base64.b64encode(item).decode("ascii")}
        if isinstance(item, float):
            # Store the IEEE 754 binary64 bytes, preserving NaN payloads.
            return {"type": "float", "value": struct.pack(">d", item).hex()}
        if isinstance(item, complex):
            return {"type": "complex", "real": encode(item.real), "imag": encode(item.imag)}
        if item is None or isinstance(item, (str, bool, int)):
            return {"type": "scalar", "value": item}
        raise TypeError(f"Unsupported evidence value: {type(item).__name__}")

    tree = encode(value)
    np.savez_compressed(directory / "arrays.npz", **arrays)
    document = {
        "schema_version": SCHEMA_VERSION,
        "arrays_sha256": _sha(directory / "arrays.npz"),
        "value": tree,
    }
    (directory / "analysis.json").write_text(
        json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8"
    )
    return {
        name: {"path": str((directory / name).resolve()), "sha256": _sha(directory / name)}
        for name in ("analysis.json", "arrays.npz")
    }


def load_analysis(directory: str | Path) -> Any:
    """Load the output of :func:`save_analysis`.

    The array archive is checked against its recorded SHA-256 digest and read
    with ``allow_pickle=False``.

    Raises
    ------
    ValueError
        Unsupported schema, checksum mismatch or unknown value tag.
    """
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
    "Readable JSON form; only persistence above promises exact type retention."
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


def evidence_bundle(
    summary: Mapping,
    *,
    ld: Mapping[str, Any] | None = None,
    comparison: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    annotations: Mapping[str, Any] | None = None,
) -> dict:
    """Collect scalar facts of an analysis, each with a stable evidence ID.

    Each ID is derived from the JSON pointer of the fact. Only the listed
    summary tables, LD diagnostics, comparison and caller-supplied
    provenance and annotations are included; dense iteration histories,
    effect moments and model matrices are not. Review the bundle, including
    the caller-supplied fields, before transmitting it to an external
    provider.
    """
    allowed = (
        "variants",
        "components",
        "credible_sets",
        "cs_audit",
        "proxies",
        "cs_relations",
        "prior_influence",
        "input_qc",
        "convergence",
        "refinement",
        "warnings",
    )
    selected = {k: _plain(summary[k]) for k in allowed if k in summary and k != "refinement"}
    if "refinement" in summary:
        refinement = summary["refinement"]
        selected["refinement"] = {
            k: _plain(refinement[k])
            for k in (
                "status",
                "stop_reason",
                "total_fits",
                "best_objective",
                "config",
                "requested_convergence",
                "convergence_resolution",
                "objective_definition",
            )
            if k in refinement
        }
        if "history" in refinement:
            selected["refinement"]["history"] = []
            for attempt in refinement["history"]:
                row = {
                    k: _plain(v)
                    for k, v in attempt.items()
                    if k
                    in (
                        "round",
                        "component",
                        "excluded_indices",
                        "excluded_ids",
                        "accepted",
                        "reason",
                        "objective",
                        "improvement",
                        "relative_improvement",
                        "incumbent_objective",
                    )
                }
                if "stages" in attempt:
                    row["stages"] = {
                        phase: {
                            k: _plain(v)
                            for k, v in stage.items()
                            if k
                            in (
                                "niter",
                                "stop_reason",
                                "ibss_converged",
                                "seconds",
                                "logged_elbo",
                                "objective",
                                "excluded_max_alpha",
                                "error",
                                "convergence",
                                "convergence_config",
                                "solver_settings",
                                "convergence_source",
                                "requested_convergence",
                                "effective_criterion",
                                "tol",
                                "max_iter",
                                "track_fit",
                            )
                        }
                        for phase, stage in attempt["stages"].items()
                    }
                selected["refinement"]["history"].append(row)
        for stage in ("before", "after"):
            if stage in refinement:
                selected["refinement"][stage] = {
                    k: _plain(v)
                    for k, v in refinement[stage].items()
                    if k
                    in (
                        "pip",
                        "cs",
                        "cs_index",
                        "ibss_converged",
                        "elbo",
                        "logged_elbo",
                        "niter",
                        "convergence",
                    )
                }
    data = {
        "summary": selected,
        "ld_diagnostics": _plain(ld),
        "comparison": _plain(comparison),
        "provenance": _plain(provenance or {}),
        "annotations": _plain(annotations or {}),
    }
    facts = []

    def visit(value, path):
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], path + "/" + key.replace("~", "~0").replace("/", "~1"))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}/{index}")
        else:
            facts.append(
                {
                    "id": "ev_" + hashlib.sha256(path.encode()).hexdigest()[:20],
                    "path": path,
                    "value": value,
                }
            )

    visit(data, "")
    canonical = json.dumps(
        data, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    return {
        "schema_version": "prusie.evidence.v1",
        "data": data,
        "facts": facts,
        "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "transmission_scope": (
            "Selected deterministic result tables, diagnostics and "
            "comparisons; supplied provenance and annotations are "
            "included and require privacy review. Dense fit "
            "histories, effect moments and raw fit matrices are not "
            "collected from results."
        ),
    }


def export_tables(summary: Mapping, directory: str | Path) -> list[str]:
    """Write each list-of-records table of ``summary`` as CSV.

    Nested values are written as JSON text. Returns the written paths.
    """
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
                writer.writerow(
                    {
                        k: (
                            json.dumps(_plain(v), ensure_ascii=False, allow_nan=False)
                            if isinstance(v, (Mapping, list, tuple, np.ndarray))
                            else v
                        )
                        for k, v in row.items()
                    }
                )
        paths.append(str(path.resolve()))
    return paths


_METHODS = """PIP is marginal inclusion probability across active components; alpha is a
component probability. Credible sets use the recorded posterior coverage and LD purity
filters. The effective number of variants is exp(entropy of component alpha),
normalized over non-null variants; it describes concentration, not a causal count.
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


def analysis_summary(
    summary: Mapping,
    *,
    ld: Mapping[str, Any] | None = None,
    comparison: Mapping[str, Any] | None = None,
    candidate_pip: float = 0.1,
    top_candidates: int = 10,
) -> dict:
    """Return a compact summary of results, convergence, refinement, LD and sensitivity.

    Parameters
    ----------
    summary : mapping
        Output of :func:`prusie.summarize`.
    ld : dict, optional
        Output of :func:`prusie.diagnose_ld`.
    comparison : dict, optional
        Output of :func:`prusie.compare_fits`, or a saved record.
    candidate_pip : float, default 0.1
        Variants with PIP ≥ candidate_pip are listed as candidates.
    top_candidates : int, default 10
        Number of leading variants by PIP listed as candidates.

    Returns
    -------
    dict
        Candidate variants (all credible-set members, the leading variants
        and those above ``candidate_pip``), credible sets, stopping status,
        refinement outcome, LD status counts and sensitivity maxima, with
        references to the full tables. The candidate rule is a display
        choice, not a statistical threshold. No model is fitted.
    """
    if not 0 <= candidate_pip <= 1:
        raise ValueError("candidate_pip must lie in [0, 1]")
    if (
        isinstance(top_candidates, bool)
        or not isinstance(top_candidates, int)
        or top_candidates < 0
    ):
        raise ValueError("top_candidates must be a nonnegative integer")
    variants = _records(summary.get("variants"))
    components = {row.get("component_id"): row for row in _records(summary.get("components"))}
    sets = _records(summary.get("credible_sets"))
    memberships = {}
    cs_rows = []
    for row in sets:
        for variant in row.get("members", []):
            memberships.setdefault(str(variant), []).append(row.get("component_id"))
        cs = _select(
            row,
            (
                "component_id",
                "members",
                "size",
                "representative",
                "requested_coverage",
                "actual_coverage",
                "min_abs_corr",
                "mean_abs_corr",
                "median_abs_corr",
            ),
        )
        cs.setdefault("size", len(row.get("members", [])))
        cs["effective_candidates"] = components.get(row.get("component_id"), {}).get(
            "effective_candidates"
        )
        cs_rows.append(cs)

    def pip_value(row):
        value = row.get("pip")
        return (
            float(value)
            if isinstance(value, (int, float, np.number)) and np.isfinite(value)
            else -math.inf
        )

    ordered = sorted(variants, key=lambda row: (-pip_value(row), str(row.get("variant_id", ""))))
    top = {str(row.get("variant_id")) for row in ordered[:top_candidates]}
    candidates = []
    for row in ordered:
        variant = str(row.get("variant_id"))
        reasons = ["formal_CS_member"] if variant in memberships else []
        reasons += ["top_PIP"] if variant in top else []
        reasons += ["PIP_threshold"] if pip_value(row) >= candidate_pip else []
        if reasons:
            candidates.append(
                dict(
                    variant_id=variant,
                    pip=_plain(row.get("pip")),
                    rank=row.get("rank"),
                    cs_components=memberships.get(variant, []),
                    selection_sources=reasons,
                )
            )
    control = summary.get("convergence") or {}
    last = control.get("last") or {}
    fit_state = _select(
        control,
        (
            "requested_criterion",
            "effective_criterion",
            "criterion",
            "niter",
            "stop_reason",
            "tol",
            "alpha_tol",
            "pip_tol",
            "min_iter",
            "cycle_lag",
            "max_cycle_lag",
            "requested_config",
            "configuration_source",
            "config_resolution",
            "fallback_iterations",
            "elbo_decrease_count",
            "nonfinite_elbo_count",
            "finalization",
        ),
    )
    fit_state.setdefault("stop_reason", "not_assessed")
    fit_state.setdefault("niter", last.get("iteration"))
    fit_state["last"] = _select(
        last,
        (
            "iteration",
            "phase",
            "elbo",
            "delta_elbo",
            "relative_delta_elbo",
            "max_abs_delta_alpha",
            "max_abs_delta_pip",
            "max_abs_delta_bbar",
            "sigma2",
            "delta_sigma2",
            "events",
        ),
    )
    fit_state["trajectory_available"] = bool(summary.get("trace"))
    fit_state["pending"] = (
        "No stopping rule was recorded."
        if not control
        else (
            "The selected inner stopping condition was satisfied; "
            "this does not establish a global optimum."
        )
        if control.get("stop_reason") == "criterion_satisfied"
        else (
            "Review the recorded stop reason; inner convergence is not established by this summary."
        )
    )
    refinement = summary.get("refinement") or {}
    history = _records(refinement.get("history"))
    before, after = refinement.get("before") or {}, refinement.get("after") or {}
    before_objective = before.get("elbo")
    after_objective = refinement.get("best_objective", after.get("elbo"))
    improvement = None
    if all(
        isinstance(value, (float, int, np.number)) and np.isfinite(value)
        for value in (before_objective, after_objective)
    ):
        improvement = float(after_objective - before_objective)
    search = dict(
        status=refinement.get("status", "not_requested"),
        stop_reason=refinement.get("stop_reason", "not_assessed"),
        attempts=len(history),
        accepted_attempts=sum(bool(row.get("accepted")) for row in history),
        inner_fits=refinement.get("total_fits"),
        rounds=len({row.get("round") for row in history if row.get("round") is not None}),
        before_objective=before_objective,
        after_objective=after_objective,
        improvement=improvement,
        objective_definition=(
            "common original-prior objective recomputed from the complete post-trim state"
        ),
        before_logged_elbo=before.get("logged_elbo"),
        after_logged_elbo=after.get("logged_elbo"),
        config=_plain(refinement.get("config", {})),
    )
    if ld is not None:
        from .ld import diagnostic_counts

        ld_state = dict(
            status="recorded",
            **diagnostic_counts(ld),
            mixture=_select(
                ld.get("mixture") or {},
                ("status", "reason", "optimizer", "converged", "kkt_max_violation"),
            ),
            thresholds=_plain(ld.get("thresholds", {})),
            regularization=ld.get("regularization"),
            method=ld.get("method"),
            warnings=_plain(ld.get("warnings", [])),
        )
    else:
        ld_state = dict(
            status="not_assessed",
            total=0,
            assessed=0,
            flagged=0,
            not_flagged=0,
            conditional_only=0,
            unassessed=0,
            mixture={},
            thresholds={},
        )
    sensitivity = dict(
        status="not_assessed", baseline=None, settings=[], relation_counts={}, pairs=[]
    )
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
            pairs = [
                row
                for row in comparison.get("pairwise", [])
                if baseline in (row.get("setting_a"), row.get("setting_b"))
            ]
        sensitivity = dict(
            status="recorded",
            baseline=baseline,
            settings=comparison.get("settings", []),
            relation_counts=counts,
            pairs=_plain(pairs),
            layer="CS-supported signals",
            layer_summary=_plain(comparison.get("layer_summary", {})),
            legacy_format=comparison.get("schema_version") != "prusie.comparison.v2",
            mode=comparison.get("mode"),
            support=_plain(comparison.get("support", {})),
            regional_maxima=[],
        )
        # Read the saved comparison support. In union mode a pair uses only rows
        # observed in both settings; common-mode tables already contain only the
        # global intersection. Do not fill missing variants or renormalize PIPs.
        for pair in pairs:
            first, second = pair.get("setting_a"), pair.get("setting_b")
            observed = []
            for index, row in enumerate(_records(comparison.get("variants"))):
                values = row.get("pips") or {}
                a, b = values.get(first), values.get(second)
                if all(isinstance(v, (int, float, np.number)) and np.isfinite(v) for v in (a, b)):
                    observed.append(
                        (abs(float(a) - float(b)), str(row.get("variant_id")), index, row)
                    )
            if observed:
                # Prefer a displayed candidate among exact ties, avoiding an
                # "outside the view" warning when every PIP is unchanged.
                delta, variant, index, row = min(
                    observed,
                    key=lambda item: (-item[0], not bool(item[3].get("candidate")), item[1]),
                )
                sensitivity["regional_maxima"].append(
                    dict(
                        setting_a=first,
                        setting_b=second,
                        variant_id=variant,
                        max_abs_delta_pip=delta,
                        pip_a=row["pips"][first],
                        pip_b=row["pips"][second],
                        in_candidate_view=bool(row.get("candidate")),
                        tied_maxima=sum(item[0] == delta for item in observed),
                        evidence_pointer=f"/comparison/variants/{index}",
                        support_scope=pair.get("support_scope", "recorded shared variant table"),
                    )
                )
    return _plain(
        dict(
            schema_version="prusie.analysis_summary.v1",
            evidence_entries={
                "main_results": "/summary/credible_sets",
                "variants": "/summary/variants",
                "fit": "/summary/convergence",
                "refinement": "/summary/refinement",
                "sensitivity": "/comparison",
                "ld": "/ld_diagnostics",
            },
            results=dict(
                retained_cs=len(sets),
                variant_count=len(variants),
                active_components=sum(bool(row.get("active")) for row in components.values()),
                credible_sets=cs_rows,
                candidates=candidates,
                candidate_policy=dict(
                    all_CS_members=True, pip_threshold=candidate_pip, top_candidates=top_candidates
                ),
                causal_count="not_identified_by_CS_count",
            ),
            fit=fit_state,
            refinement=search,
            ld=ld_state,
            sensitivity=sensitivity,
        )
    )


def _display(value, *, probability=False):
    """Rounded text for display; exported values keep full precision."""
    from ._style import format_number

    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return "NA"
    if isinstance(value, float):
        return format_number(value, probability=probability)
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, Mapping):
        return "; ".join(f"{key}: {_display(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple, np.ndarray)):
        return ", ".join(_display(item) for item in value)
    return str(value)


def summary_text(
    summary: Mapping,
    *,
    ld: Mapping[str, Any] | None = None,
    comparison: Mapping[str, Any] | None = None,
) -> str:
    """Return a short plain-text summary of an analysis.

    Parameters
    ----------
    summary : mapping
        Output of :func:`prusie.summarize`.
    ld : dict, optional
        Output of :func:`prusie.diagnose_ld`.
    comparison : dict, optional
        Output of :func:`prusie.compare_fits` or the ``comparison`` entry of
        :func:`prusie.run_sensitivity`.

    Returns
    -------
    str
        One line each for the counts of variants, credible sets and active
        effects, every credible set, the leading PIPs, IBSS stopping,
        refinement, LD diagnostics and, if given, sensitivity maxima.

    Notes
    -----
    The text is computed from the existing tables only; no model is fitted,
    no eigendecomposition is computed and plotting libraries are not
    imported. Values are rounded for display.

    Examples
    --------
    >>> print(summary_text(summarize(fit)))  # doctest: +SKIP
    """
    compact = analysis_summary(summary, ld=ld, comparison=comparison)
    results, fit, search = (compact[key] for key in ("results", "fit", "refinement"))
    lines = [
        (
            f"{results['variant_count']}"
            f" variants; {results['retained_cs']}"
            f" retained credible sets; "
            f"{results['active_components']}"
            f" active components."
        )
    ]
    for row in results["credible_sets"]:
        members = list(row.get("members", []))
        shown = ", ".join(map(str, members[:6]))
        if len(members) > 6:
            shown += f", … (+{len(members) - 6}; full members in credible_sets)"
        lines.append(
            f"CS component {row.get('component_id')}"
            f": {shown}"
            f"; size {row['size']}"
            f"; "
            f"coverage {_display(row.get('actual_coverage'), probability=True)}"
            f"; "
            f"effective variants {_display(row.get('effective_candidates'))}"
            f"."
        )
    candidates = results["candidates"][:5]
    if candidates:
        lines.append(
            "Leading PIP: "
            + "; ".join(
                (f"{row['variant_id']} {_display(row['pip'], probability=True)}")
                for row in candidates
            )
            + "."
        )
    requested = fit.get("requested_criterion", fit.get("criterion", "not_assessed"))
    effective = fit.get("effective_criterion", fit.get("criterion", "not_assessed"))
    lines.append(
        f"IBSS: {fit['stop_reason']}"
        f"; requested {requested}"
        f", effective {effective}"
        f"; "
        f"iterations {_display(fit.get('niter'))}"
        f"."
    )
    if search["status"] == "not_requested" or search["stop_reason"] == "not_requested":
        lines.append("Refinement: not requested.")
    else:
        lines.append(
            f"Refinement: {search['accepted_attempts']}"
            f"/{search['attempts']}"
            f" attempts accepted; "
            f"full-model ELBO "
            f"change {_display(search['improvement'])}"
            f"."
        )
    counts = compact["ld"]
    lines.append(
        "LD: not assessed."
        if ld is None
        else (
            f"LD: {counts['assessed']}"
            f" assessed, {counts['flagged']}"
            f" flagged; "
            f"{counts['conditional_only']}"
            f" conditional only; {counts['unassessed']}"
            f" unassessed."
        )
    )
    if comparison is not None:
        for row in compact["sensitivity"].get("regional_maxima", []):
            lines.append(
                (
                    f"Sensitivity {row['setting_a']}"
                    f" → {row['setting_b']}"
                    f": max |ΔPIP| "
                    f"{_display(row['max_abs_delta_pip'])}"
                    f" at {row['variant_id']}"
                )
                + (" (outside candidate view)." if not row["in_candidate_view"] else ".")
            )
    return "\n".join(lines)


def write_report(
    summary: Mapping,
    directory: str | Path,
    *,
    figures: Mapping[str, Any] | None = None,
    ld: Mapping[str, Any] | None = None,
    comparison: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    annotations: Mapping[str, Any] | None = None,
    title: str = "prusie analysis",
) -> dict:
    """Write an offline HTML report with full-precision exports.

    Parameters
    ----------
    summary : mapping
        Output of :func:`prusie.summarize`.
    directory : str or Path
        Output directory.
    figures : mapping, optional
        Labels mapped to Matplotlib figures or to the tuples returned by the
        functions in :mod:`prusie.viz`.
    ld, comparison : dict, optional
        LD diagnostics and a comparison of fits.
    provenance : mapping, optional
        Data checksums, harmonization, parameters and software versions.
        Fields not supplied are recorded as 'not_assessed'.
    annotations : mapping, optional
        Additional caller-supplied facts for the evidence export.
    title : str, default 'prusie analysis'
        Report title.

    Returns
    -------
    dict
        Absolute paths under ``html``, ``tables``, ``figures``,
        ``figure_index``, ``text_summary``, ``analysis_summary`` and
        ``evidence``; ``persistence`` holds path/checksum records for
        ``analysis.json`` and ``arrays.npz``. ``evidence_sha256`` is the
        evidence bundle digest.

    Notes
    -----
    Copy the whole directory to keep the report links valid. Reports use the
    supplied fit summaries and diagnostics.
    The full provenance is written to the evidence export; the page shows
    only a short description of the sources.
    """
    from ._style import REPORT_CSS

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    required = (
        "data_checksums",
        "variant_harmonization",
        "parameters",
        "initial_state",
        "reference_sources",
        "software_versions",
        "random_seed",
        "elapsed_seconds",
        "skipped",
    )
    recorded = {key: "not_assessed" for key in required}
    recorded.update(provenance or {})
    evidence = evidence_bundle(
        summary, ld=ld, comparison=comparison, provenance=recorded, annotations=annotations
    )
    compact = analysis_summary(summary, ld=ld, comparison=comparison)
    compact["evidence_sha256"] = evidence["sha256"]
    compact["files"] = {
        "evidence": "evidence.json",
        "lossless_structure": "analysis.json",
        "lossless_arrays": "arrays.npz",
        "tables": "tables/",
        "figures": "figure_index.json",
        "text_summary": "summary.txt",
    }
    artifacts = save_analysis(
        {
            "summary": summary,
            "ld": ld,
            "comparison": comparison,
            "provenance": recorded,
            "annotations": annotations,
        },
        directory,
    )
    for name, value in (("evidence", evidence), ("analysis_summary", compact)):
        (directory / f"{name}.json").write_text(
            json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8"
        )
    (directory / "summary.txt").write_text(
        summary_text(summary, ld=ld, comparison=comparison) + "\n", encoding="utf-8"
    )
    tables = export_tables(summary, directory / "tables")
    extra_tables = {}
    if comparison is not None:
        from .comparison import comparison_relations

        extra_tables.update(
            comparison_variants=comparison.get("variants", []),
            signal_relations=comparison_relations(comparison, baseline_only=False),
            component_relations=comparison_relations(
                comparison, layer="components", baseline_only=False
            ),
        )
    if ld is not None:
        extra_tables["ld_variants"] = ld.get("variants", [])
    attempts, stages = [], []
    for index, row in enumerate((summary.get("refinement") or {}).get("history", []), 1):
        attempt = _select(
            row,
            (
                "round",
                "component",
                "excluded_ids",
                "accepted",
                "reason",
                "incumbent_objective",
                "objective",
                "improvement",
            ),
        )
        for phase, stage in row.get("stages", {}).items():
            attempt[f"{phase}_stop"] = stage.get("stop_reason")
            attempt[f"{phase}_iterations"] = stage.get("niter")
            attempt[f"{phase}_converged"] = stage.get("ibss_converged")
            # Copy every recorded stage field as is; records written without
            # solver settings or timings are not completed with defaults.
            stages.append(
                dict(
                    _plain(stage),
                    attempt=index,
                    stage=phase,
                    **_select(row, ("round", "component", "excluded_ids")),
                )
            )
        attempts.append(attempt)
    extra_tables["refinement_attempts"] = attempts
    extra_tables["refinement_stages"] = stages
    tables.extend(export_tables(extra_tables, directory / "tables"))

    def escape(value):
        return html.escape(str(value), quote=True)

    sections = [f"<header><h1>{escape(title)}</h1>"]
    supplied = dict(provenance or {}) | dict(annotations or {})
    identity = {
        key: supplied[key] for key in ("trait", "gene", "locus") if supplied.get(key) is not None
    }
    if identity:
        sections.append('<p class="analysis-input">' + escape(_display(identity)) + "</p>")
    sections.append(
        '<p class="lede">Read the candidate sets, then check '
        "the fitted solution, search and input "
        "compatibility.</p></header>"
    )
    navigation = (
        ("main-results", "Candidates & resolution"),
        ("fit-status", "Analysis status"),
        ("figures", "Figures"),
        ("sensitivity-ld", "Sensitivity & LD details"),
        ("details", "Complete records"),
        ("exports", "Downloads"),
    )
    sections.append(
        '<nav aria-label="On this page"><ul>'
        + "".join(f'<li><a href="#{key}">{escape(label)}</a></li>' for key, label in navigation)
        + "</ul></nav><main>"
    )
    probabilities = {
        "pip",
        "top_alpha",
        "actual_coverage",
        "requested_coverage",
        "biological_mass",
        "null_mass",
    }
    labels = {
        "variant_id": "Variant ID",
        "pip": "PIP",
        "size": "CS size",
        "representative": "Representative",
        "component_id": "Component ID",
        "actual_coverage": "Posterior mass",
        "requested_coverage": "CS target",
        "effective_candidates": "Effective number of variants",
        "min_abs_corr": "Minimum |r|",
        "cs_components": "CS components",
        "selection_sources": "Display selection",
        "max_abs_delta_pip": "Maximum |ΔPIP|",
        "before_objective": "ELBO before refinement",
        "after_objective": "ELBO after refinement",
    }

    def cell(value, key):
        if key == "selection_sources" and isinstance(value, (list, tuple)):
            display_names = {
                "formal_CS_member": "CS member",
                "top_PIP": "top PIP",
                "PIP_threshold": "PIP threshold",
            }
            value = [display_names.get(item, item) for item in value]
        if key == "cs_components" and isinstance(value, (list, tuple)) and not value:
            return "<td>None</td>"  # Known nonmembership; unavailable data remain NA.
        numeric = isinstance(value, (float, int, np.number)) and not isinstance(
            value, (bool, np.bool_)
        )
        attrs = ' class="numeric"' if numeric else ""
        if numeric and key in probabilities:
            attrs += ' title="Full precision: ' + escape(repr(float(value))) + '"'
        return (
            "<td"
            + attrs
            + ">"
            + escape(_display(value, probability=key in probabilities))
            + "</td>"
        )

    def table(rows, columns=None, caption=None):
        rows = list(rows)
        if not rows:
            sections.append('<p class="not-recorded">No records.</p>')
            return
        columns = columns or list(dict.fromkeys(key for row in rows for key in row))
        sections.append(
            '<div class="table-scroll" role="region" aria-label="'
            + escape(caption or "Data table")
            + '" tabindex="0"><table>'
            + ("<caption>" + escape(caption) + "</caption>" if caption else "")
            + "<thead><tr>"
            + "".join(
                '<th scope="col">' + escape(labels.get(key, str(key).replace("_", " "))) + "</th>"
                for key in columns
            )
            + "</tr></thead><tbody>"
        )
        for row in rows:
            sections.append("<tr>" + "".join(cell(row.get(key), key) for key in columns) + "</tr>")
        sections.append("</tbody></table></div>")

    def details(label, value, columns=None):
        sections.append("<details><summary>" + escape(label) + "</summary>")
        if isinstance(value, (list, tuple)) and all(isinstance(row, Mapping) for row in value):
            table(value, columns)
        else:
            sections.append(
                "<pre>"
                + escape(json.dumps(_plain(value), ensure_ascii=False, allow_nan=False, indent=2))
                + "</pre>"
            )
        sections.append("</details>")

    results, control, search = (compact[key] for key in ("results", "fit", "refinement"))
    search_requested = (
        search["status"] != "not_requested" and search["stop_reason"] != "not_requested"
    )
    sections.append(
        '<section aria-labelledby="main-results"><h2 id="main-results">'
        "Main results and resolution</h2>"
        f"<p><strong>{results['retained_cs']}"
        f" retained credible sets</strong> among "
        f"{results['variant_count']}"
        f" analyzed variants; {results['active_components']}"
        f" active components. "
        "A retained CS is a posterior candidate set, not a confirmed causal effect count.</p>"
    )
    table(
        results["credible_sets"],
        (
            "component_id",
            "representative",
            "size",
            "actual_coverage",
            "min_abs_corr",
            "effective_candidates",
        ),
    )
    sections.append(
        '<p class="caption">Smaller credible sets and effective '
        "candidate counts indicate more concentrated posterior "
        "support. LD purity describes correlations within a "
        "set. Neither measure identifies a biological "
        "mechanism.</p>"
    )
    sections.append(
        "<h3>Main candidates</h3><p>All formal CS members, the "
        "top ten variants by PIP, and all PIP ≥ 0.1 candidates "
        "are included in the display selection. Missing or "
        "excluded variants are not assigned zero PIP. Numbers "
        "use four significant digits; near-endpoint "
        "probabilities retain a bound or scientific notation. "
        "Hover over probability cells or use the full-precision "
        "exports for exact values.</p>"
    )
    candidates = results["candidates"]
    candidate_columns = ("variant_id", "pip", "cs_components", "selection_sources")
    table(candidates if len(candidates) <= 30 else candidates[:20], candidate_columns)
    if len(candidates) > 30:
        sections.append(
            f"<p>Showing the first 20 of {len(candidates)}"
            f" selected candidates in PIP order. "
            "The complete selection follows; no CS member was removed from it.</p>"
        )
        details(f"All {len(candidates)} selected candidates", candidates, candidate_columns)
    details("Complete CS membership and coverage", results["credible_sets"])
    sections.append(
        '</section><section aria-labelledby="fit-status"><h2 id="fit-status">Analysis status</h2>'
    )
    sections.append("<h3>IBSS stopping</h3><p>" + escape(control["pending"]) + "</p>")
    table(
        [
            _select(
                control,
                (
                    "requested_criterion",
                    "effective_criterion",
                    "niter",
                    "stop_reason",
                    "tol",
                    "alpha_tol",
                    "pip_tol",
                    "elbo_decrease_count",
                    "nonfinite_elbo_count",
                ),
            )
        ]
    )
    if control["last"]:
        table(
            [
                _select(
                    control["last"],
                    ("elbo", "delta_elbo", "max_abs_delta_alpha", "max_abs_delta_pip"),
                )
            ],
            caption=("Last recorded iteration; logged ELBO precedes finalization"),
        )
    if not control["trajectory_available"]:
        sections.append(
            "<p>No iteration trajectory is available in this "
            "summary; final diagnostics alone cannot reconstruct "
            "the path taken by IBSS.</p>"
        )
    sections.append("<h3>Refinement search</h3>")
    if not search_requested:
        sections.append("<p>Refinement: not requested.</p>")
    else:
        sections.append(
            "<p>Refinement: "
            + escape(_display(search["status"]))
            + ". "
            + str(search["attempts"])
            + " exclusion attempts; "
            + str(search["accepted_attempts"])
            + " accepted; "
            + escape(_display(search["inner_fits"]))
            + (
                " inner fits. The initial stopping condition and "
                "finding a better solution concern different "
                "questions.</p>"
            )
        )
    if search_requested:
        table(
            [_select(search, ("stop_reason", "before_objective", "after_objective", "improvement"))]
        )
        sections.append(
            '<p class="caption">Improvement compares full-model ELBO under the '
            "original prior, evaluated after zeroing components with "
            "V &lt; prior_tol (the post-trim state). A negative rejected candidate does "
            "not mean the accepted solution regressed. Logged "
            "pre-finalization ELBO remains separate in the full "
            "search record.</p>"
        )
    sensitivity, counts = compact["sensitivity"], compact["ld"]
    sections.append("<h3>Sensitivity</h3>")
    if comparison is None:
        sections.append("<p>Sensitivity was not assessed.</p>")
    else:
        sections.append(
            "<p>Baseline: <strong>"
            + escape(sensitivity["baseline"])
            + "</strong>. "
            + str(len(sensitivity["settings"]))
            + " settings; CS-supported relation counts: "
            + escape(_display(sensitivity["relation_counts"]))
            + ". "
            "Each fit or summary setting is shown separately.</p>"
        )
        table(
            sensitivity.get("regional_maxima", []),
            ("setting_b", "variant_id", "max_abs_delta_pip", "in_candidate_view"),
        )
        for row in sensitivity.get("regional_maxima", []):
            if not row["in_candidate_view"]:
                sections.append(
                    "<p><strong>Outside the candidate view:</strong> variant "
                    + escape(row["variant_id"])
                    + " has the largest |ΔPIP| ("
                    + escape(_display(row["max_abs_delta_pip"]))
                    + ") for "
                    + escape(row["setting_b"])
                    + (
                        '. See <a href="tables/comparison_variants.csv">the '
                        "full variant table</a> and <code>"
                    )
                    + escape(row["evidence_pointer"])
                    + "</code> in evidence.json.</p>"
                )
    sections.append("<h3>LD compatibility</h3>")
    if ld is None:
        sections.append("<p>LD diagnostics were not assessed.</p>")
    else:
        sections.append(
            "<p>LD flip assessment: "
            + str(counts["assessed"])
            + " assessed ("
            + str(counts["flagged"])
            + " flagged; "
            + str(counts["not_flagged"])
            + " assessed and not flagged); "
            + str(counts["conditional_only"])
            + " conditional-only; "
            + str(counts["unassessed"])
            + " unassessed.</p>"
        )
        if not counts["assessed"]:
            sections.append("<p>No completed flip assessment is available.</p>")
        elif not counts["flagged"]:
            sections.append(
                "<p>No variants were flagged among completed "
                "assessments. This does not establish that the LD "
                "reference is correct.</p>"
            )
        if counts.get("mixture"):
            table([counts["mixture"]], caption="Mixture solver status")
    sections.append(
        '</section><section aria-labelledby="figures"><h2 '
        'id="figures">Figures and before/after evidence</h2>'
    )
    figure_paths, figure_index, figure_groups, plot_data = [], [], {}, {}
    figure_responsive_rules = []
    for index, (name, value) in enumerate((figures or {}).items()):
        figure = value[0] if isinstance(value, tuple) else value
        stem = f"figure_{index + 1}"
        lower = str(name).lower()
        group = (
            "variance"
            if "variance detail" in lower
            else "fit"
            if any(term in lower for term in ("convergence", "refinement"))
            else "sensitivity"
            if "sensitivity" in lower
            else "ld"
            if "ld" in lower and "signal" not in lower
            else "signal"
            if any(term in lower for term in ("signal", "prior"))
            else "locus"
        )
        import matplotlib as mpl

        # SVG serialization happens after plotting's local style context exits.
        with mpl.rc_context({"svg.hashsalt": "prusie-offline-report-v1", "svg.fonttype": "none"}):
            for suffix in ("svg", "png"):
                path = directory / f"{stem}.{suffix}"
                metadata = {"Date": None} if suffix == "svg" else {"Software": "prusie"}
                figure.savefig(
                    path,
                    bbox_inches="tight",
                    metadata=metadata,
                    **({"dpi": 150} if suffix == "png" else {}),
                )
                figure_paths.append(str(path.resolve()))
        # Use the intended display size, not the raster's export resolution.
        # Only wide multi-panel figures retain a local scroll area on phones.
        display_width = round(float(figure.get_figwidth()) * 96)
        axes_count = (
            len(np.asarray(value[1], dtype=object).ravel())
            if isinstance(value, tuple)
            else len(figure.axes)
        )
        complex_figure = axes_count >= 2 and group != "sensitivity" and display_width >= 640
        minimum_width = min(display_width, 720) if complex_figure else 0
        figure_index.append(
            dict(
                title=str(name),
                png=f"{stem}.png",
                svg=f"{stem}.svg",
                section=group,
                display_width_px=display_width,
                minimum_width_px=minimum_width,
            )
        )
        if isinstance(value, tuple) and len(value) > 2:
            plot_data[str(name)] = _plain(value[2])
        captions = {
            "locus": (
                "All analyzed variants are retained. Physical "
                "coordinates are used only when supplied; otherwise the "
                "x axis is variant index."
            ),
            "fit": (
                "The inner stopping condition and improvement after "
                "search answer different questions. Refinement "
                "improvements use the full-model ELBO defined above; "
                "the iteration log records ELBO before finalization."
            ),
            "variance": (
                "Prior-variance and residual-variance paths describe "
                "solver states; full component records remain in the "
                "exports."
            ),
            "signal": (
                "CS members and explicitly selected additional "
                "candidates describe localization. Component "
                "probabilities and Bayes factors are conditional on the "
                "other fitted components."
            ),
            "sensitivity": (
                "Each column shows its recorded fit or summary setting, "
                "with baseline first. Missing cells are unobserved "
                "variants, not zero PIP. The full variant table also "
                "identifies changes outside the candidate view."
            ),
            "ld": (
                "Flags use the joint recorded likelihood-ratio and "
                "absolute-z rule. Extreme residuals alone are not "
                "allele-flip flags; conditional-only and unassessed "
                "states stay separate."
            ),
        }
        caption = captions[group]
        if group == "sensitivity" and "variant_ids" in plot_data.get(str(name), {}):
            shown = plot_data[str(name)]
            selected = shown.get("variant_ids", [])
            caption += (
                f" The heatmap shows {len(selected)}"
                f" selected variants; full-precision "
                f"values are in plot_data.json "
                f"and the variant table."
            )
        if complex_figure:
            figure_responsive_rules.append(
                f"@container (max-width:{minimum_width - 1}"
                f"px){{#{stem}"
                f" .figure-scroll-hint{{display:block}}}}"
            )
        encoded = base64.b64encode((directory / f"{stem}.png").read_bytes()).decode("ascii")
        figure_groups.setdefault(group, []).append(
            (
                f'<figure id="{stem}'
                f'" class="figure-frame" style="--figure-width:{display_width}'
                f"px;"
                f"--figure-min-width:{minimum_width}"
                f'px"><div class="figure-scroll" role="region" '
                f'aria-label="{escape(name)}'
                f'" tabindex="0"><img alt="{escape(name)}'
                f'" src="data:image/png;base64,{encoded}'
                f'" '
                f'loading="lazy"></div>'
            )
            + (
                (
                    '<p class="figure-scroll-hint">Scroll the figure '
                    "horizontally, or open the original below.</p>"
                )
                if complex_figure
                else ""
            )
            + (
                f"<figcaption>{escape(name)}"
                f' · <a href="{stem}'
                f'.png">Original PNG</a> · '
                f'<a href="{stem}'
                f'.svg">SVG</a><br>{escape(caption)}'
                f"</figcaption></figure>"
            )
        )
    (directory / "figure_index.json").write_text(
        json.dumps(figure_index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if plot_data:
        (directory / "plot_data.json").write_text(
            json.dumps(plot_data, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
            encoding="utf-8",
        )
    questions = (
        ("locus", "Where is the posterior support?"),
        ("fit", "Did fitting stop, and did search improve the solution?"),
        ("variance", "Variance details"),
        ("signal", "How well are individual signals resolved?"),
        ("sensitivity", "Which conclusions change across settings?"),
        ("ld", "Are the statistics compatible with the supplied LD?"),
    )
    for group, question in questions:
        if group not in figure_groups:
            continue
        sections.append(
            "<details><summary>" + question + "</summary>"
            if group == "variance"
            else "<h3>" + question + "</h3>"
        )
        sections.extend(figure_groups[group])
        if group == "variance":
            sections.append("</details>")
    if not figure_paths:
        sections.append("<p>No figures were supplied.</p>")
    sections.append(
        "</section><section "
        'aria-labelledby="sensitivity-ld"><h2 '
        'id="sensitivity-ld">Sensitivity and LD diagnostics</h2>'
    )
    if comparison is not None:
        sections.append(
            ("<p>Main comparisons use CS-supported signals against baseline ")
            + escape(sensitivity["baseline"])
            + (
                ". Active diffuse components remain in the complete "
                "component view; posterior overlap alone does not "
                "confirm a localized split or merge. Missing settings "
                "indicate variants outside that fit, not low PIP.</p>"
            )
        )
        details(
            "Baseline comparisons and support policy",
            {
                "mode": sensitivity.get("mode"),
                "support": sensitivity.get("support"),
                "pairs": sensitivity["pairs"],
            },
        )
        details(
            "Baseline CS-supported signal relations",
            comparison_relations(comparison, baseline_only=True),
        )
    if ld is not None:
        sections.append(
            "<p>A finite conditional residual is not a completed "
            "flip assessment. Missing-z working zeros are not "
            "observed z statistics.</p>"
        )
        table(
            [_select(counts, ("method", "regularization", "thresholds"))],
            caption="Recorded LD method and joint flag rule",
        )
        from .ld import diagnostic_input_missing, diagnostic_status

        needs_review = []
        for row in ld.get("variants", []):
            if diagnostic_status(row) == "not_flagged":
                continue
            displayed = dict(row, assessment=diagnostic_status(row))
            if diagnostic_input_missing(row):
                displayed.update(observed_z=None, standardized_residual=None)
            needs_review.append(displayed)
        if needs_review:
            details(
                "Flagged, conditional-only and unassessed variants",
                needs_review,
                (
                    "variant_id",
                    "assessment",
                    "observed_z",
                    "expected_z",
                    "standardized_residual",
                    "log_likelihood_ratio",
                    "flag",
                    "unavailable_reason",
                ),
            )
        if counts.get("warnings"):
            details("LD diagnostic warnings", counts["warnings"])
    sections.append(
        '</section><section aria-labelledby="details"><h2 '
        'id="details">Complete records and provenance</h2>'
    )
    details("Complete stopping controls and finalization", control)
    if search_requested:
        details("Refinement objective, logged ELBO and controls", search)
        accepted = [row for row in attempts if row.get("accepted")]
        if accepted:
            details("Accepted attempts", accepted)
        details("Complete refinement attempts and stage stops", attempts)
        if stages:
            sections.append(
                "<details><summary>Complete A/B stage objectives and "
                "stopping records</summary><p>Stage A changes the prior "
                "to exclude a CS; its objective is not comparable to "
                "stage B. Stage B restores the original prior. Each "
                "recomputed objective and logged sweep ELBO is kept in "
                "a separate column. <a "
                'href="tables/refinement_stages.csv">Download all '
                "recorded stage fields</a>; fields absent from the "
                "original analysis remain unavailable.</p>"
            )
            table(
                stages,
                (
                    "attempt",
                    "round",
                    "component",
                    "excluded_ids",
                    "stage",
                    "objective",
                    "logged_elbo",
                    "niter",
                    "stop_reason",
                    "ibss_converged",
                    "error",
                ),
            )
            sections.append("</details>")
    details(
        "All components, including inactive and diffuse components",
        summary.get("components", []),
        (
            "component_id",
            "active",
            "cs_retained",
            "V",
            "top_variant",
            "top_alpha",
            "effective_candidates",
            "biological_mass",
            "null_mass",
            "component_log_bf",
        ),
    )
    sections.append(
        "<p>Complete component rankings and alpha arrays are in "
        '<a href="evidence.json">evidence.json</a> under '
        "<code>data.summary.components</code>.</p>"
    )
    for label, name in (
        ("CS audit", "cs_audit"),
        ("CS-outside direct proxies", "proxies"),
        ("Between-CS genotype correlations", "cs_relations"),
    ):
        details(label, summary.get(name, []))
    if comparison is not None:
        details(
            "All setting pairs: CS-supported relations",
            comparison_relations(comparison, baseline_only=False),
        )
        details(
            "All active-component relations, including diffuse overlap",
            comparison_relations(comparison, layer="components", baseline_only=False),
        )
        details(
            "Complete parameters for each sensitivity setting", comparison.get("fit_settings", {})
        )
    details("Fit parameters", recorded.get("parameters"))
    # Keep long data hashes, server paths and process details in the supplied
    # provenance export, not in the report body. Never invent input metadata.
    sources = _select(
        recorded,
        (
            "numeric_source_sha",
            "render_source_sha",
            "render_version",
            "software_versions",
            "reference_sources",
            "random_seed",
        ),
    )
    details("Source and software identity", sources)
    sections.append(
        "<p>Full source records, checksums and supplied "
        "annotations: <code>data.provenance</code> and "
        "<code>data.annotations</code> in <a "
        'href="evidence.json">evidence.json</a>.</p>'
    )
    details(
        "Input QC and warnings",
        {"input_qc": summary.get("input_qc", {}), "warnings": summary.get("warnings", [])},
    )
    sections.append(
        "<details><summary>Methods and interpretation</summary><p>"
        + escape(_METHODS)
        + "</p></details>"
    )
    sections.append(
        (
            '</section><section aria-labelledby="exports"><h2 '
            'id="exports">Machine-readable evidence '
            "references</h2><p>Start with <a "
            'href="analysis_summary.json">analysis_summary.json</a> '
            'or <a href="summary.txt">summary.txt</a>; query <a '
            'href="evidence.json">evidence.json</a> using data '
            "paths or stable fact IDs. All "
        )
        + str(len(evidence["facts"]))
        + (
            " scalar facts remain in that file and are not "
            "duplicated into this page. No browser scripts or "
            "remote resources are required.</p><p>Lossless saved "
            'structure: <a href="analysis.json">analysis.json</a> '
            'with <a href="arrays.npz">arrays.npz</a>. Figure '
            "mapping: <a "
            'href="figure_index.json">figure_index.json</a>. Open '
            "index.html locally and keep its directory together to "
            "retain export links. GitHub file pages do not execute "
            "this report.</p>"
        )
    )
    if plot_data:
        sections.append(
            "<p>Full plotted values and selection records: <a "
            'href="plot_data.json">plot_data.json</a>.</p>'
        )
    sections.append(
        '<ul class="downloads">'
        + "".join(
            '<li><a href="tables/'
            + escape(Path(path).name)
            + '">'
            + escape(Path(path).name)
            + "</a></li>"
            for path in tables
        )
        + "</ul></section></main>"
    )
    css = (
        REPORT_CSS
        + """
    nav ul,.downloads{display:flex;flex-wrap:wrap;gap:.5rem 1.25rem;list-style:none;padding:0}
    header{padding-bottom:.5rem}nav{border-bottom:1px solid #dde2e6;padding:.25rem 0 .75rem}
    section{margin:2.2rem 0}h3{margin-top:1.6rem}.numeric{text-align:right;font-variant-numeric:tabular-nums}
    .table-scroll{max-width:100%;overflow:auto}th{white-space:normal}td{overflow-wrap:anywhere}
    caption{text-align:left;font-weight:600;margin:.5rem 0}figure{margin:1.4rem 0 2rem}
    img{display:block;height:auto}figcaption,.caption{font-size:.92rem;color:#52606d}
    .figure-frame{container-type:inline-size}.figure-scroll{max-width:100%;overflow-x:auto}
    .figure-scroll img{width:min(100%,var(--figure-width));min-width:var(--figure-min-width);max-width:none}
    .figure-scroll-hint{display:none;font-size:.85rem;color:#52606d;margin:.35rem 0}
    details{border-top:1px solid #e6e9ec;padding:.65rem 0}summary{cursor:pointer;font-weight:600}
    pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:34rem;overflow:auto}
    @media(max-width:800px){body{padding:1rem}table{font-size:.9rem}th,td{padding:.4rem}h1{font-size:1.7rem}}
    @media print{body{max-width:none;padding:0;font-size:11pt}nav{display:none}img{max-height:24cm;object-fit:contain}
    figure{break-inside:avoid}.table-scroll{overflow:visible}details{break-inside:avoid}a{color:inherit}
    .figure-scroll{overflow:visible}.figure-scroll img{min-width:0;max-width:100%}.figure-scroll-hint{display:none!important}}
    """  # noqa: E501 - Preserve rendered HTML/CSS bytes.
        + "\n".join(figure_responsive_rules)
    )
    content = (
        (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
            "img-src data:; style-src 'unsafe-inline'\">"
            f"<title>{escape(title)}"
            f"</title><style>{css}"
            f"</style></head><body>"
        )
        + "\n".join(sections)
        + "</body></html>"
    )
    path = directory / "index.html"
    path.write_text(content, encoding="utf-8")
    return {
        "html": str(path.resolve()),
        "persistence": artifacts,
        "tables": tables,
        "figures": figure_paths,
        "evidence_sha256": evidence["sha256"],
        "figure_index": str((directory / "figure_index.json").resolve()),
        "text_summary": str((directory / "summary.txt").resolve()),
        "analysis_summary": str((directory / "analysis_summary.json").resolve()),
        "evidence": str((directory / "evidence.json").resolve()),
    }
