"""Export data-safe complete-panel records from the 0.3.0 measurement archive.

This module reads retained measurements. It never fits, filters by convergence,
or substitutes historical R timings. Private input and installation paths are
intentionally absent from the exported schema.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import statistics
import zipfile
from collections import Counter
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def distribution(values):
    values = sorted(v for v in values if finite(v))
    if not values:
        return dict(n=0, minimum=None, q1=None, median=None, q3=None, maximum=None)

    # Linear interpolation agrees with numpy.quantile(method="linear").
    def quantile(p):
        position = (len(values) - 1) * p
        lo = math.floor(position)
        hi = math.ceil(position)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)

    return dict(
        n=len(values),
        minimum=values[0],
        q1=quantile(0.25),
        median=quantile(0.5),
        q3=quantile(0.75),
        maximum=values[-1],
    )


def expected_tasks(rows):
    """The panel and applicability rules were fixed before candidate outcomes."""
    for row in rows:
        yield row, "primary", 1, False, None
        if row["panel"] == "GWAS" or row["p"] >= 1024:
            for count in (2, 4, 8):
                yield row, f"threads{count}", count, False, None
        if row["panel"] == "GWAS":
            yield row, "overwrite", 1, True, None
    largest = max((r for r in rows if r["panel"] == "GWAS"), key=lambda r: (r["p"], r["case_id"]))
    for p in dict.fromkeys(
        p for p in (500, 1000, 2000, 4000, 10000, largest["p"]) if p <= largest["p"]
    ):
        yield largest, f"window{p}", 1, False, p


def backend(directory, name):
    time_path = directory / f"{name}_time.json"
    memory_path = directory / f"{name}_memory.json"
    timing = read(time_path) if time_path.exists() else {}
    memory = read(memory_path) if memory_path.exists() else {}
    retained = [s for s in timing.get("samples", []) if s.get("warmup") is False]
    times = [s.get("time_seconds") for s in retained]
    complete = (
        timing.get("status") == "ok" and len(times) == 7 and all(finite(v) and v > 0 for v in times)
    )
    stats = distribution(times)
    peak = memory.get("peak_rss_bytes")
    memory_complete = memory.get("status") == "ok" and finite(peak) and peak > 0
    samples = retained or memory.get("samples", [])
    return dict(
        time_complete=complete,
        memory_complete=memory_complete,
        retained_seconds=times,
        runtime=stats,
        repeat_iqr_over_median=(
            (stats["q3"] - stats["q1"]) / stats["median"] if complete else None
        ),
        peak_rss_bytes=peak if memory_complete else None,
        niter=[s.get("niter") for s in samples],
        converged=[s.get("converged") for s in samples],
        cs_count=[s.get("cs_count") for s in samples],
        time_status=timing.get("status", "missing"),
        memory_status=memory.get("status", "missing"),
        runtime_identity={
            "versions": timing.get("runtime_versions"),
            "R": timing.get("environment", {}).get("R"),
            "R_blas": {
                k: timing.get("environment", {}).get("blas_probe", {}).get(k)
                for k in ("config", "threads")
            },
            "numerical_libraries": [
                dict(name=Path(item["path"]).name, sha256=item["sha256"], threads=item["threads"])
                for item in timing.get("numerical_libraries", [])
            ],
        },
    )


def measurement_freeze(run):
    """Use the immutable measured build when a later opt-in feature is qualified."""
    retained = run / "status/measurement_freeze.json"
    return read(retained if retained.exists() else run / "status/code_freeze.json")


def load_task(run, row, kind, count, overwrite, window):
    freeze = measurement_freeze(run)
    root = Path(freeze.get("m2_output_root", str(run / "measurements/M2")))
    assert root.resolve().is_relative_to(run / "measurements")
    directory = root / row["panel"] / row["case_id"] / kind
    result_path = directory / "result.json"
    result = read(result_path) if result_path.exists() else {}
    if result:
        request = read(directory / "request.json")
        unsigned = {k: v for k, v in request.items() if k != "signature_sha256"}
        assert (
            hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest()
            == request["signature_sha256"]
        )
        assert result["signature_sha256"] == request["signature_sha256"]
        assert request["identity"]["freeze"] == freeze
        task = result["task"]
        assert (
            task["panel"],
            task["case_id"],
            task["kind"],
            task["n_threads"],
            task["overwrite"],
            task["window"],
        ) == (row["panel"], row["case_id"], kind, count, overwrite, window)
        assert task["parameters"] == row["parameters"]
        # Check each retained raw JSON before using its measurements. Binary
        # validation outputs remain in the private scientific archive.
        outputs = result.get("outputs", {})
        for path in directory.glob("*.json"):
            if str(path) in outputs:
                assert digest(path) == outputs[str(path)], f"Changed raw output: {path.name}"
            if path.stem in {
                f"{b}_{m}" for b in ("R", "prusie") for m in ("validate", "time", "memory")
            }:
                assert read(path)["signature_sha256"] == request["signature_sha256"]
    raw_native = []
    hosts = [result.get("host_start", {}), result.get("host_end", {})]
    for command in result.get("commands", []):
        hosts.extend((command.get("host_start", {}), command.get("host_end", {})))
    for mode in ("validate", "time", "memory"):
        file = directory / f"prusie_{mode}.json"
        if file.exists():
            raw_native.extend(read(file).get("samples", []))
    errors = {
        key: max((s[key] for s in raw_native if finite(s.get(key))), default=None)
        for key in ("max_abs_alpha_error", "max_abs_pip_error")
    }
    reference = result.get("fresh_R_integrity") or {}
    native = backend(directory, "prusie")
    reference_backend = backend(directory, "R") if kind == "primary" or window is not None else None
    paired = bool(
        reference_backend and native["time_complete"] and reference_backend["time_complete"]
    )
    paired_memory = bool(
        reference_backend and native["memory_complete"] and reference_backend["memory_complete"]
    )
    return dict(
        case_id=row["case_id"],
        panel=row["panel"],
        source_index=row["index"],
        kind=kind,
        p=window or row["p"],
        full_p=row["p"],
        n=row["parameters"]["n"],
        max_iter=row["parameters"]["max_iter"],
        n_threads=count,
        overwrite_ld=overwrite,
        task_state=result.get("state", "MISSING"),
        result_sha256=digest(result_path) if result_path.exists() else None,
        hardware=request["identity"].get("hardware") if result else None,
        signature_sha256=result.get("signature_sha256"),
        measurement_software_sha256=(
            {Path(path).name: value for path, value in request["identity"]["scripts"].items()}
            if result
            else None
        ),
        alpha_pass=result.get("alpha_pass"),
        validity_failure_count=len(result.get("validity_failures", [])),
        **errors,
        nonfinite_error_samples=sum(
            not finite(s.get("max_abs_alpha_error")) or not finite(s.get("max_abs_pip_error"))
            for s in raw_native
        ),
        fresh_R_integrity={
            k: reference.get(k)
            for k in (
                "available",
                "alpha_bitwise_equal",
                "alpha_values_equal",
                "max_abs_alpha_error",
            )
        },
        timing_pair=paired,
        memory_pair=paired_memory,
        speedup=(
            reference_backend["runtime"]["median"] / native["runtime"]["median"] if paired else None
        ),
        memory_ratio=(
            reference_backend["peak_rss_bytes"] / native["peak_rss_bytes"]
            if paired_memory
            else None
        ),
        prusie=native,
        susieR=reference_backend,
        host_load_1min=distribution(h["load_average"][0] for h in hosts if h.get("load_average")),
        host_memory_occupancy=distribution(h.get("occupancy") for h in hosts),
    )


def group_summary(rows):
    paired = [r for r in rows if r["timing_pair"]]
    sums = {b: sum(sum(r[b]["retained_seconds"]) for r in paired) for b in ("prusie", "susieR")}
    medians = {b: sum(r[b]["runtime"]["median"] for r in paired) for b in ("prusie", "susieR")}
    unavailable = [r["case_id"] for r in rows if r["alpha_pass"] is None]
    return dict(
        cases=len(rows),
        runtime_identities={
            b: [
                json.loads(value) | {"cases": count}
                for value, count in Counter(
                    json.dumps(r[b]["runtime_identity"], sort_keys=True) for r in rows if r[b]
                ).items()
            ]
            for b in ("prusie", "susieR")
        },
        p_range=[min(r["p"] for r in rows), max(r["p"] for r in rows)],
        task_states=dict(Counter(r["task_state"] for r in rows)),
        timing_pairs=len(paired),
        memory_pairs=sum(r["memory_pair"] for r in rows),
        retained_time_totals_seconds=sums,
        total_time_speedup=sums["susieR"] / sums["prusie"] if paired else None,
        sum_case_medians_speedup=medians["susieR"] / medians["prusie"] if paired else None,
        geometric_mean_speedup=math.exp(statistics.mean(math.log(r["speedup"]) for r in paired))
        if paired
        else None,
        wins=sum(r["speedup"] > 1 for r in paired),
        ties=sum(r["speedup"] == 1 for r in paired),
        regressions=sum(r["speedup"] < 1 for r in paired),
        alpha_passes=sum(r["alpha_pass"] is True for r in rows),
        alpha_failures=[r["case_id"] for r in rows if r["alpha_pass"] is False],
        alpha_unavailable=unavailable,
        operational_failures=[r["case_id"] for r in rows if r["task_state"] != "COMPLETE"],
        frozen_R_deviations=[
            r["case_id"]
            for r in rows
            if r["fresh_R_integrity"]["available"]
            and r["fresh_R_integrity"]["alpha_bitwise_equal"] is not True
        ],
        frozen_R_unavailable=[
            r["case_id"] for r in rows if not r["fresh_R_integrity"]["available"]
        ],
        frozen_R_max_abs_alpha_error=max(
            (
                r["fresh_R_integrity"]["max_abs_alpha_error"]
                for r in rows
                if finite(r["fresh_R_integrity"]["max_abs_alpha_error"])
            ),
            default=None,
        ),
        max_abs_alpha_error=max(
            (r["max_abs_alpha_error"] for r in rows if finite(r["max_abs_alpha_error"])),
            default=None,
        ),
        max_abs_pip_error=max(
            (r["max_abs_pip_error"] for r in rows if finite(r["max_abs_pip_error"])), default=None
        ),
        speedup_distribution=distribution(r["speedup"] for r in paired),
        memory_ratio_distribution=distribution(r["memory_ratio"] for r in rows),
        host_load_1min_range=distribution(
            v
            for r in rows
            for v in (r["host_load_1min"]["minimum"], r["host_load_1min"]["maximum"])
        ),
        host_memory_occupancy_range=distribution(
            v
            for r in rows
            for v in (r["host_memory_occupancy"]["minimum"], r["host_memory_occupancy"]["maximum"])
        ),
        backends={
            b: dict(
                runtime_seconds=distribution(r[b]["runtime"]["median"] for r in paired),
                repeat_iqr_over_median=distribution(r[b]["repeat_iqr_over_median"] for r in paired),
                full_process_peak_mib=distribution(
                    r[b]["peak_rss_bytes"] / 2**20
                    for r in rows
                    if r.get(b) and r[b]["memory_complete"]
                ),
                iteration_limit_cases=[
                    r["case_id"]
                    for r in rows
                    if r.get(b)
                    and any(
                        n == r["max_iter"] and c is False
                        for n, c in zip(r[b]["niter"], r[b]["converged"], strict=True)
                    )
                ],
                unconverged_cases=[
                    r["case_id"]
                    for r in rows
                    if r.get(b) and any(v is False for v in r[b]["converged"])
                ],
            )
            for b in ("prusie", "susieR")
        },
    )


def export_m1(run):
    comparisons, rows, bindings, identities = [], [], {}, {}
    initial_bindings, initial_decisions = {}, []
    environments = {}
    for window in ("cumulative", "scaling", "overwrite"):
        path = (
            Path(measurement_freeze(run).get("m1_root", str(run / "m1")))
            / window
            / "window1.aggregate.json"
        )
        initial = read(path)
        assert initial["state"] == "SLURM_V10_COMPLETE"
        initial_bindings[window] = digest(path)
        plan = read(path.parent / "bundle/plan.json")
        pooled = path.parent / "pooled.aggregate.json"
        if pooled.exists():
            authorization = read(path.parent / "quality_retry.json")
            assert authorization["one_retry_only"]
            assert authorization["original_aggregate_sha256"] == initial_bindings[window]
            assert authorization["plan_sha256"] == digest(path.parent / "bundle/plan.json")
            path = pooled
        data = read(path)
        assert data["state"] == "SLURM_V10_COMPLETE" and data["all_raw_outputs_verified"]
        assert data["plan_sha256"] == digest(path.parent / "bundle/plan.json")
        if pooled.exists():
            assert data["windows"] == ["window1", "window2"]
            assert data["pool_comparisons"] == authorization["nontrustworthy_comparisons"]
        for item in initial["comparisons"]:
            decision = item["decision"]
            initial_decisions.append(
                dict(
                    window=window,
                    comparison=item["id"],
                    panel=item["panel"],
                    state=decision["state"],
                    rounds=decision["rounds"],
                    accepted=decision["accepted"],
                    trustworthy=decision["trustworthy"],
                    cases=decision["cases"],
                    applicable_cases=decision["applicable_cases"],
                    runtime_counts={
                        key: decision["runtime"][key] for key in ("wins", "ties", "regressions")
                    },
                    runtime_agreement=decision["runtime"]["decision_agreement"],
                    memory_agreement=decision["memory"]["decision_agreement"],
                    veto_agreement=decision["runtime"]["complete_veto_quality"]["veto_agreement"],
                )
            )
        probes = [
            p for p in (run / "m1/preflight").glob("*.json") if digest(p) == plan["topology_sha256"]
        ]
        assert len(probes) == 1, "Authenticate the M1 topology probe"
        topology = read(probes[0])
        environments[window] = {k: topology[k] for k in ("cpu_model", "glibc", "python")} | dict(
            lanes=plan["lanes"],
            retained_rounds=plan["rounds"],
            retained_rounds_by_comparison={
                item["id"]: item["decision"]["rounds"] for item in data["comparisons"]
            },
            numa_domains=len({n for row in topology["l3_domains"] for n in row["numa_nodes"]}),
            l3_domains=len(topology["l3_domains"]),
        )
        bindings[window] = digest(path)
        backends = read(
            Path(measurement_freeze(run).get("m1_root", str(run / "m1")))
            / window
            / "bundle/backends.json"
        )["backends"]
        identities[window] = {
            name: {
                k: backend[k]
                for k in (
                    "commit",
                    "build_commit",
                    "wheel_sha256",
                    "native_sha256",
                    "call_options",
                    "numpy_identity",
                )
            }
            for name, backend in backends.items()
        }
        for item in data["comparisons"]:
            decision = item["decision"]
            applicable = [r for r in decision["raw"]["rows"] if r["applicable"]]
            median_counts = dict(
                wins=sum(
                    r["candidate_times"]["median"] < r["parent_times"]["median"] for r in applicable
                ),
                ties=sum(
                    r["candidate_times"]["median"] == r["parent_times"]["median"]
                    for r in applicable
                ),
                regressions=sum(
                    r["candidate_times"]["median"] > r["parent_times"]["median"] for r in applicable
                ),
            )
            comparisons.append(
                {
                    k: item[k]
                    for k in ("id", "parent", "candidate", "kind", "panel", "diagnostic_only")
                }
                | {
                    "case_median_runtime_counts": median_counts,
                    "windows": item["windows"],
                    "decision": {
                        k: decision[k]
                        for k in (
                            "state",
                            "accepted",
                            "trustworthy",
                            "cases",
                            "applicable_cases",
                            "rounds",
                            "runtime",
                            "memory",
                            "resampling",
                        )
                        if k in decision
                    },
                }
            )
            for index, raw in enumerate(decision["raw"]["rows"]):
                rows.append(
                    dict(
                        comparison=item["id"],
                        panel=item["panel"],
                        parent=item["parent"],
                        candidate=item["candidate"],
                        **raw,
                        parent_retained_seconds=decision["raw"]["parent_times"][index],
                        candidate_retained_seconds=decision["raw"]["candidate_times"][index],
                        parent_retained_incremental_bytes=decision["raw"][
                            "parent_incremental_peaks"
                        ][index],
                        candidate_retained_incremental_bytes=decision["raw"][
                            "candidate_incremental_peaks"
                        ][index],
                    )
                )
    # The selected v10 schema carries only case identifiers, scalars and arrays.
    # Reject accidental private paths before creating a public export.
    assert "/home/" not in json.dumps(dict(comparisons=comparisons, rows=rows))
    return dict(
        protocol="v10",
        source_aggregate_sha256=bindings,
        source_window1_aggregate_sha256=initial_bindings,
        initial_window1_decisions=initial_decisions,
        backend_identities=identities,
        environments=environments,
        comparisons=comparisons,
        records=rows,
        memory_definition=(
            "post-fit VmHWM minus immediately pre-fit VmRSS; "
            "pre-fit HWM excess is retained separately"
        ),
    )


def export(run, output):
    run, output = Path(run).resolve(), Path(output)
    manifest = read(run / "inputs/m2_manifest.json")
    freeze = measurement_freeze(run)
    tasks = list(expected_tasks(manifest["rows"]))
    records = [load_task(run, *task) for task in tasks]
    assert not any(r["task_state"] == "MISSING" for r in records), (
        "Complete the fixed panel before public export"
    )
    primary = [r for r in records if r["kind"] == "primary"]
    assert Counter(r["panel"] for r in primary) == {"GWAS": 200, "eQTL": 14157}
    groups = {g: group_summary([r for r in primary if r["panel"] == g]) for g in ("GWAS", "eQTL")}
    provenance = dict(
        run_id=run.name,
        version="0.3.0",
        compute_code_commit=freeze["code_sha"],
        wheel_sha256=freeze["wheel_sha256"],
        native_sha256=freeze["native_sha256"],
        reference_commit=manifest["reference_commit"],
        reference_version="0.16.6",
        manifest_sha256=digest(run / "inputs/m2_manifest.json"),
        source_panel_sha256=sorted(manifest["source_panels"].values()),
        protocol=manifest["protocol"],
        maximum_physical_cores=64,
        measurement_software_versions=[
            dict(sha256=json.loads(value), tasks=count)
            for value, count in sorted(
                Counter(
                    json.dumps(row["measurement_software_sha256"], sort_keys=True)
                    for row in records
                ).items()
            )
        ],
        panels={
            "GWAS": "FinnGen R13 type 2 diabetes, n=486019",
            "eQTL": "GTEx v11 Whole_Blood, n=800",
        },
    )
    summary = dict(
        schema_version=1,
        provenance=provenance,
        groups=groups,
        series_counts=dict(Counter(r["kind"] for r in records)),
        all_tasks_accounted=len(records),
        alpha_failures=[
            dict(
                case_id=r["case_id"],
                panel=r["panel"],
                series=r["kind"],
                error=r["max_abs_alpha_error"],
            )
            for r in records
            if r["alpha_pass"] is False
        ],
        definitions=dict(
            total_time_speedup=(
                "sum of all seven retained R fit times / sum of all seven retained "
                "prusie fit times on complete positive timing pairs"
            ),
            geometric_mean_speedup=(
                "exp(mean(log(within-case median R time / within-case median prusie time)))"
            ),
            sum_case_medians_speedup=(
                "sum of within-case R medians / sum of within-case "
                "prusie medians; separate diagnostic"
            ),
            memory=(
                "independent fresh-process post-fit HWM RSS including interpreter, imports, "
                "inputs, validation reference and fit; no subtraction of prior HWM"
            ),
            inclusion=(
                "all complete positive timing pairs, independently of alpha differences, "
                "iteration limit, CS presence and memory-mode failures"
            ),
        ),
    )
    m1 = export_m1(run)
    output.mkdir(parents=True, exist_ok=True)
    write(output / "v030_summary.json", summary)
    write(output / "v030_m1.json", m1)
    write_records(output, records)
    return summary


def write_records(output, records):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    primary = [r for r in records if r["kind"] == "primary"]
    # Keep browsing tables small. A deterministic compressed JSON-lines archive
    # preserves all seven observations and the structured status/host records.
    with (
        (output / "v030_records.jsonl.gz").open("wb") as binary,
        gzip.GzipFile(filename="", fileobj=binary, mode="wb", mtime=0) as stream,
    ):
        for row in records:
            stream.write((json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n").encode())
    columns = [
        "case_id",
        "panel",
        "source_index",
        "kind",
        "p",
        "n_threads",
        "overwrite_ld",
        "task_state",
        "alpha_pass",
        "max_abs_alpha_error",
        "max_abs_pip_error",
        "timing_pair",
        "memory_pair",
        "speedup",
        "memory_ratio",
        "prusie_seconds",
        "susieR_seconds",
        "prusie_peak_rss_bytes",
        "susieR_peak_rss_bytes",
        "prusie_repeat_iqr_over_median",
        "susieR_repeat_iqr_over_median",
        "frozen_R_alpha_bitwise_equal",
        "frozen_R_max_abs_alpha_error",
        "result_sha256",
    ]
    for name, selected in (
        ("primary", primary),
        ("threads", [r for r in records if r["kind"].startswith("threads")]),
        ("overwrite", [r for r in records if r["kind"] == "overwrite"]),
        ("windows", [r for r in records if r["kind"].startswith("window")]),
    ):
        with (output / f"v030_{name}.tsv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for row in selected:
                flat = {k: row[k] for k in columns if k in row}
                for b in ("prusie", "susieR"):
                    value = row.get(b) or {}
                    flat[b + "_seconds"] = value.get("runtime", {}).get("median")
                    flat[b + "_peak_rss_bytes"] = value.get("peak_rss_bytes")
                    flat[b + "_repeat_iqr_over_median"] = value.get("repeat_iqr_over_median")
                flat["frozen_R_alpha_bitwise_equal"] = row["fresh_R_integrity"][
                    "alpha_bitwise_equal"
                ]
                flat["frozen_R_max_abs_alpha_error"] = row["fresh_R_integrity"][
                    "max_abs_alpha_error"
                ]
                writer.writerow(flat)


def bind_release(run, output):
    """Identify the qualified release without relabelling retained measurements."""
    run, output = Path(run).resolve(), Path(output)
    summary_path = output / "v030_summary.json"
    summary = read(summary_path)
    measured = measurement_freeze(run)
    release = read(run / "status/code_freeze.json")
    for key, measured_key in (
        ("compute_code_commit", "code_sha"),
        ("wheel_sha256", "wheel_sha256"),
        ("native_sha256", "native_sha256"),
    ):
        assert summary["provenance"][key] == measured[measured_key]
    assert release["code_sha"] != measured["code_sha"], "No distinct release to bind"
    for item in (measured, release):
        assert digest(item["wheel"]) == item["wheel_sha256"]
    assert release["native_sha256"] == measured["native_sha256"]
    numerical_path = run / release["numerical_report"]
    assert digest(numerical_path) == release["numerical_report_sha256"]
    numerical = read(numerical_path)
    assert numerical["passed"] and numerical["completed_cells"] == 1472
    assert not numerical["failures"] and not numerical["missing"]
    assert numerical["commit"] == release["numerical_qualified_code_sha"]
    certificate = read(run / release["production_identity_certificate"])
    assert certificate["runtime_members_equal"]
    assert certificate["final_wheel_sha256"] == release["wheel_sha256"]
    assert certificate["qualified_wheel_sha256"] == numerical["wheel_sha256"]
    with zipfile.ZipFile(measured["wheel"]) as old, zipfile.ZipFile(release["wheel"]) as new:
        assert set(old.namelist()) == set(new.namelist())
        changed = sorted(name for name in old.namelist() if old.read(name) != new.read(name))
    assert set(changed) <= {
        "prusie/refinement.py",
        "prusie/report.py",
        "prusie-0.3.0.dist-info/METADATA",
        "prusie-0.3.0.dist-info/RECORD",
    }, "The ordinary fitting payload changed after formal measurement"
    summary["provenance"]["qualified_release"] = dict(
        code_commit=release["code_sha"],
        wheel_sha256=release["wheel_sha256"],
        native_sha256=release["native_sha256"],
        numerical_qualified_commit=numerical["commit"],
        numerical_report_sha256=release["numerical_report_sha256"],
        changed_wheel_members=changed,
        formally_timed=False,
        fresh_process_memory_profiled=False,
        ordinary_fitting_modules_byte_identical=True,
        relationship=(
            "The retained measurements used refine=False and the measured wheel above. "
            "The qualified release adds opt-in refinement tracing and report handling. "
            "Its native extension and ordinary fitting modules are byte-identical, "
            "but the release wheel was not separately timed or memory-profiled."
        ),
    )
    write(summary_path, summary)
    return summary["provenance"]["qualified_release"]


def load_public(output):
    output = Path(output)
    with gzip.open(output / "v030_records.jsonl.gz", "rt") as stream:
        records = [json.loads(line) for line in stream]
    return read(output / "v030_summary.json"), read(output / "v030_m1.json"), records
