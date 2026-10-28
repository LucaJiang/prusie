"""Render current 0.3.0 marker blocks from public retained measurements."""

from __future__ import annotations

import re
from pathlib import Path

from v030_records import finite, load_public, read


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "&#124;")

    return (
        "\n| "
        + " | ".join(map(cell, headers))
        + " |\n| "
        + " | ".join("---" for _ in headers)
        + " |\n"
        + "".join("| " + " | ".join(map(cell, row)) + " |\n" for row in rows)
    )


def number(value, digits=3):
    return f"{value:.{digits}g}" if finite(value) else "unavailable"


def milliseconds(value):
    return number(value * 1000 if finite(value) else None)


def acceptance_counts(group):
    return (
        f"{group['alpha_passes']} / {len(group['alpha_failures'])} / "
        f"{len(group['alpha_unavailable'])}"
    )


def render(root, pages=None, check=False):
    root = Path(root)
    summary, m1, records = load_public(root / "docs/assets/benchmark")
    groups = summary["groups"]
    allowed = set(pages) if pages is not None else None
    outputs = {}

    def fragment(page, key, value, prefix="generated"):
        if allowed is not None and page not in allowed:
            return
        path = root / page
        text = outputs.get(path, path.read_text())
        start, end = f"<!-- {prefix}:{key}:start -->", f"<!-- {prefix}:{key}:end -->"
        assert text.count(start) == text.count(end) == 1, f"Missing/duplicate marker {page}:{key}"
        pattern = re.escape(start) + r".*?" + re.escape(end)
        outputs[path] = re.sub(
            pattern, lambda _: start + "\n" + value.rstrip() + "\n" + end, text, flags=re.S
        )

    rows = []
    for panel, group in groups.items():
        rows.append(
            [
                panel,
                group["cases"],
                number(group["total_time_speedup"]) + "×",
                number(group["fitting"]["headline_R_over_prusie"]) + "×",
                number(group["sum_case_medians_speedup"]) + "×",
                number(group["memory_ratio_distribution"]["median"]) + "×",
            ]
        )
    highlights = "Measured **prusie 0.3.0** against fresh **susieR 0.16.6** fits.\n"
    highlights += table(
        [
            "Analysis",
            "Cases",
            "Complete-call speed-up¹",
            "Fitting-stage speed-up¹",
            "Complete-call sum-of-medians²",
            "Peak RSS ratio³",
        ],
        rows,
    )
    highlights += (
        "\n¹ Sum of three retained R times divided by the corresponding prusie total, "
        "on exclusive AMD EPYC 7742 nodes. The fitting stage excludes input/LD preparation. "
        "² Sum of within-case R medians divided by the sum of prusie medians. "
        "³ Median paired R/prusie fresh-process peak RSS from the earlier local AMD EPYC 9754 "
        "measurements, including interpreter and inputs. "
        "Runtime and memory have separate origins.\n"
    )
    for page in ("README.md", "docs/index.md", "docs/performance.md"):
        fragment(page, "highlights", highlights, "current")

    panel = (
        "The fixed panel contains **200 FinnGen R13 type 2 diabetes GWAS loci** "
        "(n = 486,019; 89,727 cases and 396,292 controls) and **14,157 GTEx v11 "
        "Whole_Blood genes** (n = 800). GWAS is fitted with the frozen Gaussian "
        "RSS approximation. All cases remain accounted for, including iteration-limit "
        "outputs, empty credible sets, numerical discrepancies and operational failures. "
        "Gene windows and study participants overlap; cases are not independent "
        "biological replications.\n"
    )
    panel += table(
        [
            "Analysis",
            "Cases",
            "Variants per case",
            "Alpha passes / failures / unavailable",
            "Unresolved case failures",
        ],
        [
            [
                p,
                g["cases"],
                " to ".join(map(str, g["p_range"])),
                acceptance_counts(g),
                len(g["operational_failures"]),
            ]
            for p, g in groups.items()
        ],
    )
    fragment("docs/performance.md", "panel", panel)
    runtime_rows = []
    for panel in groups:
        current = summary["burgundy"]["panels"][panel]
        for label, group in (("Complete call", current), ("Fitting stage", current["fitting"])):
            runtime_rows.append(
                [
                    panel,
                    label,
                    milliseconds(group["backends"]["prusie"]["case_median_seconds"]["median"]),
                    milliseconds(group["backends"]["R"]["case_median_seconds"]["median"]),
                    number(group["headline_R_over_prusie"]) + "×",
                    number(group["sum_of_medians_R_over_prusie"]) + "×",
                    number(group["geometric_mean_case_median_ratio"]) + "×",
                    f"{group['wins']} / {group['ties']} / {group['regressions']}",
                ]
            )
    fragment(
        "docs/performance.md",
        "runtime",
        table(
            [
                "Analysis",
                "Metric",
                "prusie median (ms)",
                "R median (ms)",
                "Total R/prusie",
                "Sum-of-medians",
                "Geometric mean",
                "Faster / tied / slower",
            ],
            runtime_rows,
        ),
        "current",
    )
    variability = []
    for panel in groups:
        current = summary["burgundy"]["panels"][panel]
        old = summary["burgundy"]["local_baseline_first3"][panel]
        for label, group in (("Burgundy, 3 calls", current), ("Local, first 3 calls", old)):
            for public, backend in (("prusie", "prusie"), ("susieR", "R")):
                values = group["backends"][backend]
                variability.append(
                    [
                        panel,
                        label,
                        public,
                        number(values["repeat_iqr_over_median"]["median"]),
                        values["cases_with_repeat_above_3x"],
                        number(100 * values["total_time_fraction_from_repeats_above_2x"]) + "%",
                    ]
                )
    fragment(
        "docs/performance.md",
        "variability",
        table(
            [
                "Analysis",
                "Complete-call timing",
                "Backend",
                "Median IQR/median",
                "Cases with a repeat >3× median",
                "Time in repeats >2× median",
            ],
            variability,
        ),
        "current",
    )
    fragment(
        "docs/performance.md",
        "memory",
        table(
            [
                "Analysis / backend",
                "Fresh-process peak RSS, across-case median (MiB)",
                "Measured cases",
            ],
            [
                [
                    f"{p} / {b}",
                    number(g["backends"][b]["full_process_peak_mib"]["median"]),
                    g["backends"][b]["full_process_peak_mib"]["n"],
                ]
                for p, g in groups.items()
                for b in ("prusie", "susieR")
            ],
        ),
        "current",
    )
    windows = sorted([r for r in records if r["kind"].startswith("window")], key=lambda r: r["p"])
    scaling = []
    for row in windows:
        for backend in ("prusie", "susieR"):
            value = row[backend]
            niter = sorted(set(v for v in value["niter"] if v is not None))
            converged = sorted(set(v for v in value["converged"] if v is not None))
            scaling.append(
                [
                    row["p"],
                    backend,
                    number(value["runtime"]["median"]),
                    number(value["fitting_runtime"]["median"]),
                    ", ".join(map(str, niter)),
                    ", ".join(map(str, converged)),
                    number(value["peak_rss_bytes"] / 2**20 if value["peak_rss_bytes"] else None),
                ]
            )
    fragment(
        "docs/performance.md",
        "scaling",
        table(
            [
                "Leading variants",
                "Backend",
                "Complete-call median (s)",
                "Fitting-stage median (s)",
                "Observed iterations",
                "Converged",
                "Earlier EPYC 9754 peak RSS (MiB)",
            ],
            scaling,
        ),
        "current",
    )
    protocol = summary["burgundy_protocol"]
    original = protocol["original_code_under_test"]
    identity = (
        f"- Frozen release code: `{original['code_sha']}`\n"
        f"- Original release wheel SHA-256: `{original['wheel_sha256']}`\n"
        f"- Original native extension SHA-256: `{original['native_sha256']}`\n"
        f"- Burgundy protocol SHA-256: `{summary['burgundy']['protocol_sha256']}`\n"
        "- Statistical reference: R 4.4.0; susieR 0.16.6, "
        "`8e56a8e038e989856d106d9ca5175cc664fea9d2`.\n\n"
        "Both runtime metrics use run-local timing instrumentation derived from these frozen "
        "artifacts. The native binary differs from the release binary; the original wheel is "
        "preserved. The instrumentation patch, derived binary hashes, source boundaries and "
        "qualification results are in the "
        "[measurement archive](reproducibility.md#burgundy-protocol). "
        "Earlier local memory measurements retain their original measured build identity.\n"
    )
    fragment("docs/performance.md", "identity", identity, "current")
    comparison_labels = {
        "C1_V030_vs_BASE": "0.3.0 vs intermediate baseline",
        "C2_V030_vs_MAIN": "0.3.0 vs historical baseline",
        "C3_V030_vs_ENG01": "0.3.0 vs previously optimized baseline",
        "C4_t2_vs_t1": "n_threads=2 vs 1",
        "C4_t4_vs_t1": "n_threads=4 vs 1",
        "C4_t8_vs_t1": "n_threads=8 vs 1",
        "C5_true_vs_false": "overwrite_ld=True vs False",
    }
    role_labels = {
        "default-veto": "non-regression check",
        "thread-option": "four-thread option",
        "memory-only": "memory comparison",
    }
    gate_rows = []
    for item in m1["comparisons"]:
        if item["id"].startswith("AA_"):
            continue
        d = item["decision"]
        runtime = d.get("runtime", {})
        memory = d.get("memory", {})
        gate_rows.append(
            [
                comparison_labels[item["id"]],
                {
                    "panel": "Primary (87)",
                    "fresh_panel": "Confirmation (80)",
                    "largep": "Large-p (17)",
                }.get(item["panel"], item["panel"]),
                d["state"],
                d["rounds"],
                f"{d['applicable_cases']} / {d['cases']}",
                " / ".join(str(runtime.get(key, "—")) for key in ("wins", "ties", "regressions")),
                " / ".join(
                    str(item["case_median_runtime_counts"][key])
                    for key in ("wins", "ties", "regressions")
                ),
                " / ".join(str(memory.get(key, "—")) for key in ("wins", "ties", "regressions")),
                "descriptive" if item["diagnostic_only"] else role_labels[item["kind"]],
            ]
        )
    fragment(
        "docs/performance.md",
        "earlier",
        table(
            [
                "Comparison",
                "Fixed panel",
                "v10 decision",
                "Retained rounds",
                "Applicable / all cases",
                "v10 paired-round runtime W / T / R",
                "Case-median runtime W / T / R",
                "Incremental-memory W / T / R",
                "Role",
            ],
            gate_rows,
        )
        + "\nThese are earlier controlled Burgundy measurements. Cumulative totals do not replace "
        "the fixed-panel majority, stratified non-regression check and robustness criteria. "
        "The v10 runtime count uses the median of paired-round time ratios; the additional "
        "case-median count compares the two within-case median times. These can differ. "
        "Memory counts compare within-case medians of incremental peaks. "
        "W / T / R means lower, equal and higher candidate measurements. "
        "A comparison with 16 retained rounds uses the protocol's single quality retry, "
        "pooling both complete windows only for comparisons whose first-window decision "
        "was unresolved. Other comparisons retain the original eight rounds. Initial "
        "decisions remain in the paired measurement records.\n",
    )
    fragment(
        "docs/numerical_accuracy.md",
        "maxima",
        table(
            [
                "Analysis",
                "Cases",
                "Alpha passes / failures / unavailable",
                "Maximum |Δalpha|",
                "Maximum |ΔPIP|",
            ],
            [
                [
                    p,
                    g["cases"],
                    acceptance_counts(g),
                    number(g["max_abs_alpha_error"]),
                    number(g["max_abs_pip_error"]),
                ]
                for p, g in groups.items()
            ],
        )
        + "\nThe gate remains absolute alpha error ≤ 1e-5, with relative tolerance zero. "
        "Iteration-limit exceedances remain failures of this numerical threshold.\n",
    )
    failure_rows = []

    def outcomes(values):
        return (
            ", ".join(sorted({f"{v['niter']} / {v['converged']}" for v in values})) or "unavailable"
        )

    for item in summary["alpha_failures"]:
        failure_rows.append(
            [
                item["case_id"],
                item["series"],
                number(item["error"], 8),
                outcomes(item["prusie_fit_outcomes"]),
                outcomes(item["R_fit_outcomes"]),
                item["both_backends_nonconverged_at_iteration_limit"],
                item["disposition"],
            ]
        )
    failures = (
        table(
            [
                "Case",
                "Setting",
                "Maximum |Δalpha|",
                "prusie: iterations / converged",
                "R: iterations / converged",
                "Both at iteration limit",
                "Disposition",
            ],
            failure_rows,
        )
        if failure_rows
        else "No alpha failures were recorded.\n"
    )
    converged_failures = [
        r
        for p in summary["burgundy"]["panels"].values()
        for r in p["exceedances"]
        if r["any_backend_call_converged"]
    ]
    if converged_failures:
        failures = (
            "**Alpha exceedances involving a converged backend: "
            + ", ".join(r["case_id"] for r in converged_failures)
            + ".**\n\n"
            + failures
        )
    else:
        failures = "No alpha exceedance involved a converged backend.\n\n" + failures
    fragment("docs/numerical_accuracy.md", "errors", failures)
    limits = table(
        ["Analysis", "prusie iteration-limit cases", "R iteration-limit cases"],
        [
            [
                p,
                len(g["backends"]["prusie"]["iteration_limit_cases"]),
                len(g["backends"]["susieR"]["iteration_limit_cases"]),
            ]
            for p, g in groups.items()
        ],
    )
    limits += (
        "\nAll completed positive timing pairs enter the performance summaries, "
        "independently of convergence, alpha agreement and credible-set presence. "
        "Per-case records list every status.\n"
    )
    fragment("docs/numerical_accuracy.md", "limits", limits)
    cs = []
    for p in groups:
        rows = [r for r in records if r["panel"] == p and r["kind"] == "primary"]
        for b in ("prusie", "susieR"):
            available = [r for r in rows if r[b] and r[b]["cs_count"]]
            counts = [r[b]["cs_count"][0] for r in available]
            cs.append([p, b, len(available), sum(counts), sum(v == 0 for v in counts)])
    fragment(
        "docs/numerical_accuracy.md",
        "cs",
        table(
            [
                "Analysis",
                "Backend",
                "Cases with recorded counts",
                "Credible sets (first retained fit)",
                "Empty-CS cases",
            ],
            cs,
        )
        + "\nThis full-panel campaign records set counts; it does not compare full member "
        "identities, returned mass or purity. "
        "The separate packaged example and regression suites check those quantities.\n",
    )
    toy = root / "docs/assets/toy_result.json"
    if toy.exists():
        result = read(toy)
        pip_error = result["comparison"]["errors"]["pip"]["max_abs_error"]
        text = (
            f"The independent 500-variant toy example converged in {result['niter']} iterations "
            f"and returned {len(result['cs'])} credible sets. Its maximum absolute PIP difference "
            f"from the separately generated susieR 0.16.6 reference was {pip_error:.3g}.\n"
        )
        for page in ("docs/example.md", "docs/numerical_accuracy.md"):
            fragment(page, "toy", text)
        cs = table(
            ["Original component", "Synthetic variants", "Returned posterior mass", "Minimum |r|"],
            [
                [
                    c["component_id_0based"],
                    ", ".join(f"syn{j + 1:04d}" for j in c["members_0based"]),
                    f"{c['coverage']:.6f}",
                    f"{c['purity']['min_abs_corr']:.6f}",
                ]
                for c in result["cs"]
            ],
        )
        fragment("docs/example.md", "toy-cs", cs)
    stale = [str(p.relative_to(root)) for p, text in outputs.items() if p.read_text() != text]
    if check:
        assert not stale, "Stale generated 0.3.0 blocks: " + ", ".join(stale)
    else:
        for path, text in outputs.items():
            path.write_text(text)
    return stale
