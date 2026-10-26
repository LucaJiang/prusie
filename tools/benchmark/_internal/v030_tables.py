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
                group["timing_pairs"],
                number(group["total_time_speedup"]) + "×",
                number(group["geometric_mean_speedup"]) + "×",
                number(group["memory_ratio_distribution"]["median"]) + "×",
            ]
        )
    highlights = "Measured **prusie 0.3.0** against fresh **susieR 0.16.6** fits.\n"
    highlights += table(
        [
            "Analysis",
            "Cases",
            "Timing pairs",
            "Total-time speed-up¹",
            "Per-case geometric mean²",
            "Peak RSS ratio³",
        ],
        rows,
    )
    highlights += (
        "\n¹ Sum of all seven retained R fit times divided by the corresponding prusie total. "
        "² Geometric mean of within-case median-time ratios. "
        "³ Median paired R/prusie fresh-process peak RSS, including interpreter and inputs. "
        "The two speed-up summaries use different weighting.\n"
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
            "Operational failures",
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
    fragment(
        "docs/performance.md",
        "runtime",
        table(
            [
                "Analysis",
                "prusie median (ms)",
                "R median (ms)",
                "Faster / tied / slower",
                "prusie repeat IQR/median",
                "R repeat IQR/median",
            ],
            [
                [
                    p,
                    milliseconds(g["backends"]["prusie"]["runtime_seconds"]["median"]),
                    milliseconds(g["backends"]["susieR"]["runtime_seconds"]["median"]),
                    f"{g['wins']} / {g['ties']} / {g['regressions']}",
                    number(g["backends"]["prusie"]["repeat_iqr_over_median"]["median"]),
                    number(g["backends"]["susieR"]["repeat_iqr_over_median"]["median"]),
                ]
                for p, g in groups.items()
            ],
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
                "Median fit time (s)",
                "Observed iterations",
                "Converged",
                "Peak RSS (MiB)",
            ],
            scaling,
        ),
        "current",
    )
    provenance = summary["provenance"]
    fragment(
        "docs/performance.md",
        "identity",
        f"- Compute commit: `{provenance['compute_code_commit']}`\n"
        f"- Frozen CPython 3.12 wheel SHA-256: `{provenance['wheel_sha256']}`\n"
        f"- Native extension SHA-256: `{provenance['native_sha256']}`\n"
        f"- Statistical reference: susieR 0.16.6, `{provenance['reference_commit']}`.\n",
        "current",
    )
    gate_rows = []
    for item in m1["comparisons"]:
        if item["id"].startswith("AA_"):
            continue
        d = item["decision"]
        runtime = d.get("runtime", {})
        memory = d.get("memory", {})
        gate_rows.append(
            [
                item["id"],
                {
                    "panel": "Primary (87)",
                    "fresh_panel": "Confirmation (80)",
                    "largep": "Large-p (17)",
                }.get(item["panel"], item["panel"]),
                d["state"],
                f"{d['applicable_cases']} / {d['cases']}",
                " / ".join(str(runtime.get(key, "—")) for key in ("wins", "ties", "regressions")),
                " / ".join(
                    str(item["case_median_runtime_counts"][key])
                    for key in ("wins", "ties", "regressions")
                ),
                " / ".join(str(memory.get(key, "—")) for key in ("wins", "ties", "regressions")),
                "diagnostic" if item["diagnostic_only"] else item["kind"],
            ]
        )
    fragment(
        "docs/performance.md",
        "earlier",
        table(
            [
                "New M1 comparison",
                "Fixed panel",
                "v10 decision",
                "Applicable / all cases",
                "v10 paired-round runtime W / T / R",
                "Case-median runtime W / T / R",
                "Incremental-memory W / T / R",
                "Role",
            ],
            gate_rows,
        )
        + "\nThese are new paired Burgundy measurements. Cumulative totals do not replace "
        "the fixed-panel majority, stratified veto and robustness gates. "
        "The v10 runtime count uses the median of paired-round time ratios; the additional "
        "case-median count compares the two within-case median times. These can differ. "
        "Memory counts compare within-case medians of incremental peaks. "
        "W / T / R means lower, equal and higher candidate measurements.\n",
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
        ),
    )
    fragment(
        "docs/numerical_accuracy.md",
        "errors",
        table(
            [
                "Analysis",
                "Frozen-R integrity deviations",
                "Integrity unavailable",
                "Maximum fresh/frozen |Δalpha|",
            ],
            [
                [
                    p,
                    len(g["frozen_R_deviations"]),
                    len(g["frozen_R_unavailable"]),
                    number(g["frozen_R_max_abs_alpha_error"]),
                ]
                for p, g in groups.items()
            ],
        )
        + "\nFresh R is checked before timing against the stored frozen alpha values. "
        "Every nonzero difference remains recorded; the full-precision new R alpha is used "
        "for the prusie comparison. Frozen decimal serialization can prevent bitwise identity.\n",
    )
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
