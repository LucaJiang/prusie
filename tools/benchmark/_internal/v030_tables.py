"""Render generated documentation blocks from the published 0.3.0 benchmark records."""

from __future__ import annotations

import gzip
import json
import re
import statistics
from pathlib import Path

from v030_records import digest, finite, read

ASSETS = Path("docs/assets/benchmark")
PANELS = ("GWAS", "eQTL")
THREADS = (("C4_t2_vs_t1", 2), ("C4_t4_vs_t1", 4), ("C4_t8_vs_t1", 8))


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


def quartiles(values):
    q1, median, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return q1, median, q3


def records_of(assets):
    summary = read(assets / "benchmark_summary.json")
    path = assets / "benchmark_records.jsonl.gz"
    assert digest(path) == summary["records_sha256"]
    with gzip.open(path, "rt") as stream:
        records = [json.loads(line) for line in stream]
    retime = read(assets / "window10000_retime.json")
    for row in records:
        if row["kind"] == "window10000":
            row["R"] = dict(
                row["R"],
                times=retime["retained_complete_seconds"],
                fitting_times=retime["retained_fitting_seconds"],
            )
    return summary, records


def ratios(rows, field):
    return [statistics.median(r["R"][field]) / statistics.median(r["prusie"][field]) for r in rows]


def render(root, pages=None, check=False):
    root = Path(root)
    assets = root / ASSETS
    summary, records = records_of(assets)
    primary = {
        p: [r for r in records if r["kind"] == "primary" and r["panel"] == p] for p in PANELS
    }
    windows = sorted((r for r in records if r["kind"].startswith("window")), key=lambda r: r["p"])
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

    def ratio_cell(values):
        q1, median, q3 = quartiles(values)
        return f"{median:.1f}× [{q1:.1f}-{q3:.1f}]"

    def cumulative(cases, field):
        return sum(sum(r["R"][field]) for r in cases) / sum(sum(r["prusie"][field]) for r in cases)

    rows = []
    for panel, cases in primary.items():
        memory = [
            r["R"]["peak_rss_diagnostic_bytes"] / r["prusie"]["peak_rss_diagnostic_bytes"]
            for r in cases
        ]
        rows.append(
            [
                panel,
                f"{len(cases):,}",
                f"{cumulative(cases, 'times'):.1f}×",
                f"{cumulative(cases, 'fitting_times'):.1f}×",
                f"{statistics.median(memory):.2f}×",
            ]
        )
    highlights = "Measured **prusie 0.3.0** against **susieR 0.16.6** on the same inputs.\n"
    highlights += table(
        [
            "Evaluated cases",
            "Cases",
            "Cumulative runtime ratio",
            "Cumulative fitting-stage ratio",
            "Median peak-memory ratio",
        ],
        rows,
    )
    highlights += (
        "\nThe cumulative ratios divide the summed susieR time by the summed prusie time over "
        "three retained complete calls of every evaluated case, with one numerical thread; "
        "the fitting stage excludes input and LD preparation. The memory ratio is the median "
        "of paired susieR/prusie ratios of peak backend-process memory. All measurements are "
        "from exclusive AMD EPYC 7742 nodes. Ratios for other traits, tissues, region sizes "
        "and LD references may differ; see [performance](docs/performance.md).\n"
    )
    for page in ("README.md", "docs/index.md", "docs/performance.md"):
        link = "docs/performance.md" if page == "README.md" else "performance.md"
        fragment(
            page,
            "highlights",
            highlights.replace("(docs/performance.md)", f"({link})"),
            "current",
        )

    panel_text = (
        "The fixed panel contains **200 FinnGen R13 type 2 diabetes GWAS loci** "
        "(n = 486,019; 89,727 cases and 396,292 controls) and **14,157 GTEx v11 "
        "Whole_Blood genes** (n = 800). GWAS is fitted with the Gaussian RSS "
        "approximation. Every case is included, including iteration-limit fits and "
        "cases with no credible set. Gene windows and study participants "
        "overlap, so cases are not independent biological replicates.\n"
    )
    panel_text += table(
        ["Analysis", "Cases", "Variants per case", "Cases where neither program converged"],
        [
            [
                p,
                f"{len(cases):,}",
                f"{min(r['p'] for r in cases):,} to {max(r['p'] for r in cases):,}",
                sum(r["both_nonconverged_at_limit"] for r in cases),
            ]
            for p, cases in primary.items()
        ],
    )
    fragment("docs/performance.md", "panel", panel_text)

    runtime_rows = []
    for panel, cases in primary.items():
        for label, field in (("Complete call", "times"), ("Fitting stage", "fitting_times")):
            values = ratios(cases, field)
            q1, median, q3 = quartiles(values)
            native = sum(statistics.median(r["prusie"][field]) for r in cases)
            reference = sum(statistics.median(r["R"][field]) for r in cases)
            runtime_rows.append(
                [
                    panel,
                    label,
                    f"{cumulative(cases, field):.1f}×",
                    f"{median:.1f}× [{q1:.1f}-{q3:.1f}]",
                    f"{native / 60:.1f} min",
                    f"{reference / 3600:.2f} h",
                    f"{sum(v > 1 for v in values):,} / {len(values):,}",
                ]
            )
    fragment(
        "docs/performance.md",
        "runtime",
        table(
            [
                "Analysis",
                "Metric",
                "Cumulative susieR/prusie ratio",
                "Per-case ratio, median [IQR]",
                "prusie total",
                "susieR total",
                "Cases faster with prusie",
            ],
            runtime_rows,
        ),
        "current",
    )

    variability = []
    for panel in PANELS:
        for public, backend in (("prusie", "prusie"), ("susieR", "R")):
            values = summary["panels"][panel]["backends"][backend]
            variability.append(
                [
                    panel,
                    public,
                    number(values["repeat_iqr_over_median"]["median"]),
                    values["cases_with_repeat_above_2x"],
                    number(100 * values["total_time_fraction_from_repeats_above_2x"]) + "%",
                ]
            )
    fragment(
        "docs/performance.md",
        "variability",
        table(
            [
                "Analysis",
                "Backend",
                "Median repeat IQR/median",
                "Cases with a call >2× its case median",
                "Share of total time in those calls",
            ],
            variability,
        ),
        "current",
    )

    memory_rows = []
    for panel, cases in primary.items():
        for public, backend in (("prusie", "prusie"), ("susieR", "R")):
            peaks = [r[backend]["peak_rss_diagnostic_bytes"] / 2**20 for r in cases]
            memory_rows.append([panel, public, number(statistics.median(peaks)), f"{len(cases):,}"])
    fragment(
        "docs/performance.md",
        "memory",
        table(
            ["Analysis", "Backend", "Peak RSS, median across cases (MiB)", "Cases"],
            memory_rows,
        ),
        "current",
    )

    thread_records = read(assets / "thread_scaling.json")["records"]
    thread_rows = []
    for comparison, count in THREADS:
        values = [r["ratio"] for r in thread_records if r["comparison"] == comparison]
        thread_rows.append(
            [
                count,
                f"{statistics.median(values):.3f}",
                f"{sum(v < 1 for v in values)} / {sum(v == 1 for v in values)} / "
                f"{sum(v > 1 for v in values)}",
            ]
        )
    fragment(
        "docs/performance.md",
        "threads",
        table(
            ["Threads", "Median time / one-thread time", "Faster / tied / slower, out of 87"],
            thread_rows,
        ),
        "current",
    )

    with gzip.open(assets / "benchmark_overwrite_records.jsonl.gz", "rt") as stream:
        overwrite = [json.loads(line) for line in stream]
    differences = [
        r["measurements"]["overwrite"]["peak_rss_bytes"]
        - r["measurements"]["default"]["peak_rss_bytes"]
        for r in overwrite
    ]
    fragment(
        "docs/performance.md",
        "overwrite",
        table(
            ["Loci", "Equal peaks", "Lower with overwrite", "Higher with overwrite", "Largest gap"],
            [
                [
                    len(overwrite),
                    sum(d == 0 for d in differences),
                    sum(d < 0 for d in differences),
                    sum(d > 0 for d in differences),
                    f"{max(abs(d) for d in differences) / 1024:.0f} KiB",
                ]
            ],
        ),
        "current",
    )

    scaling = []
    for row in windows:
        for public, backend in (("prusie", "prusie"), ("susieR", "R")):
            value = row[backend]
            scaling.append(
                [
                    f"{row['p']:,}",
                    public,
                    number(statistics.median(value["times"])),
                    number(statistics.median(value["fitting_times"])),
                    value["niter"][1],
                    number(value["peak_rss_diagnostic_bytes"] / 2**20),
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
                "Iterations",
                "Peak RSS (MiB)",
            ],
            scaling,
        ),
        "current",
    )

    protocol = read(assets / "benchmark_protocol.json")
    original = protocol["original_code_under_test"]
    identity = (
        f"- Release code: `{original['code_sha']}`\n"
        f"- Release wheel SHA-256: `{original['wheel_sha256']}`\n"
        f"- Release native extension SHA-256: `{original['native_sha256']}`\n"
        f"- Benchmark protocol SHA-256: `{summary['protocol_sha256']}` "
        "(hash of the protocol file as executed; the published copy was later edited "
        "only to remove host names, so its bytes differ)\n"
        "- Statistical reference: R 4.4.0; susieR 0.16.6, "
        "`8e56a8e038e989856d106d9ca5175cc664fea9d2`.\n\n"
        "Both runtime metrics use timing-instrumented copies of these release artifacts; the "
        "instrumented native binary is not byte-identical to the release wheel. The patch, "
        "derived binary hashes and timing boundaries are in the "
        "[measurement archive](reproducibility.md#benchmark-protocol).\n"
    )
    fragment("docs/performance.md", "identity", identity, "current")

    def percent_at_most(values, bound):
        return f"{100 * sum(v <= bound for v in values) / len(values):.1f}%"

    alpha_rows = []
    for panel, cases in primary.items():
        values = [r["max_abs_alpha_error"] for r in cases]
        q1, median, q3 = quartiles(values)
        alpha_rows.append(
            [
                panel,
                f"{len(cases):,}",
                f"{number(median)} [{number(q1)}, {number(q3)}]",
                percent_at_most(values, 1e-8),
                number(max(values)),
                number(statistics.median(r["max_abs_pip_error"] for r in cases)),
                number(max(r["max_abs_pip_error"] for r in cases)),
            ]
        )
    fragment(
        "docs/numerical_accuracy.md",
        "maxima",
        table(
            [
                "Analysis",
                "Cases",
                "Per-case max |Δalpha|, median [IQR]",
                "Cases ≤ 1e-8",
                "Largest",
                "Per-case max |ΔPIP|, median",
                "Largest",
            ],
            alpha_rows,
        ),
    )

    largest = sorted(
        (r for r in records if r["kind"] == "primary"), key=lambda r: -r["max_abs_alpha_error"]
    )[:6]
    largest_text = table(
        [
            "Case",
            "Variants",
            "Max |Δalpha|",
            "Max |ΔPIP|",
            "Iterations (prusie / susieR)",
            "Converged",
            "Credible sets (prusie / susieR)",
        ],
        [
            [
                r["case_id"],
                f"{r['p']:,}",
                number(r["max_abs_alpha_error"]),
                number(r["max_abs_pip_error"]),
                f"{r['prusie']['niter'][1]} / {r['R']['niter'][1]}",
                "neither" if r["both_nonconverged_at_limit"] else "both",
                f"{r['prusie']['cs_count'][1]} / {r['R']['cs_count'][1]}",
            ]
            for r in largest
        ],
    )
    fragment("docs/numerical_accuracy.md", "errors", largest_text)

    def at_limit(cases, backend):
        return sum(not all(r[backend]["converged"]) for r in cases)

    fragment(
        "docs/numerical_accuracy.md",
        "limits",
        table(
            ["Analysis", "prusie nonconverged", "susieR nonconverged"],
            [[p, at_limit(cases, "prusie"), at_limit(cases, "R")] for p, cases in primary.items()],
        )
        + "\nNonconverged fits stopped at `max_iter=100`. They remain in every runtime and "
        "memory summary.\n",
    )

    cs = []
    for panel, cases in primary.items():
        for public, backend in (("prusie", "prusie"), ("susieR", "R")):
            counts = [r[backend]["cs_count"][1] for r in cases]
            cs.append([panel, public, f"{sum(counts):,}", f"{sum(v == 0 for v in counts):,}"])
    fragment(
        "docs/numerical_accuracy.md",
        "cs",
        table(["Analysis", "Backend", "Credible sets", "Cases with no credible set"], cs)
        + "\nCounts come from the first retained call. The full-panel comparison does not "
        "compare set members, coverage or purity; the packaged example and the regression "
        "tests check those.\n",
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
        assert not stale, "Stale generated blocks: " + ", ".join(stale)
    else:
        for path, text in outputs.items():
            path.write_text(text)
    return stale
