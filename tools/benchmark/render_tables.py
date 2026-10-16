"""Format the recorded benchmark summaries without running or modifying fits."""

import json
from pathlib import Path


def render(root: Path, fragment, table):
    """Update marked prose from the full-precision, read-only JSON records."""
    folder = root / "docs/assets/benchmark"
    real = json.loads((folder / "real_summary.json").read_text())
    scaling = json.loads((folder / "scaling_summary.json").read_text())
    rows = []
    for key, label in (("gwas", "GWAS"), ("eqtl", "eQTL")):
        group = real["groups"][key]
        difference = max(
            row["numerical_pip_max_abs_error"] for row in real["records"] if row["group"] == key
        )
        rows.append(
            [
                label,
                group["cases"],
                f"{difference:.2e}",
                f"{group['runtime_speedup_geometric_mean']:.2g}×",
                f"{group['full_process_memory_ratio']['median']:.2g}×",
            ]
        )
    highlights = f"Measured **{real['version']}** against susieR 0.16.6.\n"
    highlights += table(
        [
            "Analysis",
            "Loci",
            "Max. absolute PIP difference vs susieR",
            "Speed-up¹",
            "Memory ratio²",
        ],
        rows,
    )
    highlights += (
        "\n¹ Geometric mean of per-locus susieR/prusie runtime ratios. "
        "² Median per-locus susieR/prusie ratio of peak resident memory (RSS), "
        "including the interpreter and inputs.\n"
    )
    for page in ("README.md", "docs/index.md", "docs/performance.md"):
        fragment(page, "highlights", highlights, prefix="current")
    rows = []
    memory = []
    for key, label in (("gwas", "GWAS"), ("eqtl", "eQTL")):
        group = real["groups"][key]
        native, reference = (group["backends"][name] for name in ("prusie", "susieR"))
        rows.append(
            [
                label,
                f"{native['runtime_seconds']['median'] * 1000:.3g}",
                f"{reference['runtime_seconds']['median'] * 1000:.3g}",
                f"{group['faster']} / {group['tied']} / {group['slower']}",
                f"{native['repeat_iqr_over_median']['median']:.3g}",
                f"{reference['repeat_iqr_over_median']['median']:.3g}",
            ]
        )
        for backend in ("prusie", "susieR"):
            data = group["backends"][backend]
            memory.append(
                [f"{label} / {backend}"]
                + [
                    f"{data[field]['median']:.3g}"
                    for field in (
                        "full_process_peak_mib",
                        "baseline_rss_mib",
                        "sampled_fit_peak_mib",
                        "sampled_incremental_peak_mib",
                    )
                ]
            )
    fragment(
        "docs/performance.md",
        "runtime",
        table(
            [
                "Analysis",
                "prusie (ms)",
                "susieR (ms)",
                "Faster / tied / slower",
                "prusie repeat IQR/median",
                "susieR repeat IQR/median",
            ],
            rows,
        ),
        prefix="current",
    )
    fragment(
        "docs/performance.md",
        "memory",
        table(
            [
                "Analysis / backend",
                "Peak RSS (MiB)",
                "Before fitting (MiB)",
                "Sampled fit peak (MiB)",
                "Sampled increment (MiB)",
            ],
            memory,
        ),
        prefix="current",
    )
    rows = []
    for case in sorted(scaling["records"], key=lambda row: row["n_variants"]):
        for name in ("prusie", "susieR"):
            data = case["backends"][name]
            rows.append(
                [
                    case["n_variants"],
                    name,
                    f"{data['runtime_seconds']['median']:.3g}",
                    data["niter"],
                    data["converged"],
                    f"{data['peak_memory_mib']:.3g}",
                    f"{data['sampled_memory']['sampled_incremental_peak_bytes'] / 2**20:.3g}",
                ]
            )
    fragment(
        "docs/performance.md",
        "scaling",
        table(
            [
                "Variants",
                "Backend",
                "Median (s)",
                "Iterations",
                "Converged",
                "Peak RSS (MiB)",
                "Sampled increment (MiB)",
            ],
            rows,
        ),
        prefix="current",
    )
    fragment(
        "docs/performance.md",
        "identity",
        f"- Compute commit: `{real['compute_code_commit']}`\n"
        f"- CPython 3.12 wheel SHA-256: `{real['wheel_sha256']}`\n"
        f"- Native extension SHA-256: `{real['native_sha256']}`\n",
        prefix="current",
    )
