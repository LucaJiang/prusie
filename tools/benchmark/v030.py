#!/usr/bin/env python3
"""Import or redraw the frozen 0.3.0 measurements without repeating fits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent / "_internal"))

from v030_alpha_vs_iterations import load as load_alpha_rows  # noqa: E402
from v030_alpha_vs_iterations import plot as plot_alpha  # noqa: E402
from v030_plots import plot  # noqa: E402
from v030_records import bind_release, export, load_public  # noqa: E402
from v030_tables import render  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("import", "bind-release", "figures", "tables", "check"))
    parser.add_argument(
        "--run", type=Path, help="Private completed archive; required for import and bind-release"
    )
    parser.add_argument("--assets", type=Path, default=ROOT / "docs/assets/benchmark")
    parser.add_argument(
        "--pages",
        help="Comma-separated repository-relative page paths; omit for all current blocks",
    )
    parser.add_argument("--output-dir", type=Path, help="Figure destination; defaults to --assets")
    args = parser.parse_args()
    pages = args.pages.split(",") if args.pages else None
    if args.action == "import":
        if args.run is None:
            parser.error("import requires --run")
        summary = export(args.run, args.assets)
        print("Accounted tasks:", summary["all_tasks_accounted"])
    elif args.action == "bind-release":
        if args.run is None:
            parser.error("bind-release requires --run")
        release = bind_release(args.run, args.assets)
        print("Qualified, separately untimed release:", release["code_commit"])
    elif args.action == "figures":
        summary, m1, records = load_public(args.assets)
        output = args.output_dir or args.assets
        plot(summary, m1, records, output)
        plot_alpha(load_alpha_rows(args.assets / "benchmark_records.jsonl.gz"), output)
        print("Rendered SVG/PDF figure pairs from retained measurements")
    else:
        assert args.assets.resolve() == (ROOT / "docs/assets/benchmark").resolve(), (
            "Tables use the checkout's published records"
        )
        changed = render(ROOT, pages=pages, check=args.action == "check")
        print(
            "Current 0.3.0 blocks:", len(changed), "changed" if args.action == "tables" else "stale"
        )


if __name__ == "__main__":
    main()
