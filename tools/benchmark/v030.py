#!/usr/bin/env python3
"""Import or redraw the frozen 0.3.0 measurements without repeating fits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent / "_internal"))

from v030_plots import plot  # noqa: E402
from v030_records import export, load_public  # noqa: E402
from v030_tables import render  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("import", "figures", "tables", "check"))
    parser.add_argument(
        "--run", type=Path, help="Private completed measurement archive; required only for import"
    )
    parser.add_argument("--assets", type=Path, default=ROOT / "docs/assets/benchmark")
    parser.add_argument(
        "--pages",
        help="Comma-separated repository-relative page paths; omit for all current blocks",
    )
    args = parser.parse_args()
    pages = args.pages.split(",") if args.pages else None
    if args.action == "import":
        if args.run is None:
            parser.error("import requires --run")
        summary = export(args.run, args.assets)
        print("Accounted tasks:", summary["all_tasks_accounted"])
    elif args.action == "figures":
        summary, m1, records = load_public(args.assets)
        plot(summary, m1, records, args.assets)
        print("Rendered eight SVG/PDF figure pairs from retained measurements")
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
