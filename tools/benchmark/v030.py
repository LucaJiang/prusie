#!/usr/bin/env python3
"""Redraw the 0.3.0 benchmark figures or regenerate their tables without fitting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent / "_internal"))

from v030_alpha_vs_iterations import plot as plot_alpha  # noqa: E402
from v030_plots import plot  # noqa: E402
from v030_tables import render  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("figures", "tables", "check"))
    parser.add_argument("--assets", type=Path, default=ROOT / "docs/assets/benchmark")
    parser.add_argument(
        "--pages",
        help="Comma-separated repository-relative page paths; omit for all current blocks",
    )
    parser.add_argument("--output-dir", type=Path, help="Figure destination; defaults to --assets")
    args = parser.parse_args()
    pages = args.pages.split(",") if args.pages else None
    if args.action == "figures":
        output = args.output_dir or args.assets
        plot(args.assets, output)
        plot_alpha(args.assets, output)
        print("Rendered SVG/PDF figure pairs from the published records")
    else:
        assert args.assets.resolve() == (ROOT / "docs/assets/benchmark").resolve(), (
            "Tables use the checkout's published records"
        )
        changed = render(ROOT, pages=pages, check=args.action == "check")
        print("Generated blocks:", len(changed), "changed" if args.action == "tables" else "stale")


if __name__ == "__main__":
    main()
