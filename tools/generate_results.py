#!/usr/bin/env python3
"""Regenerate current documentation tables from retained benchmark records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/benchmark/_internal"))
from v030_tables import render  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", help="Comma-separated repository-relative pages")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    changed = render(ROOT, pages=args.pages.split(",") if args.pages else None, check=args.check)
    print("Current blocks:", len(changed), "stale" if args.check else "updated")


if __name__ == "__main__":
    main()
