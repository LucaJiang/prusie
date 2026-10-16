#!/usr/bin/env python3
"""Check the installation against the frozen coloc examples and R outputs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from reference_coloc import main

if __name__ == "__main__":
    raise SystemExit(main())
