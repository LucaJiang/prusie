#!/usr/bin/env python3
"""Run the pinned-R comparison with all generated files in one new directory."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", required=True)
    parser.add_argument("--r-build", default="R", help="Matching R executable for CMD SHLIB")
    parser.add_argument("--output-dir", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--panel", choices=("toy", "regression", "scaling"), default="toy")
    inputs.add_argument("--case-manifest", type=Path, help="Frozen list of case_path records")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--sample-rss", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(HERE / "compare_r.py"),
        "--rscript",
        args.rscript,
        "--output-dir",
        str(out / "results"),
    ]
    if args.validation_only:
        command.append("--validation-only")
    else:
        build = out / "build"
        build.mkdir()
        shutil.copyfile(HERE / "blas_probe.c", build / "blas_probe.c")
        subprocess.run(
            [args.r_build, "CMD", "SHLIB", "blas_probe.c", "-o", "blas_probe.so", "-ldl"],
            cwd=build,
            check=True,
        )
        command.extend(["--blas-probe", str(build / "blas_probe.so")])
    if args.case_manifest is not None:
        command.extend(["--case-manifest", str(args.case_manifest.resolve())])
    elif args.panel == "scaling":
        subprocess.run(
            [sys.executable, str(HERE / "prepare_scaling.py"), "--output", str(out / "inputs")],
            check=True,
        )
        command.extend(["--case-manifest", str(out / "inputs/tasks.json")])
    elif args.panel == "regression":
        command.extend(["--fixture", "regression"])
    if args.sample_rss:
        command.append("--sample-rss")
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
