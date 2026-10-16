#!/usr/bin/env python3
"""Inspect and install a wheel, rebuild its sdist, and check the rebuilt wheel.

Run manually or in CI with Python 3.12, a build toolchain and auditwheel. Runtime
checks use fresh environments whose PATH contains no Rust, Cargo or maturin.
No package, tag or Release is published.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import tomllib
from license_check import check_license_texts, check_licenses, license_paths

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ignored(path):
    return "__pycache__" in path.parts or path.suffix in (".pyc", ".so", ".pyd")


def inspect_wheel(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        dist = next(n.rsplit("/", 1)[0] for n in names if n.endswith(".dist-info/WHEEL"))
        metadata = archive.read(dist + "/METADATA").decode()
        requirements = [line for line in metadata.splitlines() if line.startswith("Requires-Dist:")]
        assert not any(
            name in metadata.lower()
            for name in (
                "requires-dist: markdown",
                "requires-dist: pymdown",
                "requires-dist: pygments",
            )
        )
        wheel = archive.read(dist + "/WHEEL").decode()
        # The installed package must carry every Python source and frozen data file.
        runtime = ROOT / "src/prusie"
        required = [p for p in runtime.rglob("*") if p.is_file() and not ignored(p)]
        for source in required:
            member = "prusie/" + source.relative_to(runtime).as_posix()
            assert member in names, f"Missing wheel runtime file: {member}"
            assert archive.read(member) == source.read_bytes(), f"Altered runtime file: {member}"
        native = [
            n for n in names if n.startswith("prusie/_native.") and n.endswith((".so", ".pyd"))
        ]
        assert len(native) == 1, "Wheel must include one native extension"
        check_licenses(ROOT)
        payload = {}
        for name in license_paths(ROOT):
            member = f"{dist}/licenses/{name}"
            assert names.count(member) == 1, f"Missing or duplicated distribution notice: {name}"
            payload[name] = archive.read(member)
            assert payload[name] == (ROOT / name).read_bytes(), f"Altered notice: {name}"
            assert f"License-File: {name}" in metadata, f"Undeclared license payload: {name}"
        # Validate the terms read from the artifact itself, not just from the checkout.
        license_check = check_license_texts(lambda name: payload[name].decode())
        assert "License-Expression: GPL-3.0-or-later" in metadata
        assert not any(n.startswith(("docs/", "tests/", "dev/")) for n in names)
        return dict(
            filename=path.name,
            sha256=sha(path),
            compressed_bytes=path.stat().st_size,
            unpacked_bytes=sum(x.file_size for x in archive.infolist()),
            member_count=sum(not x.is_dir() for x in archive.infolist()),
            WHEEL=wheel,
            METADATA=metadata,
            runtime_requirements=requirements,
            docs_dependencies_absent=True,
            native_members=native,
            native_sha256=hashlib.sha256(archive.read(native[0])).hexdigest(),
            license_content_verified=license_check,
            license_members=[n for n in names if "license" in n.lower() or "notices" in n.lower()],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    wheel, sdist, work = args.wheel.resolve(), args.sdist.resolve(), args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    for name in ("tmp", "cache", "logs", "unpacked", "rebuilt"):
        (work / name).mkdir()
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        TMPDIR=str(work / "tmp"),
        PIP_CACHE_DIR=os.environ.get("PIP_CACHE_DIR", str(work / "cache")),
        CARGO_TARGET_DIR=str(work / "target"),
    )
    for name in (
        "PRUSIE_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        env[name] = "1"
    record = dict(
        passed=False,
        platform=platform.platform(),
        python=platform.python_version(),
        wheel=inspect_wheel(wheel),
        sdist=dict(filename=sdist.name, sha256=sha(sdist)),
        steps=[],
        builds=dict(
            RUSTFLAGS=env.get("RUSTFLAGS"),
            CARGO_ENCODED_RUSTFLAGS=env.get("CARGO_ENCODED_RUSTFLAGS"),
            CC=env.get("CC"),
            CFLAGS=env.get("CFLAGS"),
        ),
    )

    def save():
        (work / "artifact_checks.json").write_text(json.dumps(record, indent=2) + "\n")

    def run(label, argv, cwd=work, overrides=None):
        active_env = env | (overrides or {})
        log = work / "logs" / (label + ".log")
        start = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with log.open("w") as output:
            process = subprocess.run(
                list(map(str, argv)),
                cwd=cwd,
                env=active_env,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record["steps"].append(
            dict(
                label=label,
                argv=list(map(str, argv)),
                cwd=str(cwd),
                exit_code=process.returncode,
                environment={
                    key: active_env[key]
                    for key in (
                        "PATH",
                        "TMPDIR",
                        "PIP_CACHE_DIR",
                        "CARGO_HOME",
                        "CARGO_TARGET_DIR",
                        "RUSTUP_TOOLCHAIN",
                        "RUSTFLAGS",
                        "PYTHONDONTWRITEBYTECODE",
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "PRUSIE_NUM_THREADS",
                        "PRUSIE_LD_ORACLE",
                        "PRUSIE_LD_CERTIFIED_ORACLE",
                        "PYTEST_ADDOPTS",
                    )
                    if key in active_env
                },
                log=str(log),
                start=start,
                end=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            )
        )
        save()
        if process.returncode:
            raise RuntimeError(f"{label} failed; see {log}")

    try:
        with tarfile.open(sdist) as archive:
            record["sdist"].update(
                compressed_bytes=sdist.stat().st_size,
                unpacked_bytes=sum(x.size for x in archive.getmembers() if x.isfile()),
            )
            archive.extractall(work / "unpacked", filter="data")
        sources = list((work / "unpacked").glob("*/pyproject.toml"))
        assert len(sources) == 1, "Expected exactly one source project"
        source = sources[0].parent
        config = tomllib.loads(sources[0].read_text())
        version = config["project"]["version"]
        # Directory contracts follow the source responsibilities; individual files
        # are discovered instead of maintained in a second packaging manifest.
        required = set(license_paths(ROOT)) | {
            "pyproject.toml",
            "Cargo.toml",
            "Cargo.lock",
            "build.rs",
            "README.md",
            "rust-toolchain.toml",
            "rustfmt.toml",
            "requirements-dev.lock",
            "examples/check_example.py",
            "examples/quickstart.py",
            "examples/post_analysis.py",
        }
        for directory in (
            "src/prusie",
            "rust",
            "bindings",
            "tests",
            "dev/references",
            "examples/data",
            "tools",
            "docs/assets",
            "docs/licenses",
        ):
            required.update(
                p.relative_to(ROOT).as_posix()
                for p in (ROOT / directory).rglob("*")
                if p.is_file() and not ignored(p)
            )
        required.update(
            p.relative_to(ROOT).as_posix()
            for pattern in ("*.md", "*.html")
            for p in (ROOT / "docs").glob(pattern)
        )
        for name in sorted(required):
            assert (source / name).is_file(), f"Incomplete buildable/testable sdist: {name}"
            if name == "Cargo.toml":
                # Maturin copies the Python readme into Cargo's package metadata.
                # Permit only that documented packaging addition, not dependency,
                # feature, license or build-source changes.
                expected = tomllib.loads((ROOT / name).read_text())
                expected["package"].setdefault("readme", config["project"]["readme"])
                assert tomllib.loads((source / name).read_text()) == expected, (
                    "Unexpected source Cargo manifest change"
                )
            else:
                assert (source / name).read_bytes() == (ROOT / name).read_bytes(), (
                    f"Source payload mismatch: {name}"
                )
        assert not (source / "docs/examples").exists(), "Website examples belong in the checkout"
        record["sdist"]["required_source_payload"] = {n: sha(source / n) for n in sorted(required)}
        record["sdist"]["license_content_verified"] = check_licenses(source)
        math_assets = json.loads((source / "tools/math-assets.lock.json").read_text())
        assert all(
            sha(source / name) == expected for name, expected in math_assets["files"].items()
        )
        record["sdist"]["math_assets_verified"] = len(math_assets["files"])
        record["sdist"]["math_license"] = math_assets["license"]

        def install_and_check(label, artifact):
            runtime = work / (label + "-env")
            run(label + "-venv", [sys.executable, "-m", "venv", runtime])
            python = runtime / "bin/python"
            path = str(runtime / "bin")
            # Explicit wheel and binary-only dependencies: pip cannot compile source.
            run(
                label + "-install",
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "--no-compile",
                    "--only-binary=:all:",
                    "numpy==2.2.6",
                    artifact,
                ],
                overrides={"PATH": path},
            )
            probe = """import hashlib, importlib.metadata, json, pathlib, shutil, sys
import importlib.util
import prusie
from prusie import _native
prefix=pathlib.Path(sys.prefix).resolve()
assert pathlib.Path(prusie.__file__).resolve().is_relative_to(prefix)
assert pathlib.Path(_native.__file__).resolve().is_relative_to(prefix)
assert not any(shutil.which(x) for x in ('rustc','cargo','maturin','R','Rscript'))
assert importlib.util.find_spec('scipy') is None
assert importlib.util.find_spec('matplotlib') is None
assert prusie.__version__ == sys.argv[1] == importlib.metadata.version('prusie')
example=prusie.load_example()
fit=prusie.susie_rss(**example['inputs'], **example['parameters'])
assert fit.pip.shape == (500,) and fit.converged
print(json.dumps(dict(version=prusie.__version__,module=prusie.__file__,native=_native.__file__,native_sha256=hashlib.sha256(pathlib.Path(_native.__file__).read_bytes()).hexdigest(),rust_on_path=False)))
"""
            run(label + "-identity", [python, "-I", "-c", probe, version], overrides={"PATH": path})
            run(label + "-pip-check", [python, "-m", "pip", "check"], overrides={"PATH": path})
            run(
                label + "-example",
                [
                    python,
                    "-I",
                    source / "examples/check_example.py",
                    "--output-dir",
                    work / (label + "-example"),
                ],
                overrides={"PATH": path},
            )
            run(
                label + "-quickstart",
                [python, "-I", source / "examples/quickstart.py"],
                overrides={"PATH": path},
            )

        install_and_check("wheel", wheel)
        run(
            "sdist-rebuild",
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--wheel-dir",
                work / "rebuilt",
                source,
            ],
            cwd=work,
        )
        rebuilt = list((work / "rebuilt").glob("prusie-*.whl"))
        assert len(rebuilt) == 1
        record["rebuilt_wheel"] = inspect_wheel(rebuilt[0])
        install_and_check("rebuilt", rebuilt[0])
        # Run the complete existing suite from the extracted source distribution.
        # The runtime-only installations above remain separate and minimal.
        test_env = work / "sdist-test-env"
        run("sdist-test-venv", [sys.executable, "-m", "venv", test_env])
        test_python = test_env / "bin/python"
        run(
            "sdist-test-dependencies",
            [
                test_python,
                "-m",
                "pip",
                "install",
                "--no-compile",
                rebuilt[0],
                "-r",
                source / "requirements-dev.lock",
                "-r",
                source / "tools/docs-requirements.txt",
                "-r",
                source / "tools/results-requirements.txt",
            ],
        )
        run(
            "sdist-pytest",
            [test_python, "-m", "pytest", "tests", "tools/benchmark/test_helpers.py"],
            cwd=source,
        )
        run("sdist-pip-check", [test_python, "-m", "pip", "check"])
        if sys.platform.startswith("linux"):
            run("auditwheel", [sys.executable, "-m", "auditwheel", "show", wheel])
            run("auditwheel-rebuilt", [sys.executable, "-m", "auditwheel", "show", rebuilt[0]])
        record["passed"] = True
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    save()
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
