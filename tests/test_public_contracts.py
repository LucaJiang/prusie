"""Public typing, optional imports and compatibility of the visualization API."""

import ast
import hashlib
import importlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, get_args, get_type_hints

import prusie


def test_exported_functions_have_resolvable_annotations():
    for name in prusie.__all__:
        function = getattr(prusie, name)
        if not inspect.isfunction(function):
            continue
        signature = inspect.signature(function)
        hints = get_type_hints(function)
        assert "return" in hints, name
        for parameter in signature.parameters.values():
            assert parameter.name in hints, (name, parameter.name)


def test_public_module_functions_have_annotations():
    package = Path(prusie.__file__).parent
    for path in package.rglob("*.py"):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
                continue
            if path.stem.startswith("_") and path.parent.name != "viz":
                continue
            arguments = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            arguments += [argument for argument in (node.args.vararg, node.args.kwarg) if argument]
            assert node.returns is not None, (path.name, node.name)
            assert all(argument.annotation is not None for argument in arguments), (
                path.name,
                node.name,
            )


def test_import_does_not_load_optional_analysis_modules(tmp_path):
    package_root = str(Path(prusie.__file__).parent.parent)
    environment = dict(os.environ, PYTHONPATH=package_root)
    command = """
import sys
import prusie
for name in ('scipy', 'matplotlib', 'prusie.analysis', 'prusie.comparison',
             'prusie.ld', 'prusie.report', 'prusie.viz'):
    assert name not in sys.modules, name
assert callable(prusie.susie_rss)
assert callable(prusie.summary_text)
assert 'prusie.report' in sys.modules
"""
    subprocess.run(
        [sys.executable, "-B", "-c", command],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_visualization_compatibility_imports():
    viz = importlib.import_module("prusie.viz")
    plots = importlib.import_module("prusie.plots")
    for name in viz.__all__:
        assert getattr(plots, name) is getattr(viz, name)


def test_result_metadata_has_named_schemas():
    hints = get_type_hints(prusie.SusieResult)
    assert hints["params"].__name__ == "FitParameters"
    assert hints["qc"].__name__ == "InputChecks"
    assert get_args(hints["cs_audit"])[0].__name__ == "CredibleSetAudit"
    assert Any not in get_args(hints["_prepared"])


def test_installed_package_declares_inline_types():
    assert (Path(prusie.__file__).parent / "py.typed").is_file()


def test_frozen_parity_manifest_checksums():
    directory = Path(__file__).parent / "fixtures/susier_parity_0166"
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected, name
