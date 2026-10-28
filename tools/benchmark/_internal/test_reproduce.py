"""The public benchmark command delegates without changing fitting protocols."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def reproduce():
    path = Path(__file__).resolve().parents[1] / "reproduce.py"
    spec = importlib.util.spec_from_file_location("benchmark_reproduce", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validation_only_uses_internal_driver(reproduce, monkeypatch, tmp_path):
    output = tmp_path / "new-results"
    monkeypatch.setattr(
        reproduce.sys,
        "argv",
        ["reproduce.py", "--rscript", "Rscript", "--validation-only", "--output-dir", str(output)],
    )
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(reproduce.subprocess, "run", run)
    assert reproduce.main() == 0
    assert len(calls) == 1
    assert Path(calls[0][1]).name == "compare_r.py"
    assert "_internal" in calls[0][1]
    assert "--validation-only" in calls[0]
    assert not (output / "build").exists()


def test_render_records_does_not_require_r(reproduce, monkeypatch, tmp_path):
    output = tmp_path / "figures"
    monkeypatch.setattr(
        reproduce.sys, "argv", ["reproduce.py", "--render-records", "--output-dir", str(output)]
    )
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(reproduce.subprocess, "run", run)
    assert reproduce.main() == 0
    assert len(calls) == 1
    assert Path(calls[0][1]).name == "v030.py"
    assert calls[0][2] == "figures"
    assert calls[0][calls[0].index("--output-dir") + 1] == str(output)
    assert "--real" not in calls[0] and "--scaling" not in calls[0]
    assert "Rscript" not in calls[0]
    assert not (output / "build").exists()


def test_missing_rscript_fails_before_creating_output(reproduce, monkeypatch, tmp_path):
    output = tmp_path / "no-results"
    monkeypatch.setattr(reproduce.sys, "argv", ["reproduce.py", "--output-dir", str(output)])
    with pytest.raises(SystemExit) as error:
        reproduce.main()
    assert error.value.code == 2
    assert not output.exists()


def test_existing_output_is_not_overwritten(reproduce, monkeypatch, tmp_path):
    monkeypatch.setattr(
        reproduce.sys, "argv", ["reproduce.py", "--render-records", "--output-dir", str(tmp_path)]
    )
    with pytest.raises(FileExistsError):
        reproduce.main()
