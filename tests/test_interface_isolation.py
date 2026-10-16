"""Public display and offline import behavior."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import prusie


def example_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "post_analysis.py"
    spec = importlib.util.spec_from_file_location("prusie_post_analysis_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_needs_no_optional_provider_or_process():
    script = """
import subprocess, socket, sys
def forbidden(*args, **kwargs):
    raise AssertionError('Import must not start a process or network request')
subprocess.Popen = forbidden
socket.create_connection = forbidden
import prusie
assert 'matplotlib' not in sys.modules
assert 'scipy' not in sys.modules
print(prusie.__file__)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script], text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "prusie" in result.stdout


def test_result_repr_only_reads_stored_shape_and_status():
    fit = prusie.susie_rss(z=np.array([4.0, 0.2, -0.3]), R=np.eye(3), n=200, L=2)
    original = fit.as_dict()
    text = repr(fit)
    assert text.startswith("SusieResult(variables=3, components=2, credible_sets=")
    assert f"niter={fit.niter}" in text
    assert f"converged={fit.converged}" in text
    assert fit.convergence["stop_reason"] in text
    assert "array(" not in text and len(text) < 200
    assert all(fit.as_dict()[key] is value for key, value in original.items())

    class ShapeOnly:
        shape = (2, 3)

        def __array__(self, *args, **kwargs):
            raise AssertionError("repr must not materialize or reduce arrays")

        def __repr__(self):
            raise AssertionError("repr must not format arrays")

    fit.alpha = ShapeOnly()
    fit.sets = {"cs": {"L0": [0], "L7": [1]}}
    fit.convergence = {"stop_reason": "max_iter"}
    fit.converged = False
    assert "credible_sets=2" in repr(fit)
    assert "stop_reason='max_iter'" in repr(fit)
    fit.sets = {"cs": None}
    fit.convergence = {}
    assert "credible_sets=0" in repr(fit)
    assert "stop_reason='not_available'" in repr(fit)


def test_example_preserves_statistics_offline(tmp_path, monkeypatch):
    pytest.importorskip("scipy")
    pytest.importorskip("matplotlib")
    from prusie.report import load_analysis

    example = example_module()
    monkeypatch.setattr(example, "make_figures", lambda *args: {})

    def forbidden(*args, **kwargs):
        raise AssertionError("No external process may run during statistical analysis")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    result = example.execute("stable", tmp_path)
    assert Path(result["html"]).is_file()
    fits = load_analysis(tmp_path / "fits")
    assert fits["refined"].pip.shape == fits["baseline"].pip.shape
    assert np.isfinite(fits["refined"].pip).all()
