"""Public display and optional-service isolation; all AI calls are mocked."""
import hashlib
import importlib.util
import json
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
assert 'prusie.ai' not in sys.modules
assert 'matplotlib' not in sys.modules
assert 'scipy' not in sys.modules
print(prusie.__file__)
"""
    result = subprocess.run([sys.executable, "-B", "-c", script],
                            text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "prusie" in result.stdout


def test_result_repr_only_reads_stored_shape_and_status():
    fit = prusie.susie_rss(z=np.array([4., .2, -.3]), R=np.eye(3), n=200, L=2)
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


@pytest.mark.parametrize("request_ai", [False, True])
def test_example_preserves_statistics_before_optional_failure(tmp_path, monkeypatch, request_ai):
    pytest.importorskip("scipy")
    pytest.importorskip("matplotlib")
    import prusie.ai as ai
    from prusie.report import load_analysis
    example = example_module()
    monkeypatch.setattr(example, "make_figures", lambda *args: {})
    before = {}
    calls = []
    def unavailable(self, bundle, *, consent=False, **kwargs):
        assert request_ai, "Offline example must never call the AI provider"
        assert consent is True and self.model == "test-explicit-model"
        for name in ("index.html", "evidence.json", "analysis_summary.json", "summary.txt",
                     "receipt.json", "fits/analysis.json", "fits/arrays.npz"):
            before[name] = (tmp_path / name).read_bytes()
        assert json.loads(before["evidence.json"])["sha256"] == bundle["sha256"]
        calls.append(bundle["sha256"])
        raise RuntimeError("Mock service unavailable")
    monkeypatch.setattr(ai.CodexCLI, "interpret", unavailable)
    def forbidden(*args, **kwargs):
        raise AssertionError("No external process may run during statistical analysis")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    result = example.execute("stable", tmp_path, interpret=request_ai,
                             model="test-explicit-model" if request_ai else None)
    assert Path(result["html"]).is_file()
    fits = load_analysis(tmp_path / "fits")
    assert fits["refined"].pip.shape == fits["baseline"].pip.shape
    assert np.isfinite(fits["refined"].pip).all()
    if request_ai:
        assert len(calls) == 1
        assert all((tmp_path / name).read_bytes() == data for name, data in before.items())
        record = json.loads((tmp_path / "ai_interpretation.json").read_text())
        assert record["status"] == "failed" and record["error_type"] == "RuntimeError"
        assert record["evidence_sha256"] == calls[0]
        assert record["source_files_sha256"]["evidence.json"] == hashlib.sha256(before["evidence.json"]).hexdigest()
        assert json.loads((tmp_path / "analysis.json").read_text())["schema_version"]
    else:
        assert not calls and not (tmp_path / "ai_interpretation.json").exists()
        assert not (tmp_path / "ai").exists()


def test_example_rejects_implicit_model_and_nonpublic_transmission(tmp_path):
    example = example_module()
    with pytest.raises(ValueError, match=r"explicit model"):
        example.execute("stable", tmp_path, interpret=True)
    with pytest.raises(ValueError, match=r"synthetic stable"):
        example.execute("problem", tmp_path, interpret=True, model="test-model")
    assert not list(tmp_path.iterdir())


def test_plot_does_not_call_optional_provider(monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import prusie.ai as ai
    from prusie.plots import plot_convergence
    def forbidden(*args, **kwargs):
        raise AssertionError("Plots must not request AI interpretation")
    monkeypatch.setattr(ai.CodexCLI, "interpret", forbidden)
    fit = prusie.susie_rss(z=np.array([4., .2, -.3]), R=np.eye(3), n=200,
                         L=2, track_fit=True)
    plotted = plot_convergence(fit)
    figure = plotted[0] if isinstance(plotted, tuple) else plotted
    figure.canvas.draw()
    plt.close("all")
