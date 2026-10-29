"""Requested controls survive continuation, search, persistence and hashing."""

import hashlib
import json

import numpy as np
import pytest

from prusie import ConvergenceConfig, RefinementConfig, continue_fit, refine_fit, susie_rss
from prusie.diagnostics import resolve_convergence
from prusie.refinement import _HASH_BUFFER_ELEMENTS, _hash_float64_c_order, data_signature
from prusie.report import load_analysis, save_analysis


def fit_small(**overrides):
    options = dict(
        z=[6.0, 3.0, 0.1],
        R=[[1.0, 0.5, 0.0], [0.5, 1.0, 0.1], [0.0, 0.1, 1.0]],
        n=200,
        L=2,
        retain_data=True,
        check_prior=False,
    )
    options.update(overrides)
    return susie_rss(**options)


STRICT = ConvergenceConfig("joint", alpha_tol=2e-8, pip_tol=3e-8, min_iter=5, cycle_lag=3)


def test_requested_controls_inherit_override_and_survive_persistence(tmp_path):
    before = fit_small(convergence=STRICT)
    continued = continue_fit(before, tol=1e-7)
    assert continued.params["convergence"] == STRICT.options()
    assert continued.convergence["requested_config"] == STRICT.options()
    assert continued.convergence["config_resolution"]["source"] == "inherited_params"
    assert continued.niter >= STRICT.min_iter
    replacement = ConvergenceConfig("alpha_pip", 1e-6, 2e-6, 3, 2)
    changed = continue_fit(continued, convergence=replacement)
    assert changed.params["convergence"] == replacement.options()
    assert changed.params["convergence_resolution"]["previous_requested"] == STRICT.options()
    assert changed.params["tol"] == continued.params["tol"] == 1e-7
    assert changed.convergence["tol_resolution"] == "inherited_params"
    # The immutable data originated with STRICT, but a later explicit request wins.
    assert continue_fit(changed).params["convergence"] == replacement.options()
    assert before.params["convergence"] == STRICT.options()
    json.dumps(changed.params["convergence"])
    save_analysis(changed, tmp_path / "fit")
    restored = load_analysis(tmp_path / "fit")
    assert resolve_convergence(restored)[0] == replacement
    assert fit_small(s_init=restored).params["convergence"] == replacement.options()


def test_omitted_continuation_tol_preserves_strict_baseline():
    before = fit_small(tol=1e-8)
    inherited = continue_fit(before)
    assert inherited.params["tol"] == inherited.convergence["tol"] == 1e-8
    overridden = continue_fit(before, tol=0.002)
    assert overridden.params["tol"] == 0.002
    assert overridden.convergence["tol_resolution"] == "explicit_override"


def test_nonfinite_effective_fallback_is_not_next_request():
    fit = fit_small(convergence=STRICT)
    fit.convergence["effective_criterion"] = "alpha_pip"
    fit.convergence["fallback_iterations"] = [2]
    assert continue_fit(fit).convergence["requested_criterion"] == "joint"
    # Legacy diagnostic-only results recover REQUESTED rather than effective.
    del fit.params["convergence"]
    control, source = resolve_convergence(fit)
    assert control == STRICT
    assert source["source"] == "recovered_from_diagnostics"
    fit.convergence = {}
    assert resolve_convergence(fit)[1]["source"] == "recovered_prepared_options"
    fit._prepared = None
    control, source = resolve_convergence(fit)
    assert control == ConvergenceConfig() and source["source"] == "default_unrecorded"
    assert "No inheritance" in source["compatibility_note"]


@pytest.mark.parametrize("override", [None, ConvergenceConfig("alpha_pip", 1e-7, 2e-7, 4, 2)])
def test_both_refinement_stages_record_inherited_controls_and_inner_tol(override):
    fit = fit_small(convergence=STRICT, track_fit="compact")
    expected = override or STRICT
    result = refine_fit(fit, RefinementConfig(max_rounds=1, inner_tol=7e-5), convergence=override)
    assert result.refinement["requested_convergence"] == expected.options()
    stages = [stage for row in result.refinement["history"] for stage in row["stages"].values()]
    assert len(stages) >= 2
    for stage in stages:
        assert stage["solver_settings"]["convergence"] == expected.options()
        assert stage["solver_settings"]["tol"] == 7e-5
        assert stage["convergence"]["tol"] == 7e-5
        assert stage["convergence"]["requested_config"] == expected.options()
        assert stage["niter"] >= expected.min_iter
    # If no candidate wins, the actual returned baseline retains its own request.
    accepted = any(row["accepted"] for row in result.refinement["history"])
    assert result.params["convergence"] == (expected if accepted else STRICT).options()


def test_cycle_warning_uses_actual_iterations_and_reason(monkeypatch):
    from prusie import _native

    original = _native.fit

    def stopped(*args, **kwargs):
        native = original(*args, **kwargs)
        native["niter"] = 3
        native["converged"] = False
        native["diagnostics"]["stop_reason"] = "cycle_detected"
        return native

    monkeypatch.setattr(_native, "fit", stopped)
    with pytest.warns(RuntimeWarning, match=r"after 3 iterations.*cycle_detected"):
        fit = fit_small(max_iter=100)
    assert not any("100 iterations" in warning for warning in fit.warnings)


def unstreamed_signature(matrix, response, yty, n, scales, multiplier, ids, metadata):
    h = hashlib.sha256()
    for a in (matrix, response, np.asarray([yty, n, multiplier]), scales):
        a = np.asarray(a, dtype=np.float64)
        h.update(str(a.shape).encode())
        h.update(a.tobytes(order="C"))
    for value in ids:
        h.update(str(value).encode())
        h.update(b"\0")
    for key in ("effect_allele", "other_allele"):
        h.update(key.encode())
        for value in metadata.get(key, []):
            h.update(str(value).encode())
            h.update(b"\0")
    return h.hexdigest()


@pytest.mark.parametrize(
    "layout", ["C", "F", "transpose", "strided", "reversed", "float32", "empty"]
)
def test_streamed_signature_matches_unstreamed_digest(layout):
    matrix = np.arange(30.0, dtype=np.float64).reshape(5, 6)
    if layout == "F":
        matrix = np.asfortranarray(matrix)
    elif layout == "transpose":
        matrix = matrix.T
    elif layout == "strided":
        matrix = matrix[::2, ::2]
    elif layout == "reversed":
        matrix = matrix[::-1, ::-1]
    elif layout == "float32":
        matrix = matrix.astype("float32")
    elif layout == "empty":
        matrix = matrix[:0]
    args = (
        matrix,
        np.arange(5.0)[::2],
        40.0,
        50,
        np.ones(5),
        1.0,
        ["a", "b"],
        {"effect_allele": ["A", "T"], "other_allele": ["C", "G"]},
    )
    assert data_signature(*args) == unstreamed_signature(*args)


def test_identity_changes_remain_detectable():
    args = dict(
        matrix=np.eye(3),
        response=np.arange(3.0),
        yty=50.0,
        n=100,
        scales=np.ones(3),
        multiplier=1.0,
        ids=["a", "b", "c"],
        metadata={"effect_allele": ["A", "C", "G"], "other_allele": ["G", "T", "A"]},
    )
    original = data_signature(**args)
    changes = [
        dict(matrix=np.eye(3) * 2),
        dict(response=np.arange(3.0)[::-1]),
        dict(ids=["b", "a", "c"]),
        dict(scales=np.full(3, 2.0)),
        dict(multiplier=2.0),
        dict(metadata={"effect_allele": ["G", "C", "G"], "other_allele": ["A", "T", "A"]}),
    ]
    assert all(data_signature(**(args | change)) != original for change in changes)


@pytest.mark.parametrize("order", ["C", "F"])
def test_hash_update_buffers_are_bounded(order):
    class Recorder:
        def __init__(self):
            self.sizes = []

        def update(self, chunk):
            self.sizes.append(len(chunk))

    h = Recorder()
    _hash_float64_c_order(h, np.ones((512, 512), dtype="float32", order=order))
    assert max(h.sizes) <= _HASH_BUFFER_ELEMENTS * 8
    assert sum(h.sizes[1:]) == 512 * 512 * 8


def test_hash_implementation_and_tracking_leave_default_posterior_exact(monkeypatch):
    import prusie.refinement as signature_module

    current = fit_small(retain_data=False)
    compact = fit_small(track_fit="compact")
    dense = fit_small(track_fit="dense")
    monkeypatch.setattr(signature_module, "data_signature", unstreamed_signature)
    unstreamed = fit_small(retain_data=True)
    for fit in (unstreamed, compact, dense):
        for name in ("pip", "alpha", "mu", "mu2", "V", "elbo", "KL", "XtXr", "lbf", "lbf_variable"):
            assert getattr(fit, name).tobytes() == getattr(current, name).tobytes(), name
        assert fit.sigma2 == current.sigma2
        assert fit.data_signature == compact.data_signature
    assert current.data_signature is None


def test_development_identity_is_distinct_and_installed():
    import importlib.metadata

    import prusie

    assert prusie.__version__ == importlib.metadata.version("prusie")
    assert prusie.__version__ != "0.2.3rc6"
