"""Preparation-only trend constructors match the frozen susieR boundaries."""

import json
from pathlib import Path
from typing import get_type_hints

import numpy as np
import pytest

import prusie
from prusie import _native

FIXTURES = Path(__file__).parent / "fixtures/susier_parity_0166"
REFERENCE = json.loads((FIXTURES / "reference.json").read_text())
BOUNDARIES = json.loads((FIXTURES / "trend_constructor_boundaries.json").read_text())


@pytest.fixture
def no_native_fit(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("preparation must not invoke a native fit or SER update")

    for name in ("fit", "fit_design_dense", "fit_design_csc", "fit_design_trend", "ser_update"):
        monkeypatch.setattr(_native, name, fail)


def test_trendfilter_init_only_without_mad(no_native_fit):
    response = REFERENCE["datasets"][BOUNDARIES["input"]["dataset"]]["y"]
    reference = BOUNDARIES["cases"]["trend_noMAD_init_only"]
    assert reference["status"] == "ok"
    prepared = prusie.susie_trendfilter(response, L=3, use_mad=False, init_only=True)
    assert isinstance(prepared, prusie.SusieInputs)
    assert prepared.params["init_only"] is True
    assert prepared._request.deferred_model_init is None
    expected = reference["structure"]["fields"]
    assert list(prepared.data["X"].shape) == expected["data"]["fields"]["X"]["dim"]
    assert prepared.params.get("model_init") is None
    assert expected["params"]["fields"]["model_init"]["typeof"] == "NULL"


def test_trendfilter_mad_init_only_retains_nested_preparation(no_native_fit):
    response = np.asarray(REFERENCE["datasets"][BOUNDARIES["input"]["dataset"]]["y"])
    reference = BOUNDARIES["cases"]["trend_MAD_init_only"]
    assert reference["status"] == "ok"
    assert BOUNDARIES["scientific_fits"] == 0
    assert BOUNDARIES["ibss_update_guard_calls"] == 0
    prepared = prusie.susie_trendfilter(response, L=3, init_only=True)
    assert isinstance(prepared, prusie.SusieInputs)
    initial = prepared.params["model_init"]
    assert isinstance(initial, prusie.SusieInputs)
    assert prepared._request.deferred_model_init is initial
    assert prepared._request.params["model_init"] is initial
    assert prepared.params["init_only"] is True
    assert initial.params["init_only"] is True
    assert prepared.params["estimate_residual_variance"] is True
    for name, value in BOUNDARIES["MAD_initial_params"].items():
        if name == "residual_variance":
            assert initial.params[name] == pytest.approx(value, abs=1e-13, rel=0)
        else:
            assert initial.params.get(name) == value
    expected = reference["structure"]["fields"]["params"]["fields"]["model_init"]
    assert expected["names"] == ["data", "params"]
    assert list(prepared.data["X"].shape) == [len(response), len(response)]
    assert prepared.data["X"].shape == initial.data["X"].shape
    assert prepared.params["residual_variance"] is None
    assert BOUNDARIES["cases"]["trend_MAD_workhorse"]["status"] == "R_ERROR"
    with pytest.raises(ValueError, match="model_init must be a fitted SusieResult"):
        prusie.susie_workhorse(prepared)


def test_trendfilter_preparation_return_type_and_documentation():
    assert get_type_hints(prusie.susie_trendfilter)["return"] == (
        prusie.SusieResult | prusie.SusieInputs
    )
    assert 'params["model_init"]' in prusie.susie_trendfilter.__doc__
    assert BOUNDARIES["cases"]["auto_init_only"]["status"] == "R_ERROR"
    assert "init_only=True" in prusie.susie_auto.__doc__
