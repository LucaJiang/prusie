"""Single-effect and utility comparisons against pinned susieR outputs."""

import json
from pathlib import Path

import numpy as np
import pytest

import prusie

DIRECTORY = Path(__file__).parent / "fixtures/susier_parity_0166"
REFERENCE = json.loads((DIRECTORY / "ser_utilities.json").read_text())
DATA = json.loads((DIRECTORY / "reference.json").read_text())["datasets"]["N2finemapping"]


@pytest.mark.parametrize("name", list(REFERENCE["cases"]))
def test_ser_alpha(name):
    case = REFERENCE["cases"][name]
    fit = prusie.susie_ser(**case["input"], **(case["options"] or {}))
    np.testing.assert_allclose(fit.alpha, case["reference"]["alpha"], atol=1e-5, rtol=0)
    np.testing.assert_allclose(fit.alpha.sum(axis=1), 1, atol=1e-10, rtol=0)
    np.testing.assert_allclose(fit.mu, case["reference"]["mu"], atol=1e-7, rtol=0)


def test_predictor_order_helpers():
    np.testing.assert_array_equal(prusie.absolute_order([1, -3, 2, 0]), [1, 2, 0, 3])
    path = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1]])
    np.testing.assert_array_equal(prusie.path_order(path), [2, 0, 1])
    assert prusie.is_symmetric_matrix(np.eye(2))
    assert not prusie.is_symmetric_matrix(np.array([[1.0, 2.0], [0.0, 1.0]]))


@pytest.mark.parametrize("name", list(REFERENCE["utilities"]))
def test_frozen_R_utilities(name):
    case = REFERENCE["utilities"][name]
    function = {
        "marginal_default": prusie.compute_marginal_bhat_shat,
        "marginal_sigma2": prusie.compute_marginal_bhat_shat,
        "absolute_order": prusie.absolute_order,
        "univar_order": prusie.univar_order,
        "path_order": prusie.path_order,
    }[name]
    inputs = case["input"]
    observed = function(**inputs) if isinstance(inputs, dict) else function(inputs)
    if isinstance(observed, dict):
        for key in observed:
            np.testing.assert_allclose(
                observed[key], case["reference"][key], atol=1e-10, rtol=1e-12
            )
    else:
        np.testing.assert_array_equal(observed, case["reference"])


def test_block_ascent_rejects_decreasing_mutating_callback():
    fit = prusie.susie_ser([1.0, 2.0], coverage=None)
    original = fit.alpha.copy()
    objective = fit.elbo.copy()

    def decrease(model, data, iteration):
        model.alpha[:] = 0.5
        model.elbo = objective - 1
        return {"model": model}

    accepted = prusie.block_coordinate_ascent(fit, None, decrease)
    assert accepted.converged
    np.testing.assert_array_equal(accepted.alpha, original)
    np.testing.assert_array_equal(accepted.elbo, objective)
