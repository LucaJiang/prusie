"""Single-effect and utility comparisons against pinned susieR outputs."""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

import prusie

DIRECTORY = Path(__file__).parent / "fixtures/susier_parity_0166"
REFERENCE = json.loads((DIRECTORY / "ser_utilities.json").read_text())
BOUNDARIES = json.loads((DIRECTORY / "auxiliary_boundaries.json").read_text())
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


@pytest.mark.parametrize("name", ["scalar_shat", "raw_prior_weights_null"])
def test_ser_boundary_inputs_match_R(name):
    fixture = json.loads((DIRECTORY / "ser_boundaries.json").read_text())
    case = fixture["cases"][name]
    fit = prusie.susie_ser(**case["input"], **(case["options"] or {}))
    expected = case["reference"]
    np.testing.assert_allclose(fit.alpha, expected["alpha"], atol=1e-5, rtol=0)
    np.testing.assert_allclose(fit.V, expected["V"], atol=1e-7, rtol=0)
    np.testing.assert_allclose(fit.mu, expected["mu"], atol=1e-7, rtol=0)
    assert fit.null_index == (expected["null_index"] - 1 if expected["null_index"] else None)


@pytest.mark.parametrize("name", list(BOUNDARIES["symmetry"]))
def test_symmetry_boundaries_match_R(name):
    case = BOUNDARIES["symmetry"][name]
    special = {"NA": np.nan, "Inf": np.inf, "-Inf": -np.inf}
    matrix = np.asarray(
        [[special.get(value, value) for value in row] for row in case["input"]["x"]],
        dtype=float,
    )
    if case["input"]["sparse"]:
        matrix = sparse.csc_matrix(matrix)
    assert case["status"] == "ok"
    assert prusie.is_symmetric_matrix(matrix) is case["value"]


@pytest.mark.parametrize("size", [1, 2])
@pytest.mark.parametrize("diagonal", [np.nan, np.inf])
def test_dense_symmetry_ignores_diagonal_nonfinite_values(size, diagonal):
    # Independent Rfast boundary probes also retain TRUE for these inputs.
    matrix = np.eye(size)
    matrix[0, 0] = diagonal
    assert prusie.is_symmetric_matrix(matrix)


def test_sparse_symmetry_scale_excludes_matching_values_and_never_densifies(monkeypatch):
    matrix = sparse.csc_matrix([[1e20, 0.2 + 1e-12], [0.2, 1e20]])

    def reject_densification(*args, **kwargs):
        raise AssertionError("Sparse symmetry checking must not densify")

    monkeypatch.setattr(sparse.csc_matrix, "toarray", reject_densification)
    # Large equal diagonals must not dilute the unequal off-diagonal error.
    assert not prusie.is_symmetric_matrix(matrix)


def _objective_only_result(initial):
    return prusie.SusieResult(
        variant_ids=np.array(["manual"]),
        alpha=np.ones((1, 1)),
        mu=np.zeros((1, 1)),
        mu2=np.ones((1, 1)),
        lbf_variable=np.zeros((1, 1)),
        lbf=np.zeros(1),
        V=np.ones(1),
        sigma2=1.0,
        pip=np.ones(1),
        elbo=np.array([initial], dtype=float),
        niter=0,
        converged=False,
        sets={"cs": []},
        null_index=None,
        params={},
        backend_version="manual objective-only helper check",
        warnings=[],
        qc={},
        variant_metadata={},
        X_column_scale_factors=np.ones(1),
        intercept=0.0,
        KL=np.zeros(1),
        XtXr=np.zeros(1),
    )


@pytest.mark.parametrize("name", list(BOUNDARIES["block_coordinate_ascent"]))
def test_block_ascent_boundaries_match_R_without_fitting(name):
    case = BOUNDARIES["block_coordinate_ascent"][name]
    config, trace = case["config"], []
    model = _objective_only_result(config["initial"])

    def step(current, data, iteration):
        trace.append(
            dict(
                iter=iteration,
                data=data,
                objective_before=float(current.elbo[-1]),
                marker_before=current.niter,
            )
        )
        increment = (
            data
            if name == "data_update"
            else config["increments"][min(iteration, len(config["increments"])) - 1]
        )
        current.elbo = np.append(current.elbo, current.elbo[-1] + increment)
        current.niter += 1
        update = dict(model=current, converged=iteration == config.get("stop_iter"))
        if name == "data_update":
            update["data"] = data / 2
        return update

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = prusie.block_coordinate_ascent(
            model,
            config.get("data"),
            step,
            max_iter=config["max_iter"],
            tol=config["tol"],
        )
    assert case["status"] == "ok"
    assert result.elbo.tolist() == case["value"]["elbo"]
    assert result.niter == case["value"]["marker"]
    assert result.converged is case["value"]["converged"]
    assert trace == case["trace"]
    assert [str(item.message) for item in caught] == [
        message.removeprefix("WARNING: ").strip() for message in case["messages"]
    ]
