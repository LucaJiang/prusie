"""Credible-set refinement agrees with susieR 0.16.6 on final alpha."""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

import prusie

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/susier_refine_0166/refine_convergence.json").read_text(
        encoding="utf-8"
    )
)
OPTIONS = {
    "refine_pip": {"convergence_method": "pip", "refine": True},
    "refine_mixture": {
        "prior_variance_grid": np.array([0.02, 0.2, 1.0]),
        "mixture_weights": np.array([0.2, 0.5, 0.3]),
        "refine": True,
    },
    "refine_NIG": {"estimate_residual_method": "NIG", "refine": True},
}


def fit(case):
    inputs = prusie.load_convergence_example()["inputs"]
    options = dict(OPTIONS[case], **FIXTURE["common"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return prusie.susie_ss(**inputs, **options)


@pytest.mark.parametrize("case", sorted(OPTIONS))
def test_final_alpha_matches_susier(case):
    result = fit(case)
    expected = np.asarray(FIXTURE["cases"][case]["alpha"])
    assert result.alpha.shape == expected.shape
    np.testing.assert_allclose(result.alpha, expected, rtol=0, atol=1e-5)


@pytest.mark.parametrize("case", ["refine_pip", "refine_mixture", "refine_NIG"])
def test_refinement_improves_and_records_the_search(case):
    result = fit(case)
    assert result.refinement["method"] == "block_coordinate_ascent"
    assert result.refinement_status == "converged"
    history = result.refinement["history"]
    assert max(history[0]["candidate_elbos"]) > history[0]["elbo"]
    assert result.elbo[-1] == pytest.approx(FIXTURE["cases"][case]["elbo"], abs=1e-4)
