"""Frozen susieR checks for post-hoc outcome configuration models."""

import json
from pathlib import Path

import numpy as np
import pytest

import prusie

DIRECTORY = Path(__file__).parent / "fixtures/susier_parity_0166"
REFERENCE = json.loads((DIRECTORY / "reference.json").read_text(encoding="utf-8"))
POST = json.loads((DIRECTORY / "post_outcome.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fits():
    data = REFERENCE["datasets"]["N2finemapping"]
    controls = dict(
        L=3,
        max_iter=100,
        tol=0.001,
        residual_variance=1,
        estimate_residual_variance=False,
        variant_ids=data["variant_ids"],
        n_purity="auto",
    )
    first = prusie.susie(data["X"], data["y"], **controls)
    second = prusie.susie(data["X"], POST["y2"], **controls)
    np.testing.assert_allclose(second.alpha, POST["fit2_reference"]["alpha"], atol=1e-5, rtol=0)
    return {"first": first, "second": second}


@pytest.mark.parametrize("name", list(POST["cases"]))
def test_post_outcome_R_agreement(name, fits):
    case = POST["cases"][name]
    observed = prusie.susie_post_outcome_configuration(
        fits, by=case["by"], method=case["method"], cs_only=case["cs_only"]
    )
    expected = case["reference"]
    if case["method"] == "susiex":
        assert list(observed["susiex"]) == list(expected["susiex"])
        for key, value in observed["susiex"].items():
            oracle = expected["susiex"][key]
            np.testing.assert_allclose(
                [r["config_prob"] for r in value["config_probability"]],
                [r["config_prob"] for r in oracle["config_probability"]],
                atol=1e-7,
            )
            np.testing.assert_allclose(
                [r["posthoc_prob"] for r in value["config_summary"]],
                [r["posthoc_prob"] for r in oracle["config_summary"]],
                atol=1e-7,
            )
    else:
        assert len(observed["coloc_pairwise"]) == len(expected["coloc_pairwise"])
        for record, oracle in zip(
            observed["coloc_pairwise"], expected["coloc_pairwise"], strict=True
        ):
            assert record["l1"] == oracle["l1"] - 1
            assert record["l2"] == oracle["l2"] - 1
            assert record["hit1"] == oracle["hit1"]
            assert record["hit2"] == oracle["hit2"]
            np.testing.assert_allclose(
                [record[f"PP.H{k}"] for k in range(5)],
                [oracle[f"PP.H{k}"] for k in range(5)],
                atol=1e-7,
            )
