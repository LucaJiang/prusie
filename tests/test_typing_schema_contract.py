"""Returned fit metadata is covered by the exported TypedDict schemas."""

import json
from pathlib import Path
from typing import get_type_hints

import numpy as np
import pytest

import prusie


@pytest.fixture(scope="module")
def frozen_data():
    path = Path(__file__).parent / "fixtures/susier_parity_0166/reference.json"
    return json.loads(path.read_text())["datasets"]["N2finemapping"]


@pytest.mark.parametrize("route", ["susie", "suff_stat", "rss", "rss_overwrite", "rss_lambda"])
def test_returned_metadata_matches_exported_schemas(route, frozen_data):
    data = frozen_data
    options = dict(
        L=2,
        max_iter=100,
        tol=1e-3,
        n_threads=2,
        estimate_prior_variance=False,
        estimate_residual_variance=False,
        coverage=None,
        variant_ids=data["variant_ids"],
    )
    if route == "susie":
        fit = prusie.susie(data["X"], data["y"], **options)
    elif route == "suff_stat":
        fit = prusie.susie_suff_stat(
            data["XtX"], data["Xty"], data["yty"], data["n"], check_input=True, **options
        )
    elif route == "rss_lambda":
        fit = prusie.susie_rss_lambda(
            data["z"], data["R"], n=data["n"], lambda_=0.2, check_prior=False, **options
        )
    else:
        fit = prusie.susie_rss(
            data["z"],
            np.array(data["R"], dtype=np.float64, order="C"),
            n=data["n"],
            overwrite_ld=route == "rss_overwrite",
            check_input=route == "rss",
            check_prior=False,
            **options,
        )

    assert isinstance(fit, prusie.SusieResult)
    assert set(fit.params) <= prusie.FitParameters.__annotations__.keys()
    assert set(fit.qc) <= prusie.InputChecks.__annotations__.keys()
    assert fit.params["n_threads"] == 2
    if route in ("rss", "rss_overwrite"):
        assert fit.params["overwrite_ld"] is (route == "rss_overwrite")
    if route == "rss_overwrite":
        assert set(fit.qc["ld_overwrite"]) == {"action", "original_rss_scale"}
    if route == "rss_lambda":
        assert fit.params["lambda"] == 0.2


def test_fit_schema_includes_inherited_ridge_parameter():
    assert "lambda" in prusie.FitParameters.__annotations__
    assert get_type_hints(prusie.FitParameters)["lambda"] is float
