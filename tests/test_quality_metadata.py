"""Regression checks for public documentation and typed input metadata."""

import inspect
from dataclasses import replace

import numpy as np
import pytest

import prusie

FIXED_OPTIONS = dict(
    L=1,
    residual_variance=1,
    estimate_residual_variance=False,
    estimate_prior_variance=False,
    coverage=None,
)


def test_public_sufficient_statistics_docstring_is_preserved() -> None:
    doc = inspect.getdoc(prusie.susie_suff_stat)
    assert doc is not None
    assert "Parameters\n----------" in doc
    assert "XtX :" in doc
    assert "Returns\n-------" in doc
    assert prusie.susie_ss is prusie.susie_suff_stat


@pytest.mark.parametrize("check_input, policy", [(False, "zero"), (True, "reject")])
def test_typed_summary_missing_policy_matches_validation(check_input: bool, policy: str) -> None:
    options = dict(
        FIXED_OPTIONS,
        XtX=99 * np.eye(2),
        Xty=[3.0, 0.5],
        yty=99.0,
        n=100,
        check_input=check_input,
    )
    fit = prusie.susie_suff_stat(**options)
    assert fit.input_quality.missing_summary_values == policy
    assert fit.qc["input_missing_policy"] == policy
    exported = fit.as_dict()
    assert exported["qc"] is fit.qc
    assert exported["input_quality"]["missing_summary_values"] == policy
    options["Xty"] = [np.nan, 0.5]
    if check_input:
        with pytest.raises(ValueError, match="Xty must contain only finite values"):
            prusie.susie_suff_stat(**options)
    else:
        with pytest.warns(RuntimeWarning, match="NA values in Xty are replaced with 0"):
            missing = prusie.susie_suff_stat(**options)
        assert missing.input_quality.missing_summary_values == "zero"


@pytest.mark.parametrize("check_input", [False, True])
def test_rss_policy_describes_original_input_instead_of_cleaned_Xty(check_input: bool) -> None:
    options = dict(FIXED_OPTIONS, R=np.eye(2), n=100, check_input=check_input)
    with pytest.warns(RuntimeWarning, match="NA values in z-scores are replaced with 0"):
        scores = prusie.susie_rss(z=[np.nan, 2.0], **options)
    assert scores.input_quality.missing_summary_values == "zero"
    assert scores.qc["input_missing_policy"] == "zero"
    effects = prusie.susie_rss(bhat=[0.0, 0.2], shat=0.1, **options)
    assert effects.input_quality.missing_summary_values == "reject"
    assert effects.qc["input_missing_policy"] == "reject"
    with pytest.raises(ValueError, match="bhat must contain only finite values"):
        prusie.susie_rss(bhat=[np.nan, 0.2], shat=0.1, **options)


@pytest.mark.parametrize("variant_ids", [None, ["rs1", "rs2"]])
def test_ser_typed_variant_ids_preserve_legacy_qc(variant_ids: list[str] | None) -> None:
    fit = prusie.susie_ser([2.0, -1.0], variant_ids=variant_ids, coverage=None)
    positional = variant_ids is None
    assert fit.qc["p"] == 2
    assert fit.qc["variant_ids_positional"] is positional
    assert fit.input_quality.predictor_count == 2
    assert fit.input_quality.positional_variant_ids is positional
    exported = fit.as_dict()
    assert exported["qc"] is fit.qc
    assert exported["input_quality"]["positional_variant_ids"] is positional
    restored = prusie.SusieResult(**exported)
    assert restored.input_quality.positional_variant_ids is positional
    legacy = replace(fit, qc={"p": 2, "variant_ids_positional": positional}, input_quality=None)
    assert legacy.input_quality.positional_variant_ids is positional


def test_ser_bhat_missing_policy_matches_its_finite_input_requirement() -> None:
    fit = prusie.susie_ser(bhat=[0.2, -0.1], shat=0.1, coverage=None)
    assert fit.input_quality.missing_summary_values == "reject"
    with pytest.raises(ValueError, match="bhat must contain only finite values"):
        prusie.susie_ser(bhat=[np.nan, -0.1], shat=0.1, coverage=None)


def test_legacy_qc_policy_remains_available_while_typed_policy_is_normalized() -> None:
    fit = prusie.susie_ser([2.0, -1.0], coverage=None)
    legacy_qc = {
        "input_missing_policy": "susieR_zero_z_and_Xty",
        "variant_ids_are_positional": False,
    }
    restored = replace(fit, qc=legacy_qc, input_quality=None)
    assert restored.qc is legacy_qc
    assert restored.qc["input_missing_policy"] == "susieR_zero_z_and_Xty"
    assert restored.input_quality.missing_summary_values == "zero"
    assert restored.input_quality.positional_variant_ids is False
