"""Pinned R entry-point oracles beyond the original z/R fitters."""

import json
from pathlib import Path

import numpy as np
import pytest

import prusie

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/susier_parity_0166/reference.json").read_text()
)


@pytest.mark.parametrize(
    "name",
    [
        "rss_default",
        "rss_X",
        "rss_score",
        "rss_maf",
        "auto",
        "trendfilter_order0",
        "trendfilter_order1",
        "trendfilter_mad",
    ],
)
def test_extended_entry_point_alpha(name):
    case = FIXTURE["cases"][name]
    data = FIXTURE["datasets"][case["data"]]
    options = dict(
        L=3,
        max_iter=100,
        tol=0.001,
        estimate_residual_variance=False,
        residual_variance=1,
        n_purity="auto",
    )
    options.update(case["options"] or {})
    if case["api"] == "susie_rss":
        inputs = dict(z=data["z"], R=data["R"], n=data["n"])
        if case["special"] == "rss_X":
            inputs.pop("R")
            inputs["X"] = data["X"]
        if case["special"] == "maf":
            inputs["maf"] = data["maf"]
        fit = prusie.susie_rss(**inputs, **options)
    elif case["api"] == "susie_auto":
        fit = prusie.susie_auto(data["X"], data["y"], **case["options"])
    else:
        fit = prusie.susie_trendfilter(data["y"][:48], **case["options"])
    expected = np.asarray(case["reference"]["alpha"])
    assert fit.alpha.shape == expected.shape
    assert np.isfinite(fit.alpha).all()
    np.testing.assert_allclose(fit.alpha.sum(axis=1), 1, atol=1e-10, rtol=0)
    np.testing.assert_allclose(fit.alpha, expected, atol=1e-5, rtol=0)


def test_control_matches_R_and_rejects_invalid_options():
    assert dict(prusie.susie_rss_control()) == FIXTURE["auxiliary"]["control"]
    with pytest.raises(ValueError):
        prusie.susie_rss_control(qc_eigen_tol_abs=-1)
    with pytest.raises(ValueError):
        prusie.susie_rss_control(mixture_reference_p=0)


@pytest.mark.parametrize(
    "name",
    [
        "rss_mismatch_none",
        "rss_mismatch_eb",
        "rss_mismatch_eb_mix",
        "rss_mismatch_eb_ser_init",
        "rss_mismatch_eb_force_init",
        "rss_mismatch_eb_no_init",
    ],
)
def test_mismatch_alpha(name):
    case = FIXTURE["cases"][name]
    data = FIXTURE["datasets"][case["data"]]
    fit = prusie.susie_rss(
        z=data["z"],
        R=data["R"],
        n=data["n"],
        L=3,
        residual_variance=1,
        estimate_residual_variance=False,
        n_purity="auto",
        **case["options"],
    )
    np.testing.assert_allclose(fit.alpha, case["reference"]["alpha"], atol=1e-5, rtol=0)


@pytest.mark.parametrize(
    "name",
    ["rss_R_finite_numeric", "rss_R_finite_false", "rss_R_finite_X_true", "rss_mismatch_map"],
)
def test_finite_reference_alpha(name):
    reference = json.loads(
        (Path(__file__).parent / "fixtures/susier_parity_0166/supplement.json").read_text()
    )
    case = reference["cases"][name]
    data = FIXTURE["datasets"]["N2finemapping"]
    inputs = {"X": data["X"]} if case.get("special") == "X" else {"R": data["R"]}
    fit = prusie.susie_rss(
        z=data["z"],
        n=data["n"],
        L=3,
        residual_variance=1,
        estimate_residual_variance=False,
        n_purity="auto",
        **inputs,
        **case["options"],
    )
    np.testing.assert_allclose(fit.alpha, case["reference"]["alpha"], atol=1e-5, rtol=0)


@pytest.mark.parametrize("name", ["rss_lambda_fixed", "rss_lambda_estimate", "rss_lambda_X"])
def test_rss_lambda_alpha(name):
    case = FIXTURE["cases"][name]
    data = FIXTURE["datasets"][case["data"]]
    options = dict(case["options"])
    options["lambda_"] = options.pop("lambda")
    inputs = {"X": data["X"]} if case["special"] == "rss_X" else {"R": data["R"]}
    fit = prusie.susie_rss_lambda(
        data["z"],
        n=data["n"],
        L=3,
        max_iter=100,
        tol=0.001,
        residual_variance=1,
        estimate_residual_variance=False,
        **inputs,
        **options,
    )
    np.testing.assert_allclose(fit.alpha, case["reference"]["alpha"], atol=1e-5, rtol=0)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize(
    "case_id",
    ["rss_X_single_n1000_dense", "rss_X_single_nmissing_dense", "rss_X_multiple_n1000_dense"],
)
def test_reference_genotype_sample_size(case_id, sparse):
    from scipy.sparse import csc_matrix

    fixture = json.loads(
        (Path(__file__).parent / "fixtures/susier_parity_0166/operator_inputs.json").read_text()
    )
    case = fixture["cases"][case_id]
    inputs = fixture["inputs"][case["input"]]
    X = inputs["X"]
    if case["input"] == "multiple":
        if sparse:
            X = [csc_matrix(panel) for panel in X]
    elif sparse:
        X = csc_matrix(X)
    fit = prusie.susie_rss(inputs["z"], X=X, **case["options"])
    np.testing.assert_allclose(fit.alpha, case["reference"]["alpha"], atol=1e-5, rtol=0)


@pytest.mark.parametrize("name", ["rss_R_finite_numeric", "rss_mismatch_eb"])
def test_finite_reference_reliability_matches_R(name):
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/susier_parity_0166/qc_public.json").read_text()
    )
    case = fixture["cases"][name]
    data = FIXTURE["datasets"]["N2finemapping"]
    fit = prusie.susie_rss(data["z"], data["R"], n=data["n"], **case["options"])
    expected = case["reference"]["diagnostics"]["R_finite_diagnostics"]
    observed = fit.R_finite_diagnostics
    for field in (
        "B",
        "p",
        "R_frob_sq_debiased",
        "effective_rank",
        "r_over_B",
        "per_variable_penalty",
        "Rhat_diag_deviation",
    ):
        reference = expected[field]
        reference = np.inf if isinstance(reference, str) and reference == "Inf" else reference
        np.testing.assert_allclose(observed[field], reference, atol=1e-7, rtol=1e-7)
    for field in (
        "Q_art",
        "artifact_flag",
        "artifact_evaluable",
        "low_eigen_count",
        "low_eigen_fraction",
        "eig_delta",
        "lambda_bias",
    ):
        if field in expected:
            np.testing.assert_allclose(observed[field], expected[field], atol=1e-7, rtol=1e-7)
    if "bf_attenuation" in expected:
        np.testing.assert_allclose(
            observed["bf_attenuation"]["variant_max"],
            expected["bf_attenuation"]["variant_max"],
            atol=1e-7,
            rtol=1e-7,
        )
    np.testing.assert_allclose(fit.alpha, case["reference"]["alpha"], atol=1e-5, rtol=0)


def test_reliability_control_thresholds_only_change_flags():
    data = FIXTURE["datasets"]["N2finemapping"]
    options = dict(n=data["n"], L=3, residual_variance=1, R_mismatch="eb", max_iter=100, tol=0.001)
    ordinary = prusie.susie_rss(data["z"], data["R"], **options)
    with pytest.warns(RuntimeWarning, match="sensitive to LD-reference"):
        flagged = prusie.susie_rss(
            data["z"], data["R"], control=prusie.susie_rss_control(artifact_threshold=0), **options
        )
    np.testing.assert_array_equal(ordinary.alpha, flagged.alpha)
    assert not ordinary.R_finite_diagnostics["artifact_flag"]
    assert flagged.R_finite_diagnostics["artifact_flag"]
    assert flagged.R_finite_diagnostics["R_reliability_flag"]


def test_finite_reference_nig_residual_posterior_matches_R():
    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures/susier_parity_0166/nig_finite_interaction.json"
        ).read_text()
    )
    case = fixture["cases"]["N2_finite_NIG"]
    fit = prusie.susie_rss(**case["input"], **case["options"])
    expected = case["reference"]
    np.testing.assert_allclose(fit.alpha, expected["alpha"], atol=1e-5, rtol=0)
    np.testing.assert_allclose(fit.sigma2, expected["sigma2"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(fit.elbo, expected["elbo"], atol=1e-8, rtol=0)
    np.testing.assert_array_equal(fit.KL, np.zeros(fit.alpha.shape[0]))
    assert "bf_attenuation" not in fit.R_finite_diagnostics
