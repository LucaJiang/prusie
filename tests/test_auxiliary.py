"""Independent frozen susieR oracles for public posterior accessors."""

import json
from pathlib import Path

import numpy as np
import pytest

import prusie
from prusie import SusieResult
from prusie.posterior import credible_sets


@pytest.fixture(scope="module")
def oracle():
    path = Path(__file__).parent / "fixtures/susier_parity_0166/reference.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rfit(oracle):
    r = oracle["cases"]["individual_default"]["reference"]
    data = oracle["datasets"]["N2finemapping"]
    sets = r["sets"]
    purity = {
        key.replace(".", "_"): np.asarray([row[key] for row in sets["purity"]])
        for key in sets["purity"][0]
        if key != "_row"
    }
    members = sets.get("cs") or {}
    members = list(members.values()) if isinstance(members, dict) else members
    normalized = dict(
        cs=[np.asarray(row, dtype=int) - 1 for row in members],
        cs_index=np.atleast_1d(sets.get("cs_index", [])).astype(int) - 1,
        purity=purity,
        coverage=np.atleast_1d(sets.get("coverage", [])),
    )
    return SusieResult(
        variant_ids=np.asarray(data["variant_ids"]),
        **{
            key: np.asarray(r[key])
            for key in (
                "alpha",
                "mu",
                "mu2",
                "V",
                "lbf",
                "lbf_variable",
                "pip",
                "elbo",
                "KL",
                "X_column_scale_factors",
            )
        },
        sigma2=r["sigma2"],
        niter=r["niter"],
        converged=r["converged"],
        sets=normalized,
        intercept=r["intercept"],
        null_index=None,
        params={},
        backend_version="frozen-R",
        warnings=[],
        qc={},
        variant_metadata={},
        XtXr=np.zeros(len(data["variant_ids"])),
    )


@pytest.mark.parametrize(
    "name",
    [
        "susie_get_pip",
        "susie_get_posterior_mean",
        "susie_get_posterior_sd",
        "susie_get_niter",
        "susie_get_objective",
        "susie_get_prior_variance",
        "susie_get_residual_variance",
        "coef",
    ],
)
def test_getter_matches_independent_R(name, oracle, rfit):
    np.testing.assert_allclose(
        getattr(prusie, name)(rfit), oracle["auxiliary"][name], atol=1e-12, rtol=1e-12
    )


def test_predict_and_summary_match_R(oracle, rfit):
    data = oracle["datasets"]["N2finemapping"]
    np.testing.assert_allclose(
        prusie.predict(rfit, np.asarray(data["X"])[:4]),
        oracle["auxiliary"]["predict"],
        atol=1e-12,
        rtol=1e-12,
    )
    observed = prusie.summary(rfit)
    reference = oracle["auxiliary"]["summary"]
    assert [row["variable"] + 1 for row in observed["vars"]] == [
        row["variable"] for row in reference["vars"]
    ]
    assert [row["cs"] + 1 if row["cs"] >= 0 else -1 for row in observed["vars"]] == [
        row["cs"] for row in reference["vars"]
    ]
    np.testing.assert_allclose(
        [row["cs_log10bf"] for row in observed["cs"]],
        [row["cs_log10bf"] for row in reference["cs"]],
        atol=1e-12,
    )


def test_CS_OR_extension_and_squared_match_R(oracle, rfit):
    data = oracle["datasets"]["N2finemapping"]
    observed = [
        prusie.susie_get_cs(
            rfit, X=np.asarray(data["X"]), median_abs_corr=0.1, cs_extension_corr=0.9
        ),
        prusie.susie_get_cs(rfit, Xcorr=data["R"], squared=True),
    ]
    for value, name in zip(observed, ("cs", "cs_squared"), strict=True):
        reference = oracle["auxiliary"][name]
        np.testing.assert_array_equal(value["cs_index"] + 1, np.atleast_1d(reference["cs_index"]))
        members = list(reference["cs"].values())
        for actual, expected in zip(value["cs"], members, strict=True):
            np.testing.assert_array_equal(actual + 1, expected)
        np.testing.assert_allclose(value["coverage"], reference["coverage"], atol=1e-12)
        for key in reference["purity"][0]:
            if key == "_row":
                continue
            expected = [row[key] for row in reference["purity"]]
            normalized = key.replace(".", "_").replace("sq_corr", "abs_corr")
            np.testing.assert_allclose(value["purity"][normalized], expected, atol=1e-12)


def test_attainable_preserves_original_rows_and_R_members(oracle, rfit):
    with pytest.warns(RuntimeWarning, match="purity filtering is skipped"):
        observed = prusie.susie_get_cs_attainable(rfit)
    reference = oracle["auxiliary"]["cs_attainable"]
    expected = reference["cs"] or {}
    assert len(observed["cs"]) == len(expected)
    for i, (name, members) in enumerate(expected.items()):
        assert observed["cs_index"][i] == int(name[1:]) - 1
        np.testing.assert_array_equal(observed["cs"][i] + 1, members)


def test_sampling_is_seeded_and_matches_posterior_moments(rfit):
    draws = prusie.susie_get_posterior_samples(rfit, 30000, seed=842)
    again = prusie.susie_get_posterior_samples(rfit, 30000, seed=842)
    np.testing.assert_array_equal(draws["b"], again["b"])
    mean, sd = prusie.susie_get_posterior_mean(rfit), prusie.susie_get_posterior_sd(rfit)
    assert np.all(np.abs(draws["b"].mean(axis=1) - mean) <= 7 * sd / np.sqrt(30000) + 1e-12)
    np.testing.assert_array_equal(draws["gamma"], draws["b"] != 0)
    with pytest.raises(TypeError, match="seed"):
        prusie.susie_get_posterior_samples(rfit, 1)


def test_init_coef_and_inactive_component_semantics():
    initial = prusie.susie_init_coef([1, 0], [2.0, -3.0], 3)
    np.testing.assert_array_equal(initial["alpha"], [[0, 1, 0], [1, 0, 0]])
    np.testing.assert_array_equal(initial["mu2"], initial["mu"] ** 2)
    with pytest.raises(ValueError):
        prusie.susie_init_coef([-1], [2.0], 3)
    alpha = np.array([[0.4, 0.3, 0.3], [0.4, 0.3, 0.3]])
    correlation = np.array([[1, 0.8, 0.1], [0.8, 1, 0.8], [0.1, 0.8, 1]])
    out = credible_sets(
        alpha, [1, 1], correlation, coverage=0.95, min_abs_corr=0.5, median_abs_corr=0.7
    )
    assert len(out["cs"]) == 1
    np.testing.assert_array_equal(out["cs_index"], [0])


def test_lfsr_uses_matching_per_cell_gaussian_moments(rfit, oracle):
    from scipy.special import ndtr

    sd = np.sqrt(np.maximum(rfit.mu2 - rfit.mu**2, 0))
    ratio = np.divide(-rfit.mu, sd, out=np.zeros_like(sd), where=sd > 0)
    cdf = np.where(sd == 0, rfit.mu <= 0, ndtr(ratio))
    expected = np.sum(rfit.alpha * np.minimum(cdf, 1 - cdf), axis=1)
    np.testing.assert_allclose(prusie.susie_get_lfsr(rfit), expected, atol=1e-15, rtol=0)
    # Pinned R transposes mu but not its SD matrix, recycling mismatched cells.
    # Record that upstream diagnostic difference without reproducing it.
    assert not np.allclose(expected, oracle["auxiliary"]["susie_get_lfsr"], atol=1e-5, rtol=0)
