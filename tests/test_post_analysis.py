"""Posterior semantics, exact pipeline audit, and immutable report re-extraction."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from prusie.analysis import _normal_log10p, summarize
from prusie.posterior import credible_sets, marginal_pip


def toy_fit(alpha=None, *, V=None, R=None, ids=None, null_index=None, priors=None):
    alpha = np.asarray([[0.7, 0.2, 0.1], [0.05, 0.15, 0.8]] if alpha is None else alpha, float)
    L, q = alpha.shape
    p = q - (null_index is not None)
    V = np.ones(L) if V is None else np.asarray(V, float)
    R = np.eye(q) if R is None else R
    priors = np.ones(q) / q if priors is None else np.asarray(priors, float)
    bf = np.full_like(alpha, -np.inf)
    np.log(
        np.divide(alpha, priors, out=np.zeros_like(alpha), where=priors > 0),
        out=bf,
        where=alpha > 0,
    )
    audit = []
    sets = credible_sets(
        alpha, V, R, coverage=0.8, min_abs_corr=0.1, null_index=null_index, audit=audit
    )
    return SimpleNamespace(
        variant_ids=np.asarray(ids or [f"v{j}" for j in range(p)]),
        alpha=alpha,
        V=V,
        mu=np.ones_like(alpha),
        mu2=np.ones_like(alpha),
        lbf_variable=bf,
        lbf=np.arange(L),
        pip=marginal_pip(alpha, V, null_index=null_index),
        null_index=null_index,
        sets=sets,
        params=dict(prior_tol=1e-9, prior_weights=priors.tolist(), coverage=0.8, min_abs_corr=0.1),
        variant_metadata={},
        qc={},
        warnings=[],
        cs_audit=audit,
    )


def test_audit_is_actual_pipeline_and_does_not_change_results():
    alpha = np.array(
        [[0.8, 0.1, 0.1], [0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.45, 0.45, 0.1], [0.1, 0.1, 0.8]]
    )
    V = [0, 1, 1, 1, 1]
    R = np.eye(3)
    audit = []
    ordinary = credible_sets(alpha, V, R, coverage=0.7, min_abs_corr=0.5, null_index=2)
    audited = credible_sets(alpha, V, R, coverage=0.7, min_abs_corr=0.5, null_index=2, audit=audit)
    np.testing.assert_array_equal(ordinary["cs_index"], audited["cs_index"])
    assert ordinary["cs_index"].tolist() == [2]
    assert audit[0]["reasons"] == ["inactive"]
    assert audit[1]["reasons"] == ["duplicate"]  # dedup precedes activity in real pipeline
    assert audit[2]["reasons"] == ["retained"]
    assert audit[3]["reasons"] == ["low_purity_or_unavailable"]
    assert "contains_null" in audit[4]["reasons"]
    disabled = []
    assert not credible_sets(alpha, V, R, coverage=None, audit=disabled)["cs"]
    assert all(row["reasons"] == ["disabled"] for row in disabled)


def test_summary_resolution_null_and_prior_odds():
    fit = toy_fit([[0.45, 0.45, 0.1]], null_index=2, priors=[0.3, 0.6, 0.1], R=np.ones((3, 3)))
    summary = summarize(fit, np.ones((2, 2)))
    component = summary["components"][0]
    assert component["effective_candidates"] == pytest.approx(2)
    assert component["null_mass"] == 0.1
    assert component["top_variant"] == "v0"
    assert component["cumulative_posterior"] == [0.45, 0.9]
    assert (
        max(abs(row["raw_odds_decomposition_error"]) for row in summary["prior_influence"]) < 1e-15
    )
    assert summary["prior_influence"][0]["effective_prior_weight"] > 0.3
    assert summary["independent_causal_signal_count"] is None


def test_proxies_are_direct_and_do_not_change_formal_cs():
    # This matrix is PSD; r02 < .99 despite high-LD edges 0--1--2.
    theta = np.array([0.0, 0.1, 0.2])
    R = np.cos(theta[:, None] - theta[None, :])
    fit = toy_fit([[0.96, 0.02, 0.02]], R=R)
    before = copy.deepcopy(fit)
    summary = summarize(fit, R, proxy_r=0.99)
    assert summary["credible_sets"][0]["members"] == ["v0"]
    assert [x["variant_id"] for x in summary["proxies"]] == ["v1"]
    np.testing.assert_array_equal(fit.alpha, before.alpha)
    np.testing.assert_array_equal(fit.sets["cs"][0], before.sets["cs"][0])


def test_posterior_filter_reextract_does_not_run_ibss(monkeypatch):
    import prusie.rss

    monkeypatch.setattr(prusie.rss, "susie_rss", lambda *a, **k: pytest.fail("summary ran IBSS"))
    monkeypatch.setattr(
        prusie.rss._native, "fit", lambda *a, **k: pytest.fail("summary ran native IBSS")
    )
    fit = toy_fit([[0.7, 0.2, 0.1]], R=np.ones((3, 3)))
    summary = summarize(fit, np.ones((3, 3)), coverage=0.95, min_abs_corr=0.2)
    assert summary["credible_sets"][0]["size"] == 3
    assert len(fit.sets["cs"][0]) == 2
    assert summary["cs_extraction"] == "posterior_only_no_IBSS"
    with pytest.raises(ValueError, match=r"explicit R"):
        summarize(fit, coverage=0.95)
    assert not summarize(fit, coverage=None)["credible_sets"]


def test_metadata_missing_pvalue_underflow_and_zero_prior():
    fit = toy_fit([[0, 0.3, 0.7]], priors=[0, 0.5, 0.5])
    s = summarize(fit, z=[40, 2, 0], metadata={"N": [100, np.nan, 120], "INFO": [0.9, 0.8, np.nan]})
    assert np.isfinite(s["variants"][0]["neg_log10_p"])
    assert s["variants"][0]["neg_log10_p"] > 300
    assert s["input_qc"]["N"]["missing"] == 1
    assert s["input_qc"]["EAF"]["status"] == "not_assessed"
    assert s["prior_influence"][0]["status"] == "zero_prior"
    assert s["prior_influence"][0]["log_prior"] is None
    assert not s["trace"]
    from scipy.special import log_ndtr

    for z in [0, 2, 34, 35, 40, 100]:
        assert _normal_log10p(z) == pytest.approx(
            -(np.log(2) + log_ndtr(-z)) / np.log(10), abs=1e-11
        )


def test_summary_does_not_decompose_ld(monkeypatch):
    fit = toy_fit()
    monkeypatch.setattr(np.linalg, "eigh", lambda *a, **k: pytest.fail("unexpected eigen work"))
    s = summarize(fit, np.eye(3))
    assert s["proxy_status"] == "assessed"


def test_actual_fit_prior_odds_expose_epsilon_and_trim():
    from prusie import susie_rss

    fit = susie_rss(
        [7.0, 3.0, 0.0],
        np.eye(3),
        n=200,
        L=2,
        prior_weights=[1e-14, 0.5, 0.5],
        scaled_prior_variance=0.2,
        estimate_prior_variance=False,
    )
    rows = summarize(fit)["prior_influence"]
    assert all(x["status"] == "effective_odds_verified" for x in rows)
    assert max(abs(x["odds_decomposition_error"]) for x in rows) < 1e-12
    assert max(abs(x["raw_odds_decomposition_error"]) for x in rows) > 10
    trimmed = toy_fit([[0.2, 0.3, 0.5]], V=[0], priors=[0.2, 0.3, 0.5])
    assert {x["status"] for x in summarize(trimmed)["prior_influence"]} == {"trimmed_prior_state"}
