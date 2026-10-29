"""Independent conditional-model checks and optional pinned official R oracles."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from prusie.ld import estimate_s_rss, kriging_rss, prepare_ld


def test_conditional_formula_matches_gaussian_schur_complement():
    R = 0.65 ** np.abs(np.arange(6)[:, None] - np.arange(6)[None, :])
    z = np.array([3.2, 2.7, -1.2, -1.8, 0.4, 0.2])
    result = kriging_rss(z, R, n=200, s=0.1)
    adjusted = z * np.sqrt(199 / (z * z + 198))
    C = 0.9 * R + 0.1 * np.eye(6)
    for j, row in enumerate(result["variants"]):
        others = np.arange(6) != j
        mean = C[j, others] @ np.linalg.solve(C[np.ix_(others, others)], adjusted[others])
        variance = C[j, j] - C[j, others] @ np.linalg.solve(C[np.ix_(others, others)], C[others, j])
        assert row["expected_z"] == pytest.approx(mean, abs=3e-14)
        assert row["conditional_variance"] == pytest.approx(variance, abs=3e-14)
    assert result["mixture"]["kkt_max_violation"] < 1e-5
    # Independent R full-simplex KKT certificate, not default mixsqp's
    # demonstrably suboptimal boundary value (-2.4664035362056413).
    assert result["variants"][0]["log_likelihood_ratio"] == pytest.approx(
        -3.3052166268911964, abs=1e-10
    )


def test_singular_partial_mle_boundary_and_cache():
    R = np.ones((3, 3))
    z = np.array([3.0, -3.0, 0.0])
    cache = {}
    assert estimate_s_rss(z, R, method="null-partialmle", cache=cache) == pytest.approx(9)
    result = kriging_rss(z, R, s=9, cache=cache)
    assert result["regularization"] == 0.8
    assert result["requested_regularization"] == 9
    assert result["cache_hit"]
    assert result["warnings"]
    singular = kriging_rss([2, 2, 2], R, s=0)
    assert singular["singular_pseudoinverse"]
    np.testing.assert_array_equal(R, np.ones((3, 3)))
    altered = np.eye(3)
    assert not prepare_ld(altered, cache=cache)["cache_hit"]


def test_missing_z_is_recorded_without_automatic_flip():
    result = kriging_rss([np.nan, 3.0, -3.0], np.eye(3), s=0.1)
    assert result["variants"][0]["status"] == "not_assessed"
    assert result["variants"][0]["flag"] is None
    assert result["variants"][0]["observed_z"] is None
    assert result["variants"][0]["working_z"] == 0
    assert result["variants"][0]["standardized_residual"] is None
    assert np.isfinite(result["variants"][0]["working_standardized_residual"])
    assert result["status_counts"]["unassessed"] == 1
    with pytest.raises(ValueError, match=r"n must"):
        estimate_s_rss([1, 2], np.eye(2), n=1)
    with pytest.raises(ValueError, match=r"nonnegative"):
        kriging_rss([1, 2], np.eye(2), s=-1)


def test_official_r_oracles():
    path = os.environ.get("PRUSIE_LD_ORACLE")
    if not path:
        pytest.skip("set PRUSIE_LD_ORACLE to independently generated pinned R oracle JSON")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["source_commit"] == "8e56a8e038e989856d106d9ca5175cc664fea9d2"
    for name, case in payload["cases"].items():
        R, z = np.array(case["R"]), np.array(case["z"])
        n = case["n"]
        n = None if n in ("Inf", None, {}) else n
        for method, expected in case["estimates"].items():
            assert estimate_s_rss(z, R, n, method=method) == pytest.approx(expected, abs=2e-7), (
                name,
                method,
            )
        for kind, oracle in case["kriging"].items():
            if "error" in oracle:
                continue
            result = kriging_rss(z, R, n, s=None if kind == "estimated" else oracle["s"])
            for field, original in (
                ("observed_z", "z"),
                ("expected_z", "condmean"),
                ("conditional_variance", "condvar"),
                ("standardized_residual", "z_std_diff"),
            ):
                np.testing.assert_allclose(
                    [x[field] for x in result["variants"]],
                    [x[original] for x in oracle["conditional_dist"]],
                    rtol=0,
                    atol=2e-5,
                    err_msg=f"{name}/{kind}/{field}",
                )


def test_independent_r_full_likelihood_mixture_certificate():
    raw_path = os.environ.get("PRUSIE_LD_ORACLE")
    certified_path = os.environ.get("PRUSIE_LD_CERTIFIED_ORACLE")
    if not raw_path or not certified_path:
        pytest.skip("set pinned raw and independently certified R oracle paths")
    raw = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    certified = json.loads(Path(certified_path).read_text(encoding="utf-8"))
    disagreements = []
    for key, oracle in certified["cases"].items():
        name, _kind = key.split("/")
        case = raw["cases"][name]
        n = None if case["n"] in ("Inf", None, {}) else case["n"]
        result = kriging_rss(case["z"], case["R"], n, s=oracle["s"])
        assert oracle["certificate_passed"]
        assert oracle["certified"]["kkt_violation"] <= 1e-6
        np.testing.assert_allclose(
            [r["log_likelihood_ratio"] for r in result["variants"]],
            oracle["logLR"],
            rtol=0,
            atol=1e-6,
            err_msg=key,
        )
        assert result["mixture"]["kkt_max_violation"] <= 1e-5
        if oracle["default"]["kkt_violation"] > 1e-5:
            # Preserve and explain the upstream discrepancy, never call this
            # all-field agreement with the original default R run.
            assert oracle["certified"]["loglik"] > oracle["default"]["loglik"]
            disagreements.append(key)
    assert disagreements


def test_mixture_failure_retains_conditional_results_with_unassessed_flags(monkeypatch):
    import prusie.ld

    def fail(_):
        raise RuntimeError("controlled optimizer failure")

    monkeypatch.setattr(prusie.ld, "_mixture", fail)
    result = kriging_rss([3.0, -2.0], np.eye(2), s=0.1)
    assert result["mixture"]["status"] == "not_assessed"
    assert result["variants"][0]["expected_z"] == 0
    assert result["variants"][0]["flag"] is None
    assert result["variants"][0]["status"] == "conditional_only"
