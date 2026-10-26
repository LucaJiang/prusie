"""Comparisons keep IDs, directions, ambiguity and unobserved posterior mass."""

import copy

import numpy as np
import pytest
from test_post_analysis import toy_fit

from prusie.comparison import compare_fits, comparison_relations, run_sensitivity, sensitivity_key


def test_permutations_component_order_and_allele_recode():
    a = toy_fit([[0.8, 0.15, 0.05], [0.1, 0.1, 0.8]], R=np.ones((3, 3)))
    b = copy.deepcopy(a)
    order = [2, 0, 1]
    b.variant_ids = b.variant_ids[order]
    b.alpha = b.alpha[::-1, order]
    b.pip = b.pip[order]
    b.sets = dict(cs=[], cs_index=[], purity={})
    a.variant_metadata = {
        "effect_allele": np.array(["A", "G", "A"]),
        "other_allele": np.array(["C", "T", "G"]),
    }
    b.variant_metadata = {
        "effect_allele": a.variant_metadata["other_allele"][order],
        "other_allele": a.variant_metadata["effect_allele"][order],
    }
    compared = compare_fits({"a": a, "b": b}, overlap_threshold=0.7)
    assert all(row["pip_range"] == 0 for row in compared["variants"])
    assert {(e["component_a"], e["component_b"]) for e in compared["signal_relations"]} == {
        (0, 1),
        (1, 0),
    }
    assert all(x == -1 for x in compared["alignments"][0]["directions"].values())
    b.variant_metadata["effect_allele"][0] = "T"
    with pytest.raises(ValueError, match=r"incompatible"):
        compare_fits({"a": a, "b": b})


def test_missing_variants_not_zero_and_unobserved_mass_not_renormalized():
    a = toy_fit([[0.4, 0.3, 0.3]], ids=["a", "b", "c"])
    b = toy_fit([[0.6, 0.4]], ids=["a", "b"])
    out = compare_fits({"a": a, "b": b}, overlap_threshold=0.5)
    missing = next(x for x in out["variants"] if x["variant_id"] == "c")
    assert missing["pips"]["b"] is None
    # Neither low-purity component retains a formal CS. Its unrenormalized
    # posterior evidence remains available in the full component layer.
    assert not out["signal_relations"]
    edge = out["component_relations"][0]
    assert edge["posterior_overlap"] == pytest.approx(0.7)
    assert edge["unobserved_mass_a"] == 0.3
    assert [v["variant_id"] for v in compare_fits({"a": a, "b": b}, mode="common")["variants"]] == [
        "a",
        "b",
    ]


def test_high_ld_candidate_exchange_split_merge_and_ambiguity():
    a = toy_fit([[0.99, 0.005, 0.005]])
    b = toy_fit([[0.005, 0.99, 0.005], [0.005, 0.005, 0.99]])
    R = np.full((3, 3), 0.95)
    np.fill_diagonal(R, 1)
    split = compare_fits({"a": a, "b": b}, R=R)
    assert {e["status"] for e in split["signal_relations"]} == {"split"}
    assert all(e["candidate_exchange_high_ld"] for e in split["signal_relations"])
    merge = compare_fits({"b": b, "a": a}, R=R)
    assert {e["status"] for e in merge["signal_relations"]} == {"merge"}
    ambiguous = compare_fits({"a": b, "b": b}, R=R)
    assert {e["status"] for e in ambiguous["signal_relations"]} == {"ambiguous"}
    assert len(ambiguous["signal_relations"]) == 4
    assert len(ambiguous["display_matches"]) == 2
    unrelated = compare_fits({"a": a, "b": toy_fit([[0.005, 0.005, 0.99]])})
    assert all(e["status"] == "unmatched" for e in unrelated["signal_relations"])


def test_explicit_plan_cache_fixed_variance_and_report_only():
    calls = []
    inputs = dict(
        z=np.array([2.0, 1.0, 0.0]), R=np.ones((3, 3)), n=100, variant_ids=["v0", "v1", "v2"]
    )

    def fitting(**kw):
        calls.append(kw)
        return toy_fit(R=kw["R"])

    cache = {}
    plan = [
        dict(name="L2", changes={"L": 2}),
        dict(name="L2_again", changes={"L": 2}),
        dict(
            name="fixed", changes={"scaled_prior_variance": 0.1, "estimate_prior_variance": False}
        ),
        dict(name="init", changes={"scaled_prior_variance": 0.1}),
        dict(name="coverage", report={"coverage": 0.9}),
    ]
    out = run_sensitivity(inputs, plan, fit_function=fitting, cache=cache, software_version="test")
    assert len(calls) == 4
    assert out["records"][2]["cache_hit"]
    assert out["records"][3]["analysis"] == "fixed_effect_variance_prior"
    assert out["records"][4]["analysis"] == "computational_initialization_sensitivity"
    assert not out["records"][5]["fit_performed"]
    assert out["fits"]["coverage"] is not out["fits"]["baseline"]
    assert out["fits"]["coverage"].params["coverage"] == 0.9
    assert out["fits"]["baseline"].params["coverage"] == 0.8
    np.testing.assert_array_equal(out["fits"]["coverage"].alpha, out["fits"]["baseline"].alpha)
    assert not any("s_init" in c for c in calls)
    before = len(calls)
    run_sensitivity(inputs, plan, fit_function=fitting, cache=cache, software_version="test")
    assert len(calls) == before
    bad = [dict(name="both", changes={"L": 2, "prior_weights": [0.5, 0.3, 0.2]})]
    with pytest.raises(ValueError, match=r"single"):
        run_sensitivity(inputs, bad, fit_function=fitting)
    assert len(calls) == before


def test_cache_hash_contains_order_direction_version_and_initialization():
    data = dict(
        z=np.array([1.0, 2.0]),
        R=np.eye(2),
        variant_ids=["a", "b"],
        variant_metadata={"effect_allele": ["A", "C"], "other_allele": ["G", "T"]},
    )
    base = sensitivity_key(data, {"L": 2}, software_version="1")
    altered = copy.deepcopy(data)
    altered["variant_ids"].reverse()
    allele = copy.deepcopy(data)
    allele["variant_metadata"]["effect_allele"].reverse()
    assert (
        len(
            {
                base,
                sensitivity_key(altered, {"L": 2}, software_version="1"),
                sensitivity_key(allele, {"L": 2}, software_version="1"),
                sensitivity_key(data, {"L": 2}, software_version="2"),
                sensitivity_key(
                    data, {"L": 2}, software_version="1", initialization="continuation"
                ),
            }
        )
        == 5
    )


def test_actual_plan_continuation_refine_and_fixed_prior():
    inputs = dict(z=np.array([6.0, 1.0, 0.0]), R=np.eye(3), n=200, variant_ids=["a", "b", "c"])
    out = run_sensitivity(
        inputs,
        [
            dict(name="continue", continuation={"max_iter": 20}),
            dict(name="refine", changes={"refine": True}),
            dict(
                name="fixed",
                changes={"scaled_prior_variance": 0.1, "estimate_prior_variance": False},
            ),
            dict(name="L2", changes={"L": 2}),
        ],
        baseline_settings={"L": 1},
    )
    assert out["fits"]["continue"].converged
    assert out["fits"]["refine"].refinement["status"] != "not_executed"
    assert not out["fits"]["fixed"].params["estimate_prior_variance"]
    assert out["records"][1]["initialization"] == "baseline_posterior"


def test_provided_baseline_preserves_settings_and_rejects_mismatches():
    from prusie import susie_rss

    inputs = dict(z=np.array([6.0, 1.0, 0.0]), R=np.eye(3), n=200, variant_ids=["a", "b", "c"])
    fit = susie_rss(**inputs, L=1, estimate_prior_variance=False, scaled_prior_variance=0.1)
    out = run_sensitivity(inputs, [dict(name="L2", changes={"L": 2})], baseline=fit)
    assert out["fits"]["L2"].params["L"] == 2
    assert not out["fits"]["L2"].params["estimate_prior_variance"]
    assert out["fits"]["L2"].params["scaled_prior_variance"] == 0.1
    assert out["baseline_validation"]["raw_input_identity"].startswith("not_assessed")
    with pytest.raises(ValueError, match=r"variant_ids"):
        run_sensitivity(dict(inputs, variant_ids=["b", "a", "c"]), [], baseline=fit)
    with pytest.raises(ValueError, match=r"L differs"):
        run_sensitivity(inputs, [], baseline=fit, baseline_settings={"L": 2})
    with pytest.raises(ValueError, match=r"sample size"):
        run_sensitivity(dict(inputs, n=201), [], baseline=fit)
    with pytest.raises(ValueError, match=r"duplicate"):
        run_sensitivity(inputs, [], baseline_settings={"n": 200})


def test_missing_n_uses_noncentrality_variance_parameter():
    inputs = dict(z=np.array([3.0, 0.0]), R=np.eye(2), variant_ids=["a", "b"])
    with pytest.warns(RuntimeWarning, match="n is missing"):
        out = run_sensitivity(
            inputs,
            [dict(name="fixed", changes={"prior_variance": 5.0, "estimate_prior_variance": False})],
            baseline_settings={"L": 1},
        )
    assert out["fits"]["fixed"].params["prior_variance"] == 5.0
    assert out["records"][-1]["analysis"] == "fixed_effect_variance_prior"
    with pytest.raises(ValueError, match=r"uses prior_variance"):
        run_sensitivity(inputs, [dict(name="wrong", changes={"scaled_prior_variance": 0.1})])


def test_uniform_diffuse_components_are_not_localized_signal_matches():
    p = 20
    a = toy_fit(np.full((2, p), 1 / p))
    b = toy_fit(np.full((3, p), 1 / p))
    before = copy.deepcopy(a)
    out = compare_fits({"baseline": a, "more_components": b}, R=np.eye(p))
    assert out["signal_relations"] == []
    assert out["baseline_signal_relations"] == []
    assert len(out["component_relations"]) == 6
    assert {row["status"] for row in out["component_relations"]} == {"component_overlap"}
    assert all(row["posterior_overlap"] == pytest.approx(1) for row in out["component_relations"])
    assert all(not row["cs_retained"] for row in out["components"]["baseline"])
    assert all(
        row["resolution"]["effective_candidates"] == pytest.approx(p)
        for row in out["components"]["baseline"]
    )
    assert out["layer_summary"]["settings"]["baseline"]["unretained_active_components"] == 2
    np.testing.assert_array_equal(a.alpha, before.alpha)
    np.testing.assert_array_equal(a.pip, before.pip)
    np.testing.assert_array_equal(a.V, before.V)


def test_added_diffuse_components_cannot_pollute_formal_match_degrees():
    p = 20
    peak = np.r_[0.97, np.repeat(0.03 / (p - 1), p - 1)]
    a = toy_fit(np.vstack([peak, np.full((7, p), 1 / p)]))
    b = toy_fit(np.vstack([peak, np.full((11, p), 1 / p)]))
    out = compare_fits({"baseline": a, "L12": b}, R=np.eye(p))
    assert len(out["signal_relations"]) == 1
    edge = out["signal_relations"][0]
    assert (edge["component_a"], edge["component_b"], edge["status"]) == (0, 0, "matched")
    assert (edge["degree_a"], edge["degree_b"]) == (1, 1)
    assert edge["cs_retained_a"] and edge["cs_retained_b"]
    assert "formal_cs_overlap" in edge["support_sources"]
    assert len(out["component_relations"]) > 70
    assert all(
        e["status"] not in {"split", "merge", "ambiguous"}
        for e in out["component_relations"]
        if not (e["cs_retained_a"] and e["cs_retained_b"])
    )


def test_cs_lost_is_unresolved_and_unrelated_retained_cs_is_unmatched():
    a = toy_fit([[0.99, 0.005, 0.005]])
    lost = copy.deepcopy(a)
    lost.sets = dict(cs=[], cs_index=[], purity={})
    out = compare_fits({"baseline": a, "no_cs": lost})
    row = out["signal_relations"][0]
    assert row["status"] == "unresolved"
    assert row["cs_retained_a"] and not row["cs_retained_b"]
    assert row["posterior_overlap"] == pytest.approx(1)
    assert out["display_matches"] == []
    unrelated = toy_fit([[0.005, 0.005, 0.99]])
    out = compare_fits({"baseline": a, "unrelated": unrelated})
    assert len(out["signal_relations"]) == 2
    assert {r["status"] for r in out["signal_relations"]} == {"unmatched"}
    assert out["signal_relations"][0]["cs_retained_a"]
    assert out["signal_relations"][1]["cs_retained_b"]


def test_large_formal_cs_members_below_pip_threshold_remain_candidates():
    p = 20
    a = toy_fit(np.full((1, p), 1 / p), R=np.ones((p, p)))
    out = compare_fits({"baseline": a}, pip_threshold=0.1, top_candidates=0)
    members = {a.variant_ids[i] for i in a.sets["cs"][0]}
    assert len(members) > 10
    assert all(x["pip_max"] < 0.1 for x in out["variants"])
    assert {x["variant_id"] for x in out["variants"] if x["candidate"]} == members
    assert all(
        x["candidate_sources"]["baseline"] == ["formal_cs_member"]
        for x in out["variants"]
        if x["candidate"]
    )
    with pytest.raises(ValueError, match=r"top_candidates"):
        compare_fits({"a": a}, top_candidates=True)


def test_formal_cs_at_distinct_pip_activity_boundary_is_preserved():
    fit = toy_fit([[0.99, 0.005, 0.005]], V=[2e-9])
    fit.params["prior_tol"] = 3e-9
    fit.pip[:] = 0.0
    out = compare_fits({"baseline": fit})
    component = out["components"]["baseline"][0]
    assert component["cs_retained"] and not component["active"]
    candidate = next(x for x in out["variants"] if x["variant_id"] == "v0")
    assert candidate["candidate"] and candidate["pip_max"] == 0
    assert "formal_cs_member" in candidate["candidate_sources"]["baseline"]


def test_baseline_pairwise_view_component_permutation_and_missing_ids():
    a = toy_fit([[0.99, 0.005, 0.005], [0.005, 0.005, 0.99]])
    reordered = toy_fit(a.alpha[::-1])
    smaller = toy_fit([[0.99, 0.01]], ids=["v0", "v1"])
    out = compare_fits(
        {"reordered": reordered, "baseline": a, "smaller": smaller}, baseline="baseline"
    )
    assert out["settings"][0] == "baseline"
    assert len(out["pairwise"]) == 3 and len(out["baseline_pairwise"]) == 2
    assert all(x["setting_a"] == "baseline" for x in out["baseline_signal_relations"])
    permutation = [x for x in out["baseline_signal_relations"] if x["setting_b"] == "reordered"]
    assert {(x["component_a"], x["component_b"], x["status"]) for x in permutation} == {
        (0, 1, "matched"),
        (1, 0, "matched"),
    }
    assert next(x for x in out["variants"] if x["variant_id"] == "v2")["pips"]["smaller"] is None
    assert comparison_relations(out) == out["baseline_signal_relations"]
    assert len(comparison_relations(out, layer="components", baseline_only=False)) == len(
        out["component_relations"]
    )


def test_legacy_comparison_reader_reclassifies_available_cs_support():
    old = dict(
        schema_version="prusie.comparison.v1",
        baseline="base",
        thresholds=dict(posterior_overlap=0.5, abs_r=0.9),
        signal_relations=[
            dict(
                setting_a="other",
                setting_b="base",
                component_a=0,
                component_b=0,
                posterior_overlap=1.0,
                cs_overlap=["x"],
                cs_size_a=1,
                cs_size_b=1,
                signed_r=None,
                status="ambiguous",
            ),
            dict(
                setting_a="other",
                setting_b="base",
                component_a=1,
                component_b=0,
                posterior_overlap=0.01,
                cs_overlap=[],
                cs_size_a=None,
                cs_size_b=1,
                signed_r=1.0,
                status="ambiguous",
            ),
            dict(
                setting_a="other",
                setting_b="base",
                component_a=1,
                component_b=1,
                posterior_overlap=1.0,
                cs_overlap=[],
                cs_size_a=None,
                cs_size_b=None,
                signed_r=None,
                status="ambiguous",
            ),
        ],
    )
    before = copy.deepcopy(old)
    main = comparison_relations(old)
    assert len(main) == 1 and main[0]["status"] == "matched"
    assert main[0]["setting_a"] == "base"
    full = comparison_relations(old, layer="components")
    assert len(full) == 3
    assert full[-1]["status"] == "component_overlap"
    assert old == before
    only_ld = dict(old, signal_relations=[old["signal_relations"][1]])
    unmatched = comparison_relations(only_ld)
    assert len(unmatched) == 1 and unmatched[0]["status"] == "unmatched"
    assert unmatched[0]["component_a"] == 0 and unmatched[0]["component_b"] is None


def test_strict_baseline_convergence_survives_L_prior_and_report_changes():
    from prusie import susie_rss
    from prusie.diagnostics import ConvergenceConfig

    inputs = dict(z=np.array([6.0, 1.0, 0.0]), R=np.eye(3), n=200, variant_ids=["a", "b", "c"])
    control = ConvergenceConfig("joint", alpha_tol=2e-7, pip_tol=3e-8, min_iter=4, cycle_lag=3)
    fit = susie_rss(**inputs, L=1, convergence=control, track_fit="compact")
    before = copy.deepcopy(fit.convergence)
    plan = [
        dict(name="L2", changes={"L": 2}),
        dict(name="prior", changes={"prior_weights": [0.6, 0.3, 0.1]}),
        dict(name="coverage", report={"coverage": 0.8}),
    ]
    out = run_sensitivity(inputs, plan, baseline=fit)
    assert out["convergence_resolution"]["source"] == "inherited_params"
    assert out["baseline_settings"]["convergence"] == control.options()
    for name in ("L2", "prior"):
        result = out["fits"][name]
        assert result.params["convergence"] == control.options()
        assert result.convergence["requested_criterion"] == "joint"
        assert result.convergence["requested_config"] == control.options()
    assert out["fits"]["coverage"].convergence == before
    assert fit.convergence == before
    assert not out["records"][-1]["fit_performed"]
    assert out["records"][-1]["convergence_resolution"]["source"] == "unchanged_report_only"


def test_sensitivity_explicit_override_and_serialized_config_roundtrip():
    from prusie import susie_rss
    from prusie.diagnostics import ConvergenceConfig

    inputs = dict(z=np.array([6.0, 1.0, 0.0]), R=np.eye(3), n=200, variant_ids=["a", "b", "c"])
    original = ConvergenceConfig("joint", alpha_tol=1e-6, pip_tol=2e-6, min_iter=4, cycle_lag=2)
    override = ConvergenceConfig("alpha_pip", alpha_tol=3e-6, pip_tol=4e-6, min_iter=3, cycle_lag=1)
    fit = susie_rss(**inputs, L=1, convergence=original, retain_data=True)
    out = run_sensitivity(
        inputs,
        [dict(name="L2", changes={"L": 2}), dict(name="continue", continuation={"max_iter": 10})],
        baseline=fit,
        baseline_settings={"convergence": override.options()},
    )
    for name in ("L2", "continue"):
        assert out["fits"][name].params["convergence"] == override.options()
    assert out["convergence_resolution"] == dict(
        source="explicit_override", previous_requested=original.options()
    )
    assert out["records"][0]["requested_convergence"] == original.options()
    assert out["fits"]["baseline"].params["convergence"] == original.options()
    # Saved settings restore a Config object rather than leaking an ordinary
    # dict to the native public API on a later call.
    replay = run_sensitivity(inputs, [], baseline_settings=out["baseline_settings"])
    assert replay["fits"]["baseline"].params["convergence"] == override.options()
    item_override = run_sensitivity(
        inputs,
        [dict(name="continue", continuation={"max_iter": 10, "convergence": override.options()})],
        baseline=fit,
    )
    assert item_override["fits"]["continue"].params["convergence"] == override.options()
    assert item_override["plan"][0]["continuation"]["convergence"] == override.options()
    assert (
        item_override["records"][-1]["convergence_resolution"]["previous_requested"]
        == original.options()
    )


def test_legacy_baseline_convergence_recovery_never_uses_effective_fallback():
    from prusie import susie_rss
    from prusie.diagnostics import ConvergenceConfig

    inputs = dict(z=np.array([6.0, 1.0, 0.0]), R=np.eye(3), n=200, variant_ids=["a", "b", "c"])
    control = ConvergenceConfig("joint", alpha_tol=4e-7, pip_tol=8e-8, min_iter=4, cycle_lag=2)
    fit = susie_rss(**inputs, L=1, convergence=control)
    fit.params.pop("convergence")
    fit.convergence["effective_criterion"] = "alpha_pip"
    recovered = run_sensitivity(inputs, [dict(name="L2", changes={"L": 2})], baseline=fit)
    assert recovered["convergence_resolution"]["source"] == "recovered_legacy_diagnostics"
    assert recovered["fits"]["L2"].params["convergence"] == control.options()
    fit.convergence = {}
    fit._prepared = None
    fallback = run_sensitivity(inputs, [dict(name="L2", changes={"L": 2})], baseline=fit)
    assert fallback["convergence_resolution"]["source"] == "legacy_default"
    assert "No inheritance" in fallback["convergence_resolution"]["compatibility_note"]
    assert fallback["fits"]["L2"].params["convergence"] == ConvergenceConfig().options()


@pytest.mark.filterwarnings("ignore:IBSS did not converge:RuntimeWarning")
def test_sensitivity_without_requested_controls_matches_direct_fits():
    from prusie import susie_rss

    R = np.array(
        [
            [1.0, 0.6, 0.2, 0.0],
            [0.6, 1.0, 0.3, 0.1],
            [0.2, 0.3, 1.0, 0.4],
            [0.0, 0.1, 0.4, 1.0],
        ]
    )
    inputs = dict(z=np.array([1.0, 4.0, 1.0, -0.5]), R=R, n=500, variant_ids=["a", "b", "c", "d"])
    settings = dict(L=2, R_mismatch="eb", max_iter=20, check_prior=False)
    plan = [dict(name="L1", changes={"L": 1})]
    calls = []

    def fit_function(**arguments):
        calls.append(arguments)
        return susie_rss(**arguments)

    out = run_sensitivity(inputs, plan, baseline_settings=settings, fit_function=fit_function)
    assert calls and all("convergence" not in arguments for arguments in calls)
    assert "convergence" not in out["baseline_settings"]
    direct = susie_rss(**inputs, **settings)
    assert direct.convergence["requested_criterion"] == "pip"
    assert out["fits"]["baseline"].convergence["requested_criterion"] == "pip"
    np.testing.assert_array_equal(out["fits"]["baseline"].alpha, direct.alpha)
    supplied = run_sensitivity(inputs, plan, baseline=direct)
    assert supplied["fits"]["L1"].convergence["requested_criterion"] == "pip"


def test_sensitivity_cache_uses_installed_development_identity(monkeypatch):
    import prusie.comparison as comparison

    values = dict(z=np.array([2.0, 0.0]), R=np.eye(2))
    monkeypatch.setattr(comparison, "version", lambda _: "0.2.3rc6")
    old = sensitivity_key(values, {"L": 1})
    monkeypatch.setattr(comparison, "version", lambda _: "0.2.4.dev1")
    assert sensitivity_key(values, {"L": 1}) != old
