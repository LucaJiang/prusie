"""Comparisons keep IDs, directions, ambiguity and unobserved posterior mass."""
import copy
import numpy as np
import pytest
from test_post_analysis import toy_fit
from prusie.comparison import compare_fits, run_sensitivity, sensitivity_key


def test_permutations_component_order_and_allele_recode():
    a = toy_fit([[.8,.15,.05],[.1,.1,.8]],R=np.ones((3,3)))
    b = copy.deepcopy(a)
    order = [2,0,1]
    b.variant_ids = b.variant_ids[order]
    b.alpha = b.alpha[::-1,order]
    b.pip = b.pip[order]
    b.sets = dict(cs=[],cs_index=[],purity={})
    a.variant_metadata = {"effect_allele":np.array(["A","G","A"]),"other_allele":np.array(["C","T","G"])}
    b.variant_metadata = {"effect_allele":a.variant_metadata["other_allele"][order],
                          "other_allele":a.variant_metadata["effect_allele"][order]}
    compared = compare_fits({"a":a,"b":b},overlap_threshold=.7)
    assert all(row["pip_range"] == 0 for row in compared["variants"])
    assert {(e["component_a"],e["component_b"]) for e in compared["signal_relations"]} == {(0,1),(1,0)}
    assert all(x == -1 for x in compared["alignments"][0]["directions"].values())
    b.variant_metadata["effect_allele"][0] = "T"
    with pytest.raises(ValueError,match="incompatible"):
        compare_fits({"a":a,"b":b})


def test_missing_variants_not_zero_and_unobserved_mass_not_renormalized():
    a = toy_fit([[.4,.3,.3]],ids=["a","b","c"])
    b = toy_fit([[.6,.4]],ids=["a","b"])
    out = compare_fits({"a":a,"b":b},overlap_threshold=.5)
    missing = next(x for x in out["variants"] if x["variant_id"] == "c")
    assert missing["pips"]["b"] is None
    edge = out["signal_relations"][0]
    assert edge["posterior_overlap"] == pytest.approx(.7)
    assert edge["unobserved_mass_a"] == .3
    assert [v["variant_id"] for v in compare_fits({"a":a,"b":b},mode="common")["variants"]] == ["a","b"]


def test_high_ld_candidate_exchange_split_merge_and_ambiguity():
    a = toy_fit([[.99,.005,.005]])
    b = toy_fit([[.005,.99,.005],[.005,.005,.99]])
    R = np.full((3,3),.95); np.fill_diagonal(R,1)
    split = compare_fits({"a":a,"b":b},R=R)
    assert {e["status"] for e in split["signal_relations"]} == {"split"}
    assert all(e["candidate_exchange_high_ld"] for e in split["signal_relations"])
    merge = compare_fits({"b":b,"a":a},R=R)
    assert {e["status"] for e in merge["signal_relations"]} == {"merge"}
    ambiguous = compare_fits({"a":b,"b":b},R=R)
    assert {e["status"] for e in ambiguous["signal_relations"]} == {"ambiguous"}
    assert len(ambiguous["signal_relations"]) == 4
    assert len(ambiguous["display_matches"]) == 2
    unrelated = compare_fits({"a":a,"b":toy_fit([[.005,.005,.99]])})
    assert all(e["status"] == "unmatched" for e in unrelated["signal_relations"])


def test_explicit_plan_cache_fixed_variance_and_report_only():
    calls=[]
    inputs=dict(z=np.array([2.,1.,0.]),R=np.ones((3,3)),n=100,variant_ids=["v0","v1","v2"])
    def fitting(**kw):
        calls.append(kw)
        return toy_fit(R=kw["R"])
    cache={}
    plan=[dict(name="L2",changes={"L":2}),dict(name="L2_again",changes={"L":2}),
          dict(name="fixed",changes={"scaled_prior_variance":.1,"estimate_prior_variance":False}),
          dict(name="init",changes={"scaled_prior_variance":.1}),
          dict(name="coverage",report={"coverage":.9})]
    out=run_sensitivity(inputs,plan,fit_function=fitting,cache=cache,software_version="test")
    assert len(calls) == 4
    assert out["records"][2]["cache_hit"]
    assert out["records"][3]["analysis"] == "fixed_effect_variance_prior"
    assert out["records"][4]["analysis"] == "computational_initialization_sensitivity"
    assert not out["records"][5]["fit_performed"]
    assert out["fits"]["coverage"] is not out["fits"]["baseline"]
    assert out["fits"]["coverage"].params["coverage"] == .9
    assert out["fits"]["baseline"].params["coverage"] == .8
    np.testing.assert_array_equal(out["fits"]["coverage"].alpha,out["fits"]["baseline"].alpha)
    assert not any("s_init" in c for c in calls)
    before=len(calls)
    run_sensitivity(inputs,plan,fit_function=fitting,cache=cache,software_version="test")
    assert len(calls) == before
    bad=[dict(name="both",changes={"L":2,"prior_weights":[.5,.3,.2]})]
    with pytest.raises(ValueError,match="single"):
        run_sensitivity(inputs,bad,fit_function=fitting)
    assert len(calls) == before


def test_cache_hash_contains_order_direction_version_and_initialization():
    data=dict(z=np.array([1.,2.]),R=np.eye(2),variant_ids=["a","b"],
              variant_metadata={"effect_allele":["A","C"],"other_allele":["G","T"]})
    base=sensitivity_key(data,{"L":2},software_version="1")
    altered=copy.deepcopy(data);altered["variant_ids"].reverse()
    allele=copy.deepcopy(data);allele["variant_metadata"]["effect_allele"].reverse()
    assert len({base,sensitivity_key(altered,{"L":2},software_version="1"),
        sensitivity_key(allele,{"L":2},software_version="1"),
        sensitivity_key(data,{"L":2},software_version="2"),
        sensitivity_key(data,{"L":2},software_version="1",initialization="continuation")}) == 5


def test_actual_plan_continuation_refine_and_fixed_prior():
    inputs=dict(z=np.array([6.,1.,0.]),R=np.eye(3),n=200,variant_ids=["a","b","c"])
    out=run_sensitivity(inputs,[dict(name="continue",continuation={"max_iter":20}),
        dict(name="refine",changes={"refine":True}),
        dict(name="fixed",changes={"scaled_prior_variance":.1,"estimate_prior_variance":False}),
        dict(name="L2",changes={"L":2})],baseline_settings={"L":1})
    assert out["fits"]["continue"].converged
    assert out["fits"]["refine"].refinement["status"] != "not_executed"
    assert not out["fits"]["fixed"].params["estimate_prior_variance"]
    assert out["records"][1]["initialization"] == "baseline_posterior"


def test_provided_baseline_preserves_settings_and_rejects_mismatches():
    from prusie import susie_rss
    inputs=dict(z=np.array([6.,1.,0.]),R=np.eye(3),n=200,variant_ids=["a","b","c"])
    fit=susie_rss(**inputs,L=1,estimate_prior_variance=False,scaled_prior_variance=.1)
    out=run_sensitivity(inputs,[dict(name="L2",changes={"L":2})],baseline=fit)
    assert out["fits"]["L2"].params["L"] == 2
    assert not out["fits"]["L2"].params["estimate_prior_variance"]
    assert out["fits"]["L2"].params["scaled_prior_variance"] == .1
    assert out["baseline_validation"]["raw_input_identity"].startswith("not_assessed")
    with pytest.raises(ValueError,match="variant_ids"):
        run_sensitivity(dict(inputs,variant_ids=["b","a","c"]),[],baseline=fit)
    with pytest.raises(ValueError,match="L differs"):
        run_sensitivity(inputs,[],baseline=fit,baseline_settings={"L":2})
    with pytest.raises(ValueError,match="sample size"):
        run_sensitivity(dict(inputs,n=201),[],baseline=fit)
    with pytest.raises(ValueError,match="duplicate"):
        run_sensitivity(inputs,[],baseline_settings={"n":200})


def test_missing_n_uses_noncentrality_variance_parameter():
    inputs=dict(z=np.array([3.,0.]),R=np.eye(2),variant_ids=["a","b"])
    with pytest.warns(RuntimeWarning,match="n is missing"):
        out=run_sensitivity(inputs,[dict(name="fixed",changes={"prior_variance":5.,"estimate_prior_variance":False})],
                            baseline_settings={"L":1})
    assert out["fits"]["fixed"].params["prior_variance"] == 5.
    assert out["records"][-1]["analysis"] == "fixed_effect_variance_prior"
    with pytest.raises(ValueError,match="uses prior_variance"):
        run_sensitivity(inputs,[dict(name="wrong",changes={"scaled_prior_variance":.1})])
