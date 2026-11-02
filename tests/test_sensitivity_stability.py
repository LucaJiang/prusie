"""Sensitivity records describe retained CS recovery and sample-size changes."""

import copy

import numpy as np
import pytest
from test_post_analysis import toy_fit

from prusie import susie_rss
from prusie.comparison import compare_fits, run_sensitivity, standard_sensitivity_plan


def test_cs_recovery_retains_component_identity_under_cs_permutations():
    baseline = toy_fit([[0.99, 0.005, 0.005], [0.005, 0.005, 0.99]])
    alternative = copy.deepcopy(baseline)
    alternative.sets["cs"] = alternative.sets["cs"][::-1]
    alternative.sets["cs_index"] = alternative.sets["cs_index"][::-1]
    result = compare_fits({"baseline": baseline, "reordered": alternative})
    assert [row["baseline_component"] for row in result["cs_stability"]] == [0, 1]
    for row in result["cs_stability"]:
        assert row["matched_components"] == [row["baseline_component"]]
        assert row["relation"] == "matched"
        assert row["jaccard"] == 1.0 and row["same_representative"]
        assert row["sum_abs_delta_pip"] == 0.0 and row["missing_members"] == []
    assert all(
        row["recovered_in"] == row["assessed_in"] == 1 for row in result["cs_stability_summary"]
    )


def test_lost_cs_unresolved_effects_and_absent_variants_are_distinct():
    baseline = toy_fit([[0.6, 0.3, 0.1]], R=np.ones((3, 3)))
    unresolved = copy.deepcopy(baseline)
    unresolved.sets = dict(cs=[], cs_index=[], purity={})
    missing = toy_fit([[1.0]], ids=["v2"])
    result = compare_fits({"baseline": baseline, "unresolved": unresolved, "missing": missing})
    rows = {row["setting"]: row for row in result["cs_stability"]}
    assert rows["unresolved"]["relation"] == "unresolved"
    assert rows["unresolved"]["jaccard"] is None
    assert rows["unresolved"]["same_representative"] is None
    assert rows["unresolved"]["sum_abs_delta_pip"] == 0.0
    assert rows["missing"]["relation"] == "unmatched"
    assert rows["missing"]["missing_members"] == ["v0", "v1"]
    assert rows["missing"]["sum_abs_delta_pip"] == 0.0
    assert result["cs_stability_summary"][0]["recovered_in"] == 0
    assert result["cs_stability_summary"][0]["assessed_in"] == 2


def test_split_merge_ambiguity_reuse_existing_relations():
    a = toy_fit([[0.99, 0.005, 0.005]])
    b = toy_fit([[0.005, 0.99, 0.005], [0.005, 0.005, 0.99]])
    R = np.full((3, 3), 0.95)
    np.fill_diagonal(R, 1.0)
    split = compare_fits({"a": a, "b": b}, R=R)
    row = split["cs_stability"][0]
    assert row["relation"] == "split" and row["matched_components"] == [0, 1]
    assert row["jaccard"] == 0.0 and not row["same_representative"]
    assert split["cs_stability_summary"][0]["recovered_in"] == 1
    assert {row["relation"] for row in compare_fits({"b": b, "a": a}, R=R)["cs_stability"]} == {
        "merge"
    }
    assert {row["relation"] for row in compare_fits({"b": b, "copy": b}, R=R)["cs_stability"]} == {
        "ambiguous"
    }


def test_jaccard_and_pip_delta_use_original_baseline_cs_members():
    a = toy_fit([[0.5, 0.3, 0.2]], R=np.ones((3, 3)))
    b = toy_fit([[0.6, 0.4]], ids=["v0", "v2"], R=np.ones((2, 2)))
    result = compare_fits({"a": a, "b": b}, mode="common")
    row = result["cs_stability"][0]
    assert row["relation"] == "matched"
    assert row["jaccard"] == 1 / 3
    assert row["sum_abs_delta_pip"] == pytest.approx(abs(a.pip[0] - b.pip[0]))
    assert row["missing_members"] == ["v1"]


def test_fit_status_separates_outer_and_inner_convergence():
    fit = susie_rss(z=[6.0, 0.0], R=np.eye(2), n=200, L=1)
    fit.converged = True
    fit.convergence["inner_ibss_converged"] = False
    fit.refinement = dict(status="completed")
    row = compare_fits({"refined": fit})["fit_status"][0]
    assert row == dict(
        setting="refined",
        converged=True,
        inner_ibss_converged=False,
        refinement_status="completed",
        niter=fit.niter,
        stop_reason=fit.convergence["stop_reason"],
        cs_count=len(fit.sets["cs"]),
    )
    assert compare_fits({"refined": fit})["cs_stability_summary"][0]["assessed_in"] == 0


def _inputs():
    return dict(z=np.array([6.0, 0.0, 2.0]), R=np.eye(3), n=200, variant_ids=["a", "b", "c"])


def test_n_change_is_an_input_factor_with_distinct_cached_fits():
    data, calls, cache = _inputs(), [], {}
    snapshot = copy.deepcopy(data)
    plan = [dict(name="n100", changes={"n": 100}), dict(name="same", changes={"n": 100})]

    def fit_function(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        return susie_rss(**kwargs)

    out = run_sensitivity(
        data, plan, baseline_settings={"L": 1}, fit_function=fit_function, cache=cache
    )
    assert len(calls) == 2 and [call["n"] for call in calls] == [200, 100]
    assert out["records"][1]["analysis"] == out["records"][2]["analysis"] == "sample_size_change"
    assert out["records"][1]["cache_key"] != out["records"][0]["cache_key"]
    assert out["records"][2]["cache_hit"]
    assert "n" not in out["baseline_settings"]
    assert out["fits"]["n100"].params["input_n"] == 100
    direct = susie_rss(**dict(data, n=100), L=1, prior_weights=np.repeat(1 / 3, 3))
    for key in ("alpha", "mu", "mu2", "pip", "V", "elbo"):
        assert getattr(direct, key).tobytes() == getattr(out["fits"]["n100"], key).tobytes()
    again = run_sensitivity(
        data, plan, baseline_settings={"L": 1}, fit_function=fit_function, cache=cache
    )
    assert all(record["cache_hit"] for record in again["records"])
    assert len(calls) == 2
    for key in ("z", "R"):
        assert data[key].tobytes() == snapshot[key].tobytes()
    assert data["n"] == 200


@pytest.mark.parametrize("bad_n", [None, 1, 0, -10, np.nan, np.inf])
def test_invalid_n_change_rejects_entire_plan_before_fitting(bad_n):
    calls = []
    with pytest.raises((TypeError, ValueError)):
        run_sensitivity(
            _inputs(),
            [dict(name="ok", changes={"L": 1}), dict(name="bad", changes={"n": bad_n})],
            fit_function=lambda **kwargs: calls.append(kwargs),
        )
    assert calls == []


@pytest.mark.parametrize(
    "data", [{k: v for k, v in _inputs().items() if k != "n"}, dict(_inputs(), n=None)]
)
def test_n_change_requires_baseline_sample_size(data):
    with pytest.raises(ValueError, match="requires baseline inputs with n"):
        run_sensitivity(data, [dict(name="n100", changes={"n": 100})])
    with pytest.raises(ValueError, match="single supported factor"):
        run_sensitivity(_inputs(), [dict(name="both", changes={"n": 100, "L": 1})])


@pytest.mark.parametrize("refine", [False, True])
def test_standard_plan_is_accepted_unchanged(refine):
    plan = standard_sensitivity_plan(2, refine=refine)
    before = copy.deepcopy(plan)
    result = run_sensitivity(_inputs(), plan, baseline_settings={"L": 2})
    assert plan == before == result["plan"]
    assert [item["name"] for item in plan] == [
        "L_half",
        "L_double",
        "coverage_0.90",
        "coverage_0.99",
        "purity_0.8",
        *(["refine"] if refine else []),
    ]
    assert result["fits"]["L_half"].params["L"] == 1
    assert result["fits"]["L_double"].params["L"] == 4
    for name in ("coverage_0.90", "coverage_0.99", "purity_0.8"):
        assert result["fits"][name].alpha.tobytes() == result["fits"]["baseline"].alpha.tobytes()


@pytest.mark.parametrize("L", [0, -1, True, 1.5])
def test_standard_plan_validates_L(L):
    with pytest.raises(ValueError, match="positive integer"):
        standard_sensitivity_plan(L)


def test_standard_plan_smallest_L_and_optional_refinement():
    assert standard_sensitivity_plan(1)[0] == dict(name="L_half", changes={"L": 1})
    assert standard_sensitivity_plan()[0] == dict(name="L_half", changes={"L": 5})
    with pytest.raises(ValueError, match="refine must be bool"):
        standard_sensitivity_plan(refine="yes")
