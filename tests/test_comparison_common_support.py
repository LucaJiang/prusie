"""Installed native-fit regressions for coherent heterogeneous comparison support."""
import copy
import math

import numpy as np
import pytest

from prusie import susie_rss
from prusie.comparison import compare_fits


def native_fit(ids, *, z=None, L=1):
    """Tiny actual fitted loci; positive fixed V keeps diagnostic components active."""
    return susie_rss(z=np.ones(len(ids)) if z is None else np.asarray(z, dtype=float),
                     R=np.eye(len(ids)), n=200, variant_ids=ids, L=L,
                     estimate_prior_variance=False, scaled_prior_variance=.2,
                     min_abs_corr=0., max_iter=100)


@pytest.fixture(scope="module")
def heterogeneous():
    return {"A": native_fit(["rsA", "rsB", "rsC"]),
            "B": native_fit(["rsA", "rsB"]),
            "C": native_fit(["rsA", "rsC"]),
            "D": native_fit(["rsA", "rsD"])}


def assert_global_support(out, fits, common):
    assert [row["variant_id"] for row in out["variants"]] == sorted(common)
    assert out["support"]["global_common_ids"] == sorted(common)
    assert out["support"]["pairwise_metrics"] == "global_intersection"
    for alignment in out["alignments"]:
        assert alignment["shared_ids"] == sorted(common)
        assert alignment["support_scope"] == "global_intersection"
        for side in ("a", "b"):
            fit = fits[alignment[f"setting_{side}"]]
            assert alignment[f"omitted_{side}"] == sorted(set(fit.variant_ids) - common)
        assert alignment["status"] == "not_assessed_missing_alleles"
        assert alignment["directions"] == {variant: None for variant in sorted(common)}
    for pair in out["pairwise"]:
        assert pair["shared_variant_count"] == len(common)
        assert pair["support_scope"] == "global_intersection"
        a, b = fits[pair["setting_a"]], fits[pair["setting_b"]]
        pa = dict(zip(a.variant_ids, a.pip, strict=True))
        pb = dict(zip(b.variant_ids, b.pip, strict=True))
        expected = max((abs(pa[x] - pb[x]) for x in common), default=None)
        assert pair["max_abs_delta_pip"] == expected
        if not common:
            assert pair["candidate_max_abs_delta_pip"] is None
        if len(common) < 2:
            assert pair["full_region_pip_correlation"] is None
    for relation in out["component_relations"]:
        assert relation["support_scope"] == "global_intersection"
        assert relation["shared_variant_count"] == len(common)
        for side in ("a", "b"):
            component = relation.get(f"component_{side}")
            if component is None:
                continue
            fit = fits[relation[f"setting_{side}"]]
            alpha = dict(zip(fit.variant_ids, fit.alpha[component], strict=True))
            omitted = math.fsum(weight for variant, weight in alpha.items() if variant not in common)
            key = "unobserved_mass" if relation.get("side") == side else f"unobserved_mass_{side}"
            assert relation[key] == pytest.approx(omitted, abs=1e-15, rel=0)
        if "posterior_overlap" in relation:
            a, b = fits[relation["setting_a"]], fits[relation["setting_b"]]
            aa = dict(zip(a.variant_ids, a.alpha[relation["component_a"]], strict=True))
            ab = dict(zip(b.variant_ids, b.alpha[relation["component_b"]], strict=True))
            assert relation["posterior_overlap"] == math.fsum(min(aa[x], ab[x]) for x in common)
            assert set(relation["cs_overlap"]) <= common


@pytest.mark.parametrize("count", [3, 4])
def test_three_and_four_native_fits_use_global_support_everywhere(heterogeneous, count):
    fits = dict(list(heterogeneous.items())[:count])
    before = {name: (fit.alpha.copy(), fit.pip.copy(), copy.deepcopy(fit.sets)) for name, fit in fits.items()}
    out = compare_fits(fits, mode="common")
    assert_global_support(out, fits, {"rsA"})
    assert len(out["pairwise"]) == count * (count - 1) // 2
    assert all(row["candidate_max_abs_delta_pip"] == row["max_abs_delta_pip"] for row in out["pairwise"])
    for name, fit in fits.items():
        np.testing.assert_array_equal(fit.alpha, before[name][0])
        np.testing.assert_array_equal(fit.pip, before[name][1])
        for actual, original in zip(fit.sets["cs"], before[name][2]["cs"], strict=True):
            np.testing.assert_array_equal(actual, original)
        record = out["components"][name][0]
        assert record["alpha"] == dict(zip(fit.variant_ids, fit.alpha[0], strict=True))
        assert record["resolution"]["biological_mass"] == pytest.approx(1.)
    # A/B each observe rsB, but it is outside the declared all-fit support.
    ab = next(row for row in out["component_relations"] if row["setting_a"] == "A" and row["setting_b"] == "B")
    assert ab["posterior_overlap"] == pytest.approx(1/3)
    assert ab["unobserved_mass_a"] == pytest.approx(2/3)
    assert ab["unobserved_mass_b"] == pytest.approx(1/2)


def test_two_fit_common_support_matches_their_intersection(heterogeneous):
    fits = {"A": heterogeneous["A"], "B": heterogeneous["B"]}
    common = compare_fits(fits, mode="common")
    union = compare_fits(fits, mode="union")
    assert_global_support(common, fits, {"rsA", "rsB"})
    assert common["pairwise"][0]["max_abs_delta_pip"] == union["pairwise"][0]["max_abs_delta_pip"]
    assert common["component_relations"][0]["posterior_overlap"] == union["component_relations"][0]["posterior_overlap"]
    assert common["component_relations"][0]["unobserved_mass_a"] == pytest.approx(1/3)


@pytest.mark.parametrize("threshold", [0., .5])
def test_empty_global_support_never_invents_matching_edges(threshold):
    fits = {"A": native_fit(["rsA", "rsB"]), "B": native_fit(["rsB", "rsC"]),
            "C": native_fit(["rsA", "rsC"])}
    assert all(set(a.variant_ids) & set(b.variant_ids) for i, a in enumerate(fits.values())
               for b in list(fits.values())[i + 1:])
    out = compare_fits(fits, mode="common", overlap_threshold=threshold,
                       R=np.ones((2, 2)), ld_reference="A")
    assert_global_support(out, fits, set())
    assert out["variants"] == [] and out["display_matches"] == []
    assert all(row["status"] == "unmatched" for row in out["signal_relations"])
    assert len(out["component_relations"]) == 6
    assert all(row["status"] == "unmatched_component" for row in out["component_relations"])
    assert all(row["unobserved_mass"] == pytest.approx(1.) for row in out["component_relations"])


def test_union_keeps_pairwise_support_and_missing_pip(heterogeneous):
    out = compare_fits(heterogeneous, mode="union")
    assert out["support"]["pairwise_metrics"] == "pairwise_intersection"
    assert [r["variant_id"] for r in out["variants"]] == ["rsA", "rsB", "rsC", "rsD"]
    row = next(r for r in out["variants"] if r["variant_id"] == "rsB")
    assert row["pips"]["C"] is None and row["pips"]["D"] is None
    assert row["missing_settings"] == ["C", "D"]
    for alignment, pair in zip(out["alignments"], out["pairwise"], strict=True):
        expected = set(heterogeneous[pair["setting_a"]].variant_ids) & set(heterogeneous[pair["setting_b"]].variant_ids)
        assert alignment["shared_ids"] == sorted(expected)
        assert pair["shared_variant_count"] == len(expected)
    ab = next(row for row in out["component_relations"] if row["setting_a"] == "A" and row["setting_b"] == "B")
    assert ab["posterior_overlap"] == pytest.approx(2/3)
    assert ab["unobserved_mass_a"] == pytest.approx(1/3)
    assert ab["unobserved_mass_b"] == 0.


def test_nonfirst_baseline_component_permutation_and_missing_metadata():
    base = native_fit(["rsA", "rsB", "rsC"], z=[5., 3., .2], L=2)
    permuted = copy.deepcopy(base)
    permuted.alpha = base.alpha[::-1].copy()
    permuted.mu = base.mu[::-1].copy()
    permuted.mu2 = base.mu2[::-1].copy()
    permuted.V = base.V[::-1].copy()
    permuted.sets["cs_index"] = 1 - base.sets["cs_index"]
    fits = {"permuted": permuted, "baseline": base, "subset": native_fit(["rsA", "rsB"], z=[5., 3.], L=2)}
    out = compare_fits(fits, baseline="baseline", mode="common", overlap_threshold=.5)
    assert_global_support(out, fits, {"rsA", "rsB"})
    assert out["settings"] == ["baseline", "permuted", "subset"]
    assert all(row["setting_a"] == "baseline" for row in out["baseline_pairwise"])
    assert all(row["pip_range"] >= 0 for row in out["variants"])
    for row in out["variants"]:
        assert row["pips"]["baseline"] == row["pips"]["permuted"]
    aligned = [row for row in out["component_relations"] if row["setting_a"] == "baseline"
               and row["setting_b"] == "permuted"]
    for component in range(2):
        edge = next(row for row in aligned if row.get("component_a") == component
                    and row.get("component_b") == 1 - component)
        expected = sum(base.alpha[component, :2])
        assert edge["posterior_overlap"] == pytest.approx(expected)
        assert edge["unobserved_mass_a"] == pytest.approx(base.alpha[component, 2])


def test_common_candidates_and_ranks_are_selected_within_declared_support():
    fits = {"A": native_fit(["rsA", "rsB", "rsC"], z=[0., 6., 5.]),
            "B": native_fit(["rsA", "rsB"], z=[0., 6.]),
            "C": native_fit(["rsA", "rsC"], z=[0., 5.])}
    out = compare_fits(fits, mode="common", top_candidates=1, pip_threshold=1.,
                       overlap_threshold=1., R=np.ones((3, 3)), ld_reference="A")
    assert_global_support(out, fits, {"rsA"})
    row = out["variants"][0]
    assert row["ranks"] == {"A": 1, "B": 1, "C": 1}
    assert row["candidate"]
    assert all("component_top_candidate" in reasons for reasons in row["candidate_sources"].values())
    assert all(component["top"] != "rsA" for records in out["components"].values() for component in records)
    assert all(edge["status"] == "unmatched_component" for edge in out["component_relations"])
    # Neither the shared original CS at rsB nor LD between excluded original
    # representatives may establish a common-mode match outside {rsA}.
    union = compare_fits(fits, mode="union", overlap_threshold=1., R=np.ones((3, 3)), ld_reference="A")
    assert any("representative_ld" in edge.get("support_sources", []) for edge in union["component_relations"])


def test_identical_support_retains_numeric_metrics_and_signal_layers(heterogeneous):
    fits = {"first": heterogeneous["A"], "same": copy.deepcopy(heterogeneous["A"])}
    common = compare_fits(fits, mode="common", R=np.eye(3))
    union = compare_fits(fits, mode="union", R=np.eye(3))
    for field in ("variants", "components", "layer_summary"):
        assert common[field] == union[field]
    for field in ("pairwise", "signal_relations", "component_relations", "display_matches"):
        def stripped(value):
            return [{key: item for key, item in row.items() if key != "support_scope"} for row in value]
        assert stripped(common[field]) == stripped(union[field])
