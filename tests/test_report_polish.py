"""Result-first report display, support-aware maxima and computation-free text."""
import builtins
import copy
import csv
from html.parser import HTMLParser
import json
from pathlib import Path

import numpy as np
import pytest

from prusie import summary_text
from prusie.report import analysis_summary, evidence_bundle, load_analysis, write_report


def summary():
    return dict(variants=[dict(variant_id="near_one", pip=.999999991, rank=1),
                          dict(variant_id="tiny", pip=2e-12, rank=2)],
                credible_sets=[dict(component_id=7, members=["near_one"], size=1,
                                    representative="near_one", actual_coverage=.999999991)],
                components=[dict(component_id=7, active=True, effective_candidates=1.)],
                convergence=dict(requested_criterion="elbo", effective_criterion="elbo",
                                 stop_reason="criterion_satisfied", niter=3,
                                 last=dict(elbo=-17.123456789, delta_elbo=0.)), trace=[])


def test_text_and_report_do_not_fit_compute_ld_or_import_plotting(monkeypatch, tmp_path):
    import prusie.api
    import prusie.ld
    import prusie.refinement

    def fail(*args, **kwargs):
        raise AssertionError("A display helper initiated a statistical computation")

    for module, names in ((prusie.api, ("susie_rss", "susie_suff_stat")),
                          (prusie.ld, ("diagnose_ld", "estimate_s_rss", "kriging_rss")),
                          (prusie.refinement, ("refine_fit", "continue_fit", "evaluate_objective")),
                          (np.linalg, ("eigh", "eigvalsh"))):
        for name in names:
            monkeypatch.setattr(module, name, fail)
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name.split(".")[0] in ("matplotlib", "scipy"):
            raise AssertionError("Text/report without figures imported an optional numerical or plotting dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    supplied = summary()
    before = copy.deepcopy(supplied)
    result = summary_text(supplied)
    assert "2 variants; 1 retained credible sets; 1 active components." in result
    assert "component 7" in result and "requested elbo, effective elbo" in result
    assert "Refinement: not requested." in result and "LD: not assessed." in result
    receipt = write_report(supplied, tmp_path)
    assert Path(receipt["text_summary"]).read_text() == result + "\n"
    assert supplied == before == load_analysis(tmp_path)["summary"]


def test_regional_maximum_ids_respect_observation_support_and_candidate_selection(tmp_path):
    comparison = dict(schema_version="prusie.comparison.v2", baseline="baseline", mode="union",
        settings=["baseline", "alternative"], signal_relations=[], component_relations=[],
        baseline_pairwise=[dict(setting_a="baseline", setting_b="alternative", max_abs_delta_pip=.2,
                                support_scope="pairwise_intersection")],
        variants=[dict(variant_id="candidate", candidate=True, pips={"baseline": .8, "alternative": .7}),
                  dict(variant_id="outside", candidate=False, pips={"baseline": .01, "alternative": .21}),
                  dict(variant_id="missing", candidate=False, pips={"baseline": None, "alternative": 1.})])
    before = copy.deepcopy(comparison)
    compact = analysis_summary(summary(), comparison=comparison)
    row = compact["sensitivity"]["regional_maxima"][0]
    assert row["variant_id"] == "outside" and row["max_abs_delta_pip"] == pytest.approx(.2)
    assert not row["in_candidate_view"] and row["evidence_pointer"] == "/comparison/variants/1"
    assert "outside candidate view" in summary_text(summary(), comparison=comparison)
    receipt = write_report(summary(), tmp_path, comparison=comparison)
    page = Path(receipt["html"]).read_text()
    assert "Outside the candidate view:" in page and "tables/comparison_variants.csv" in page
    assert comparison == before
    # A valid common-mode comparison supplies only its global-intersection rows.
    common = dict(comparison, mode="common", variants=[comparison["variants"][0]])
    assert analysis_summary(summary(), comparison=common)["sensitivity"]["regional_maxima"][0]["variant_id"] == "candidate"
    common["variants"] = []
    assert analysis_summary(summary(), comparison=common)["sensitivity"]["regional_maxima"] == []
    unchanged = copy.deepcopy(comparison)
    for row in unchanged["variants"]:
        row["pips"]["alternative"] = row["pips"]["baseline"]
    maximum = analysis_summary(summary(), comparison=unchanged)["sensitivity"]["regional_maxima"][0]
    assert maximum["max_abs_delta_pip"] == 0 and maximum["in_candidate_view"]
    assert maximum["tied_maxima"] == 2


def test_probability_precision_exports_and_semantic_navigation(tmp_path):
    supplied = summary()
    receipt = write_report(supplied, tmp_path, annotations={"trait": "recorded trait"})
    page = Path(receipt["html"]).read_text()
    assert "&gt;0.9999" in page or "&gt; 0.9999" in page
    assert "2e-12" in page and 'title="Full precision: 0.999999991"' in page
    assert "recorded trait" in page and '<nav aria-label="On this page">' in page
    assert '<th scope="col">' in page and '@media' in page
    assert '<th scope="col">Variant ID</th>' in page and '<th scope="col">PIP</th>' in page
    assert '<th scope="col">CS size</th>' in page and '<th scope="col">Representative</th>' in page
    assert 'CS member, top PIP, PIP threshold' in page
    assert '<td>None</td><td>top PIP</td>' in page  # Known non-CS member, not unassessed.
    assert 'href="#main-results"' in page
    compact = json.loads(Path(receipt["analysis_summary"]).read_text())
    assert compact["results"]["candidates"][0]["pip"] == .999999991
    assert compact["results"]["candidates"][0]["selection_sources"] == ['formal_CS_member', 'top_PIP', 'PIP_threshold']
    assert compact["fit"]["last"]["delta_elbo"] == 0.
    assert 'http://' not in page and 'https://' not in page and '<script' not in page


def test_cold_fit_record_reports_search_not_requested_without_changing_status(tmp_path):
    supplied = summary()
    supplied['refinement'] = dict(status='not_executed', stop_reason='not_requested',
                                  history=[], total_fits=0)
    before = copy.deepcopy(supplied)
    assert 'Refinement: not requested.' in summary_text(supplied)
    compact = analysis_summary(supplied)
    assert compact['refinement']['status'] == 'not_executed'
    assert compact['refinement']['stop_reason'] == 'not_requested'
    receipt = write_report(supplied, tmp_path)
    page = Path(receipt['html']).read_text()
    assert 'Refinement: not requested.' in page
    assert 'Refinement objective, logged ELBO and controls' not in page
    assert '0 exclusion attempts' not in page
    assert supplied == before == load_analysis(tmp_path)['summary']


def test_complete_large_cs_long_ids_and_no_cs_are_explicit(tmp_path):
    supplied = summary()
    names = ["long_variant_identifier_" + str(i) + "_" * 50 for i in range(35)]
    supplied["variants"] = [dict(variant_id=name, pip=1/35) for name in names]
    supplied["credible_sets"][0].update(members=names, size=35)
    receipt = write_report(supplied, tmp_path / "large")
    page = Path(receipt["html"]).read_text()
    assert "first 20 of 35" in page and "All 35 selected candidates" in page
    assert names[-1] in page
    no_cs = dict(variants=[], components=[], credible_sets=[])
    receipt = write_report(no_cs, tmp_path / "empty")
    page = Path(receipt["html"]).read_text()
    assert "0 retained credible sets" in page and "No iteration trajectory" in page
    assert "LD diagnostics were not assessed" in page


def test_display_and_export_leave_probabilities_and_objectives_unchanged(tmp_path):
    supplied = summary()
    supplied["refinement"] = dict(status="completed", total_fits=2,
        before={"elbo": -20., "logged_elbo": -19.99}, best_objective=-18.,
        after={"elbo": -18., "logged_elbo": -17.99},
        history=[dict(round=1, component=7, excluded_ids=["near_one"], accepted=True,
                      improvement=2., objective=-18.,
                      stages={"A": {"niter": 2, "objective": -16.123456789, "logged_elbo": -16.12},
                              "B": {"niter": 3, "objective": -18., "logged_elbo": -17.99,
                                    "solver_settings": {"tol": .001}}})])
    before = evidence_bundle(supplied)
    summary_text(supplied)
    receipt = write_report(supplied, tmp_path)
    assert evidence_bundle(supplied) == before
    assert load_analysis(tmp_path)["summary"] == supplied
    assert (tmp_path / "tables/refinement_attempts.csv").is_file()
    with (tmp_path / "tables/refinement_stages.csv").open(newline="") as stream:
        stages = list(csv.DictReader(stream))
    assert [row["stage"] for row in stages] == ["A", "B"]
    assert stages[0]["objective"] == "-16.123456789"
    assert stages[0]["logged_elbo"] == "-16.12" and stages[1]["logged_elbo"] == "-17.99"
    assert stages[0]["attempt"] == "1" and stages[0]["component"] == "7"
    assert json.loads(stages[0]["excluded_ids"]) == ["near_one"]
    assert stages[0]["solver_settings"] == ""  # Not recorded by this producer.
    assert json.loads(stages[1]["solver_settings"]) == {"tol": .001}
    assert "Stage A changes the prior" in Path(receipt["html"]).read_text()
    assert json.loads(Path(receipt["figure_index"]).read_text()) == []
