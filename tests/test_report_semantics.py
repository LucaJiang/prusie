"""Report semantics: useful summaries and complete, separate machine evidence."""

import json
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pytest

from prusie.report import analysis_summary, evidence_bundle, load_analysis, write_report


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.links = []
        self.scripts = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.links.extend(value for key, value in attrs if key in ("href", "src"))
        if tag == "script":
            self.scripts.append(attrs)


def example_summary():
    return {
        "variants": [{"variant_id": f"v{i}", "pip": 0.05, "rank": i + 1} for i in range(20)],
        "credible_sets": [
            {
                "component_id": 7,
                "members": [f"v{i}" for i in range(20)],
                "actual_coverage": 1.0,
                "min_abs_corr": 0.9,
            }
        ],
        "components": [
            {
                "component_id": 7,
                "active": True,
                "cs_retained": True,
                "effective_candidates": 20.0,
                "alpha": [0.05] * 20,
                "ranking": [f"v{i}" for i in range(20)],
            }
        ],
        "convergence": {
            "requested_criterion": "joint",
            "effective_criterion": "joint",
            "niter": 3,
            "stop_reason": "criterion_satisfied",
            "tol": 0.001,
            "alpha_tol": 0.000002,
            "pip_tol": 0.000003,
            "last": {
                "iteration": 3,
                "elbo": -17.12345678901234,
                "delta_elbo": 0.000012345678901234,
            },
        },
        "trace": [{"iteration": 3}],
    }


def test_compact_summary_preserves_all_large_cs_members_and_full_precision(tmp_path):
    summary = example_summary()
    result = analysis_summary(summary)
    assert len(result["results"]["candidates"]) == 20
    assert all(
        row["pip"] < 0.1 and "formal_CS_member" in row["selection_sources"]
        for row in result["results"]["candidates"]
    )
    assert result["results"]["credible_sets"][0]["effective_candidates"] == 20.0
    receipt = write_report(summary, tmp_path)
    compact = json.loads(Path(receipt["analysis_summary"]).read_text())
    assert compact["fit"]["last"]["elbo"] == -17.12345678901234
    assert compact["fit"]["requested_criterion"] == "joint"
    assert compact["fit"]["trajectory_available"]
    page = (tmp_path / "index.html").read_text()
    assert "-17.12" in page
    assert "-17.12345678901234" not in page[: page.index('id="details"')]
    assert "-17.12345678901234" in page[page.index('id="details"') :]
    assert "v19" in page
    restored = load_analysis(tmp_path)
    assert restored["summary"] == summary
    for kwargs in ({"candidate_pip": -1}, {"top_candidates": True}, {"top_candidates": -0.5}):
        with pytest.raises(ValueError):
            analysis_summary(summary, **kwargs)


def test_report_reading_order_safe_local_exports_and_no_fact_dom(tmp_path):
    summary = example_summary()
    summary["prior_influence"] = [
        {"component_id": 7, "variant_id": f"detail_{i}", "conditional_log_bf": float(i)}
        for i in range(1000)
    ]
    hostile = '<script src="https://invalid.example/test">bad()</script>'
    receipt = write_report(
        summary,
        tmp_path / "first",
        title=hostile,
        provenance={"reference_sources": hostile},
        annotations={"note": hostile},
    )
    again = write_report(
        summary,
        tmp_path / "again",
        title=hostile,
        provenance={"reference_sources": hostile},
        annotations={"note": hostile},
    )
    page = Path(receipt["html"]).read_text()
    assert page == Path(again["html"]).read_text()
    parser = Page(page)
    assert not parser.scripts and "&lt;script" in page and hostile not in page
    assert parser.links
    assert all(
        not value.startswith(("http:", "https:", "//", "/"))
        and ".." not in value
        and (value.startswith("#") or (Path(receipt["html"]).parent / value).is_file())
        for value in parser.links
    )
    headings = [
        f'id="{name}"'
        for name in (
            "main-results",
            "fit-status",
            "figures",
            "sensitivity-ld",
            "details",
            "exports",
        )
    ]
    assert [page.index(name) for name in headings] == sorted(page.index(name) for name in headings)
    evidence = json.loads(Path(receipt["evidence"]).read_text())
    assert len(evidence["facts"]) > 3000
    assert "detail_999" in Path(receipt["evidence"]).read_text()
    assert "detail_999" not in page
    assert not any(fact["id"] in page for fact in evidence["facts"])
    assert len(page) < len(Path(receipt["evidence"]).read_text()) / 3
    assert "query" in page and "analysis_summary.json" in page


@pytest.mark.parametrize(
    "objective_definition", ["missing", None, "last_logged_elbo; original prior; susieR refinement"]
)
def test_refinement_reports_recorded_objective_accepted_candidates_and_stage_controls(
    tmp_path, objective_definition
):
    summary = example_summary()
    control = {
        "convergence": {"criterion": "joint", "alpha_tol": 2e-6},
        "tol": 0.001,
        "max_iter": 100,
        "track_fit": True,
    }
    summary["refinement"] = {
        "status": "completed",
        "total_fits": 4,
        "best_objective": -15.0,
        "before": {"elbo": -18.0, "logged_elbo": -17.99},
        "after": {"elbo": -15.0, "logged_elbo": -14.99},
        "requested_convergence": control["convergence"],
        "history": [
            {
                "round": 1,
                "component": 7,
                "incumbent_objective": -18.0,
                "excluded_ids": ["v0"],
                "accepted": True,
                "improvement": 3.0,
                "objective": -15.0,
                "stages": {"A": {"solver_settings": control}, "B": {"solver_settings": control}},
            },
            {
                "round": 2,
                "component": 7,
                "incumbent_objective": -15.0,
                "accepted": False,
                "improvement": -2.0,
                "objective": -17.0,
            },
        ],
    }
    if objective_definition != "missing":
        summary["refinement"]["objective_definition"] = objective_definition
    compact = analysis_summary(summary)
    expected_definition = (
        objective_definition if objective_definition not in ("missing", None) else "not recorded"
    )
    assert compact["refinement"]["objective_definition"] == expected_definition
    assert compact["refinement"]["improvement"] == 3.0
    assert compact["refinement"]["attempts"] == 2
    assert compact["refinement"]["accepted_attempts"] == 1
    assert compact["refinement"]["before_logged_elbo"] == -17.99
    evidence = evidence_bundle(summary)
    search = evidence["data"]["summary"]["refinement"]
    assert search["requested_convergence"] == control["convergence"]
    assert search["history"][0]["incumbent_objective"] == -18.0
    assert search["history"][0]["stages"]["B"]["solver_settings"] == control
    write_report(summary, tmp_path)
    page = (tmp_path / "index.html").read_text()
    assert "Accepted attempts" in page
    assert "Recorded search objective: " + expected_definition in page
    assert "Recomputed post-finalization ELBO is a separate diagnostic" in page
    assert "negative rejected candidate" in page


@pytest.mark.parametrize(
    "rows,expected",
    [
        ([{"flag": None, "status": "not_assessed", "standardized_residual": 0.5}], (0, 0, 0, 1)),
        ([{"flag": False, "status": "assessed"}], (1, 0, 0, 0)),
        (
            [
                {"flag": True, "status": "assessed"},
                {"flag": False, "status": "assessed"},
                {"flag": None, "status": "conditional_only", "standardized_residual": 0.5},
                {"flag": None, "status": "not_assessed", "input_missing": True},
            ],
            (2, 1, 1, 1),
        ),
    ],
)
def test_ld_report_counts_keep_unknown_distinct(rows, expected, tmp_path):
    ld = {
        "variants": [dict(row, variant_id=str(index)) for index, row in enumerate(rows)],
        "thresholds": {"flip_log_lr": 2.0, "flip_abs_z": 2.0, "strict": True},
        "mixture": {"status": "not_assessed", "reason": "test_missing"},
    }
    result = analysis_summary(example_summary(), ld=ld)["ld"]
    assert (
        tuple(result[key] for key in ("assessed", "flagged", "conditional_only", "unassessed"))
        == expected
    )
    write_report(example_summary(), tmp_path, ld=ld)
    page = (tmp_path / "index.html").read_text()
    assert f"{expected[0]} assessed (" in page
    assert f"{expected[2]} conditional-only; {expected[3]} unassessed" in page
    if expected[0] == 0:
        assert "No variants were flagged" not in page


def test_failed_mixture_and_missing_z_diagnose_plot_report(monkeypatch, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import prusie.ld as diagnostics
    from prusie.plots import plot_ld_diagnostics

    def fail(_):
        raise RuntimeError("deliberate mixture failure")

    monkeypatch.setattr(diagnostics, "_mixture", fail)
    result = diagnostics.diagnose_ld(
        [1.0, np.nan, -0.5], np.eye(3), n=200, s=0.1, variant_ids=["a", "missing", "c"]
    )
    assert result["variants"][0]["standardized_residual"] is not None
    assert result["variants"][1]["working_standardized_residual"] is not None
    assert result["variants"][1]["observed_z"] is None
    assert result["variants"][1]["standardized_residual"] is None
    figure = plot_ld_diagnostics(result)
    receipt = write_report(example_summary(), tmp_path, ld=result, figures={"LD states": figure})
    compact = json.loads(Path(receipt["analysis_summary"]).read_text())
    assert compact["ld"]["assessed"] == 0
    assert compact["ld"]["conditional_only"] == 2
    assert compact["ld"]["unassessed"] == 1
    page = Path(receipt["html"]).read_text()
    assert (
        "0 assessed (0 flagged; 0 assessed and not flagged); 2 conditional-only; 1 unassessed"
        in page
    )
    assert "mixture_optimization_failed" in page
    assert "missing_z_replaced_by_zero_upstream" in page
    assert "No variants were flagged" not in page
    plt.close(figure[0])


def test_report_prefers_baseline_cs_layer_but_retains_complete_component_relations(tmp_path):
    comparison = {
        "schema_version": "prusie.comparison.v2",
        "baseline": "baseline",
        "settings": ["baseline", "L12"],
        "signal_relations": [
            {
                "setting_a": "baseline",
                "setting_b": "L12",
                "component_a": 7,
                "component_b": 3,
                "status": "matched",
                "matching_layer": "signals",
            }
        ],
        "component_relations": [
            {
                "setting_a": "baseline",
                "setting_b": "L12",
                "component_a": 8,
                "component_b": 9,
                "status": "component_overlap",
                "posterior_overlap": 1.0,
            }
        ],
        "pairwise": [{"setting_a": "baseline", "setting_b": "L12", "max_abs_delta_pip": 0.0}],
        "layer_summary": {
            "settings": {"baseline": {"retained_cs": 1, "active_components": 2}},
            "baseline_relation_counts": {"matched": 1},
        },
    }
    compact = analysis_summary(example_summary(), comparison=comparison)
    assert compact["sensitivity"]["relation_counts"] == {"matched": 1}
    receipt = write_report(example_summary(), tmp_path, comparison=comparison)
    evidence = json.loads(Path(receipt["evidence"]).read_text())
    assert evidence["data"]["comparison"]["component_relations"][0]["posterior_overlap"] == 1.0
    page = Path(receipt["html"]).read_text()
    assert page.index("component_overlap") > page.index('id="details"')
    assert "component_overlap" not in page[: page.index('id="details"')]


def test_identical_figure_exports_are_reproducible(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(2, 1))
    axis.plot([0, 1], [0, 1])
    write_report(example_summary(), tmp_path / "first", figures={"Trend": figure})
    write_report(example_summary(), tmp_path / "second", figures={"Trend": figure})
    for name in (
        "index.html",
        "evidence.json",
        "analysis_summary.json",
        "figure_1.svg",
        "figure_1.png",
    ):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()
    plt.close(figure)


def test_actual_native_stop_is_reported_with_requested_thresholds():
    from prusie import ConvergenceConfig, summarize, susie_rss

    fit = susie_rss(
        z=np.array([4.0, 1.0, -0.5]),
        R=np.eye(3),
        n=200,
        L=1,
        convergence=ConvergenceConfig(criterion="joint", alpha_tol=2e-6, pip_tol=3e-6, cycle_lag=4),
        track_fit=True,
        max_iter=100,
    )
    recorded = analysis_summary(summarize(fit))["fit"]
    assert fit.convergence["stop_reason"] == "criterion_satisfied"
    assert "was satisfied" in recorded["pending"]
    assert recorded["requested_criterion"] == "joint"
    assert recorded["alpha_tol"] == 2e-6 and recorded["pip_tol"] == 3e-6
    assert recorded["max_cycle_lag"] == 4


def test_legacy_reason_only_missing_working_values_never_become_observed(tmp_path):
    row = {
        "variant_id": "legacy_missing",
        "observed_z": 0.0,
        "expected_z": 2.0,
        "standardized_residual": -987.654321,
        "flag": None,
        "unavailable_reason": "missing_z_replaced_by_zero_upstream",
    }
    receipt = write_report(example_summary(), tmp_path, ld={"variants": [row]})
    page = Path(receipt["html"]).read_text()
    assert "legacy_missing" in page and "-987.654" not in page
    assert '<td>NA</td><td class="numeric">2</td><td>NA</td>' in page
    saved = json.loads(Path(receipt["evidence"]).read_text())
    assert saved["data"]["ld_diagnostics"]["variants"][0] == row
