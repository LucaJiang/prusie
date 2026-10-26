"Review regressions for plotted states, independent trajectories and selection."

import copy
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from prusie import plots
from prusie.ld import diagnose_ld
from prusie.report import write_report


@pytest.mark.parametrize("kind", ["conditional", "missing", "unknown", "false", "mixed"])
def test_ld_assessment_is_preserved_through_plot_and_report(kind, monkeypatch, tmp_path):
    import prusie.ld

    if kind == "conditional":

        def fail(_):
            raise RuntimeError("controlled mixture failure")

        monkeypatch.setattr(prusie.ld, "_mixture", fail)
        result = diagnose_ld([3.0, -2.0], np.eye(2), s=0.1)
        expected = dict(assessed=0, flagged=0, conditional_only=2, unassessed=0)
    elif kind == "missing":
        result = diagnose_ld([np.nan, 3.0], np.eye(2), s=0.1)
        assert result["variants"][0]["expected_z"] == 0
        assert np.isfinite(result["variants"][0]["working_standardized_residual"])
        expected = dict(assessed=1, flagged=0, conditional_only=0, unassessed=1)
    elif kind == "unknown":
        result = diagnose_ld([np.nan, np.nan], np.eye(2), s=0.1)
        expected = dict(assessed=0, flagged=0, conditional_only=0, unassessed=2)
    elif kind == "false":
        result = diagnose_ld([1.0, 1.0], np.eye(2), s=0.1)
        expected = dict(assessed=2, flagged=0, conditional_only=0, unassessed=0)
    else:
        # A recorded mixed state, including a legacy missing-z zero placeholder.
        result = {
            "variants": [
                {
                    "variant_id": "a",
                    "status": "assessed",
                    "flag": True,
                    "observed_z": 4.0,
                    "expected_z": -3.0,
                    "standardized_residual": 7.0,
                },
                {
                    "variant_id": "b",
                    "status": "assessed",
                    "flag": False,
                    "observed_z": 1.0,
                    "expected_z": 0.0,
                    "standardized_residual": 1.0,
                },
                {
                    "variant_id": "c",
                    "status": "conditional_only",
                    "flag": None,
                    "observed_z": 2.0,
                    "expected_z": 0.0,
                    "standardized_residual": 2.0,
                    "unavailable_reason": "mixture_optimization_failed",
                },
                {
                    "variant_id": "d",
                    "status": "not_assessed",
                    "flag": None,
                    "input_z": None,
                    "observed_z": 0.0,
                    "expected_z": 1.0,
                    "standardized_residual": -1.0,
                    "unavailable_reason": "missing_z_replaced_by_zero_upstream",
                },
            ],
            "mixture": {"status": "not_assessed"},
            "thresholds": {"flip_log_lr": 2.0, "flip_abs_z": 2.0},
        }
        expected = dict(assessed=2, flagged=1, conditional_only=1, unassessed=1)
    view = plots.plot_ld_diagnostics(result)
    for key, value in expected.items():
        assert view[2]["status_counts"][key] == value
    legend = " ".join(t.get_text() for t in view[1][0].get_legend().texts)
    if expected["conditional_only"]:
        assert "Conditional only; flip not assessed" in legend
    if expected["unassessed"]:
        assert "Not assessed / missing input" in legend
        for i, row in enumerate(result["variants"]):
            if row.get("input_missing") or ("input_z" in row and row["input_z"] is None):
                assert np.isnan(view[2]["observed_z"][i])
                assert np.isnan(view[2]["standardized_residual"][i])
    assert len(view[2]["flagged_or_unassessed"]) == sum(
        expected[k] for k in ("flagged", "conditional_only", "unassessed")
    )
    assert view[2]["thresholds"] == result["thresholds"]
    assert view[2]["mixture"] == result["mixture"]
    write_report(
        {"variants": [], "components": [], "credible_sets": []},
        tmp_path,
        ld=result,
        figures={"LD assessment": view},
    )
    page = (tmp_path / "index.html").read_text()
    assert "conditional" in page.lower() and "unassessed" in page.lower()
    assert "No flagged variants" not in page
    plt.close(view[0])


def test_before_after_iterations_thresholds_and_zero_components():
    def stage(length, criterion, alpha_tol):
        return {
            "convergence": {
                "requested_criterion": criterion,
                "effective_criterion": criterion,
                "stop_reason": "elbo_tolerance",
                "alpha_tol": alpha_tol,
                "tol": 0.01,
            },
            "trace": [
                {
                    "iteration": j + 1,
                    "elbo": -10.0 + j,
                    "delta_elbo": 0.01,
                    "max_abs_delta_alpha": 0.02,
                    "max_abs_delta_pip": 0.01,
                    "V": [0.0, 0.2 - j * 0.01, 0.0],
                    "sigma2": 1.0,
                    "events": [],
                }
                for j in range(length)
            ],
        }

    before, after = stage(5, "elbo", 0.003), stage(2, "joint", 0.0007)
    view = plots.plot_convergence(before, after=after)
    np.testing.assert_array_equal(view[1][0].lines[0].get_xdata(), [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(view[1][6].lines[0].get_xdata(), [1, 2])
    assert "Before" in view[1][0].get_xlabel() and "After" in view[1][6].get_xlabel()
    assert view[2]["stages"]["Before"]["zero_variance_components"] == [0, 2]
    assert view[2]["stages"]["After"]["variance_components_shown"] == [1]
    assert view[1][2].lines[1].get_ydata() == [0.003, 0.003]
    assert view[1][8].lines[1].get_ydata() == [0.0007, 0.0007]
    assert len(view[2]["stages"]["Before"]["trace"]) == 5
    assert not any(
        np.array_equal(line.get_xdata(), np.arange(1, 8)) for ax in view[1] for line in ax.lines
    )
    plt.close(view[0])


def test_unaccepted_search_is_not_labelled_an_accepted_fit():
    summary = {
        "convergence": {},
        "trace": [],
        "refinement": {"history": [{"accepted": False}], "status": "no_improvement"},
    }
    view = plots.plot_convergence(summary, after=summary)
    assert view[2]["after_label"] == "After search (unchanged)"
    assert view[2]["refinement_candidate_accepted"] is False
    assert "Accepted after" not in view[1][6].get_title()
    plt.close(view[0])


def test_refinement_uses_ids_correct_columns_and_accepted_incumbents():
    ids = np.array(["rs100", "rs200", "rs300"])
    before = SimpleNamespace(
        variant_ids=ids,
        pip=np.array([1.0, 0.1, 0.0]),
        sets={"cs_index": [4], "cs": [np.array([0])]},
        refinement={},
    )
    after = SimpleNamespace(
        variant_ids=ids,
        pip=np.array([0.0, 0.1, 1.0]),
        sets={"cs_index": [8], "cs": [np.array([2])]},
        refinement={
            "status": "completed",
            "best_objective": -5.0,
            "history": [
                {
                    "round": 1,
                    "objective": -5.0,
                    "improvement": 5.0,
                    "accepted": True,
                    "excluded_ids": ["rs100"],
                    "reason": "accepted_best_original_objective",
                },
                {
                    "round": 2,
                    "objective": -7.0,
                    "improvement": -2.0,
                    "accepted": False,
                    "excluded_ids": ["rs300"],
                    "reason": "no_improvement_above_absolute_tol",
                },
            ],
        },
    )
    view = plots.plot_refinement(before, after)
    assert view[2]["before_cs"][0]["members"] == ["rs100"]
    assert view[2]["after_cs"][0]["members"] == ["rs300"]
    before_row = next(row for row in view[2]["table_rows"] if row[0] == "Before CS")
    after_row = next(row for row in view[2]["table_rows"] if row[0] == "After CS")
    assert before_row[1:3] == ["4", ""] and after_row[1:3] == ["", "8"]
    assert "ACCEPTED" in view[2]["table_rows"][0][-1]
    objectives = [row["objective"] for row in view[2]["incumbent_path"]]
    assert objectives == [-10.0, -5.0, -5.0]
    assert view[2]["candidate_records"][1]["improvement"] == -2.0
    plt.close(view[0])
    fig, axes = plt.subplots(3, 1)
    compatibility = plots.plot_refinement(before, after, axes=axes)
    assert compatibility[0] is fig and len(compatibility[1]) == 3
    assert "3-axis layout" in axes[1].get_title()
    plt.close(fig)


def test_tables_default_complete_and_explicit_pages():
    rows = [{"component_id": i, "members": [f"rs{i}"]} for i in range(16)]
    summary = {"variants": [], "credible_sets": rows}
    view = plots.plot_refinement(summary, summary)
    assert len(view[2]["table_rows"]) == 32
    assert len(view[1][-1].tables[0].get_celld()) == (32 + 1) * 5
    plt.close(view[0])
    view = plots.plot_refinement(summary, summary, table_page=1, table_page_size=10)
    assert len(view[2]["table_rows"]) == 32
    assert len(view[1][-1].tables[0].get_celld()) == 11 * 5
    assert "Rows 11–20 of 32" in view[1][-1].texts[0].get_text()
    plt.close(view[0])


def test_variant_selection_chooses_supported_component_and_prior_subset():
    summary = {
        "variants": [
            {"variant_id": "rs33", "pip": 1.0},
            {"variant_id": "rs86", "pip": 0.18},
            {"variant_id": "rs90", "pip": 0.82},
        ],
        "credible_sets": [
            {"component_id": 1, "members": ["rs86", "rs90"], "requested_coverage": 0.95},
            {"component_id": 2, "members": ["rs33"], "requested_coverage": 0.95},
        ],
        "components": [
            {
                "component_id": 1,
                "ranking": ["rs90", "rs86", "rs33"],
                "alpha": [0.82, 0.18, 0.0],
                "cumulative_posterior": [0.82, 1.0, 1.0],
            },
            {
                "component_id": 2,
                "ranking": ["rs33", "rs86", "rs90"],
                "alpha": [1.0, 0.0, 0.0],
                "cumulative_posterior": [1.0, 1.0, 1.0],
            },
        ],
        "prior_influence": [
            {
                "component_id": c,
                "variant_id": v,
                "prior_weight": 1 / 3,
                "conditional_log_bf": 1.0,
                "alpha": 0.3,
            }
            for c in [1, 2]
            for v in ["rs33", "rs86", "rs90"]
        ],
    }
    first = plots.plot_signal(summary)
    pair = plots.plot_signal(summary, variant_id="rs86")
    prior = plots.plot_prior_influence(summary, variant_id="rs86", variant_ids=["rs86", "rs90"])
    assert first[2]["component"]["component_id"] == 2  # highest supported candidate, not first CS
    assert pair[2]["component"]["component_id"] == prior[2]["component_id"] == 1
    assert {"rs86", "rs90"} <= {text.get_text() for text in pair[1][0].get_xticklabels()}
    assert [r["variant_id"] for r in prior[2]["prior_influence"]] == ["rs86", "rs90"]
    assert len(prior[2]["all_prior_influence"]) == 3
    for view in (first, pair, prior):
        plt.close(view[0])


def test_gene_locus_filters_and_known_mismatch_with_rsid(tmp_path):
    gtf = tmp_path / "small.gtf"
    gtf.write_text(
        "\n".join(
            (
                f"{chrom}"
                f"\tlocal\ttranscript\t{start}"
                f"\t{end}"
                f'\t.\t+\t.\tgene_id "g{index}'
                f'"; transcript_id "t{index}'
                f'";'
            )
            for index, (chrom, start, end) in enumerate(
                [("1", 100, 200), ("1", 300, 400), ("2", 100, 200)]
            )
        )
        + "\n"
    )
    genes = plots.read_gene_track(
        gtf,
        genome_build="toy_build",
        source="synthetic test",
        chromosome="chr1",
        start=150,
        end=250,
    )
    assert [r["transcript_id"] for r in genes["records"]] == ["t0"]
    summary = {
        "variants": [
            {
                "variant_id": "rs999",
                "position": 160,
                "pip": 0.9,
                "metadata": {"chromosome": "chr1", "genome_build": "toy_build"},
            }
        ]
    }
    view = plots.plot_locus(summary, genes=genes)
    assert view[2]["physical_positions"]
    assert view[2]["coordinate_status"] == {"chromosome": "matched", "genome_build": "matched"}
    plt.close(view[0])
    changed = copy.deepcopy(summary)
    changed["variants"][0]["metadata"]["chromosome"] = "2"
    with pytest.raises(ValueError, match=r"chromosomes differ"):
        plots.plot_locus(changed, genes=genes)
    empty_track = plots.read_gene_track(
        gtf, genome_build="toy_build", source="test", chromosome="1", start=1000, end=2000
    )
    assert not empty_track["records"]
    with pytest.raises(ValueError, match=r"chromosomes differ"):
        plots.plot_locus(changed, genes=empty_track)
    changed["variants"][0]["metadata"] = {"chromosome": "1", "genome_build": "other_build"}
    with pytest.raises(ValueError, match=r"genome builds differ"):
        plots.plot_locus(changed, genes=genes)
    changed["variants"][0]["metadata"] = {}
    view = plots.plot_locus(changed, genes=genes)
    assert view[2]["coordinate_status"] == {
        "chromosome": "not_assessed",
        "genome_build": "not_assessed",
    }
    assert "not_assessed" in view[1][-1].get_title()
    plt.close(view[0])
    with pytest.raises(ValueError, match=r"requires chromosome"):
        plots.read_gene_track(gtf, genome_build="toy", source="test", start=10)


def test_sensitivity_main_layer_retains_large_cs_and_missing_values():
    comparison = {
        "schema_version": "prusie.comparison.v2",
        "baseline": "baseline",
        "settings": ["baseline", "changed", "third"],
        "variants": [
            {
                "variant_id": f"rs{i}",
                "candidate": i < 20,
                "pips": {"baseline": 0.05, "changed": None if i == 0 else 0.05, "third": 0.04},
            }
            for i in range(21)
        ],
        "signal_relations": [
            {
                "setting_a": "baseline",
                "setting_b": "changed",
                "component_a": 2,
                "component_b": 1,
                "status": "matched",
                "posterior_overlap": 1.0,
            },
            {
                "setting_a": "changed",
                "setting_b": "third",
                "component_a": 1,
                "component_b": 4,
                "status": "matched",
                "posterior_overlap": 1.0,
            },
        ],
        "component_relations": [
            {
                "setting_a": "baseline",
                "setting_b": "changed",
                "component_a": i,
                "component_b": j,
                "status": "component_overlap",
                "posterior_overlap": 1.0,
            }
            for i in range(7)
            for j in range(9)
        ],
    }
    view = plots.plot_sensitivity(comparison)
    assert len(view[2]["variant_ids"]) == 20  # complete CS membership, all PIP < .1
    assert len(view[2]["signal_relations"]) == 1
    assert len(view[2]["all_component_relations"]) == 63
    assert np.isnan(view[2]["pip_matrix"][0, 1])
    assert view[2]["pip_matrix"][1, 1] == 0.05
    assert "baseline versus each setting" in view[1][1].get_title()
    plt.close(view[0])
    complete = plots.plot_sensitivity(
        comparison, layer="components", candidates_only=False, table_page=2, table_page_size=10
    )
    assert len(complete[2]["signal_relations"]) == 63
    assert len(complete[2]["variant_ids"]) == 21
    plt.close(complete[0])


def test_prior_explicit_component_keeps_minimal_legacy_table_support():
    summary = {
        "prior_influence": [
            {
                "component_id": 7,
                "variant_id": "rs1",
                "prior_weight": 0.2,
                "conditional_log_bf": 1.0,
                "alpha": 0.9,
            }
        ]
    }
    view = plots.plot_prior_influence(summary, 7)
    assert view[2]["prior_influence"] == summary["prior_influence"]
    plt.close(view[0])


def test_native_summary_study_information_checks_gene_build_and_chromosome(tmp_path):
    from prusie import summarize, susie_rss

    fit = susie_rss(
        z=[3.0, 1.0],
        R=np.eye(2),
        n=100,
        L=1,
        variant_ids=["rs12", "rs13"],
        variant_metadata={"position": [100, 120]},
    )
    gtf = tmp_path / "local.gtf"
    gtf.write_text('1\tlocal\ttranscript\t90\t130\t.\t+\t.\tgene_id "g"; transcript_id "t";\n')
    genes = plots.read_gene_track(
        gtf, genome_build="build1", source="synthetic test", chromosome="1"
    )
    summary = summarize(fit, input_info={"genome_build": "build1", "chromosome": "chr1"})
    view = plots.plot_locus(summary, genes=genes)
    assert view[2]["coordinate_status"] == {"chromosome": "matched", "genome_build": "matched"}
    plt.close(view[0])
    mismatch = summarize(fit, input_info={"genome_build": "build2", "chromosome": "1"})
    with pytest.raises(ValueError, match=r"genome builds differ"):
        plots.plot_locus(mismatch, genes=genes)
    mismatch = summarize(fit, input_info={"genome_build": "build1", "chromosome": "2"})
    with pytest.raises(ValueError, match=r"chromosomes differ"):
        plots.plot_locus(mismatch, genes=genes)
    mismatch = summarize(
        fit,
        metadata={"chromosome": ["2", "2"]},
        input_info={"genome_build": "build1", "chromosome": "1"},
    )
    with pytest.raises(ValueError, match=r"chromosomes differ between metadata"):
        plots.plot_locus(mismatch, genes=genes)
    mismatch = summarize(
        fit,
        metadata={"genome_build": ["build2", "build2"]},
        input_info={"genome_build": "build1", "chromosome": "1"},
    )
    with pytest.raises(ValueError, match=r"Variant genome builds differ"):
        plots.plot_locus(mismatch, genes=genes)


def test_legacy_reason_only_missing_value_is_not_observed_in_table():
    row = {
        "variant_id": "missing",
        "observed_z": 0.0,
        "expected_z": 1.0,
        "standardized_residual": -123.45,
        "flag": None,
        "unavailable_reason": "missing_z_replaced_by_zero_upstream",
    }
    view = plots.plot_ld_diagnostics({"variants": [row]})
    assert np.isnan(view[2]["observed_z"][0]) and np.isnan(view[2]["standardized_residual"][0])
    texts = [cell.get_text().get_text() for cell in view[1][2].tables[0].get_celld().values()]
    assert "-123.45" not in texts
    assert (
        view[2]["variants"][0]["standardized_residual"] == -123.45
    )  # unchanged complete original evidence
    assert view[2]["status_counts"]["unassessed"] == 1
    plt.close(view[0])


def test_nonfinite_fallback_selects_alpha_pip_thresholds():
    summary = {
        "convergence": {
            "requested_criterion": "elbo",
            "effective_criterion": "alpha_pip_nonfinite_fallback",
            "tol": 0.001,
            "alpha_tol": 0.0002,
            "pip_tol": 0.0003,
            "stop_reason": "criterion_satisfied",
        },
        "trace": [
            {
                "iteration": 1,
                "elbo": np.nan,
                "max_abs_delta_alpha": 0.0001,
                "max_abs_delta_pip": 0.0001,
                "V": [1.0],
                "sigma2": 1.0,
            }
        ],
    }
    view = plots.plot_convergence(summary)
    assert "not selected" in view[1][1].lines[2].get_label()
    assert view[1][2].lines[1].get_label() == "Selected stopping threshold: 0.0002"
    assert view[1][3].lines[1].get_label() == "Selected stopping threshold: 0.0003"
    plt.close(view[0])


def test_compatible_pip_rule_uses_fitting_tol_for_both_posterior_thresholds():
    summary = {
        "convergence": {
            "requested_criterion": "elbo",
            "effective_criterion": "pip",
            "tol": 0.001,
            "alpha_tol": 0.0002,
            "pip_tol": 0.0003,
        },
        "trace": [
            {
                "iteration": 1,
                "elbo": np.nan,
                "max_abs_delta_alpha": 0.0001,
                "max_abs_delta_pip": 0.0001,
                "V": [1.0],
                "sigma2": 1.0,
            }
        ],
    }
    view = plots.plot_convergence(summary)
    for axis in view[1][2:4]:
        assert axis.lines[1].get_label() == "Selected stopping threshold: 0.001"
        assert axis.lines[1].get_ydata() == [0.001, 0.001]
    plt.close(view[0])
    # A saved summary that omits the actual fitting tolerance must not use
    # unrelated configured alpha/PIP thresholds as the selected PIP rule.
    del summary["convergence"]["tol"]
    view = plots.plot_convergence(summary)
    assert len(view[1][2].lines) == len(view[1][3].lines) == 1
    plt.close(view[0])


@pytest.mark.parametrize(
    "definition",
    [
        "last_logged_elbo; original prior; susieR refinement",
        "original-prior post-trim ELBO",
        None,
    ],
)
def test_refinement_plot_preserves_recorded_objective_definition(definition):
    before = {"variants": [], "credible_sets": []}
    after = dict(before, refinement={"objective_definition": definition, "history": []})
    view = plots.plot_refinement(before, after)
    assert view[2]["objective_definition"] == (definition or "not recorded")
    plt.close(view[0])


def test_explicit_signal_view_keeps_cs_and_full_posterior_without_renormalizing():
    ids = [f"rs{i}" for i in range(30)]
    alpha = np.full(30, 1 / 30)
    summary = {
        "variants": [{"variant_id": v, "pip": float(a)} for v, a in zip(ids, alpha, strict=True)],
        "components": [
            {
                "component_id": 0,
                "ranking": ids,
                "alpha": alpha,
                "cumulative_posterior": np.cumsum(alpha),
                "null_mass": 0,
            }
        ],
        "credible_sets": [{"component_id": 0, "members": ids[:29], "requested_coverage": 0.95}],
        "proxies": [],
    }
    view = plots.plot_signal(summary, 0, max_variants=3)
    assert view[2]["displayed_variant_ids"] == ids[:29]
    assert len(view[2]["component"]["alpha"]) == 30
    assert view[2]["cumulative_posterior_renormalized"] is False
    assert view[1][1].lines[0].get_ydata()[-1] == pytest.approx(29 / 30)
    plt.close(view[0])
    summary["credible_sets"][0]["members"] = ids[:2]
    view = plots.plot_signal(summary, 0, variant_id="rs29", max_variants=3)
    assert view[2]["displayed_variant_ids"] == [*ids[:3], "rs29"]
    plt.close(view[0])
