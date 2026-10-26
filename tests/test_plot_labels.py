"""Small deterministic presentation cases; no fitting or numerical policy changes."""

import copy
import json
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from prusie import plots
from prusie._style import COLORS, format_number


@pytest.fixture
def posterior():
    ids = [f"variant_{i:02d}_with_long_local_identifier" for i in range(24)]
    alpha = np.arange(24, 0, -1, dtype=float)
    alpha /= alpha.sum()
    rows = [
        {
            "variant_id": v,
            "pip": float(a),
            "position": 1_000_000 + 100 * i,
            "neg_log10_p": float(i) / 4,
        }
        for i, (v, a) in enumerate(zip(ids, alpha, strict=True))
    ]
    components = [
        {
            "component_id": 2,
            "ranking": ids,
            "alpha": alpha.tolist(),
            "cumulative_posterior": np.cumsum(alpha).tolist(),
        }
    ]
    return {
        "variants": rows,
        "components": components,
        "credible_sets": [{"component_id": 2, "members": ids[:19], "requested_coverage": 0.95}],
        "prior_influence": [
            {
                "component_id": 2,
                "variant_id": v,
                "alpha": float(a),
                "prior_weight": 1 / 24,
                "effective_prior_weight": 1 / 24,
                "conditional_log_bf": float(i),
            }
            for i, (v, a) in enumerate(zip(ids[::-1], alpha[::-1], strict=True))
        ],
        "convergence": {
            "requested_criterion": "elbo",
            "effective_criterion": "elbo",
            "stop_reason": "elbo_tolerance",
            "tol": 0.01,
            "alpha_tol": 0.002,
            "pip_tol": 0.003,
        },
        "trace": [
            {
                "iteration": i + 1,
                "elbo": -1_000_000.0 + delta,
                "delta_elbo": change,
                "max_abs_delta_alpha": 0.002,
                "max_abs_delta_pip": 0.0,
                "V": [float(j + i) / 10 for j in range(10)],
                "sigma2": 1.0,
            }
            for i, (delta, change) in enumerate([(0.0, None), (1.0, 1.0), (0.9, -0.1), (0.9, 0.0)])
        ],
        "refinement": {"history": [{"accepted": False}], "status": "no_improvement"},
    }


def incomplete_ld():
    return {
        "variants": [
            {
                "variant_id": "conditional_extreme",
                "observed_z": 8.0,
                "expected_z": 0.0,
                "standardized_residual": 8.0,
                "status": "conditional_only",
                "flag": None,
                "log_likelihood_ratio": None,
                "unavailable_reason": "mixture_optimization_failed",
            },
            {
                "variant_id": "missing_observation",
                "input_missing": True,
                "observed_z": 0.0,
                "working_z": 0.0,
                "expected_z": 2.0,
                "standardized_residual": -2.0,
                "status": "not_assessed",
                "flag": None,
                "unavailable_reason": "missing_z",
            },
        ],
        "mixture": {"status": "not_assessed", "reason": "controlled mixture failure"},
        "thresholds": {"flip_log_lr": 2.0, "flip_abs_z": 2.0},
    }


def test_style_is_lazy_and_probability_endpoints_are_honest():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            'import sys; import prusie._style; assert "matplotlib" not in sys.modules',
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert format_number(0.0, probability=True) == "0"
    assert format_number(1.0, probability=True) == "1"
    assert format_number(1.0 - 1e-12, probability=True).startswith("> ")
    assert format_number(1e-12, probability=True) not in ("0", "1")
    assert format_number(None) == format_number(np.nan) == "NA"


def test_locus_coordinate_units_aligned_tracks_and_all_observations(posterior):
    view = plots.plot_locus(posterior, R=np.eye(24), layout="compact", color_by_ld=True)
    figure, axes, data = view
    assert data["position_unit"] == "Mb"
    np.testing.assert_allclose(data["x"], np.array(data["raw_positions"]) / 1e6)
    assert all(ax.get_xlim() == axes[0].get_xlim() for ax in axes)
    assert not any(ax.get_xlabel() for ax in axes[:-1])
    assert "(Mb)" in axes[-1].get_xlabel()
    assert len(axes[0].collections[0].get_offsets()) == 24
    assert len(data["labelled_variants"]) == 3
    assert axes[0].collections[0].get_clim() == (0.0, 1.0)
    plt.close(figure)


def test_compact_signal_and_prior_keep_large_cs_and_full_cumulative(posterior):
    R = np.eye(24)
    R[0, 1] = R[1, 0] = -0.9
    original = copy.deepcopy(posterior)
    signal = plots.plot_signal(posterior, layout="compact", max_variants=3, ld_measure="r2", R=R)
    prior = plots.plot_prior_influence(posterior, layout="compact", max_variants=3)
    expected = posterior["credible_sets"][0]["members"]
    assert signal[2]["displayed_variant_ids"] == prior[2]["displayed_variant_ids"] == expected
    assert len(signal[1]) == 3 and not any(ax.tables for ax in signal[1])
    assert signal[1][0].get_xlim() == signal[1][1].get_xlim() == (-0.5, 18.5)
    assert signal[2]["within_cs_r"][0, 1] == -0.9
    assert set(signal[2]["variant_label_map"]) == set(expected)
    assert len(set(signal[2]["variant_label_map"].values())) == len(expected)
    assert signal[2]["displayed_ld"][0, 1] == pytest.approx(0.81)
    assert signal[1][2].images[0].get_clim() == (0.0, 1.0)
    np.testing.assert_array_equal(
        signal[1][1].lines[0].get_ydata(),
        np.asarray(posterior["components"][0]["cumulative_posterior"])[:19],
    )
    assert signal[2]["selected_posterior_mass"] == pytest.approx(
        sum(posterior["components"][0]["alpha"][:19])
    )
    assert prior[2]["all_prior_influence"] == posterior["prior_influence"]
    assert posterior == original
    for view in (signal, prior):
        view[0].canvas.draw()
        plt.close(view[0])


def test_compact_convergence_signed_zero_relative_and_independent_axes(posterior):
    after = copy.deepcopy(posterior)
    after["trace"] = after["trace"][:2]
    view = plots.plot_convergence(posterior, after=after, layout="compact", relative_elbo=True)
    assert len(view[1]) == 8
    assert view[2]["after_label"] == "After search (unchanged)"
    np.testing.assert_allclose(view[1][0].lines[0].get_ydata(), [0.0, 1.0, 0.9, 0.9])
    np.testing.assert_allclose(view[1][1].lines[0].get_ydata()[1:], [1.0, -0.1, 0.0])
    assert "first recorded ELBO" in view[1][0].get_ylabel()
    assert view[1][1].lines[2].get_linestyle() == "--"
    assert view[1][2].lines[1].get_linestyle() == ":"
    assert len(view[1][4].lines[0].get_xdata()) == 2
    assert all(ax.get_xlim() == view[1][0].get_xlim() for ax in view[1][:4])
    assert all(ax.get_xlim() == view[1][4].get_xlim() for ax in view[1][4:])
    assert all(float(tick).is_integer() for tick in view[1][3].get_xticks())
    assert view[2]["stages"]["Before"]["trace"] == posterior["trace"]
    plt.close(view[0])
    variance = plots.plot_convergence(posterior, after=after, layout="variance")
    assert len(variance[1]) == 4
    assert len(variance[2]["stages"]["Before"]["variance_components_shown"]) == 6
    assert len(variance[2]["stages"]["Before"]["omitted_variance_components"]) == 4
    assert all(line.get_color() == COLORS["before"] for line in variance[1][0].lines[:-1])
    assert all(line.get_color() == COLORS["after"] for line in variance[1][2].lines[:-1])
    assert "component IDs are local" in variance[0]._suptitle.get_text()
    assert "4 further components" in variance[1][0].get_title()
    plt.close(variance[0])


def test_refinement_statuses_subset_and_no_connecting_variant_lines(posterior):
    before, after = copy.deepcopy(posterior), copy.deepcopy(posterior)
    after["refinement"] = {
        "config": {"absolute_tol": 0.01},
        "history": [
            {"accepted": True, "round": 1, "objective": -5.0, "improvement": 5.0},
            {"accepted": False, "round": 1, "objective": -6.0, "improvement": 4.0},
            {"accepted": False, "round": 1, "objective": None, "improvement": None},
            {
                "accepted": False,
                "round": 1,
                "objective": -6.0,
                "improvement": 4.0,
                "stages": {"B": {"ibss_converged": False}},
            },
            {"accepted": False, "round": 2, "objective": -5.0, "improvement": 0.0},
        ],
    }
    selected = posterior["variants"][-1]["variant_id"]
    view = plots.plot_refinement(before, after, layout="compact", variant_ids=[selected])
    assert len(view[1]) == 3 and not any(ax.tables for ax in view[1])
    assert all(line.get_linestyle() == "None" for line in view[1][0].lines)
    assert view[2]["candidate_selection"] == "explicit variant IDs + all formal CS members"
    assert set(view[2]["displayed_variant_ids"]) == {selected} | set(
        posterior["credible_sets"][0]["members"]
    )
    assert view[2]["candidate_statuses"] == [
        "Accepted",
        "Valid, not selected",
        "Failed / unavailable",
        "Not converged",
        "Below acceptance threshold",
    ]
    assert [r["objective"] for r in view[2]["incumbent_path"]] == [-10.0, -5.0, -5.0]
    plt.close(view[0])


def test_sensitivity_baseline_first_missing_zero_support_and_outside_maximum():
    comparison = {
        "settings": ["alternative_with_an_extremely_long_setting_name", "baseline"],
        "baseline": "baseline",
        "mode": "union",
        "support": {"pairwise_metrics": "pairwise_intersection"},
        "variants": [
            {
                "variant_id": "candidate",
                "candidate": True,
                "pips": {"baseline": 0.0, "alternative_with_an_extremely_long_setting_name": None},
                "max_abs_delta_from_baseline": 0.0,
            },
            {
                "variant_id": "omitted_extreme",
                "candidate": False,
                "pips": {"baseline": 0.01, "alternative_with_an_extremely_long_setting_name": 0.08},
                "max_abs_delta_from_baseline": 0.07,
            },
        ],
        "fit_settings": {"baseline": {"prior_weights": [0.7, 0.3]}},
        "sensitivity": {"prior_alignment": {"policy": "reindex_existing_weights"}},
        "signal_relations": [],
        "component_relations": [{"status": "component_overlap"}],
    }
    before = copy.deepcopy(comparison)
    view = plots.plot_sensitivity(
        comparison,
        layout="compact",
        display_names={
            "alternative_with_an_extremely_long_setting_name": "Fixed scaled prior variance = 0.05"
        },
    )
    assert view[2]["settings"][0] == "baseline"
    assert view[2]["pip_matrix"][0, 0] == 0.0 and np.isnan(view[2]["pip_matrix"][0, 1])
    assert view[2]["maximum_change_outside_candidates"]["variant_id"] == "omitted_extreme"
    assert view[2]["source_comparison"]["sensitivity"] == comparison["sensitivity"]
    assert view[2]["support"]["pairwise_metrics"] == "pairwise_intersection"
    assert view[2]["all_component_relations"] == comparison["component_relations"]
    assert "NA" in [t.get_text() for t in view[1][0].texts]
    assert view[2]["baseline_changes"][0]["variant_id"] == "omitted_extreme"
    assert view[2]["baseline_changes"][0]["max_abs_delta_pip"] == pytest.approx(0.07)
    assert view[2]["baseline_changes"][0]["outside_displayed_candidates"]
    assert not any(ax.tables for ax in view[1])
    assert comparison == before
    plt.close(view[0])


def test_incomplete_ld_missing_is_not_zero_and_extreme_is_not_flag():
    diagnostics = incomplete_ld()
    for auxiliary in ("residual", "log_lr"):
        view = plots.plot_ld_diagnostics(diagnostics, layout="compact", auxiliary=auxiliary)
        assert len(view[1]) == 2
        assert np.isnan(view[2]["observed_z"][1])
        assert np.isnan(view[2]["standardized_residual"][1])
        assert np.isnan(view[2]["log_likelihood_ratio"]).all()
        assert view[2]["status_counts"]["assessed"] == 0
        assert view[2]["annotations"][0]["reason"] == "extreme residual; not a flip flag"
        assert "AND" in view[2]["flag_rule"]
        assert "controlled mixture failure" in view[1][0].get_title()
        plt.close(view[0])


def test_no_cs_trace_and_styles_preserve_source_and_global_configuration(posterior):
    posterior["credible_sets"], posterior["trace"] = [], []
    snapshot = json.dumps(posterior, sort_keys=True)
    backend = matplotlib.get_backend()
    rc = dict(plt.rcParams)
    for style in (None, False, {"axes.facecolor": "#eeeeee"}):
        for function, options in (
            (plots.plot_signal, {}),
            (plots.plot_convergence, {}),
            (plots.plot_locus, {}),
            (plots.plot_prior_influence, {}),
        ):
            view = function(posterior, layout="compact", style=style, **options)
            if function is plots.plot_convergence:
                assert view[2]["missing_trace"]
            if function is plots.plot_signal:
                assert view[2]["credible_set"] is None
            assert plt.rcParams == rc
            assert matplotlib.get_backend() == backend
            plt.close(view[0])
    assert json.dumps(posterior, sort_keys=True) == snapshot
    # Matplotlib rc_context deliberately does not restore backend; reject this
    # non-local setting before it can alter the caller's plotting environment.
    with pytest.raises(ValueError, match=r"backend is not a local style option"):
        plots.plot_locus(posterior, style={"backend": "svg"})
    assert plt.rcParams == rc
    assert matplotlib.get_backend() == backend


def test_small_candidate_axes_label_every_id_and_positive_delta_has_no_empty_negative_half():
    figure, ax = plt.subplots()
    labels = [f"v{i}" for i in range(12)]
    assert plots._variant_ticks(ax, labels, priority_indices=range(3)) == list(range(12))
    assert [tick.get_text() for tick in ax.get_xticklabels()] == labels
    plt.close(figure)
    view = plots.plot_convergence(
        {
            "convergence": {"tol": 0.01},
            "trace": [
                {"iteration": 1, "elbo": -2.0, "delta_elbo": None},
                {"iteration": 2, "elbo": -1.0, "delta_elbo": 1.0},
                {"iteration": 3, "elbo": -1.0, "delta_elbo": 0.0},
            ],
        },
        layout="compact",
    )
    assert view[1][1].get_ylim()[0] == 0.0
    np.testing.assert_array_equal(view[1][1].lines[0].get_ydata()[1:], [1.0, 0.0])
    plt.close(view[0])


def test_signal_self_correlation_display_preserves_raw_ld(posterior):
    R = np.eye(24)
    R[0, 0] = np.nextafter(1.0, 0.0)
    original = R.copy()
    figure, _, data = plots.plot_signal(posterior, 2, R=R, layout="compact", ld_measure="r2")
    try:
        assert data["within_cs_r"][0, 0] == original[0, 0]
        assert data["displayed_ld"][0, 0] == 1.0
        assert data["ld_display_text"][0][0] == "1"
        np.testing.assert_array_equal(R, original)
    finally:
        plt.close(figure)
