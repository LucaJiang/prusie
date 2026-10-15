"""Bounded semantic tests for plots, offline evidence and persistence."""
import copy
from html.parser import HTMLParser
from typing import ClassVar

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from prusie import plots
from prusie.report import evidence_bundle, load_analysis, save_analysis, write_report


@pytest.fixture
def summary():
    return {
        "variants": [{"variant_id": name, "pip": pip, "position": 100 + i * 10, "neg_log10_p": i}
                     for i, (name, pip) in enumerate(zip(("vα", "v2", "v3"), (.8, .4, 0.), strict=True))],
        "components": [{"component_id": 4, "ranking": ["vα", "v2", "v3"],
                        "alpha": [.7, .3, 0.], "cumulative_posterior": [.7, 1., 1.], "null_mass": 0.}],
        "credible_sets": [{"component_id": 4, "members": ["vα", "v2"], "requested_coverage": .95},
                          {"component_id": 7, "members": ["v2"], "requested_coverage": .95}],
        "proxies": [{"component_id": 4, "variant_id": "v3", "cs_variant_id": "v2", "signed_r": -.999}],
        "prior_influence": [{"component_id": 4, "variant_id": "vα", "prior_weight": .7,
                             "conditional_log_bf": 2., "alpha": .7}],
        "convergence": {"stop_reason": "iteration_limit", "tol": .01, "alpha_tol": .002, "pip_tol": .003},
        "trace": [{"iteration": 1, "phase": "post_sweep_pre_sigma2_update", "elbo": -20.,
                   "delta_elbo": None, "max_abs_delta_alpha": None, "max_abs_delta_pip": None,
                   "V": [.4], "sigma2": 1., "events": []},
                  {"iteration": 2, "phase": "post_sweep_pre_sigma2_update", "elbo": -20.001,
                   "delta_elbo": -.001, "max_abs_delta_alpha": .04, "max_abs_delta_pip": 0.,
                   "V": [.3], "sigma2": 1., "events": ["elbo_decrease"]}],
        "refinement": {"status": "no_improvement", "history": [], "stop_reason": "no_improvement"},
    }


def test_seven_views_preserve_semantics_and_inputs(summary, tmp_path):
    original = copy.deepcopy(summary)
    R = np.array([[1., .99, -.99], [.99, 1., -.999], [-.99, -.999, 1.]])
    comparison = {"settings": ["baseline", "filtered"], "variants": [
        {"variant_id": "vα", "pips": {"baseline": .8, "filtered": .7}},
        {"variant_id": "v3", "pips": {"baseline": 0., "filtered": None}}],
        "signal_relations": [{"status": "ambiguous", "components": [4, 7]}]}
    diagnostics = {"method": "kriging_rss", "regularization": .01, "variants": [
        {"variant_id": "vα", "observed_z": 2., "expected_z": 1., "standardized_residual": 1., "flag": False},
        {"variant_id": "v2", "observed_z": -3., "expected_z": 2., "standardized_residual": -8.,
         "flag": True, "log_likelihood_ratio": 7.}]}
    rc_before = plt.rcParams["axes.facecolor"]
    views = [plots.plot_locus(summary, R=R, style={"axes.facecolor": "white"}),
             plots.plot_signal(summary, 4, R=R), plots.plot_convergence(summary),
             plots.plot_refinement(summary, summary, comparison=comparison),
             plots.plot_ld_diagnostics(diagnostics), plots.plot_sensitivity(comparison),
             plots.plot_prior_influence(summary, 4)]
    assert plt.rcParams["axes.facecolor"] == rc_before
    assert [row["component_id"] for row in views[0][2]["cs_memberships"] if row["variant_id"] == "v2"] == [4, 7]
    assert views[2][1][1].get_yscale() == "symlog"
    assert views[2][1][1].lines[0].get_ydata()[1] == -.001
    assert np.isnan(views[5][2]["pip_matrix"][1, 1])
    assert views[5][2]["pip_matrix"][1, 0] == 0.
    assert views[1][2]["within_cs_r"].shape == (2, 2)
    for i, (figure, axes, data) in enumerate(views):
        figure.savefig(tmp_path / f"view_{i}.svg")
        figure.savefig(tmp_path / f"view_{i}.png")
        assert len(axes) and data
        plt.close(figure)
    assert summary == original


def test_supplied_axes_missing_trace_and_coordinate_rules(summary, tmp_path):
    figure, axes = plt.subplots(6, 1)
    output = plots.plot_convergence({"convergence": {"stop_reason": "not_assessed"}}, axes=axes)
    assert output[0] is figure and output[2]["missing_trace"]
    assert all(not ax.lines for ax in axes[4:]) or not len(axes[5].lines[0].get_xdata())
    assert "cannot be reconstructed" in axes[0].texts[0].get_text()
    plt.close(figure)
    for row in summary["variants"]:
        row["position"] = None
    figure, axes, data = plots.plot_locus(summary)
    assert not data["physical_positions"]
    assert "Variant index" in axes[-1].get_xlabel()
    assert all(not ax.get_xlabel() for ax in axes[:-1])
    plt.close(figure)
    with pytest.raises(ValueError, match=r"physical"):
        plots.plot_locus(summary, genes={})
    gtf = tmp_path / "toy.gtf"
    gtf.write_text('1\tlocal\ttranscript\t100\t200\t.\t+\t.\tgene_id "g"; transcript_id "t";\n')
    genes = plots.read_gene_track(gtf, genome_build="toy", source="local", transcript_ids=["t"])
    assert genes["records"][0]["start"] == 100
    assert genes["coordinate_convention"] == "1-based closed"


def test_lossless_json_npz_all_types_and_nan_payload(tmp_path):
    values = np.array([0x7ff8000000001234, 0x8000000000000000, 0x3ff0000000000001], dtype=np.uint64).view(np.float64)
    original = {"float_bytes": values, "ids": np.array(["00α", "vβ"], dtype=object),
                "ints": np.array([2**62, -3], dtype=">i8"), "f32": np.array([.1], dtype=np.float32),
                "scalar": np.float32(.125), "none": None, "nan": float("nan"),
                "tuple": (b"\x00x", 2**100, -0.), 7: {"complex": 2 + 3j}}
    save_analysis(original, tmp_path)
    restored = load_analysis(tmp_path)
    for key in ("float_bytes", "ints", "f32"):
        assert restored[key].dtype == original[key].dtype
        assert restored[key].tobytes() == original[key].tobytes()
    assert restored["ids"].dtype == np.dtype(object)
    np.testing.assert_array_equal(restored["ids"], original["ids"])
    assert type(restored["scalar"]) is np.float32
    assert type(restored["tuple"]) is tuple and restored["tuple"] == original["tuple"]
    assert restored[7] == original[7] and np.isnan(restored["nan"])
    with np.load(tmp_path / "arrays.npz", allow_pickle=False) as archive:
        assert all(not archive[key].dtype.hasobject for key in archive.files)
    (tmp_path / "arrays.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match=r"checksum"):
        load_analysis(tmp_path)


def test_evidence_and_offline_html(summary, tmp_path):
    summary["refinement"]["before"] = {"pip": np.array([.8]), "trace": [{"alpha": [[.8, .2]]}],
                                           "mu": np.array([2.])}
    bundle = evidence_bundle(summary)
    again = evidence_bundle(copy.deepcopy(summary))
    assert bundle == again
    assert len({f["id"] for f in bundle["facts"]}) == len(bundle["facts"])
    assert set(bundle["data"]["summary"]["refinement"]["before"]) == {"pip"}
    assert next(f["value"] for f in bundle["facts"] if f["path"] == "/summary/variants/0/pip") == .8
    figure = plots.plot_locus(summary)
    receipt = write_report(summary, tmp_path, figures={"Locus <toy>": figure},
                           provenance={"data_checksums": {"toy": "known"}}, title="<toy>")
    class Links(HTMLParser):
        references: ClassVar[list] = []
        def handle_starttag(self, tag, attrs):
            self.references.extend(v for k, v in attrs if k in ("href", "src"))
    page = (tmp_path / "index.html").read_text()
    parser = Links()
    parser.feed(page)
    assert parser.references
    assert all(link.startswith(("data:", "#")) or (not link.startswith(("/", "http:", "https:", "//")) and
               ".." not in link.split("/") and (tmp_path / link).exists()) for link in parser.references)
    assert "&lt;toy&gt;" in page and "not_assessed" in page
    assert "Methods and interpretation" in page and "Machine-readable evidence references" in page
    restored = load_analysis(tmp_path)["summary"]
    np.testing.assert_array_equal(restored["refinement"]["before"]["pip"], [.8])
    assert restored["variants"] == summary["variants"]
    assert receipt["tables"] and receipt["figures"]
    plt.close(figure[0])


def test_real_fit_all_views_and_result_roundtrip(tmp_path):
    from prusie import susie_rss
    from prusie.analysis import summarize
    from prusie.comparison import compare_fits
    from prusie.ld import diagnose_ld
    from prusie.refinement import RefinementConfig, refine_fit
    z = np.array([4., 1., -.5, .2])
    R = np.eye(4)
    fit = susie_rss(z=z, R=R, n=200, L=2, variant_ids=["toy_a", "toy_b", "toy_c", "toy_d"],
                    track_fit=True, max_iter=100)
    after = refine_fit(fit, RefinementConfig(max_rounds=1, candidate_budget=1))
    summary = summarize(fit, R=R, z=z)
    refined = summarize(after, R=R, z=z)
    comparison = compare_fits({"before": fit, "after": after}, R=R, ld_reference="before")
    ld = diagnose_ld(z, R, n=200, variant_ids=fit.variant_ids)
    figures = {"locus": plots.plot_locus(summary, R=R),
               "signal": plots.plot_signal(summary, 0, R=R),
               "convergence": plots.plot_convergence(fit),
               "refinement": plots.plot_refinement(summary, refined, comparison=comparison),
               "LD diagnostic": plots.plot_ld_diagnostics(ld),
               "sensitivity": plots.plot_sensitivity(comparison),
               "prior": plots.plot_prior_influence(summary, 0)}
    output = write_report(summary, tmp_path / "report", figures=figures, ld=ld,
                          comparison=comparison, provenance={"parameters": fit.params,
                          "initial_state": "cold", "data_checksums": {}, "random_seed": None})
    assert len(output["figures"]) == 14
    assert figures["convergence"][2]["trace"] == fit.trace
    save_analysis(fit, tmp_path / "fit")
    restored = load_analysis(tmp_path / "fit")
    for key in ("alpha", "mu", "mu2", "pip", "V", "elbo", "variant_ids"):
        assert getattr(restored, key).dtype == getattr(fit, key).dtype
        np.testing.assert_array_equal(getattr(restored, key), getattr(fit, key))
    assert restored.trace == fit.trace and restored.data_signature == fit.data_signature
    assert restored._prepared is None  # Prepared matrices are not public posterior evidence.
    for value in figures.values():
        plt.close(value[0])


def test_regional_labels_thin_without_dropping_observations():
    ids = [f"toy_{i}" for i in range(100)]
    summary = {"variants": [{"variant_id": v, "pip": .01} for v in ids],
               "components": [{"component_id": 0, "ranking": ids, "alpha": [.01]*100,
                               "cumulative_posterior": np.arange(1,101)/100}],
               "prior_influence": [{"component_id": 0, "variant_id": v,
                                    "prior_weight": .01, "conditional_log_bf": 0., "alpha": .01} for v in ids]}
    refinement = plots.plot_refinement(summary, summary)
    signal = plots.plot_signal(summary, 0)
    prior = plots.plot_prior_influence(summary, 0)
    assert len(refinement[1][0].lines[0].get_xdata()) == 100
    assert len(signal[1][0].patches) == len(prior[1][0].patches) == 100
    assert len(refinement[2]["variant_ids"]) == 100
    for view in (refinement, signal, prior):
        assert len(view[1][0].get_xticklabels()) <= 12
        plt.close(view[0])
