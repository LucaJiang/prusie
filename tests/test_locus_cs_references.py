"""CS-specific references use component posterior support and preserve LD values."""

import copy
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from prusie.report import write_report
from prusie.viz import plot_locus


def posterior():
    return {
        "variants": [
            {"variant_id": v, "pip": p, "position": 1_000_000 + 100 * i}
            for i, (v, p) in enumerate(
                zip(["a", "b", "c", "d"], [0.99, 0.2, 0.8, 0.1], strict=True)
            )
        ],
        "credible_sets": [
            {"component_id": 7, "members": ["c", "d"]},
            {"component_id": 2, "members": ["a", "b"]},
        ],
        "components": [
            {"component_id": 7, "ranking": ["d", "c", "a", "b"], "alpha": [0.6, 0.3, 0.1, 0]},
            {"component_id": 2, "ranking": ["b", "a", "c", "d"], "alpha": [0.7, 0.2, 0.1, 0]},
        ],
    }


def test_each_cs_reference_uses_own_alpha_and_exports_complete_ordered_series(tmp_path):
    summary = posterior()
    before = copy.deepcopy(summary)
    R = np.array(
        [[1, -0.3, 0.2, 0.1], [-0.3, 1, -0.4, 0.2], [0.2, -0.4, 1, 0.6], [0.1, 0.2, 0.6, 1]]
    )
    original = R.copy()
    figure, axes, data = plot_locus(summary, R=R, layout="compact")
    try:
        assert len(axes) == 5 and data["ld_status"] == "per_cs"
        assert [r["component_id"] for r in data["ld_references"]] == [2, 7]
        assert [r["top_variant_id"] for r in data["ld_references"]] == ["b", "d"]
        for ax, series in zip(axes[3:], data["ld_references"], strict=True):
            index = series["top_index"]
            np.testing.assert_array_equal(series["signed_r"], R[index])
            np.testing.assert_array_equal(series["r_squared"], R[index] ** 2)
            np.testing.assert_allclose(ax.collections[0].get_offsets()[:, 1], R[index] ** 2)
            assert series["top_variant_id"] in ax.get_title()
            assert ax.get_ylim() == (-0.03, 1.03)
        assert data["ld_reference"] == "b"  # Legacy alias identifies first displayed series.
        receipt = write_report(summary, tmp_path, figures={"locus overview": (figure, axes, data)})
        exported = json.loads((tmp_path / "plot_data.json").read_text())["locus overview"]
        assert len(exported["ld_references"]) == 2
        assert exported["variant_ids"] == ["a", "b", "c", "d"]
        page = (tmp_path / "index.html").read_text()
        assert "zero-based fit row" in page and "highest-alpha" in page
        assert receipt["html"]
    finally:
        plt.close(figure)
    assert summary == before
    np.testing.assert_array_equal(R, original)


@pytest.mark.parametrize("mode", ["zero", "one", "missing", "explicit", "ties", "same_top"])
def test_ld_reference_fallbacks_and_exact_ties(mode):
    summary = posterior()
    R = np.eye(4)
    kwargs = {}
    if mode == "zero":
        summary["credible_sets"] = []
    elif mode == "one":
        summary["credible_sets"] = summary["credible_sets"][:1]
    elif mode == "missing":
        R = None
    elif mode == "explicit":
        kwargs["reference_variant"] = "c"
    elif mode == "ties":
        summary["components"][1]["alpha"] = [0.5, 0.5, 0, 0]
        summary["credible_sets"][1]["members"] = ["b", "a"]
    elif mode == "same_top":
        summary["credible_sets"][0]["members"] = ["a", "b"]
        summary["components"][0]["alpha"] = [0, 0, 0.1, 0.9]
    figure, axes, data = plot_locus(summary, R=R, **kwargs)
    try:
        refs = data["ld_references"]
        if mode == "zero":
            assert len(axes) == 4 and refs[0]["top_variant_id"] == "a"
            assert data["ld_status"] == "global_pip_fallback"
            assert "no retained CS" in axes[3].get_title()
        elif mode == "one":
            assert len(axes) == 4 and refs[0]["top_variant_id"] == "d"
        elif mode == "missing":
            assert len(axes) == 4 and refs == [] and data["r_squared"] is None
            assert data["ld_status"] == "not_supplied"
        elif mode == "explicit":
            assert len(axes) == 4 and refs[0]["top_variant_id"] == "c"
            assert data["ld_status"] == "explicit_reference"
        elif mode == "ties":
            assert refs[0]["top_variant_id"] == "a"
            assert refs[0]["tied_variant_ids"] == ["a", "b"]
        else:
            assert len(refs) == 2 and all(r["top_variant_id"] == "b" for r in refs)
            assert [r["component_id"] for r in refs] == [2, 7]
    finally:
        plt.close(figure)


def test_incomplete_cs_metadata_is_not_silently_replaced_by_global_pip():
    summary = posterior()
    summary["credible_sets"][0]["members"].append("absent")
    with pytest.raises(ValueError, match="member is absent"):
        plot_locus(summary, R=np.eye(4))
    summary = posterior()
    summary["components"] = []
    with pytest.raises(ValueError, match="finite component alpha"):
        plot_locus(summary, R=np.eye(4))
