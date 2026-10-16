"""Heatmap rendering semantics on bounded saved-style evidence; no fits required."""

import copy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import Normalize

from prusie import plots
from prusie._style import annotation_color, format_ld_cell, format_pip_cell


def comparison(values, settings=("baseline", "alternative")):
    return {
        "settings": list(settings),
        "baseline": settings[0],
        "mode": "union",
        "support": {"pairwise_metrics": "pairwise_intersection"},
        "variants": [
            {
                "variant_id": f"v{i}",
                "pips": dict(zip(settings, row, strict=True)),
                "candidate": True,
            }
            for i, row in enumerate(values)
        ],
        "signal_relations": [],
    }


def signal(count=2):
    ids = [f"v{i}" for i in range(count)]
    alpha = np.ones(count) / count
    return {
        "variants": [
            {"variant_id": value, "pip": float(a)} for value, a in zip(ids, alpha, strict=True)
        ],
        "credible_sets": [{"component_id": 4, "members": ids, "requested_coverage": 0.95}],
        "components": [
            {
                "component_id": 4,
                "ranking": ids,
                "alpha": alpha.tolist(),
                "cumulative_posterior": np.cumsum(alpha).tolist(),
            }
        ],
    }


def test_probability_text_and_mapped_contrast_keep_endpoint_semantics():
    values = [0.0, 1e-12, 0.009, 0.01, 0.99, 0.999999, 1.0, None]
    expected = ["0", "<0.01", "<0.01", "0.01", "0.99", ">0.99", "1", "NA"]
    assert list(map(format_pip_cell, values)) == expected
    source = comparison([[v, v] for v in values])
    before = copy.deepcopy(source)
    for cmap in ("Blues", "Blues_r", "Reds", "binary"):
        figure, axes, data = plots.plot_sensitivity(source, layout="compact", cmap=cmap)
        image = axes[0].images[0]
        assert type(image.norm) is Normalize and image.get_clim() == (0.0, 1.0)
        assert image.get_interpolation() == "none"
        assert data["display_policy"]["scale"] == "linear"
        for i, variant in enumerate(data["variant_ids"]):
            original_index = int(variant[1:])
            assert data["display_text"][i] == [expected[original_index]] * 2
            raw = data["pip_matrix"][i, 0]
            if values[original_index] is None:
                assert np.isnan(raw)
                rgba = image.cmap.get_bad()
            else:
                assert raw == values[original_index]
                rgba = image.cmap(image.norm(raw))
            assert data["annotation_colors"][i][0] == annotation_color(rgba)
        assert not np.array_equal(image.cmap(image.norm(0.0)), image.cmap.get_bad())
        # Light colors get dark text, and reversed maps reverse that choice.
        zero = data["variant_ids"].index("v0")
        one = data["variant_ids"].index("v6")
        if cmap == "Blues":
            assert data["annotation_colors"][zero][0] == "#000000"
            assert data["annotation_colors"][one][0] == "#ffffff"
        if cmap == "Blues_r":
            assert data["annotation_colors"][zero][0] == "#ffffff"
            assert data["annotation_colors"][one][0] == "#000000"
        plt.close(figure)
    assert source == before


@pytest.mark.parametrize(
    "rows, columns, all_missing", [(1, 1, False), (1, 4, True), (15, 3, False), (24, 2, False)]
)
def test_adaptive_heatmap_keeps_every_candidate_and_handles_degenerate_shapes(
    rows, columns, all_missing
):
    settings = ["baseline"] + [f"setting{i}" for i in range(columns - 1)]
    source = comparison(
        [[None if all_missing else i / (rows + 1)] * columns for i in range(rows)], settings
    )
    view = plots.plot_sensitivity(source, layout="compact")
    assert view[2]["pip_matrix"].shape == (rows, columns)
    assert len(view[2]["variant_ids"]) == rows
    assert len(view[1][0].get_yticklabels()) == rows
    assert set(tick.get_text() for tick in view[1][0].get_yticklabels()) == set(
        view[2]["variant_ids"]
    )
    assert view[2]["cell_annotations"] is (rows <= 18 and columns <= 7 and rows * columns <= 100)
    assert view[0].get_figwidth() < 7.0 if columns <= 4 else True
    if all_missing:
        assert np.ma.getmaskarray(view[1][0].images[0].get_array()).all()
        assert {text.get_text() for text in view[1][0].texts} == {"NA"}
        assert "NA" in view[1][1].get_legend().texts[0].get_text()
    view[0].canvas.draw()
    plt.close(view[0])


def test_long_names_all_rows_and_supplied_axes_keep_global_style_and_raw_data():
    names = ["baseline", "long_alternative_name_which_requires_line_wrapping"]
    source = comparison([[0.1, None], [0.3, 0.4]], names)
    source["variants"][0]["variant_id"] = (
        "a_very_long_variant_identifier_that_must_remain_traceable"
    )
    before = copy.deepcopy(source)
    rc, backend = dict(plt.rcParams), matplotlib.get_backend()
    for style in (False, {"font.size": 11}):
        figure, axes = plt.subplots(2, 1, figsize=(6, 5))
        size = figure.get_size_inches().copy()
        view = plots.plot_sensitivity(
            source,
            layout="compact",
            axes=axes,
            style=style,
            display_names={names[1]: "Scaled prior variance\n0.05 (fixed)"},
        )
        assert view[0] is figure
        np.testing.assert_array_equal(figure.get_size_inches(), size)
        assert set(view[2]["variant_label_map"]) == {r["variant_id"] for r in source["variants"]}
        assert "0.05 (fixed)" in view[2]["display_policy"]["rendered_setting_labels"][names[1]]
        assert view[2]["source_comparison"] == before
        assert plt.rcParams == rc and matplotlib.get_backend() == backend
        plt.close(figure)
    assert source == before


def test_small_ld_is_square_uses_separate_palette_and_four_decimal_annotations():
    summary = signal()
    R = np.array([[1.0, -0.99903], [-0.99903, 1.0]])
    before = copy.deepcopy(summary)
    for measure, cmap, limits in [("r", "RdBu_r", (-1.0, 1.0)), ("r2", "Reds", (0.0, 1.0))]:
        view = plots.plot_signal(summary, R=R, layout="compact", ld_measure=measure)
        image = view[1][2].images[0]
        assert image.get_clim() == limits and type(image.norm) is Normalize
        assert image.cmap.name == cmap
        np.testing.assert_array_equal(view[2]["within_cs_r"], R)
        np.testing.assert_array_equal(view[2]["displayed_ld"], R if measure == "r" else R**2)
        assert view[2]["ld_display_text"][0][0] == "1"
        assert view[2]["ld_display_text"][0][1] == format_ld_cell(
            R[0, 1] if measure == "r" else R[0, 1] ** 2
        )
        assert view[2]["ld_display_text"][0][1] not in ("1", "-1")
        view[0].canvas.draw()
        bounds = view[1][2].get_window_extent().transformed(view[0].dpi_scale_trans.inverted())
        assert bounds.width == pytest.approx(bounds.height, abs=1e-10)
        assert 1.2 <= bounds.width <= 1.8
        assert [t.get_text() for t in view[1][2].get_xticklabels()] == ["v0", "v1"]
        assert [t.get_text() for t in view[1][2].get_yticklabels()] == ["v0", "v1"]
        plt.close(view[0])
    assert summary == before
    assert format_ld_cell(0.99999999) == ">0.9999"
    assert format_ld_cell(-0.99999999) == "<-0.9999"


def test_singleton_large_cs_and_custom_ld_axes_keep_members_and_matrix():
    for n in (1, 20):
        summary = signal(n)
        view = plots.plot_signal(
            summary, R=np.eye(n), layout="compact", max_variants=1, ld_measure="r2"
        )
        assert len(view[2]["displayed_variant_ids"]) == n
        assert view[2]["within_cs_r"].shape == (n, n)
        assert len(view[2]["ld_display_policy"]["variant_ids"]) == n
        view[0].canvas.draw()
        bounds = view[1][2].get_window_extent().transformed(view[0].dpi_scale_trans.inverted())
        assert bounds.width == pytest.approx(bounds.height)
        if n == 1:
            assert "Singleton CS" in view[1][2].get_title()
            assert bounds.width < 1.2
        plt.close(view[0])
    fig, axes = plt.subplots(3, 1, figsize=(6, 7))
    view = plots.plot_signal(signal(), R=np.eye(2), layout="compact", axes=axes, cmap="Greys")
    assert view[0] is fig
    assert axes[2].images[0].cmap.name == "Greys"
    assert axes[2].images[0].get_clim() == (-1.0, 1.0)
    fig.canvas.draw()
    box = axes[2].get_window_extent()
    assert box.width == pytest.approx(box.height)
    np.testing.assert_array_equal(fig.get_size_inches(), [6.0, 7.0])
    plt.close(fig)
