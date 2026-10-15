"""Credible-set metadata must follow returned sets and original components."""

import warnings

import numpy as np
import pytest

from prusie import susie_rss
from prusie.posterior import credible_sets


@pytest.mark.parametrize(
    "alpha,V,correlation,indices,members,mass,purity",
    [
        (
            [[0.96, 0.02, 0.02], [0.01, 0.97, 0.02]],
            [1, 1],
            np.eye(3),
            [0, 1],
            [[0], [1]],
            [0.96, 0.97],
            [1, 1],
        ),
        (
            [[0.52, 0.45, 0.03], [0.49, 0.49, 0.02], [0.01, 0.01, 0.98]],
            [1, 1, 1],
            np.ones((3, 3)),
            [0, 2],
            [[0, 1], [2]],
            [0.97, 0.98],
            [1, 1],
        ),
        ([[0.48, 0.48, 0.04], [0.01, 0.01, 0.98]], [1, 1], np.eye(3), [1], [[2]], [0.98], [1]),
        ([[0.98, 0.01, 0.01], [0.01, 0.01, 0.98]], [0, 1], np.eye(3), [1], [[2]], [0.98], [1]),
        (
            [
                [0.48, 0.48, 0.04, 0, 0],
                [0, 0, 0.51, 0.45, 0.04],
                [0, 0, 0, 0.02, 0.98],
                [0.97, 0.01, 0.01, 0.01, 0],
                [0, 0, 0.51, 0.45, 0.04],
            ],
            [1, 1, 1, 0, 1],
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0.8, 0],
                [0, 0, 0.8, 1, 0],
                [0, 0, 0, 0, 1],
            ],
            [2, 1],
            [[4], [2, 3]],
            [0.98, 0.96],
            [1, 0.8],
        ),
    ],
    ids=[
        "no_filter",
        "deduplicate",
        "purity_filter",
        "original_component",
        "filter_deduplicate_and_reorder",
    ],
)
def test_coverage_tracks_original_components(alpha, V, correlation, indices, members, mass, purity):
    alpha = np.array(alpha, dtype=float)
    before = alpha.copy()
    result = credible_sets(
        alpha, np.array(V), np.array(correlation), coverage=0.95, min_abs_corr=0.5
    )
    np.testing.assert_array_equal(result["cs_index"], indices)
    assert [x.tolist() for x in result["cs"]] == members
    np.testing.assert_allclose(result["coverage"], mass, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(result["actual_coverage"], result["coverage"])
    for values in result["purity"].values():
        np.testing.assert_allclose(values, purity, rtol=0, atol=1e-15)
    assert result["requested_coverage"] == 0.95
    np.testing.assert_array_equal(alpha, before)


@pytest.mark.parametrize(
    "coverage,min_abs_corr,V",
    [
        (0.95, 0.5, [1, 1]),
        (0.95, 0.5, [0, 0]),
        (None, 0.5, [1, 1]),
        (0.95, None, [1, 1]),
    ],
    ids=["all_impure", "no_active_components", "coverage_disabled", "purity_disabled"],
)
def test_empty_coverage(coverage, min_abs_corr, V):
    result = credible_sets(
        np.array([[0.48, 0.48, 0.04], [0.04, 0.48, 0.48]]),
        np.array(V),
        np.eye(3),
        coverage=coverage,
        min_abs_corr=min_abs_corr,
    )
    assert result["cs"] == []
    assert result["cs_index"].dtype == np.int64
    for field in ("cs_index", "coverage", "actual_coverage"):
        assert result[field].shape == (0,)
    assert all(x.shape == (0,) for x in result["purity"].values())
    assert result["requested_coverage"] == coverage


def test_public_fit_coverage_does_not_change_fitting():
    args = dict(z=[5.0, 2.0, 0.0], R=np.eye(3), n=100, L=2, estimate_prior_variance=False)
    with warnings.catch_warnings(record=True) as emitted:
        with_sets = susie_rss(**args)
        without_sets = susie_rss(**args, coverage=None)
    for field in (
        "alpha",
        "mu",
        "mu2",
        "lbf_variable",
        "lbf",
        "V",
        "sigma2",
        "pip",
        "elbo",
        "KL",
        "XtXr",
        "niter",
        "converged",
    ):
        np.testing.assert_array_equal(getattr(with_sets, field), getattr(without_sets, field))
    assert with_sets.sets["cs"]
    for k, component in enumerate(with_sets.sets["cs_index"]):
        expected = sum(float(with_sets.alpha[component, j]) for j in with_sets.sets["cs"][k])
        assert with_sets.sets["coverage"][k] == pytest.approx(expected, abs=1e-15)
    np.testing.assert_array_equal(with_sets.sets["coverage"], with_sets.sets["actual_coverage"])
    assert not any("coverage indexing" in str(w.message) for w in emitted)
