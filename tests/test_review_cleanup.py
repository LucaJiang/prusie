"""Regression checks for the simplified public preparation and CS conditions."""

import numpy as np
import pytest

from prusie import api, posterior


@pytest.mark.parametrize("layout", ["C", "F", "strided", "readonly", "unaligned"])
def test_matrix_normalizes_once_before_scan(monkeypatch, layout):
    matrix = np.array([[1.0, -0.25], [-0.25, 1.0]])
    if layout == "F":
        matrix = np.asfortranarray(matrix)
    elif layout == "strided":
        backing = np.empty((4, 4))
        backing[::2, ::2] = matrix
        matrix = backing[::2, ::2]
    elif layout == "readonly":
        matrix.flags.writeable = False
    elif layout == "unaligned":
        backing = np.zeros(33, dtype=np.uint8)
        unaligned = np.ndarray((2, 2), dtype=np.float64, buffer=backing, offset=1)
        unaligned[:] = matrix
        matrix = unaligned
    before = matrix.copy()
    original = api.native_array
    calls = []

    def normalize(a):
        calls.append(a)
        return original(a)

    monkeypatch.setattr(api, "native_array", normalize)
    actual = api._matrix(matrix, "R", 2, correlation=True)
    assert len(calls) == 1
    assert actual.flags.c_contiguous and actual.flags.aligned
    np.testing.assert_array_equal(actual, before)
    np.testing.assert_array_equal(matrix, before)
    assert np.shares_memory(actual, matrix) == (layout in ("C", "readonly"))


def test_nan_purity_is_still_rejected_after_early_guard(monkeypatch):
    # Independently exercise the Python comparison, even if the Rust helper
    # would normally reject this NaN before returning a pair vector.
    monkeypatch.setattr(
        posterior._native, "cs_pairwise_abs", lambda *args: np.array([np.nan])
    )
    result = posterior.credible_sets(np.array([[0.5, 0.5]]), [1.0], np.eye(2))
    assert result["cs"] == []


def test_cs_aliases_deduplication_order_and_original_component_ids():
    alpha = np.array([[0.50, 0.48, 0.02], [0.50, 0.48, 0.02], [0.01, 0.01, 0.98]])
    # The inactive first membership suppresses its active duplicate. Preserve
    # the existing order of deduplication and activity checks.
    result = posterior.credible_sets(alpha, [0.0, 1.0, 1.0], np.ones((3, 3)))
    assert result["cs_index"].tolist() == [2]
    assert result["cs"][0].tolist() == [2]
    assert result["coverage"] is result["actual_coverage"]
    assert result["coverage"][0] == 0.98


@pytest.mark.parametrize("coverage,minimum", [(None, 0.5), (0.95, None)])
def test_disabled_sets_do_not_access_matrix(coverage, minimum):
    result = posterior.credible_sets(
        None, None, None, coverage=coverage, min_abs_corr=minimum
    )
    assert result["cs"] == [] and result["requested_coverage"] == coverage
