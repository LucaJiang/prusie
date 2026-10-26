"""Positional labels retain their exact values, Unicode width and metadata copy."""

import numpy as np
import pytest

from prusie import susie_rss
from prusie._validation import _metadata


@pytest.mark.parametrize(
    "m,width", [(1, 1), (9, 1), (10, 1), (99, 2), (100, 2), (999, 3), (1000, 3)]
)
def test_positional_id_boundaries(m, width):
    column = np.arange(m)
    ids, metadata = _metadata(None, {"position": column}, m)
    assert ids.tolist() == list(map(str, range(m)))
    assert ids.dtype == np.dtype(f"<U{width}")
    assert not np.shares_memory(column, metadata["position"])
    column[:] = -1
    np.testing.assert_array_equal(metadata["position"], np.arange(m))


def test_empty_and_explicit_ids_keep_validation():
    ids, _ = _metadata(None, None, 0)
    assert ids.shape == (0,) and ids.dtype == np.dtype("<U1")
    with pytest.raises(ValueError, match=r"nonempty"):
        susie_rss([], np.empty((0, 0)), 100)
    for ids in (["a", "a"], ["a", ""], ["a"]):
        with pytest.raises(ValueError, match=r"distinct nonempty"):
            _metadata(ids, None, 2)


def test_null_column_is_not_a_returned_positional_id():
    fit = susie_rss([2.0, 0.0], np.eye(2), 100, L=1, null_weight=0.2)
    assert fit.variant_ids.tolist() == ["0", "1"]
    assert fit.null_index == 2 and fit.alpha.shape[1] == 3


@pytest.mark.parametrize(
    "column",
    [
        np.array([{"label": "a"}, {"label": "b"}], dtype=object),
        np.array([(1.0, "a"), (2.0, "b")], dtype=[("value", "<f8"), ("label", "<U1")]),
    ],
)
def test_generated_ids_preserve_metadata_dtype_and_array_copy(column):
    expected = column.copy()
    fit = susie_rss([2.0, 0.0], np.eye(2), 100, L=1, variant_metadata={"annotation": column})
    actual = fit.variant_metadata["annotation"]
    assert type(actual) is np.ndarray
    assert actual.dtype == column.dtype
    assert actual.dtype.descr == column.dtype.descr
    assert not np.shares_memory(actual, column)
    np.testing.assert_array_equal(actual, expected)
    if column.dtype.hasobject:
        assert actual[0] is column[0]  # Preserve the established shallow copy.
    column[0] = column[1]
    np.testing.assert_array_equal(actual, expected)
