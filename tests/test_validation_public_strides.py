"""Reject malformed logical correlation matrices despite valid physical storage."""
import numpy as np
import pytest
from prusie import susie_rss


@pytest.mark.parametrize('size', [4, 5])
@pytest.mark.parametrize('layout', ['C-row', 'C-column', 'F-row', 'F-column'])
def test_public_negative_stride_keeps_logical_correlation_contract(size, layout):
    storage = np.full((size, size), .25)
    np.fill_diagonal(storage, 1.)
    if layout.startswith('F'):
        storage = np.asfortranarray(storage)
    matrix = storage[::-1, :] if layout.endswith('row') else storage[:, ::-1]
    matrix.flags.writeable = False
    before = matrix.view(np.uint64).copy()
    with pytest.raises(ValueError, match='unit diagonal'):
        susie_rss(np.arange(size, dtype=float), matrix, 100, L=1)
    np.testing.assert_array_equal(matrix.view(np.uint64), before)
