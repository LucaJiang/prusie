"""Public thread environment preserves exact outcomes; no model dispatch."""
import numpy as np
import pytest
from prusie import _native

@pytest.mark.parametrize('threads', ['1','2','4','8','invalid','0'])
def test_validation_environment_keeps_complete_decisions(monkeypatch, threads):
    monkeypatch.setenv('PRUSIE_NUM_THREADS', threads)
    matrix=np.eye(1025)
    _native.validate_rss_matrix(matrix,99.)
    matrix[0,-1]=np.nan
    with pytest.raises(ValueError, match='finite'):
        _native.validate_rss_matrix(matrix,99.)
