"""Standalone CS covariance scales retain reciprocal rounding across dtypes."""
import numpy as np
import pytest
from prusie.posterior import credible_sets

@pytest.mark.parametrize('dtype', [np.float64, np.float32, np.float16, np.dtype('>f8')])
@pytest.mark.parametrize('zero', [False, True])
def test_covariance_scale_dtype_preserves_complete_purity(dtype, zero):
    correlation=np.array([[1.,.4,.8],[.4,1.,.6],[.8,.6,1.]])
    scales=np.array([1.1,1.3,0. if zero else 1.7],dtype=dtype)
    before=scales.tobytes()
    inverse=np.divide(1.,scales,out=np.zeros_like(scales),where=scales!=0)
    expected=correlation*inverse[:,None]*inverse[None,:]
    values=np.abs(expected[np.triu_indices(3,1)])
    result=credible_sets(np.array([[.4,.3,.3]]),np.ones(1),correlation,
                         covariance_scales=scales,min_abs_corr=0.)
    np.testing.assert_array_equal(result['cs_index'],[0])
    np.testing.assert_array_equal(result['cs'][0],[0,1,2])
    np.testing.assert_array_equal(result['coverage'],[1.])
    for name,value in [('min_abs_corr',values.min()),('mean_abs_corr',values.mean()),
                       ('median_abs_corr',np.median(values))]:
        np.testing.assert_array_equal(result['purity'][name],[value])
    assert scales.tobytes()==before
