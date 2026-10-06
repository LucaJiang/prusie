"""Numeric conversion must not leave a stale typed NumPy wrapper."""
import numpy as np
import pytest
from prusie import _native


class MutatingFloat:
    def __init__(self, target, change, value=1.):
        self.target, self.change, self.value = target, change, value
        self.called = False
    def __float__(self):
        self.called = True
        if self.change == 'dtype':
            self.target.dtype = np.uint64
        else:
            self.target.shape = ((self.target.size,) if self.target.ndim == 2
                                 else (1, self.target.size))
        return self.value


@pytest.mark.parametrize('change', ['dtype', 'rank'])
@pytest.mark.parametrize('boundary', ['matrix','rss','prepare_matrix','prepare_inverse',
    'prepare_scales','cs_matrix','cs_members','cs_inverse','fit_matrix','fit_vector'])
def test_callback_rechecks_every_typed_array(change, boundary):
    matrix=np.eye(3);inverse=np.ones(3);scales=np.ones(3);members=np.arange(3,dtype=np.int64)
    targets={'matrix':matrix,'rss':matrix,'prepare_matrix':matrix,'prepare_inverse':inverse,
             'prepare_scales':scales,'cs_matrix':matrix,'cs_members':members,'cs_inverse':inverse,
             'fit_matrix':matrix,'fit_vector':inverse}
    value=MutatingFloat(targets[boundary],change)
    options=dict(l=1,prior_variance=value,residual_variance=1.,prior_weights=[1/3]*3,
                 estimate_prior_variance=False,estimate_prior_method='optim',
                 estimate_residual_variance=False,check_null_threshold=0.,max_iter=100,
                 tol=.001,check_prior=False,prior_tol=1e-9,null_index=None)
    calls={
        'matrix':lambda:_native.validate_matrix(matrix,'XtX',False,value),
        'rss':lambda:_native.validate_rss_matrix(matrix,value),
        'prepare_matrix':lambda:_native.prepare_crossproduct(matrix,inverse,scales,value),
        'prepare_inverse':lambda:_native.prepare_crossproduct(matrix,inverse,scales,value),
        'prepare_scales':lambda:_native.prepare_crossproduct(matrix,inverse,scales,value),
        'cs_matrix':lambda:_native.cs_pairwise_abs(matrix,members,inverse,value),
        'cs_members':lambda:_native.cs_pairwise_abs(matrix,members,inverse,value),
        'cs_inverse':lambda:_native.cs_pairwise_abs(matrix,members,inverse,value),
        'fit_matrix':lambda:_native.fit(matrix,inverse,99.,100.,options),
        'fit_vector':lambda:_native.fit(matrix,inverse,99.,100.,options),
    }
    with pytest.raises(ValueError,match='dtype or rank changed|float64 array after option conversion'):
        calls[boundary]()
    assert value.called
