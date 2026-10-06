"""Analytical native-boundary checks for exact matrix factorization and fallbacks."""
import numpy as np
import pytest
from prusie import _native


def options():
    return dict(l=1,prior_variance=.2,residual_variance=1.,prior_weights=[.2,0.,.8],
                estimate_prior_variance=False,estimate_prior_method='optim',
                estimate_residual_variance=False,check_null_threshold=0.,
                max_iter=100,tol=.001,check_prior=False,prior_tol=1e-9,null_index=2)


@pytest.mark.parametrize('order',['C','F'])
def test_factorized_native_preserves_orientation_and_no_information(order):
    # Deliberately asymmetric to detect an unintended transpose at this thin
    # boundary; public matrix symmetry validation is tested separately.
    raw=np.array([[2.,3.,0.],[1.,7.,0.],[0.,0.,0.]],order=order)
    scales=np.array([2.,.5,1.]);inverse=1./scales
    prepared=((raw*8.)*inverse[None,:])/scales[:,None]
    response=np.array([3.,4.,0.]);base=options()
    expected=_native.fit(np.asfortranarray(prepared),response,20.,21.,base)
    actual=_native.fit(raw,response,20.,21.,dict(base,matrix_global_scale=8.,
        matrix_inverse_scales=inverse.tolist(),matrix_scales=scales.tolist()))
    assert _native.MATRIX_OPERATOR_VERSION==1
    for key in ['alpha','mu','mu2','XtXr','V','KL','elbo']:
        np.testing.assert_allclose(actual[key],expected[key],atol=1e-12,rtol=0)
    assert actual['niter']==expected['niter']
    assert actual['mu'][0,2]==0.


@pytest.mark.parametrize('change',[
    {'matrix_global_scale':-1.}, {'matrix_global_scale':float('inf')},
    {'matrix_scales':[1.]}, {'matrix_inverse_scales':[1.,float('nan'),1.]},
])
def test_invalid_factor_metadata_rejected(change):
    settings=dict(options(),matrix_global_scale=1.,matrix_scales=[1.]*3,matrix_inverse_scales=[1.]*3)
    settings.update(change)
    with pytest.raises(ValueError):
        _native.fit(np.eye(3),np.array([.1,0.,0.]),20.,21.,settings)


def test_partial_factor_metadata_rejected():
    with pytest.raises(ValueError,match='Missing native option'):
        _native.fit(np.eye(3),np.array([.1,0.,0.]),20.,21.,dict(options(),matrix_global_scale=1.))


@pytest.mark.parametrize('operator',[False,True])
@pytest.mark.parametrize('layout',['unaligned_matrix','byte_stride_matrix','unaligned_vector','byte_stride_vector'])
def test_native_rejects_incompatible_byte_layout_before_rust_view(operator,layout):
    matrix=np.eye(3);response=np.array([.1,0.,0.])
    if layout=='unaligned_matrix':
        storage=np.zeros(3*3*8+1,dtype=np.uint8)
        matrix=np.ndarray((3,3),dtype=np.float64,buffer=storage,offset=1);matrix[:]=np.eye(3)
    elif layout=='byte_stride_matrix':
        storage=np.zeros(3*3*9,dtype=np.uint8)
        matrix=np.ndarray((3,3),dtype=np.float64,buffer=storage,strides=(27,9));matrix[:]=np.eye(3)
    elif layout=='unaligned_vector':
        storage=np.zeros(3*8+1,dtype=np.uint8)
        response=np.ndarray((3,),dtype=np.float64,buffer=storage,offset=1);response[:]=[.1,0.,0.]
    else:
        storage=np.zeros(3*9,dtype=np.uint8)
        response=np.ndarray((3,),dtype=np.float64,buffer=storage,strides=(9,));response[:]=[.1,0.,0.]
    settings=options()
    if operator:settings.update(matrix_global_scale=1.,matrix_inverse_scales=[1.]*3,matrix_scales=[1.]*3)
    with pytest.raises(ValueError,match='aligned.*byte strides'):
        _native.fit(matrix,response,20.,21.,settings)



def one_options():
    return dict(options(),prior_weights=[1.],null_index=None)


@pytest.mark.parametrize('which',['matrix','vector','both'])
@pytest.mark.parametrize('step',[-9,int(np.iinfo(np.intp).min)])
def test_empty_negative_stride_rejected_before_borrow_or_view(which,step):
    matrix=np.ones((1,1));response=np.array([.1])
    if which in ('matrix','both'):
        storage_m=np.zeros(8,dtype=np.uint8)
        matrix=np.ndarray((0,0),dtype=np.float64,buffer=storage_m,strides=(step,step))
    if which in ('vector','both'):
        storage_v=np.zeros(8,dtype=np.uint8)
        response=np.ndarray((0,),dtype=np.float64,buffer=storage_v,strides=(step,))
    with pytest.raises(ValueError):
        _native.fit(matrix,response,20.,21.,one_options())


@pytest.mark.parametrize('which',['matrix','vector','both'])
def test_singleton_minimum_stride_rejected_before_borrow(which):
    step=int(np.iinfo(np.intp).min);matrix=np.ones((1,1));response=np.array([.1])
    if which in ('matrix','both'):
        storage_m=np.ones(1)
        matrix=np.ndarray((1,1),dtype=np.float64,buffer=storage_m,strides=(step,step))
    if which in ('vector','both'):
        storage_v=np.array([.1])
        response=np.ndarray((1,),dtype=np.float64,buffer=storage_v,strides=(step,))
    with pytest.raises(ValueError,match='byte strides'):
        _native.fit(matrix,response,20.,21.,one_options())


@pytest.mark.parametrize('step',[-9,9,-int(np.iinfo(np.intp).max)])
def test_aligned_singleton_odd_stride_preserves_scalar_fit(step):
    storage_m=np.ones(1);storage_v=np.array([.1])
    matrix=np.ndarray((1,1),dtype=np.float64,buffer=storage_m,strides=(step,step))
    response=np.ndarray((1,),dtype=np.float64,buffer=storage_v,strides=(step,))
    expected=_native.fit(np.ones((1,1)),np.array([.1]),20.,21.,one_options())
    actual=_native.fit(matrix,response,20.,21.,one_options())
    for key in ['alpha','mu','mu2','XtXr','V','elbo']:
        np.testing.assert_array_equal(actual[key],expected[key])


def test_multielement_negative_stride_retains_native_contiguity_restriction():
    matrix=np.eye(3)[::-1,::-1]
    with pytest.raises(ValueError,match='contiguous'):
        _native.fit(matrix,np.array([.1,0.,0.]),20.,21.,options())


@pytest.mark.parametrize('which',['matrix','vector','both'])
def test_empty_unaligned_pointer_rejected_independently_of_numpy_flags(which):
    matrix=np.empty((0,0));response=np.empty((0,))
    if which in ('matrix','both'):
        storage_m=np.zeros(2,dtype=np.uint8)
        matrix=np.ndarray((0,0),dtype=np.float64,buffer=storage_m,offset=1)
        assert matrix.ctypes.data%8==1
    if which in ('vector','both'):
        storage_v=np.zeros(2,dtype=np.uint8)
        response=np.ndarray((0,),dtype=np.float64,buffer=storage_v,offset=1)
        assert response.ctypes.data%8==1
    with pytest.raises(ValueError,match='aligned'):
        _native.fit(matrix,response,20.,21.,one_options())


@pytest.mark.parametrize('field', ['matrix_global_scale', 'matrix_inverse_scales', 'matrix_scales'])
@pytest.mark.parametrize('which', ['matrix', 'vector'])
def test_factor_conversion_precedes_final_geometry_guard(field, which):
    # Call only a binding that finishes extraction before borrowing. The
    # mutation remains in allocated storage and the final guard must reject it.
    matrix_storage = np.zeros(81, dtype=np.uint8)
    vector_storage = np.zeros(27, dtype=np.uint8)
    matrix = np.ndarray((3, 3), dtype=float, buffer=matrix_storage)
    matrix[:] = np.eye(3)
    response = np.ndarray((3,), dtype=float, buffer=vector_storage)
    response[:] = [.1, 0., 0.]
    target = matrix if which == 'matrix' else response
    called = []
    class ChangeStride:
        def __float__(self):
            target.strides = (24, 9) if which == 'matrix' else (9,)
            called.append(True)
            return 1.
    settings = dict(options(), matrix_global_scale=1.,
                    matrix_inverse_scales=[1.]*3, matrix_scales=[1.]*3)
    settings[field] = ChangeStride() if field == 'matrix_global_scale' else [ChangeStride(), 1., 1.]
    with pytest.raises(ValueError, match='byte strides'):
        _native.fit(matrix, response, 20., 21., settings)
    assert called == [True]


def test_factor_conversion_to_valid_layout_is_seen_by_native_fit():
    matrix = np.array([[1., .25, 0.], [.25, 2., 0.], [0., 0., 0.]])
    response = np.array([.1, .2, 0.])
    settings = dict(options(), matrix_global_scale=1.,
                    matrix_inverse_scales=[1.]*3, matrix_scales=[1.]*3)
    class ChangeValues:
        def __float__(self):
            matrix[0, 0] = 1.5
            response[0] = .3
            return 1.
    actual = _native.fit(matrix, response, 20., 21.,
                        dict(settings, matrix_global_scale=ChangeValues()))
    expected = _native.fit(matrix, response, 20., 21., settings)
    for key in ['alpha', 'mu', 'mu2', 'XtXr', 'V', 'elbo']:
        np.testing.assert_array_equal(actual[key], expected[key])


@pytest.mark.parametrize('field', ['prior_variance', 'matrix_global_scale',
                                  'matrix_inverse_scales', 'matrix_scales'])
@pytest.mark.parametrize('which', ['matrix', 'vector'])
@pytest.mark.parametrize('change', ['rank', 'dtype'])
def test_option_conversion_rechecks_numpy_rank_and_dtype(field, which, change):
    matrix = np.ones((1, 1))
    response = np.array([.1])
    target = matrix if which == 'matrix' else response
    called = []
    class ChangeType:
        def __float__(self):
            if change == 'rank':
                target.shape = (1,) if which == 'matrix' else (1, 1)
            else:
                target.dtype = np.int64
            called.append(True)
            return 1.
    settings = dict(one_options(), matrix_global_scale=1.,
                    matrix_inverse_scales=[1.], matrix_scales=[1.])
    settings[field] = [ChangeType()] if field in ('matrix_inverse_scales', 'matrix_scales') else ChangeType()
    with pytest.raises(ValueError, match='must remain.*float64'):
        _native.fit(matrix, response, 20., 21., settings)
    assert called == [True]
