"""Native option snapshots preserve private sequence and callback semantics."""
import numpy as np
import pytest

from prusie import _native, susie_rss


def options(p=3):
    return dict(l=1,prior_variance=.2,residual_variance=1.,prior_weights=[1/p]*p,
                estimate_prior_variance=False,estimate_prior_method='optim',
                estimate_residual_variance=False,check_null_threshold=0.,max_iter=100,
                tol=.001,check_prior=False,prior_tol=1e-9,null_index=None,
                matrix_global_scale=99.,matrix_inverse_scales=[1.]*p,matrix_scales=[1.]*p)


def fit(settings, p=3):
    return _native.fit(np.eye(p), np.arange(p,dtype=float),99.,100.,settings)


def exact(a, b):
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], np.ndarray):
            assert a[key].dtype == b[key].dtype and a[key].shape == b[key].shape
            assert a[key].tobytes() == b[key].tobytes(), key
        elif isinstance(a[key], float):
            assert np.float64(a[key]).tobytes() == np.float64(b[key]).tobytes(), key
        else:
            assert a[key] == b[key], key


@pytest.mark.parametrize('field',['prior_weights','matrix_inverse_scales','matrix_scales'])
@pytest.mark.parametrize('kind',['C','readonly','strided','float32','subclass','unaligned'])
def test_vector_private_sequence_compatibility(field,kind):
    settings = options()
    values = np.asarray([.25,.5,.75] if field == 'prior_weights' else [1.,1.,1.])
    if kind == 'readonly': values.flags.writeable=False
    if kind == 'strided': values=np.repeat(values,2)[::2]
    if kind == 'float32': values=values.astype(np.float32)
    if kind == 'unaligned':
        raw=np.zeros(values.nbytes+1,dtype=np.uint8)
        view=np.ndarray(values.shape,dtype=float,buffer=raw,offset=1);view[:]=values;values=view
    if kind == 'subclass':
        class Reversed(np.ndarray):
            def __iter__(self): return iter(np.asarray(self)[::-1])
        values=values.view(Reversed)
    expected=fit(dict(settings,**{field:list(values)}))
    exact(fit(dict(settings,**{field:values})),expected)


@pytest.mark.parametrize('stride',[9,-9,np.iinfo(np.intp).min])
@pytest.mark.parametrize('field',['prior_weights','matrix_inverse_scales','matrix_scales'])
def test_singleton_unusual_stride_uses_sequence_fallback(stride,field):
    values=np.ndarray((1,),dtype=float,buffer=np.zeros(8,dtype=np.uint8),strides=(stride,))
    values[0]=1.
    exact(fit(dict(options(1),**{field:values}),1),fit(dict(options(1),**{field:[1.]}),1))


@pytest.mark.parametrize('change',['values','dtype','rank'])
def test_later_callback_cannot_change_prior_snapshot(change):
    weights=np.array([.25,.25,.5])
    expected=fit(dict(options(),prior_weights=weights.tolist()))
    class Later:
        def __float__(self):
            if change == 'values': weights[:]=[.8,.1,.1]
            elif change == 'dtype': weights.dtype=np.uint64
            else: weights.shape=(1,3)
            return .001
    exact(fit(dict(options(),prior_weights=weights,tol=Later())),expected)


def test_public_params_keep_normalized_python_float_list():
    weights=np.array([1.,0.,2.])
    fit=susie_rss([2.,0.,1.],np.eye(3),100,L=1,prior_weights=weights,null_weight=.2)
    assert type(fit.params['prior_weights']) is list
    assert all(type(x) is float for x in fit.params['prior_weights'])
    np.testing.assert_array_equal(weights,[1.,0.,2.])
