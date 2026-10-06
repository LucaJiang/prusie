"""Deterministic analytical cases for new layout, full validation and ownership."""
import numpy as np
import pytest
from prusie import susie_rss, susie_suff_stat

@pytest.mark.parametrize('mode',['known_n','missing_n','effects','null','EM','fixed','residual'])
def test_layout_and_no_mutation(mode):
    R=np.array([[1.,.2,-.1],[.2,1.,.3],[-.1,.3,1.]])
    z=np.array([3.,-1.,.2]);original=R.copy();original_z=z.copy()
    args=dict(z=z,R=R,n=100,L=2,check_prior=False)
    if mode=='missing_n':args['n']=None
    if mode=='effects':args.pop('z');args.update(bhat=z*.1,shat=.1,var_y=1.3)
    if mode=='null':args['null_weight']=.2
    if mode=='EM':args['estimate_prior_method']='EM'
    if mode=='fixed':args['estimate_prior_variance']=False
    if mode=='residual':args['estimate_residual_variance']=True
    a=susie_rss(**args)
    args['R']=np.asfortranarray(R)
    b=susie_rss(**args)
    np.testing.assert_array_equal(a.pip,b.pip)
    np.testing.assert_array_equal(a.alpha,b.alpha)
    np.testing.assert_array_equal(R,original)
    np.testing.assert_array_equal(z,original_z)
    np.testing.assert_array_equal(args['R'],original)

@pytest.mark.parametrize('bad',['nonfinite','asymmetric','negative_diagonal'])
def test_all_validation_blocks(bad):
    R=np.eye(130)
    if bad=='nonfinite':R[-1,-1]=np.nan
    elif bad=='asymmetric':R[-1,0]=.1
    else:R[-1,-1]=-1
    with pytest.raises(ValueError):susie_suff_stat(R,np.ones(130),129.,130)

def test_strided_input_not_mutated():
    backing=np.eye(6);R=backing[::2,::2];z=np.array([2.,1.,0.])
    f=susie_rss(z,R,100,L=1)
    g=susie_rss(z,R.copy(),100,L=1)
    np.testing.assert_array_equal(f.pip,g.pip)
    np.testing.assert_array_equal(backing,np.eye(6))
