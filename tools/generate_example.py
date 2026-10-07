#!/usr/bin/env python3
"""Generate one independent Gaussian SuSiE teaching dataset (not a study)."""
from __future__ import annotations
import argparse, hashlib, io, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
# The original manifest is immutable, independently of the regenerated archive.
FROZEN_MANIFEST_SHA256='5b2cd64a249c2df3e6508f4c1fd7ae006cab2bd1a9d4e6a4554cf9db445976a5'
# Absolute bounds, rtol=0. Observed maxima with NumPy 2.2.6 / OpenBLAS
# SkylakeX, Haswell and Sandybridge: 1.60e-14, 1.12e-15, 5.56e-16, 2.09e-17.
# These bounds allow at most about five times that measured roundoff.
REGEN_ATOL={'z':8e-14,'R':5e-15,'bhat':3e-15,'shat':1e-16}


def generate():
    seed, n, p, rho = 20261007, 1000, 500, 0.85
    rng=np.random.Generator(np.random.PCG64(seed))
    X=rng.standard_normal((n,p))
    for j in range(1,p): X[:,j]=rho*X[:,j-1]+np.sqrt(1-rho*rho)*X[:,j]
    X-=X.mean(axis=0);X/=X.std(axis=0,ddof=1)
    beta=np.zeros(p);beta[[99,349]]=[0.35,-0.30]
    y=X@beta+rng.standard_normal(n);y-=y.mean()
    Q=X.T@X;g=X.T@y;yty=float(y@y)
    bhat=g/np.diag(Q)
    shat=np.sqrt((yty-g*g/np.diag(Q))/(n-2)/np.diag(Q))
    R=Q/np.sqrt(np.outer(np.diag(Q),np.diag(Q)))
    arrays=dict(z=bhat/shat,R=R,bhat=bhat,shat=shat,true_beta=beta,
                position=np.arange(1,p+1,dtype=np.int64)*1000)
    metadata=dict(name='Independent Gaussian teaching example',synthetic=True,seed=seed,
        generator='NumPy PCG64; tools/generate_example.py',numpy_version=np.__version__,n=n,p=p,
        method='Stationary Gaussian AR(1) predictors, centered and sample-standardized; y=X beta + independent N(0,1) noise; signed z=bhat/shat from marginal t statistics; LD is sample Pearson correlation',
        population='Synthetic Gaussian predictors; not human genotypes',genome_build=None,
        chromosome='synthetic',coordinate_units='arbitrary teaching base positions at 1000-unit spacing',
        LD_population_rho=rho,residual_variance_generation=1.,true_effects=[{'index_0based':99,'beta':.35},{'index_0based':349,'beta':-.30}],
        variant_ids=[f'syn{j+1:04d}' for j in range(p)],license='GPL-3.0-or-later',author='Wenxin Jiang',
        parameters=dict(L=5,max_iter=100,tol=.001,scaled_prior_variance=.2,residual_variance=1.,
            estimate_residual_variance=False,estimate_prior_variance=True,estimate_prior_method='optim',
            coverage=.95,min_abs_corr=.5,n_purity=500))
    return arrays,metadata


def validate_arrays(frozen, regenerated):
    """Compare numerical content, independently of ZIP/DEFLATE encoding."""
    if set(frozen)!=set(regenerated):
        raise ValueError('Regenerated array names changed')
    errors={}
    for name, expected in frozen.items():
        actual=regenerated[name]
        if actual.shape!=expected.shape or actual.dtype!=expected.dtype:
            raise ValueError(f'Regenerated shape/dtype changed: {name}')
        if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
            raise ValueError(f'Non-finite teaching data: {name}')
        error=float(np.max(np.abs(actual-expected)))
        if name in REGEN_ATOL:
            if error>REGEN_ATOL[name]:
                raise ValueError(f'Regenerated {name} differs by {error:.17g}; bound {REGEN_ATOL[name]:.1g}')
        elif not np.array_equal(actual,expected):
            raise ValueError(f'Regenerated exact array changed: {name}')
        errors[name]=error
    return errors


def check_frozen(directory, arrays, metadata):
    """Check original bytes first, then scientific regeneration and metadata."""
    manifest=(directory/'checksums.json').read_bytes()
    if hashlib.sha256(manifest).hexdigest()!=FROZEN_MANIFEST_SHA256:
        raise ValueError('Frozen checksum manifest changed')
    hashes=json.loads(manifest)
    for name,expected in hashes.items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Frozen teaching checksum mismatch: {name}')
    # numpy_version identifies the original generator, not the current host.
    # Require the pinned generation version rather than relabeling provenance.
    expected_metadata=json.loads((directory/'metadata.json').read_text())
    if metadata!=expected_metadata:
        raise ValueError('Regenerated metadata changed; use the recorded NumPy version and generation parameters')
    with np.load(directory/'inputs.npz',allow_pickle=False) as frozen:
        return validate_arrays(dict(frozen),arrays)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=ROOT/'src/prusie/data/teaching');p.add_argument('--check',action='store_true');a=p.parse_args()
    arrays,metadata=generate()
    if a.check:
        errors=check_frozen(a.output_dir,arrays,metadata)
        print('Checked frozen SHA256 and numerical regeneration:',json.dumps(errors,sort_keys=True))
        return
    buffer=io.BytesIO();np.savez_compressed(buffer,**arrays)
    payload={'inputs.npz':buffer.getvalue(),'metadata.json':(json.dumps(metadata,indent=2)+'\n').encode(),
        'LICENSE.txt':b'Copyright (c) 2026 Wenxin Jiang\nThis independently generated teaching data and its generator are licensed under GPL-3.0-or-later. See the package LICENSE for the full terms.\n'}
    payload['checksums.json']=(json.dumps({k:hashlib.sha256(v).hexdigest() for k,v in payload.items()},indent=2)+'\n').encode()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for name,value in payload.items():(a.output_dir/name).write_bytes(value)
    print('Generated',len(arrays['z']),'synthetic variants')

if __name__=='__main__':main()
