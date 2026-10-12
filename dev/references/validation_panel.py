"""Outcome-independent software cases; uses only bundled synthetic fixtures."""
from pathlib import Path
import numpy as np


def cases(repository):
    repo = Path(repository)
    out = []
    def add(name, kind, args, numpy=True, expected_error=False):
        out.append(dict(name=name, kind=kind, args=args,
                        numpy_supported=numpy, expected_error=expected_error))
    p = 6
    corr = .3 ** np.abs(np.arange(p)[:, None] - np.arange(p)[None, :])
    scales = np.array([.7, 1., 1.4, .9, 1.2, .6])
    base = dict(XtX=99 * corr * scales[:, None] * scales[None, :],
                Xty=np.sqrt(99) * scales * np.array([5., 2., 0., -3., 0., 1.]),
                yty=99., n=100, L=3, max_iter=100, tol=1e-4, n_purity=10000)
    add('default_suff_stat', 'ss', base.copy())
    for standardize in (False, True):
        for estimate in (False, True):
            for method in ('optim', 'EM', 'simple', 'fixed'):
                args = dict(base, standardize=standardize,
                            estimate_residual_variance=estimate,
                            residual_variance=.9,
                            estimate_prior_variance=method != 'fixed',
                            estimate_prior_method='optim' if method == 'fixed' else method)
                add(f'ss_std{int(standardize)}_sigma{int(estimate)}_{method}', 'ss', args)
                if method in ('optim', 'fixed'):
                    add(f'weighted_std{int(standardize)}_sigma{int(estimate)}_{method}',
                        'ss', dict(args, prior_weights=np.arange(1, p+1)))
    with np.load(repo/'src/prusie/data/teaching/inputs.npz') as data:
        # The installed 500-SNP synthetic Gaussian teaching data; no new generation.
        toy = dict(z=data['z'], R=data['R'], n=1000, L=5, max_iter=100,
                   tol=1e-3, n_purity=10000)
    add('teaching500_default_rss', 'rss', toy)
    add('teaching500_estimated_sigma', 'rss', dict(toy, estimate_residual_variance=True))
    with np.load(repo/'tests/data/cycle_seed312.npz') as data:
        seed = dict(z=data['z'], R=data['R'], n=5000, L=5,
                    check_prior=False, max_iter=100, n_purity=10000)
    add('seed312_cycle_regression', 'rss', seed)
    add('seed312_max_iter30', 'rss', dict(seed, max_iter=30))
    add('initial_finite_objective', 'rss', dict(z=np.array([2.,0.]), R=np.eye(2),
        n=100, L=1, max_iter=1, n_purity=10000))
    for standardize in (False, True):
        for estimate in (False, True):
            for method in ('optim', 'EM', 'simple', 'fixed'):
                args = dict(XtX=np.diag([99.,99.,0.]), Xty=np.array([30.,2.,0.]),
                    yty=99., n=100, L=2, standardize=standardize,
                    estimate_residual_variance=estimate, estimate_prior_variance=method!='fixed',
                    estimate_prior_method='optim' if method=='fixed' else method,
                    check_prior=True, max_iter=100, n_purity=10000)
                add(f'zero_info_std{int(standardize)}_sigma{int(estimate)}_{method}',
                    'ss', args, numpy=False)
            for method in ('fixed','EM'):
                args = dict(XtX=9*np.eye(2),Xty=np.array([3.,0.]),yty=100000.,n=10,L=1,
                    scaled_prior_variance=.0045,residual_variance=1.,
                    estimate_prior_variance=method!='fixed',
                    estimate_prior_method='optim' if method=='fixed' else method,
                    estimate_residual_variance=estimate, standardize=standardize,
                    check_prior=True,max_iter=5,n_purity=10000)
                add(f'guard_std{int(standardize)}_sigma{int(estimate)}_{method}',
                    'ss',args,numpy=False,expected_error=estimate)
    # Zero V admits a direct analytic null posterior and is supported by the reference.
    add('zero_V_analytic','ss',dict(base, scaled_prior_variance=0.,
        estimate_prior_variance=False, estimate_residual_variance=False))
    # A zero-weight variant tests the native/R semantics separately from NumPy.
    add('zero_weight_boundary','ss',dict(base,prior_weights=np.array([0.,1.,2.,3.,4.,5.])),numpy=False)
    return out
