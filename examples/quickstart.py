"""Hand-written analytical API example; no biological/simulation claim."""
import numpy as np
from prusie import susie_rss
fit = susie_rss(z=[4., .5, -.2], R=[[1., .2, 0.], [.2, 1., .1], [0., .1, 1.]],
    n=191, L=2, max_iter=100, tol=.001, estimate_residual_variance=False,
    variant_ids=['a','b','c'])
assert np.all(np.isfinite(fit.pip)) and np.all((fit.pip >= 0) & (fit.pip <= 1))
assert fit.variant_ids.tolist() == ['a','b','c']
print('PIP:', fit.pip, 'converged:', fit.converged, 'CS components:', fit.sets['cs_index'])
