"""Analyze the complete, frozen 500-SNP synthetic teaching dataset D3."""
import json
from pathlib import Path

import numpy as np
import prusie

data = Path(__file__).resolve().parent / 'data'
metadata = json.loads((data / 'metadata.json').read_text())['D3']
with np.load(data / 'inputs.npz', allow_pickle=False) as arrays:
    z = arrays['D3_beta'] / np.sqrt(arrays['D3_varbeta'])
    R = arrays['D3_LD']

fit = prusie.susie_rss(
    z=z, R=R, n=metadata['N'], variant_ids=metadata['snp'],
    L=10, max_iter=100, tol=0.001, estimate_residual_variance=False,
    coverage=0.95, min_abs_corr=0.5, n_purity=500,
)
lead = int(np.argmax(fit.pip))
print(f"{fit.variant_ids[lead]}: PIP={fit.pip[lead]:.3f}")
print('Converged:', fit.converged, 'Iterations:', fit.niter)
for k, component in enumerate(fit.sets['cs_index']):
    members = fit.variant_ids[fit.sets['cs'][k]].tolist()
    print('Component:', int(component), 'SNPs:', members,
          'Posterior mass:', round(float(fit.sets['coverage'][k]), 3))
