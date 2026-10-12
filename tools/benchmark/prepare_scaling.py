#!/usr/bin/env python3
"""Freeze deterministic dense AR(1) cases for controlled scaling, not inference.

Each row is streamed to disk. The dense matrix is exactly symmetric, so R and
Python read the same bytes in their native matrix order. No stochastic study or
biological interpretation is attached to these performance-only fixtures.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sizes', type=int, nargs='+', default=[500, 1000, 2000, 4000, 10000, 20000])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = []
    for ordinal, p in enumerate(args.sizes):
        dest = args.output / str(p)
        dest.mkdir()
        positions = np.arange(p)
        # Correlation rho**abs(i-j), rho=.4, no low-rank approximation.
        powers = np.power(.4, positions.astype(float))
        with (dest/'R.bin').open('wb') as stream:
            for j in range(p):
                powers[np.abs(positions-j)].astype('<f8').tofile(stream)
        effects = [p//5, p//2, 4*p//5]
        z = sum(b*powers[np.abs(positions-j)] for j, b in zip(effects, [8., 7., 6.]))
        z += .15*np.cos(positions*.73)
        z.astype('<f8').tofile(dest/'z.bin')
        (dest/'ids.txt').write_text(''.join(f'v{j}\n' for j in positions))
        case = dict(case_id=f'scaling/{p}', n_variants=p, z_path='z.bin', R_path='R.bin',
                    R_column_major_path='R.bin', variant_ids_path='ids.txt',
                    parameters=dict(n=5000, L=5, max_iter=100, tol=.001,
                                    estimate_residual_variance=False, estimate_prior_variance=True,
                                    estimate_prior_method='optim', residual_variance=1.,
                                    scaled_prior_variance=.2, standardize=True, check_input=False,
                                    check_prior=True, coverage=.95, min_abs_corr=.5, n_purity=100000,
                                    refine=False, track_fit=False, verbose=False),
                    provenance=dict(kind='deterministic controlled performance fixture',
                                    LD='dense AR(1), rho=0.4',
                                    signal='R @ sparse effects [8,7,6] at [p//5,p//2,4p//5] + 0.15*cos(0.73*j)',
                                    frozen_before_backend_outcomes=True),
                    sha256={name:sha(dest/name) for name in ['R.bin','z.bin','ids.txt']},
                    memory_preflight=dict(dense_matrix_bytes=8*p*p,
                                          four_dense_matrices_plus_2GiB_bytes=32*p*p+2*1024**3,
                                          address_space_limit_bytes=16*1024**3,
                                          caveat='Allocation estimate, not a guarantee; failures remain recorded.'))
        path = dest/'case.json'
        path.write_text(json.dumps(case, indent=2)+'\n')
        tasks.append(dict(case_id=case['case_id'], stage='C', case_path=str(path.resolve()),
                          order=['R','Python'] if ordinal%2==0 else ['Python','R']))
        print('Prepared', p, 'dense bytes', 8*p*p, flush=True)
    (args.output/'tasks.json').write_text(json.dumps(tasks, indent=2)+'\n')


if __name__ == '__main__':
    main()
