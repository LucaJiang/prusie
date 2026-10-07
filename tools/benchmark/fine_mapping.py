#!/usr/bin/env python3
"""One complete SuSiE-RSS case per process; validation, timing or peak RSS.

This manual benchmark imports the installed package. It does not change fitting
options or use a fitted-result cache. See README.md in this directory.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import warnings

import numpy as np
import prusie

FIELDS = ('pip', 'alpha', 'mu', 'mu2', 'lbf_variable', 'lbf', 'V', 'sigma2', 'elbo', 'KL')
THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'PRUSIE_NUM_THREADS')


def read_case(path):
    path = Path(path).resolve()
    case = json.loads(path.read_text())
    for key in ('z_path', 'R_path', 'variant_ids_path'):
        case[key] = str((path.parent / case[key]).resolve())
    return case


def load_case(case):
    """Load exactly one trait, preserving every signed float64 input and SNP."""
    m = int(case['n_variants'])
    z = np.fromfile(case['z_path'], dtype='<f8')
    R = np.fromfile(case['R_path'], dtype='<f8')
    if z.size != m or R.size != m * m:
        raise ValueError('Declared variant count does not match input byte counts')
    ids = Path(case['variant_ids_path']).read_text().splitlines()
    if len(ids) != m or len(set(ids)) != m or any(not x for x in ids):
        raise ValueError('Variant IDs must be distinct, nonempty and in matrix order')
    return dict(z=z, R=R.reshape(m, m), variant_ids=ids, **case['parameters'])


def run_case(inputs, stage='C'):
    if stage != 'C':
        raise ValueError('This adapter measures fine-mapping (stage C) only')
    return prusie.susie_rss(**inputs)


def summarize_fit(fit):
    return dict(niter=int(fit.niter), converged=bool(fit.converged),
                cs_count=len(fit.sets['cs_index']),
                cs_component_ids=[int(x) for x in fit.sets['cs_index']],
                status='converged' if fit.converged else 'nonconverged')


def export_fit(fit):
    """Portable complete scientific diagnostics; call outside measurement."""
    cs = []
    for k, component in enumerate(fit.sets['cs_index']):
        cs.append(dict(component_id_0based=int(component),
                       members_0based=np.asarray(fit.sets['cs'][k]).tolist(),
                       coverage=float(fit.sets['coverage'][k]),
                       purity={key: float(values[k]) for key, values in fit.sets['purity'].items()}))
    return dict(**{key: np.asarray(getattr(fit, key)).tolist() for key in FIELDS},
                variant_ids=fit.variant_ids.tolist(), cs=cs,
                cs_return_order=[int(x) for x in fit.sets['cs_index']],
                requested_coverage=fit.sets['requested_coverage'],
                null_index=fit.null_index, niter=int(fit.niter), converged=bool(fit.converged),
                params=fit.params, warnings=fit.warnings)


def validate_fit(fit, inputs):
    """Independent probability/CS identities, without production summaries."""
    errors = []
    m = len(inputs['z'])
    q = m + int(inputs.get('null_weight', 0) > 0)
    L = min(inputs.get('L', 10), q)
    if fit.alpha.shape != (L, q) or fit.pip.shape != (m,):
        errors.append('posterior dimensions')
    if fit.variant_ids.tolist() != list(inputs['variant_ids']):
        errors.append('variant IDs/order')
    for key in FIELDS:
        if not np.all(np.isfinite(getattr(fit, key))):
            errors.append('nonfinite ' + key)
    for key in ('pip', 'alpha'):
        values = np.asarray(getattr(fit, key))
        if not np.all((values >= 0) & (values <= 1)):
            errors.append('probability bounds: ' + key)
    mass_error = float(np.max(abs(fit.alpha.sum(axis=1) - 1)))
    active = fit.V > inputs.get('prior_tol', 1e-9)
    pip = 1 - np.prod(1 - fit.alpha[active], axis=0)
    if fit.null_index is not None:
        pip = np.delete(pip, fit.null_index)
    pip_error = float(np.max(abs(pip - fit.pip)))
    if mass_error > 1e-10 or pip_error > 1e-10:
        errors.append('posterior probability identities')
    if not (np.all(fit.V >= 0) and fit.sigma2 > 0 and np.all(fit.mu2 >= 0)):
        errors.append('variance/second moment validity')
    if not (1 <= fit.niter <= inputs.get('max_iter', 50) and len(fit.elbo) == fit.niter):
        errors.append('iteration count/ELBO length')
    parameter_checks = {key: bool(fit.params[key] == value)
        for key, value in inputs.items() if key in fit.params and key != 'prior_weights'}
    weights = np.asarray(inputs.get('prior_weights', np.full(m, 1. / m)), dtype=float)
    if fit.null_index is not None:
        null_weight = inputs['null_weight']
        weights = np.r_[weights * (1 - null_weight), null_weight]
    weights = weights / weights.sum()
    parameter_checks['prior_weights'] = bool(np.max(abs(np.asarray(fit.params['prior_weights']) - weights)) <= 1e-10)
    if not all(parameter_checks.values()):
        errors.append('changed model parameters')
    cs_errors = []
    for k, component in enumerate(fit.sets['cs_index']):
        members = np.asarray(fit.sets['cs'][k], dtype=int)
        if not (0 <= component < L and len(members) and np.all((members >= 0) & (members < m))):
            errors.append('CS identity/members')
            continue
        mass = float(fit.alpha[component, members].sum())
        block = abs(inputs['R'][np.ix_(members, members)])
        pairs = block[np.triu_indices(len(members), 1)]
        purity = (float(pairs.min()), float(pairs.mean()), float(np.median(pairs))) if pairs.size else (1., 1., 1.)
        delta = dict(coverage=abs(mass - float(fit.sets['coverage'][k])),
                     **{key: abs(value - float(fit.sets['purity'][key][k]))
                        for key, value in zip(('min_abs_corr', 'mean_abs_corr', 'median_abs_corr'), purity)})
        cs_errors.append(dict(component_id_0based=int(component), errors=delta))
        if max(delta.values()) > 1e-10 or mass + 1e-10 < inputs.get('coverage', .95):
            errors.append('CS coverage/purity identity')
        if purity[0] + 1e-10 < inputs.get('min_abs_corr', .5):
            errors.append('CS purity below threshold')
    return dict(passed=not errors, failures=errors, alpha_mass_max_abs_error=mass_error,
                pip_identity_max_abs_error=pip_error, parameter_checks=parameter_checks, cs=cs_errors)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def numerical_libraries():
    """Inspect already loaded BLAS; unknown thread counts stay explicitly null."""
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[-1].startswith('/') and any(
                word in Path(fields[-1]).name.lower() for word in ('blas', 'mkl', 'libmvec')):
            paths.add(fields[-1])
    records = []
    for path in sorted(paths):
        lib = ctypes.CDLL(path)
        threads = None
        for symbol in ('openblas_get_num_threads', 'openblas_get_num_threads64_',
                       'scipy_openblas_get_num_threads64_', 'scipy_openblas_get_num_threads',
                       'MKL_Get_Max_Threads'):
            if hasattr(lib, symbol):
                func = getattr(lib, symbol)
                func.restype = ctypes.c_int
                threads = int(func())
                break
        records.append(dict(path=path, sha256=digest(path), threads=threads))
    return records


def runtime_identity():
    from prusie import _native
    package = Path(prusie.__file__).parent
    return dict(version=prusie.__version__, backend=_native.backend_version(),
                python=platform.python_version(), numpy=np.__version__,
                platform=platform.platform(), native_sha256=digest(_native.__file__),
                source_hashes={p.name: digest(p) for p in sorted(package.glob('*.py'))},
                ser_math_backend=_native.ser_math_backend(),
                blas=numerical_libraries(), threads={k: os.environ.get(k) for k in THREAD_VARS},
                affinity=sorted(os.sched_getaffinity(0)), pid=os.getpid(),
                rlimit_as=list(resource.getrlimit(resource.RLIMIT_AS)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--mode', choices=('validate', 'time', 'memory'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    record = dict(stage='C', backend='prusie', mode=args.mode, status='error',
                  time_seconds=None, peak_rss_mib=None, numerical_errors=None)
    exit_code = 1
    try:
        if any(os.environ.get(k) != '1' for k in THREAD_VARS):
            raise ValueError('Set all documented numerical thread variables to 1 before launch')
        if args.mode == 'time' and args.repeats < 7:
            raise ValueError('At least seven retained repeats are required')
        case = read_case(args.case)
        record.update(case_id=case['case_id'], n_variants=case['n_variants'],
                      L=case['parameters'].get('L', 10), parameters=case['parameters'])
        inputs = load_case(case)
        samples = []
        if args.mode == 'time':
            # One warmup, then one complete result retained at a time. Explicit
            # GC and deletion happen outside each clock, equally for both backends.
            repeats = range(args.repeats + 1)
        else:
            repeats = range(1)
        fit = None
        for repeat in repeats:
            fit = None
            gc.collect()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                start_epoch = time.time()
                start = time.perf_counter_ns()
                fit = run_case(inputs)
                elapsed = (time.perf_counter_ns() - start) / 1e9
                end_epoch = time.time()
            # Capture HWM before serialization, hash buffers, or diagnostic arrays.
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            sample = dict(repeat=repeat, warmup=args.mode == 'time' and repeat == 0,
                          time_seconds=elapsed if args.mode == 'time' else None,
                          peak_rss_mib=peak if args.mode == 'memory' else None,
                          start_epoch=start_epoch, end_epoch=end_epoch,
                          warnings=[str(w.message) for w in caught], **summarize_fit(fit))
            samples.append(sample)
        record.update(status='ok', samples=samples, **{
            k: v for k, v in summarize_fit(fit).items() if k != 'status'})
        if args.mode == 'memory':
            record['peak_rss_mib'] = samples[0]['peak_rss_mib']
        if args.mode == 'validate':
            record['fit'] = export_fit(fit)
            record['validity'] = validate_fit(fit, inputs)
            if not record['validity']['passed']:
                record['status'] = 'invalid'
        record['environment'] = runtime_identity()
        if any(x['threads'] not in (None, 1) for x in record['environment']['blas']):
            raise ValueError('Loaded numerical library reports more than one thread')
        exit_code = 0 if record['status'] == 'ok' else 1
    except Exception as exc:
        record.update(status='error', error=f'{type(exc).__name__}: {exc}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
