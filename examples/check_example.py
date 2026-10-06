#!/usr/bin/env python3
"""Fit a frozen 500-SNP teaching fixture and check it offline against official R.

Requires only the installed prusie package and NumPy. This program never
downloads data, calls R, imports source from this checkout or regenerates goldens.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import warnings

import numpy as np
import prusie

DATA = Path(__file__).resolve().parent / 'data'
PIP_ATOL = 1e-5
VALIDITY_ATOL = 1e-10
FIELDS = ('pip', 'alpha', 'mu', 'mu2', 'lbf_variable', 'lbf', 'V', 'sigma2', 'elbo', 'KL')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def array_hash(array):
    array = np.asarray(array)
    header = json.dumps({'dtype': array.dtype.str, 'shape': array.shape}, sort_keys=True).encode()
    return hashlib.sha256(header + np.ascontiguousarray(array).tobytes()).hexdigest()


def max_error(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        return {'max_abs_error': None, 'shape_match': False,
                'actual_shape': list(actual.shape), 'expected_shape': list(expected.shape)}
    return {'max_abs_error': float(np.max(np.abs(actual - expected), initial=0)), 'shape_match': True}


def runtime_identity():
    from prusie import _native
    return {
        'python': platform.python_version(), 'numpy': np.__version__,
        'prusie': prusie.__version__, 'platform': platform.platform(),
        'machine': platform.machine(), 'backend': _native.backend_version(),
        'ser_math_backend': _native.ser_math_backend(),
        'installed_module': str(Path(prusie.__file__).resolve()),
        'native_extension_sha256': sha256(_native.__file__),
        'thread_environment': {name: os.environ.get(name) for name in
            ('PRUSIE_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'MKL_NUM_THREADS', 'PRUSIE_SER_MATH')},
    }


def validate_checksums(data):
    """Validate the independent reference manifest and native snapshot separately."""
    manifest = read_json(data / 'manifest.json')
    checks = read_json(data / 'checksums.json')['files']
    if isinstance(checks, list):
        checks = {item['path']: item for item in checks}
    required = {'inputs.npz', 'r_fits.npz', 'r_fits.json', 'parameters.json',
                'metadata.json', 'variants.tsv', 'reference.lock.json', 'manifest.json'}
    require(required <= set(checks), 'Reference manifest is missing required checksum entries')
    for name, item in checks.items():
        path = (data / name).resolve()
        require(path.is_relative_to(data.resolve()), f'Unsafe checksum path: {name}')
        expected = item['sha256'] if isinstance(item, dict) else item
        require(path.is_file() and sha256(path) == expected, f'Checksum mismatch: {name}')
    for name, expected in manifest['input_reference_sha256'].items():
        path = (data / name).resolve()
        require(path.is_relative_to(data.resolve()), f'Unsafe manifest path: {name}')
        require(path.is_file() and sha256(path) == expected, f'Manifest checksum mismatch: {name}')
    require(manifest['snp_count'] == 500, 'Expected the full 500-SNP fixture')
    return manifest


def validate_order(data, metadata):
    with (data / 'variants.tsv').open(newline='') as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    for name, item in metadata.items():
        selected = [row for row in rows if row['dataset'] == name]
        ids = item['snp']
        require(len(selected) == len(ids) == 500 and len(set(ids)) == 500,
                f'{name}: variant count or uniqueness mismatch')
        require([row['variant_id'] for row in selected] == ids, f'{name}: SNP order mismatch')
        for axis in ('array_index_0based', 'ld_row_0based', 'ld_column_0based'):
            require([int(row[axis]) for row in selected] == list(range(500)),
                    f'{name}: {axis} is not the declared matrix-axis order')


def check_validity(fit, ids, correlation, params):
    """Independent identities use returned arrays, never production summary helpers."""
    m, L = len(ids), params['L']
    require(np.array_equal(fit.variant_ids, ids), 'Returned SNP IDs/order changed')
    require(fit.alpha.shape == (L, m) and fit.pip.shape == (m,), 'Posterior shape mismatch')
    for field in FIELDS:
        require(np.isfinite(np.asarray(getattr(fit, field))).all(), f'Nonfinite posterior field: {field}')
    require(((fit.alpha >= 0) & (fit.alpha <= 1)).all(), 'Alpha probabilities outside [0,1]')
    require(((fit.pip >= 0) & (fit.pip <= 1)).all(), 'PIP outside [0,1]')
    alpha_sum_error = float(np.max(abs(fit.alpha.sum(axis=1) - 1)))
    require(alpha_sum_error <= VALIDITY_ATOL, f'Alpha row mass error {alpha_sum_error:g}')
    require((fit.V >= 0).all() and fit.sigma2 > 0 and (fit.mu2 >= 0).all(), 'Invalid variance/second moment')
    require(1 <= fit.niter <= params['max_iter'] and len(fit.elbo) == fit.niter, 'Invalid iteration count/trace')
    active = np.asarray(fit.V) > params['prior_tol']
    expected_pip = np.array([1 - math.prod(1 - float(fit.alpha[l, j]) for l in np.flatnonzero(active))
                             for j in range(m)])
    pip_identity_error = float(np.max(abs(fit.pip - expected_pip)))
    require(pip_identity_error <= VALIDITY_ATOL, f'PIP probability identity error {pip_identity_error:g}')
    for name in ('L', 'n', 'max_iter', 'tol', 'coverage', 'min_abs_corr', 'standardize',
                 'estimate_residual_variance', 'estimate_prior_variance', 'estimate_prior_method',
                 'prior_tol', 'null_weight', 'check_null_threshold', 'scaled_prior_variance',
                 'n_purity', 'refine', 'track_fit'):
        require(fit.params[name] == params[name], f'Model parameter changed: {name}')
    require(fit.params['input_kind'] == 'z' and fit.null_index is None, 'Model/input route changed')
    require(fit.params['input_var_y'] is None, 'Unexpected phenotype variance assumption')
    weights = np.asarray(params.get('prior_weights', np.ones(m)), dtype=float)
    weights /= weights.sum()
    require(np.max(abs(np.asarray(fit.params['prior_weights']) - weights)) <= VALIDITY_ATOL,
            'Effective prior weights changed')
    cs = fit.sets
    require(cs['requested_coverage'] == params['coverage'], 'Requested coverage metadata changed')
    count = len(cs['cs'])
    require(len(cs['cs_index']) == len(cs['coverage']) == len(cs['actual_coverage']) == count,
            'Credible-set arrays have inconsistent lengths')
    require(len(set(map(int, cs['cs_index']))) == count, 'Duplicate original component ID')
    for values in cs['purity'].values():
        require(len(values) == count, 'Purity row count mismatch')
    records = []
    for i, members in enumerate(cs['cs']):
        members = np.asarray(members)
        component = int(cs['cs_index'][i])
        require(0 <= component < L and len(members) > 0, 'Invalid CS component or empty retained set')
        require(np.issubdtype(members.dtype, np.integer) and
                (members >= 0).all() and (members < m).all() and
                len(set(map(int, members))) == len(members), 'Invalid CS members')
        mass = math.fsum(float(fit.alpha[component, j]) for j in members)
        require(abs(float(cs['coverage'][i]) - mass) <= VALIDITY_ATOL and
                abs(float(cs['actual_coverage'][i]) - mass) <= VALIDITY_ATOL,
                f'Coverage disagrees with original component {component} posterior mass')
        require(mass + VALIDITY_ATOL >= params['coverage'] and mass <= 1 + VALIDITY_ATOL,
                f'Invalid credible-set posterior mass for component {component}')
        pair_values = np.abs(correlation[np.ix_(members, members)])[np.triu_indices(len(members), 1)]
        expected_purity = (float(pair_values.min()), float(pair_values.mean()), float(np.median(pair_values))) if len(pair_values) else (1., 1., 1.)
        purity = {key: float(cs['purity'][key][i]) for key in ('min_abs_corr', 'mean_abs_corr', 'median_abs_corr')}
        for (key, value), expected in zip(purity.items(), expected_purity):
            require(abs(value - expected) <= VALIDITY_ATOL, f'Purity {key} disagrees with signed LD for component {component}')
        require(purity['min_abs_corr'] + VALIDITY_ATOL >= params['min_abs_corr'], 'CS purity below requested threshold')
        records.append({'component_id_0based': component, 'members_0based': members.tolist(),
                        'coverage': float(cs['coverage'][i]), 'independently_summed_alpha': mass,
                        'purity': purity})
    return {'passed': True, 'alpha_row_sum_max_error': alpha_sum_error,
            'pip_identity_max_error': pip_identity_error,
            'coverage_available': bool(count), 'cs': records,
            'no_cs_interpretation': None if count else 'unavailable; no fabricated zero coverage'}


def compare_cs(actual, reference):
    a = {item['component_id_0based']: item for item in actual}
    b = {item['component_id_0based']: item for item in reference}
    rows = []
    for component in sorted(set(a) | set(b)):
        left, right = a.get(component), b.get(component)
        row = {'component_id_0based': component, 'present_native': left is not None,
               'present_R': right is not None}
        if left is not None and right is not None:
            row.update(members_match=sorted(left['members_0based']) == sorted(right['members_0based']),
                       coverage_abs_error=abs(left['coverage'] - right['coverage']),
                       purity_abs_error={name: abs(value - right['purity'][name.replace('_', '.')])
                                         for name, value in left['purity'].items()})
        rows.append(row)
    return rows


def fit_cases(data, selected=None):
    """Return executed fits and diagnostics; called separately by the maintainer freeze script."""
    metadata, parameters = read_json(data/'metadata.json'), read_json(data/'parameters.json')
    reference = read_json(data/'r_fits.json')
    validate_order(data, metadata)
    names = list(parameters) if selected is None else selected
    require(names and len(names) == len(set(names)), 'Select at least one distinct case')
    fits, records = {}, []
    with np.load(data/'inputs.npz', allow_pickle=False) as inputs, np.load(data/'r_fits.npz', allow_pickle=False) as r:
        for name in names:
            require(name in parameters, f'Unknown case {name}; choices: {", ".join(parameters)}')
            params = parameters[name]
            dataset = params['dataset']
            ids = metadata[dataset]['snp']
            matrix = inputs[f'{dataset}_LD'].copy()
            z = inputs[f'{dataset}_beta'] / np.sqrt(inputs[f'{dataset}_varbeta'])
            before = {'R': array_hash(matrix), 'z': array_hash(z)}
            options = {key: value for key, value in params.items() if key not in ('dataset', 'input_kind', 'z_definition')}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                fit = prusie.susie_rss(z=z, R=matrix, variant_ids=ids, **options)
            require(before == {'R': array_hash(matrix), 'z': array_hash(z)}, f'{name}: fitting mutated inputs')
            validity = check_validity(fit, ids, matrix, params)
            errors = {field: max_error(getattr(fit, field), r[f'{name}_{field}']) for field in FIELDS}
            pip_ok = errors['pip']['shape_match'] and errors['pip']['max_abs_error'] <= PIP_ATOL
            record = {'case': name, 'dataset': dataset, 'passed': bool(pip_ok), 'parameters': params,
                      'model_sha256': hashlib.sha256(json.dumps(params, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                      'inputs_sha256': before,
                      'variant_order_sha256': hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest(),
                      'pip_atol': PIP_ATOL, 'pip_rtol': 0, 'R_errors': errors, 'validity': validity,
                      'converged_native': bool(fit.converged), 'converged_R': reference[name]['converged'],
                      'niter_native': int(fit.niter), 'niter_R': reference[name]['niter'],
                      'cs_comparison_by_original_component': compare_cs(validity['cs'], reference[name]['cs']),
                      'warnings_native': list(dict.fromkeys([str(w.message) for w in caught] + fit.warnings)),
                      'warnings_R': reference[name]['warnings']}
            records.append(record)
            fits[name] = fit
    return fits, records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='Write report.json and actual_native.npz here; never into frozen data')
    parser.add_argument('--data-dir', type=Path, default=DATA, help='Frozen example bundle directory (default: examples/data beside this script)')
    parser.add_argument('--case', action='append', help='Run this case; may repeat. Default runs every declared case.')
    args = parser.parse_args(argv)
    data, out = args.data_dir.resolve(), args.output_dir.resolve()
    if out == data or out.is_relative_to(data):
        parser.error('--output-dir must be outside the frozen data directory')
    out.mkdir(parents=True, exist_ok=True)
    record = {'passed': False, 'runtime': runtime_identity(), 'cases': [],
              'acceptance': 'PIP atol=1e-5,rtol=0 plus input/model/probability validity; other numerical differences are diagnostics.'}
    try:
        manifest = validate_checksums(data)
        record['manifest_sha256'] = sha256(data/'manifest.json')
        record['reference_checksums_sha256'] = sha256(data/'checksums.json')
        record['source_kind'] = manifest['source_kind']
        record['reference'] = read_json(data/'reference.lock.json')
        native_checks = read_json(data/'native_checksums.json')['files']
        require({'expected_native.json', 'expected_native.npz'} <= set(native_checks),
                'Native snapshot checksum entries missing')
        for name, item in native_checks.items():
            path = (data / name).resolve()
            require(path.is_relative_to(data), f'Unsafe native checksum path: {name}')
            require(path.is_file() and sha256(path) == item['sha256'], f'Checksum mismatch: {name}')
        native_meta = read_json(data/'expected_native.json')
        require(sha256(data/'expected_native.npz') == native_meta['npz_sha256'], 'Checksum mismatch: expected_native.npz')
        # The frozen pre-rename snapshot retains its original distribution key.
        # This is a metadata lookup only; no numerical values or tolerances change.
        record['native_snapshot_distribution'] = 'pyrsusie'
        record['native_snapshot_version'] = native_meta['runtime']['pyrsusie']
        record['version_differs_from_snapshot'] = prusie.__version__ != record['native_snapshot_version']
        fits, cases = fit_cases(data, args.case)
        with np.load(data/'expected_native.npz', allow_pickle=False) as expected:
            for row in cases:
                row['native_snapshot_pip_diagnostic'] = max_error(fits[row['case']].pip, expected[f"{row['case']}_pip"])
        record['cases'] = cases
        record['passed'] = all(row['passed'] for row in cases)
        np.savez_compressed(out/'actual_native.npz', **{f'{name}_{field}': getattr(fit, field)
                                                     for name, fit in fits.items() for field in FIELDS})
        for row in cases:
            error = row['R_errors']['pip']['max_abs_error']
            error_text = f'{error:.6g}' if error is not None else 'shape mismatch'
            print(f"{'PASS' if row['passed'] else 'FAIL'} {row['case']}: max |PIP − R|={error_text}; "
                  f"validity=PASS; CS={len(row['validity']['cs'])}; niter={row['niter_native']}; "
                  f"converged={row['converged_native']}")
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
        print('FAIL ' + record['error'], file=sys.stderr)
    (out/'report.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(f"prusie={record['runtime']['prusie']}; backend={record['runtime']['backend']}; "
          f"report={out/'report.json'}")
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
