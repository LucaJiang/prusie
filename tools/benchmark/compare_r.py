#!/usr/bin/env python3
"""Manual comparison of the complete public 500-SNP fixture with pinned R.

The independent toy example is the default; --fixture regression uses
four older timed fixtures plus three predefined numerical diagnostics.
Backend processes run sequentially. This is deliberately outside default CI.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from compare_fits import compare
from prepare_fixture import prepare, ROOT
from prepare_teaching import prepare as prepare_teaching

HERE = Path(__file__).resolve().parent


def invoke(command, output, environment):
    with output.with_suffix('.log').open('w') as log:
        process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if output.exists():
        record = json.loads(output.read_text())
    else:
        record = dict(status='error', error='Backend exited without a JSON record',
                      time_seconds=None, peak_rss_mib=None)
        output.write_text(json.dumps(record, indent=2) + '\n')
    return record, process.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rscript', required=True, help='Rscript using the intended R/BLAS installation')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--blas-probe', type=Path, help='Compiled blas_probe.so; required for timing')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--validation-only', action='store_true')
    parser.add_argument('--fixture', choices=('independent', 'regression'), default='independent')
    args = parser.parse_args()
    if not args.validation_only and (args.repeats < 7 or args.blas_probe is None):
        parser.error('Timing requires --blas-probe and at least seven retained repeats')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if args.fixture == 'independent':
        case_file = prepare_teaching(ROOT / 'src/prusie/data/teaching', out / 'inputs')
        manifest = {'cases': [case_file.name], 'timed_cases': [case_file.name]}
    else:
        manifest = prepare(ROOT / 'examples/data', out / 'inputs')
    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'PRUSIE_NUM_THREADS',
                 'RAYON_NUM_THREADS'):
        env[name] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    # Freeze inclusion and execution order before observing any result.
    protocol = dict(cases=manifest['cases'], timed_cases=manifest['timed_cases'],
                    warmups=1, retained_repeats=args.repeats,
                    backend_order='Even case index: R then Python; odd: Python then R',
                    primary='All timed, completed settings-matched pairs with valid times; convergence, CS and PIP error do not select timing cases',
                    timing='Full public API; imports/input reads/exports/explicit GC excluded',
                    memory='One separate fresh process and one full fit per case/backend; process HWM includes imports/loading',
                    summary='Geometric mean paired median-R/median-Python runtime; memory ratios reported separately',
                    validation_only=args.validation_only)
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    rows = []
    commands = []
    for index, filename in enumerate(manifest['cases']):
        case_path = out / 'inputs' / filename
        case = json.loads(case_path.read_text())
        case_dir = out / case['case_id']
        case_dir.mkdir()
        measurements = {}
        modes = ['validate']
        if filename in manifest['timed_cases'] and not args.validation_only:
            modes += ['time', 'memory']
        backends = ['R', 'python'] if index % 2 == 0 else ['python', 'R']
        for mode in modes:
            for backend in backends:
                dest = case_dir / f'{backend}_{mode}.json'
                command = ([args.python, str(HERE / 'fine_mapping.py')] if backend == 'python'
                           else [args.rscript, '--vanilla', str(HERE / 'fine_mapping.R')])
                command += ['--case', str(case_path), '--mode', mode, '--output', str(dest),
                            '--repeats', str(args.repeats)]
                if backend == 'R' and args.blas_probe:
                    command += ['--blas-probe', str(args.blas_probe.resolve())]
                record, status = invoke(command, dest, env)
                measurements[backend, mode] = record
                commands.append(dict(case_id=case['case_id'], backend=backend, mode=mode,
                                     command=command, exit_code=status))
                (out / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        agreement = compare(measurements['python', 'validate'], measurements['R', 'validate'])
        (case_dir / 'agreement.json').write_text(json.dumps(agreement, indent=2) + '\n')
        row = dict(case_id=case['case_id'], n_variants=case['n_variants'], L=case['parameters']['L'],
                   settings_match=all(v.get('parameters') == case['parameters'] for v in measurements.values()),
                   measurement_status_ok=all(v.get('status') == 'ok' for v in measurements.values()),
                   PIP_passed=agreement['passed'], numerical_errors=agreement['errors'],
                   max_iter=case['parameters']['max_iter'],
                   niter={k:measurements[k, 'validate'].get('niter') for k in ('R','python')},
                   converged={k:measurements[k, 'validate'].get('converged') for k in ('R','python')},
                   mutually_converged=agreement.get('converged_python', False) and agreement.get('converged_R', False),
                   timed=filename in manifest['timed_cases'] and not args.validation_only,
                   median_seconds={}, repeat_seconds={}, peak_rss_mib={}, speedup=None, memory_ratio=None)
        if row['timed']:
            for backend in backends:
                timing = measurements[backend, 'time']
                memory = measurements[backend, 'memory']
                samples = [s['time_seconds'] for s in timing.get('samples', []) if not s['warmup']]
                row['repeat_seconds'][backend] = samples
                row['median_seconds'][backend] = float(np.median(samples)) if len(samples) >= 7 and all(t is not None and np.isfinite(t) and t > 0 for t in samples) and timing['status'] == 'ok' else None
                row['peak_rss_mib'][backend] = memory.get('peak_rss_mib') if memory['status'] == 'ok' else None
            r, p = (row['median_seconds'][k] for k in ('R', 'python'))
            if r is not None and p is not None and r > 0 and p > 0:
                row['speedup'] = r / p
            r, p = (row['peak_rss_mib'][k] for k in ('R', 'python'))
            if r is not None and p is not None and r > 0 and p > 0:
                row['memory_ratio'] = r / p
                row['memory_reduction'] = 1 - p / r
        rows.append(row)
        (out / 'cases.json').write_text(json.dumps(rows, indent=2) + '\n')
    primary = [r for r in rows if r['timed'] and r['settings_match'] and r['measurement_status_ok'] and r['speedup'] is not None]
    ratios = np.asarray([r['speedup'] for r in primary])
    summary = dict(cases=len(rows), agreement_passed=sum(r['PIP_passed'] for r in rows),
                   timing_primary_count=len(primary),
                   wins=int(np.sum(ratios > 1)), ties=int(np.sum(ratios == 1)), losses=int(np.sum(ratios < 1)),
                   geometric_mean_speedup=float(np.exp(np.mean(np.log(ratios)))) if ratios.size else None,
                   median_speedup=float(np.median(ratios)) if ratios.size else None,
                   speedup_range=[float(ratios.min()), float(ratios.max())] if ratios.size else None,
                   speedup_IQR=np.quantile(ratios, [.25, .75]).tolist() if ratios.size else None,
                   memory_note='See every per-case peak and ratio in cases.json; memory is independent of timing')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return 0 if all(r['PIP_passed'] for r in rows) and all(c['exit_code'] == 0 for c in commands) else 1


if __name__ == '__main__':
    raise SystemExit(main())
