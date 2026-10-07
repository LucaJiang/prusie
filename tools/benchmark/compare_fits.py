#!/usr/bin/env python3
"""Compare exported fits without conflating PIP agreement and other fields."""
import argparse
import json
from pathlib import Path

import numpy as np

FIELDS = ('pip', 'alpha', 'mu', 'mu2', 'lbf_variable', 'lbf', 'V', 'sigma2', 'elbo', 'KL')


def error(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape:
        return dict(shape_match=False, max_abs_error=None,
                    python_shape=list(left.shape), R_shape=list(right.shape))
    finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
    return dict(shape_match=True, finite=finite,
                max_abs_error=float(np.max(abs(left - right), initial=0)) if finite else None)


def compare(python, reference):
    """PIP ≤1e-5/rtol0 + input/model/probability validity; retain all diagnostics."""
    result = dict(case_id=python.get('case_id'), PIP_atol=1e-5, PIP_rtol=0,
                  passed=False, errors={}, failure=None)
    if python.get('status') != 'ok' or reference.get('status') != 'ok':
        result['failure'] = dict(python_status=python.get('status'), R_status=reference.get('status'),
                                 python_error=python.get('error'), R_error=reference.get('error'))
        return result
    a, b = python['fit'], reference['fit']
    result['errors'] = {key: error(a[key], b[key]) for key in FIELDS}
    result['variant_ids_order_match'] = a['variant_ids'] == b['variant_ids']
    result['model_parameters_match'] = python['parameters'] == reference['parameters']
    result['python_valid'] = bool(python['validity']['passed'])
    result['R_valid'] = bool(reference['validity']['passed'])
    result.update(niter_python=a['niter'], niter_R=b['niter'],
                  converged_python=a['converged'], converged_R=b['converged'])
    ac = {x['component_id_0based']: x for x in a['cs']}
    bc = {x['component_id_0based']: x for x in b['cs']}
    records = []
    for component in sorted(ac.keys() | bc.keys()):
        left, right = ac.get(component), bc.get(component)
        row = dict(component_id_0based=component, python_present=left is not None, R_present=right is not None)
        if left is not None and right is not None:
            row.update(members_match=sorted(left['members_0based']) == sorted(right['members_0based']),
                       coverage_abs_error=abs(left['coverage'] - right['coverage']),
                       purity_abs_error={key: abs(left['purity'][key] - right['purity'][key])
                                         for key in left['purity']})
        records.append(row)
    result['credible_sets_by_original_component'] = records
    result['no_cs_python'] = not bool(ac)
    result['no_cs_R'] = not bool(bc)
    e = result['errors']['pip']
    result['passed'] = bool(e['shape_match'] and e.get('finite') and e['max_abs_error'] <= 1e-5
                            and result['variant_ids_order_match'] and result['model_parameters_match']
                            and result['python_valid'] and result['R_valid'])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--R', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    record = compare(json.loads(args.python.read_text()), json.loads(args.R.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + '\n')
    raise SystemExit(0 if record['passed'] else 1)
