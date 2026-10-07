"""Measurement-helper checks using the existing frozen fixture, without fits."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from compare_fits import compare, FIELDS
from fine_mapping import load_case

DATA = Path(__file__).resolve().parents[2] / 'examples/data'


def exported_record():
    reference = json.loads((DATA / 'r_fits.json').read_text())['D3']
    params = json.loads((DATA / 'parameters.json').read_text())['D3']
    with np.load(DATA / 'r_fits.npz', allow_pickle=False) as arrays:
        fit = {key: arrays[f'D3_{key}'].tolist() for key in FIELDS}
    fit.update(variant_ids=reference['snp'], niter=reference['niter'],
               converged=reference['converged'], cs=[])
    for row in reference['cs']:
        fit['cs'].append(dict(component_id_0based=row['component_id_0based'],
                              members_0based=row['members_0based'], coverage=row['coverage'],
                              purity={key.replace('.', '_'): value for key, value in row['purity'].items()}))
    return dict(case_id='D3', status='ok', fit=fit, parameters=params, validity={'passed': True})


def test_load_case_preserves_complete_signed_matrix(tmp_path):
    with np.load(DATA / 'inputs.npz', allow_pickle=False) as arrays:
        matrix = arrays['D3_LD']
        z = arrays['D3_beta'] / np.sqrt(arrays['D3_varbeta'])
    ids = json.loads((DATA / 'metadata.json').read_text())['D3']['snp']
    matrix.astype('<f8').tofile(tmp_path / 'R.bin')
    z.astype('<f8').tofile(tmp_path / 'z.bin')
    (tmp_path / 'ids.txt').write_text('\n'.join(ids) + '\n')
    case = dict(n_variants=500, R_path=tmp_path / 'R.bin', z_path=tmp_path / 'z.bin',
                variant_ids_path=tmp_path / 'ids.txt', parameters={'n': 1000})
    inputs = load_case(case)
    np.testing.assert_array_equal(inputs['R'], matrix)
    np.testing.assert_array_equal(inputs['z'], z)
    assert inputs['R'].dtype == np.float64 and inputs['variant_ids'] == ids
    assert np.any(inputs['z'] < 0)
    case['n_variants'] = 499
    with pytest.raises(ValueError, match='byte counts'):
        load_case(case)


def test_component_return_order_is_not_relabeling():
    original = exported_record()
    reordered = deepcopy(original)
    reordered['fit']['cs'].reverse()
    compared = compare(reordered, original)
    assert compared['passed']
    assert all(row['members_match'] for row in compared['credible_sets_by_original_component'])


def test_pip_failure_and_intermediate_differences_remain_distinct():
    reference = exported_record()
    actual = deepcopy(reference)
    actual['fit']['lbf_variable'][0][0] += 0.25
    actual['fit']['cs'][0]['members_0based'] = actual['fit']['cs'][0]['members_0based'][1:]
    diagnostic = compare(actual, reference)
    assert diagnostic['passed']
    assert diagnostic['errors']['lbf_variable']['max_abs_error'] > .2
    assert not diagnostic['credible_sets_by_original_component'][0]['members_match']
    actual['fit']['pip'][0] += .1
    assert not compare(actual, reference)['passed']


def test_invalid_probability_record_and_failed_backend_cannot_pass():
    reference = exported_record()
    invalid = deepcopy(reference)
    invalid['validity']['passed'] = False
    assert not compare(invalid, reference)['passed']
    failure = compare({'status': 'error', 'error': 'invalid input'}, reference)
    assert not failure['passed'] and failure['failure']['python_error'] == 'invalid input'
