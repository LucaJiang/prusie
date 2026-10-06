"""Exercise public calls and prove both checksum and numerical failure paths."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT/'examples/check_example.py'
DATA = ROOT/'examples/data'


def run_example(tmp_path, data=DATA, case=None):
    # Neither the working directory nor PATH supplies R or the repository source.
    env = os.environ.copy()
    env['PATH'] = str(tmp_path/'empty-path')
    env.pop('PYTHONPATH', None)
    args = [sys.executable, '-B', str(RUNNER), '--data-dir', str(data),
            '--output-dir', str(tmp_path/'result')]
    if case is not None:
        args += ['--case', case]
    result = subprocess.run(args, cwd=tmp_path, env=env, capture_output=True, text=True)
    return result, json.loads((tmp_path/'result/report.json').read_text())


def copy_data(tmp_path):
    return Path(shutil.copytree(DATA, tmp_path/'data'))


def alter_npz(path, key):
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    arrays[key].flat[0] += .25
    np.savez_compressed(path, **arrays)


def update_checksum(data, filename):
    checks = json.loads((data/'checksums.json').read_text())
    path = data/filename
    checks['files'][filename] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                               'bytes': path.stat().st_size}
    manifest = json.loads((data/'manifest.json').read_text())
    manifest['input_reference_sha256'][filename] = checks['files'][filename]['sha256']
    (data/'manifest.json').write_text(json.dumps(manifest))
    checks['files']['manifest.json'] = {
        'sha256': hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest(),
        'bytes': (data/'manifest.json').stat().st_size}
    (data/'checksums.json').write_text(json.dumps(checks))


def test_all_public_examples_without_R(tmp_path):
    result, record = run_example(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert record['passed'] and len(record['cases']) == 7
    cases = {row['case']: row for row in record['cases']}
    assert not cases['D3_strict_purity']['validity']['coverage_available']
    assert cases['D3_strict_purity']['validity']['cs'] == []
    assert cases['D3_one_iteration']['converged_native'] is False
    assert cases['D3_one_iteration']['niter_native'] == 1
    assert all(row['R_errors']['pip']['max_abs_error'] <= 1e-5 for row in cases.values())


@pytest.mark.parametrize('filename,key', [('inputs.npz', 'D1_beta'), ('r_fits.npz', 'D1_pip')])
def test_altered_input_or_expected_value_fails_checksum(tmp_path, filename, key):
    data = copy_data(tmp_path)
    alter_npz(data/filename, key)
    result, record = run_example(tmp_path, data, 'D1')
    assert result.returncode != 0 and not record['passed']
    assert f'Checksum mismatch: {filename}' in result.stderr


def test_rehashed_wrong_reference_still_fails_numerical_comparison(tmp_path):
    data = copy_data(tmp_path)
    alter_npz(data/'r_fits.npz', 'D1_pip')
    update_checksum(data, 'r_fits.npz')
    result, record = run_example(tmp_path, data, 'D1')
    assert result.returncode != 0 and not record['passed']
    assert 'FAIL D1: max |PIP' in result.stdout
    assert record['cases'][0]['R_errors']['pip']['max_abs_error'] > .2


def test_unknown_case_has_useful_failure(tmp_path):
    result, record = run_example(tmp_path, DATA, 'not-a-case')
    assert result.returncode != 0 and 'Unknown case not-a-case' in record['error']


def test_native_version_identity_is_diagnostic(tmp_path):
    data = copy_data(tmp_path)
    path = data/'expected_native.json'
    value = json.loads(path.read_text())
    value['runtime']['pyrsusie'] = '0.0.0-reference-identity-test'
    path.write_text(json.dumps(value))
    checks = json.loads((data/'native_checksums.json').read_text())
    checks['files'][path.name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                'bytes': path.stat().st_size}
    (data/'native_checksums.json').write_text(json.dumps(checks))
    result, record = run_example(tmp_path, data, 'D1')
    assert result.returncode == 0, result.stdout + result.stderr
    assert record['passed'] and record['version_differs_from_snapshot'] is True
