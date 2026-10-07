#!/usr/bin/env python3
"""Export all seven frozen public cases for identical Python/R input loading."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def prepare(data, output):
    # Reuse checksum/order checks only; no fit, goldens update or simulation.
    sys.path.insert(0, str(ROOT / 'examples'))
    from check_example import validate_checksums, validate_order
    manifest = validate_checksums(data)
    metadata = json.loads((data / 'metadata.json').read_text())
    params = json.loads((data / 'parameters.json').read_text())
    validate_order(data, metadata)
    output.mkdir(parents=True, exist_ok=False)
    with np.load(data / 'inputs.npz', allow_pickle=False) as arrays:
        for dataset in metadata:
            ids = metadata[dataset]['snp']
            (output / f'{dataset}_ids.txt').write_text('\n'.join(ids) + '\n')
            z = arrays[f'{dataset}_beta'] / np.sqrt(arrays[f'{dataset}_varbeta'])
            z.astype('<f8', copy=False).tofile(output / f'{dataset}_z.bin')
            R = arrays[f'{dataset}_LD']
            R.astype('<f8', copy=False).tofile(output / f'{dataset}_R.bin')
            # Same matrix, different serialization to avoid R loading an extra
            # dense transpose while measuring process high-water RSS.
            R.T.astype('<f8', copy=False).tofile(output / f'{dataset}_R_column_major.bin')
    cases = []
    for name, setting in params.items():
        dataset = setting['dataset']
        case = dict(case_id=name, n_variants=500,
                    z_path=f'{dataset}_z.bin', R_path=f'{dataset}_R.bin',
                    R_column_major_path=f'{dataset}_R_column_major.bin',
                    variant_ids_path=f'{dataset}_ids.txt',
                    parameters={k: v for k, v in setting.items()
                                if k not in ('dataset', 'input_kind', 'z_definition')},
                    diagnostic_only=setting['max_iter'] == 1,
                    provenance=dict(nature=manifest['source_kind'], source_url=manifest['source_url'],
                                    genome_build=None, ancestry=None, effect_allele=None,
                                    position_meaning='synthetic variable index, not base pairs'))
        (output / f'{name}.json').write_text(json.dumps(case, indent=2) + '\n')
        cases.append(f'{name}.json')
    hashes = {}
    for path in sorted(output.iterdir()):
        with path.open('rb') as stream:
            hashes[path.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    frozen = dict(cases=cases, files_sha256=hashes,
                  inclusion='All seven pre-existing cases; no outcome-based selection',
                  timed_cases=['D1.json', 'D2.json', 'D3.json', 'D4.json'],
                  primary='D1-D4, both successful and converged',
                  diagnostics=['D1_partial_zero', 'D3_strict_purity', 'D3_one_iteration'],
                  source_input_sha256=manifest['input_reference_sha256'])
    (output / 'manifest.json').write_text(json.dumps(frozen, indent=2) + '\n')
    return frozen


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'examples/data')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.data_dir.resolve(), args.output_dir.resolve())
