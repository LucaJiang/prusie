#!/usr/bin/env python3
"""Check examples and Pages from an independent source directory with installed packages."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='New directory for copied source, logs and executed results')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        parser.error('--output-dir must be new; preserve prior check evidence')
    if out == ROOT or ROOT.is_relative_to(out):
        parser.error('--output-dir must not contain the original source')
    out.mkdir(parents=True)
    source = out/'source'

    def ignore(directory, names):
        ignored = {'.git', '.venv', '.pytest_cache', '__pycache__', 'target', 'dist', 'build'}
        return [name for name in names if name in ignored or name.endswith('.egg-info')
                or (Path(directory)/name).resolve() == out]

    shutil.copytree(ROOT, source, ignore=ignore)
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    commands = [
        [sys.executable, '-B', str(source/'examples/check_example.py'), '--output-dir', str(out/'example')],
        [sys.executable, '-B', str(source/'tools/generate_results.py'), '--check'],
        [sys.executable, '-B', str(source/'examples/quickstart.py')],
        [sys.executable, '-B', str(source/'tools/build_docs.py'), '--check'],
        [sys.executable, '-B', str(source/'tools/check_docs.py'), '--output', str(out/'docs.json')],
    ]
    records = []
    for i, command in enumerate(commands):
        call_env = env.copy()
        if i == 0:
            call_env['PATH'] = str(out/'no-executables')
        result = subprocess.run(command, cwd=source, env=call_env, capture_output=True, text=True)
        (out/f'check-{i}.log').write_text(result.stdout + result.stderr)
        print(result.stdout, end='')
        print(result.stderr, end='', file=sys.stderr)
        records.append({'command': command, 'returncode': result.returncode})
    record = {'passed': all(item['returncode'] == 0 for item in records), 'commands': records,
              'source': str(source), 'example_PATH_contains_R': False,
              'scope': 'Independent copied source; installed package; no remote CI or browser claim.'}
    (out/'record.json').write_text(json.dumps(record, indent=2)+'\n')
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
