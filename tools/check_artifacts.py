#!/usr/bin/env python3
"""Inspect and install a wheel, rebuild its sdist, and check the rebuilt wheel.

Run manually or in CI with Python 3.12, a build toolchain and auditwheel. Runtime
checks use fresh environments whose PATH contains no Rust, Cargo or maturin.
No package, tag or Release is published.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import tomllib
import zipfile


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inspect_wheel(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        dist = next(n.rsplit('/', 1)[0] for n in names if n.endswith('.dist-info/WHEEL'))
        metadata = archive.read(dist + '/METADATA').decode()
        wheel = archive.read(dist + '/WHEEL').decode()
        required = ['prusie/datasets.py', 'prusie/data/teaching/inputs.npz',
                    'prusie/data/teaching/metadata.json', 'prusie/data/teaching/reference.npz',
                    'prusie/data/teaching/reference.json', 'prusie/data/teaching/checksums.json',
                    'prusie/data/teaching/LICENSE.txt', 'prusie/__init__.py', 'prusie/api.py', 'prusie/result.py',
                    'prusie/_licenses/LICENSE-R', 'prusie/_licenses/LICENSE-susieR',
                    'prusie/_licenses/LICENSE-susieR-0.16.6',
                    'prusie/_licenses/THIRD_PARTY_NOTICES.md',
                    'prusie/_licenses/Rust_THIRD_PARTY_NOTICES.md']
        assert all(n in names for n in required), 'Missing wheel code/license payload'
        native = [n for n in names if n.startswith('prusie/_native.') and n.endswith(('.so', '.pyd'))]
        assert len(native) == 1, 'Wheel must include one native extension'
        assert any(n.startswith(dist + '/licenses/') and n.endswith('/LICENSE') for n in names), 'Missing main license'
        assert any(n.startswith('prusie/_licenses/cargo/') for n in names), 'Missing Rust dependency licenses'
        notice = archive.read('prusie/_licenses/THIRD_PARTY_NOTICES.md').decode()
        assert 'Redistribution and use in source and binary forms' in notice
        assert 'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS' in notice
        assert 'GPL-2.0-or-later' in notice and 'GPL-3.0-or-later' in notice
        return dict(filename=path.name, sha256=sha(path), WHEEL=wheel, METADATA=metadata,
                    native_members=native,
                    native_sha256=hashlib.sha256(archive.read(native[0])).hexdigest(),
                    license_members=[n for n in names if 'license' in n.lower() or 'notices' in n.lower()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--sdist', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    args = parser.parse_args()
    wheel, sdist, work = args.wheel.resolve(), args.sdist.resolve(), args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    for name in ('tmp', 'cache', 'logs', 'unpacked', 'rebuilt'):
        (work / name).mkdir()
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    env.update(PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(work / 'tmp'),
               PIP_CACHE_DIR=str(work / 'cache'), CARGO_TARGET_DIR=str(work / 'target'))
    for name in ('PRUSIE_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[name] = '1'
    record = dict(passed=False, platform=platform.platform(), python=platform.python_version(),
                  wheel=inspect_wheel(wheel), sdist=dict(filename=sdist.name, sha256=sha(sdist)),
                  steps=[], builds=dict(RUSTFLAGS=env.get('RUSTFLAGS'),
                                       CARGO_ENCODED_RUSTFLAGS=env.get('CARGO_ENCODED_RUSTFLAGS'),
                                       CC=env.get('CC'), CFLAGS=env.get('CFLAGS')))

    def save():
        (work / 'artifact_checks.json').write_text(json.dumps(record, indent=2) + '\n')

    def run(label, argv, cwd=work, overrides=None):
        active_env = env | (overrides or {})
        log = work / 'logs' / (label + '.log')
        with log.open('w') as output:
            process = subprocess.run(list(map(str, argv)), cwd=cwd, env=active_env,
                                     stdout=output, stderr=subprocess.STDOUT)
        record['steps'].append(dict(label=label, argv=list(map(str, argv)), cwd=str(cwd),
                                    exit_code=process.returncode, log=str(log)))
        save()
        if process.returncode:
            raise RuntimeError(f'{label} failed; see {log}')

    try:
        with tarfile.open(sdist) as archive:
            archive.extractall(work / 'unpacked', filter='data')
        sources = list((work / 'unpacked').glob('*/pyproject.toml'))
        assert len(sources) == 1, 'Expected exactly one source project'
        source = sources[0].parent
        config = tomllib.loads(sources[0].read_text())
        version = config['project']['version']
        required = ['src/prusie/data/teaching/inputs.npz', 'src/prusie/data/teaching/reference.npz',
                    'src/prusie/data/teaching/metadata.json', 'src/prusie/data/teaching/LICENSE.txt',
                    'tools/generate_example.py', 'tools/generate_results.py', 'Cargo.toml', 'Cargo.lock', 'build.rs', 'rust-toolchain.toml',
                    'bindings/lib.rs', 'rust/core.rs', 'rust/optimizer.rs', 'rust/vector_math.c',
                    'rust/LICENSE-R', 'rust/LICENSE-susieR', 'rust/LICENSE-susieR-0.16.6',
                    'LICENSE', 'THIRD_PARTY_NOTICES.md', 'src/prusie/_licenses/THIRD_PARTY_NOTICES.md',
                    'tests/test_reference.py', 'tests/test_offline_example.py',
                    'examples/data/inputs.npz', 'examples/data/r_fits.npz',
                    'examples/data/LICENSE.GPL-3', 'examples/check_example.py']
        assert all((source / name).is_file() for name in required), 'Incomplete buildable sdist'
        record['sdist']['required_source_payload'] = {n: sha(source / n) for n in required}

        def install_and_check(label, artifact):
            runtime = work / (label + '-env')
            run(label + '-venv', [sys.executable, '-m', 'venv', runtime])
            python = runtime / 'bin/python'
            path = str(runtime / 'bin')
            # Explicit wheel and binary-only dependencies: pip cannot compile source.
            run(label + '-install', [python, '-m', 'pip', 'install', '--only-binary=:all:',
                                    'numpy==2.2.6', artifact], overrides={'PATH': path})
            probe = """import hashlib, importlib.metadata, json, pathlib, shutil, sys
import prusie
from prusie import _native
prefix=pathlib.Path(sys.prefix).resolve()
assert pathlib.Path(prusie.__file__).resolve().is_relative_to(prefix)
assert pathlib.Path(_native.__file__).resolve().is_relative_to(prefix)
assert not any(shutil.which(x) for x in ('rustc','cargo','maturin','R'))
assert prusie.__version__ == sys.argv[1] == importlib.metadata.version('prusie')
example=prusie.load_example()
fit=prusie.susie_rss(**example['inputs'], **example['parameters'])
assert fit.pip.shape == (500,) and fit.converged
print(json.dumps(dict(version=prusie.__version__,module=prusie.__file__,native=_native.__file__,native_sha256=hashlib.sha256(pathlib.Path(_native.__file__).read_bytes()).hexdigest(),rust_on_path=False)))
"""
            run(label + '-identity', [python, '-I', '-c', probe, version], overrides={'PATH': path})
            run(label + '-pip-check', [python, '-m', 'pip', 'check'], overrides={'PATH': path})
            run(label + '-example', [python, '-I', source / 'examples/check_example.py',
                                     '--output-dir', work / (label + '-example')], overrides={'PATH': path})
            run(label + '-quickstart', [python, '-I', source / 'examples/quickstart.py'], overrides={'PATH': path})

        install_and_check('wheel', wheel)
        run('sdist-rebuild', [sys.executable, '-m', 'pip', 'wheel', '--no-deps',
                              '--wheel-dir', work / 'rebuilt', source], cwd=work)
        rebuilt = list((work / 'rebuilt').glob('prusie-*.whl'))
        assert len(rebuilt) == 1
        record['rebuilt_wheel'] = inspect_wheel(rebuilt[0])
        install_and_check('rebuilt', rebuilt[0])
        if sys.platform.startswith('linux'):
            run('auditwheel', [sys.executable, '-m', 'auditwheel', 'show', wheel])
            run('auditwheel-rebuilt', [sys.executable, '-m', 'auditwheel', 'show', rebuilt[0]])
        record['passed'] = True
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
    save()
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
