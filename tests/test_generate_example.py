"""Frozen integrity and portable numerical regeneration are separate contracts."""
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('generate_example',ROOT/'tools/generate_example.py')
generator=importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)
DATA=ROOT/'src/prusie/data/teaching'


@pytest.fixture(scope='module')
def generated():
    return generator.generate()


def test_frozen_integrity_and_regeneration(generated):
    generator.check_frozen(DATA,*generated)


def test_archive_encoding_does_not_define_numerical_reproducibility(generated):
    arrays,_=generated
    stored=io.BytesIO();compressed=io.BytesIO()
    np.savez(stored,**arrays);np.savez_compressed(compressed,**arrays)
    assert stored.getvalue()!=compressed.getvalue()
    stored.seek(0);compressed.seek(0)
    with np.load(stored,allow_pickle=False) as a,np.load(compressed,allow_pickle=False) as b:
        assert all(v==0 for v in generator.validate_arrays(dict(a),dict(b)).values())


@pytest.mark.parametrize('name',['inputs.npz','metadata.json','LICENSE.txt','checksums.json'])
def test_frozen_corruption_is_rejected_even_with_valid_regeneration(name,tmp_path,generated):
    data=tmp_path/'teaching';shutil.copytree(DATA,data)
    path=data/name;path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='Frozen'):
        generator.check_frozen(data,*generated)


def test_rehashing_altered_frozen_data_cannot_bypass_integrity(tmp_path,generated):
    import hashlib
    data=tmp_path/'teaching';shutil.copytree(DATA,data)
    path=data/'metadata.json';path.write_text(path.read_text().replace('20261007','20261008'))
    hashes=json.loads((data/'checksums.json').read_text())
    hashes['metadata.json']=hashlib.sha256(path.read_bytes()).hexdigest()
    (data/'checksums.json').write_text(json.dumps(hashes,indent=2)+'\n')
    with pytest.raises(ValueError,match='manifest'):
        generator.check_frozen(data,*generated)


@pytest.mark.parametrize('name', ['z','R','bhat','shat','true_beta','position'])
def test_numeric_change_is_rejected(name,generated):
    arrays,metadata=generated;changed={k:v.copy() for k,v in arrays.items()}
    changed[name].flat[0]+=1 if name=='position' else 1e-10
    with pytest.raises(ValueError,match=name):
        generator.check_frozen(DATA,changed,metadata)


@pytest.mark.parametrize('mutation',['dtype','shape','nan','keys'])
def test_structural_and_nonfinite_changes_rejected(mutation,generated):
    arrays,metadata=generated;changed={k:v.copy() for k,v in arrays.items()}
    if mutation=='dtype':changed['z']=changed['z'].astype(np.float32)
    elif mutation=='shape':changed['z']=changed['z'][:-1]
    elif mutation=='nan':changed['z'][0]=np.nan
    else:changed.pop('R')
    with pytest.raises(ValueError):generator.check_frozen(DATA,changed,metadata)


@pytest.mark.parametrize('name,value',[('seed',20261008),('LD_population_rho',.8),('numpy_version','unrecorded')])
def test_metadata_changes_rejected(name,value,generated):
    arrays,metadata=generated
    with pytest.raises(ValueError,match='metadata'):
        generator.check_frozen(DATA,arrays,metadata|{name:value})


@pytest.mark.skipif(platform.machine().lower() not in ('x86_64','amd64'),reason='x86 OpenBLAS kernel regression')
@pytest.mark.parametrize('core',['Haswell','Sandybridge'])
def test_generation_across_openblas_kernels(core):
    # Run in a fresh process: OpenBLAS reads its core selector when it loads.
    env=os.environ|{'OPENBLAS_CORETYPE':core,'OPENBLAS_NUM_THREADS':'1'}
    subprocess.run([sys.executable,str(ROOT/'tools/generate_example.py'),'--check'],
                   env=env,check=True,capture_output=True,text=True)
