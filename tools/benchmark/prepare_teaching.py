#!/usr/bin/env python3
"""Prepare portable binary inputs for a live-R teaching-example comparison."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def prepare(data,output):
    output.mkdir(parents=True,exist_ok=False)
    metadata=json.loads((data/'metadata.json').read_text())
    with np.load(data/'inputs.npz',allow_pickle=False) as a:
        np.asarray(a['z'],dtype='<f8').tofile(output/'z.bin')
        np.asarray(a['R'],dtype='<f8').tofile(output/'R.bin')
        np.asarray(a['R'].T,dtype='<f8').tofile(output/'R_F.bin')
    (output/'ids.txt').write_text('\n'.join(metadata['variant_ids'])+'\n')
    case=dict(case_id='independent_teaching',n_variants=metadata['p'],parameters=dict(n=metadata['n'],**metadata['parameters']),
              z_path='z.bin',R_path='R.bin',R_column_major_path='R_F.bin',variant_ids_path='ids.txt')
    (output/'case.json').write_text(json.dumps(case,indent=2)+'\n')
    (output/'checksums.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()},indent=2)+'\n')
    return output/'case.json'

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[2]/'src/prusie/data/teaching');p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();print(prepare(a.data,a.output_dir))
