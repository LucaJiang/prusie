"""Convert the R export to portable float64 NPZ and explicit JSON, without fitting.

Usage: python convert_exports.py RAW_DIRECTORY BUNDLE_DIRECTORY
The scientific arrays in inputs.npz are bit-preserving conversions of writeBin
little-endian float64. Text TSV columns are for inspection; NPZ is authoritative.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('raw',type=Path);p.add_argument('bundle',type=Path)
a=p.parse_args();a.bundle.mkdir(parents=True,exist_ok=True)
def read(name):return json.loads((a.raw/name).read_text())
def write(name,value):(a.bundle/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
meta=read('metadata.json'); inputs={}
for k,d in meta.items():
 for field in ('beta','varbeta','MAF','LD'):
  v=np.fromfile(a.raw/f'{k}_{field}.f64',dtype='<f8')
  if field=='LD':v=v.reshape((d['m'],d['m']),order='F').copy()
  inputs[f'{k}_{field}']=v
 assert np.isfinite(inputs[f'{k}_LD']).all()
 assert np.array_equal(inputs[f'{k}_LD'],inputs[f'{k}_LD'].T)
 assert (inputs[f'{k}_varbeta']>0).all()
 (a.bundle/f'{k}.tsv').write_bytes((a.raw/f'{k}.tsv').read_bytes())
np.savez_compressed(a.bundle/'inputs.npz',**inputs)
fm=read('r_fits.json');arrays={}
for key,case in fm.items():
 raw=read(f'arrays_{key}.json')
 for field,v in raw.items():
  arrays[f'{key}_{field}']=np.asarray(v,dtype=np.float64)
  if field not in ('alpha','mu','mu2','lbf_variable','sigma2'):
   arrays[f'{key}_{field}']=np.atleast_1d(arrays[f'{key}_{field}'])
  assert np.isfinite(arrays[f'{key}_{field}']).all(),(key,field)
 for cs in case['cs']:
  assert abs(float(arrays[f'{key}_alpha'][cs['component_id_0based'],cs['members_0based']].sum())-cs['coverage'])<1e-12
np.savez_compressed(a.bundle/'r_fits.npz',**arrays)
write('metadata.json',meta);write('r_fits.json',fm);write('parameters.json',read('parameters.json'))
rc=read('r_coloc.json')
for key,case in rc['cases'].items():
 if key.startswith('abf_'):
  case['result']['summary']=dict(zip(['nsnps']+[f'PP.H{i}.abf' for i in range(5)],case['result']['summary']))
  case['result']['priors']=dict(zip(['p1','p2','p12'],case['result']['priors']))
write('r_coloc.json',rc)
with (a.bundle/'variants.tsv').open('w',newline='') as f:
 w=csv.writer(f,delimiter='\t');w.writerow(['dataset','array_index_0based','upstream_position_1based','variant_id','ld_row_0based','ld_column_0based'])
 for k,d in meta.items():
  for i,(s,pos) in enumerate(zip(d['snp'],d['position'])):w.writerow([k,i,pos,s,i,i])
# No private filesystem paths in the public environment record.
env=read('reference_environment.private.json')
write('reference_environment.json',dict(R=env['R'],platform=env['platform'],
 packages={k:v['version'] for k,v in env['namespaces'].items()},
 BLAS='Netlib reference BLAS (libblas.so.3.9.0), one numerical thread',
 susie_rss_formals=env['susie_rss_formals'],
 note='Private execution paths are retained by the validation run, not required at runtime.'))
print(json.dumps({'datasets':len(meta),'snp_count_each':500,'reference_cases':list(fm),'inputs_bytes':(a.bundle/'inputs.npz').stat().st_size,'fits_bytes':(a.bundle/'r_fits.npz').stat().st_size}))
