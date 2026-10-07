#!/usr/bin/env python3
"""Export plot-ready records from separately executed toy example fits."""
import argparse,hashlib,json
from pathlib import Path
from compare_fits import compare


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--prusie',type=Path,required=True);ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    p=json.loads(a.prusie.read_text());r=json.loads(a.reference.read_text());comparison=compare(p,r)
    if not comparison['passed']:raise ValueError('Toy example comparison failed: '+json.dumps(comparison))
    assert p['case_id']==r['case_id']=='independent_teaching'
    record=dict(case_id=p['case_id'],source='Separately executed independent toy example fits',pip=p['fit']['pip'],cs=p['fit']['cs'],
        niter=p['niter'],converged=p['converged'],parameters=p['parameters'],comparison=comparison,
        raw_sha256={k:hashlib.sha256(path.read_bytes()).hexdigest() for k,path in [('prusie',a.prusie),('susieR',a.reference)]},
        input_sha256=hashlib.sha256(a.inputs.read_bytes()).hexdigest(),native_sha256=p['environment']['native_sha256'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':main()
