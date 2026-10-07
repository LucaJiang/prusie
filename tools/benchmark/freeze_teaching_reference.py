#!/usr/bin/env python3
"""Freeze a separately executed teaching reference; never touches older fixtures."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def freeze(record_path, destination):
    record=json.loads(record_path.read_text())
    assert record['status']=='ok' and record['validity']['passed']
    assert record['environment']['version']=='0.16.6'
    assert record['case_id']=='independent_teaching'
    fit=record['fit'];fields=('pip','alpha','mu','mu2','lbf_variable','lbf','V','sigma2','elbo','KL')
    np.savez_compressed(destination/'reference.npz',**{k:np.asarray(fit[k]) for k in fields})
    env=record['environment']
    meta=dict(case_id=record['case_id'],susieR_version='0.16.6',susieR_commit='8e56a8e038e989856d106d9ca5175cc664fea9d2',
        raw_record_sha256=hashlib.sha256(record_path.read_bytes()).hexdigest(),R=env['R'],
        parameters=record['parameters'],niter=fit['niter'],converged=fit['converged'],cs=fit['cs'],
        variant_ids=fit['variant_ids'],requested_coverage=fit['requested_coverage'],
        reference_arrays_sha256=hashlib.sha256((destination/'reference.npz').read_bytes()).hexdigest(),
        source='New independent teaching fit executed by tools/benchmark/fine_mapping.R',
        regeneration='Prepare inputs using tools/benchmark/prepare_teaching.py; run fine_mapping.R in a new output directory; inspect differences before changing this reference.')
    (destination/'reference.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Frozen new reference:',fit['niter'],'iterations;',len(fit['cs']),'credible sets')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--record',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args();freeze(a.record,a.destination)
