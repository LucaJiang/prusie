#!/usr/bin/env python3
"""Import retained full-call measurement records; never refit or modify inputs.

This entrance requires the original controlled input/evidence archive. The public
result-only regeneration entrance is tools/generate_results.py.
"""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path
import numpy as np
from summary import classify, paired, array_error, FIELDS, summarize


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, value): p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evidence-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--private-receipt', type=Path, required=True)
    args=ap.parse_args(); root=args.evidence_root.resolve(); hashes={}
    def tracked(p):
        p=Path(p); hashes[str(p)]=sha(p); return read(p)
    manifest=tracked(root/'shared/measurement_manifest.prefrozen.json')
    metadata={}; regions=[]
    for region in manifest['regions']:
        qc=tracked(region['qc_path']); rg=qc['region']; assert qc['passed'] and region['qc_passed']
        assert region['region']==rg['id'] and rg['rss_snps']==region['snps']
        regions.append(rg)
        for trait in region['trait_cases']:
            assert trait['trait'] in qc['traits'] and trait['n_variants']==region['snps']
            metadata[trait['case_id']]=dict(trait=trait['trait'], n=trait['n'], region=region['region'], gene=rg['gene'])
    genes={g:f'gene{j+1:03d}' for j,g in enumerate(sorted({r['gene'] for r in regions}))}
    overlaps=[(a['id'],b['id']) for i,a in enumerate(regions) for b in regions[i+1:] if a['chromosome']==b['chromosome'] and max(a['start'],b['start'])<=min(a['end'],b['end'])]
    tasks=tracked(root/'shared/tasks_C_real.prefrozen.json')+tracked(root/'shared/tasks_C_public.prefrozen.json')
    cases=[]; environments={}; source_receipts=[]
    for task in tasks:
        if task['stage']!='C': continue
        cid=task['case_id']; panel=task.get('panel','real'); case=tracked(task['case_path']); params=case['parameters']
        meta=metadata[cid] if panel=='real' else {}
        row=dict(case_id=cid,stage='C',panel=panel,group=classify(cid,panel,meta),n_variants=case['n_variants'],
                 max_iter=params['max_iter'],parameters=params,settings_match=True,measurement_invalid=False,backends={})
        if meta:
            assert params['n']==meta['n']; row['region_id']=meta['region'];row['gene_id_anonymous']=genes[meta['gene']]
        validroot=root/('raw/validation' if panel=='real' else 'raw/validation_public')/'C'/cid
        fits={}; sources={}
        for old,new in [('Python','prusie'),('R','susieR')]:
            path=validroot/f'{old}.json'
            fits[new]=tracked(path) if path.exists() else {'status':'missing'}
            sources[new]={'record':str(path.relative_to(root)), 'sha256':hashes.get(str(path))}
        a,b=(fits[k] for k in ('prusie','susieR'))
        outputs_valid=all(f.get('status')=='ok' and f.get('validity',{}).get('passed') for f in fits.values())
        numerical=dict(outputs_valid=outputs_valid,sources=sources,errors={},status='missing_or_invalid')
        if outputs_valid:
            assert a['parameters']==b['parameters']==params
            assert a['fit']['variant_ids']==b['fit']['variant_ids']
            numerical.update(status='valid',errors={k:array_error(a['fit'][k],b['fit'][k]) for k in FIELDS},
                validity={k:f['validity'] for k,f in fits.items()},credible_sets={},credible_set_differences=[])
            for name,f in fits.items(): numerical['credible_sets'][name]=f['fit']['cs']
            ac={x['component_id_0based']:x for x in a['fit']['cs']};bc={x['component_id_0based']:x for x in b['fit']['cs']}
            for c in sorted(ac.keys()|bc.keys()):
                x,y=ac.get(c),bc.get(c)
                comp=dict(component_id_0based=c,prusie_present=x is not None,susieR_present=y is not None)
                if x is not None and y is not None:
                    comp.update(members_match=sorted(x['members_0based'])==sorted(y['members_0based']),
                                posterior_mass_abs_error=abs(x['coverage']-y['coverage']),
                                purity_abs_error={k:abs(x['purity'][k]-y['purity'][k]) for k in x['purity']})
                numerical['credible_set_differences'].append(comp)
            numerical['pip_within_1e5']=numerical['errors']['pip']['max_abs_error']<=1e-5
        row['numerical']=numerical
        folder=root/'raw/formal/C'/cid
        complete=tracked(folder/'COMPLETE.json') if (folder/'COMPLETE.json').exists() else {}
        attempt=complete.get('selected_attempt')
        row['selected_attempt']=attempt
        selected=next((x for x in complete.get('attempts',[]) if x['attempt']==attempt),{})
        row['measurement_invalid']=bool(selected.get('pressure_contaminated') or selected.get('process_failed'))
        for old,new in [('Python','prusie'),('R','susieR')]:
            timing={};memory={}
            if attempt is not None:
                for mode in ('time','memory'):
                    path=folder/f'attempt{attempt:02d}'/f'{old}.{mode}.json'
                    record=tracked(path) if path.exists() else {}
                    source_receipts.append(dict(case_id=cid,backend=new,mode=mode,record=str(path.relative_to(root)),sha256=hashes.get(str(path))))
                    if mode=='time':timing=record
                    else:memory=record
            f=fits[new]; samples=timing.get('samples',[])
            if timing:row['settings_match'] &= timing.get('parameters')==params
            if memory:row['settings_match'] &= memory.get('parameters')==params
            def clean(v):
                if isinstance(v,dict):return {k:clean(x) for k,x in v.items() if k not in ('maps','loaded_maps','loaded_libraries','pid','process_status','rlimit_as')}
                if isinstance(v,list):return [clean(x) for x in v]
                if isinstance(v,str):return re.sub(r'/(?:home|opt|usr|lib|lib64|tmp|var)/[^\s\"\']+', lambda m: Path(m[0]).name, v)
                return v
            environment=clean(timing.get('environment',f.get('environment',{})))
            eid=hashlib.sha256(json.dumps(environment,sort_keys=True).encode()).hexdigest()[:16];environments[eid]=environment
            row['backends'][new]=dict(status=timing.get('status','not_timed'),memory_status=memory.get('status','not_measured'),
                repeat_seconds=[s.get('time_seconds') for s in samples if not s.get('warmup')],warmup_seconds=[s.get('time_seconds') for s in samples if s.get('warmup')],
                repeat_fit_status=[{k:s.get(k) for k in ('repeat','niter','converged','cs_count')} for s in samples if not s.get('warmup')],
                peak_memory_mib=memory.get('peak_rss_mib'),niter=timing.get('niter',f.get('niter')),
                converged=timing.get('converged',f.get('converged')),cs_count=(samples[-1].get('cs_count') if samples else len(f.get('fit',{}).get('cs',[])) if f.get('status')=='ok' else None),environment_id=eid)
        row=paired(row);row['status']='completed' if all(f.get('status')=='ok' for f in fits.values()) else 'missing_or_failed'
        if panel=='real' and any(not row['backends'][k]['converged'] for k in ('prusie','susieR')):
            assert all(row['backends'][k]['niter']==100 for k in ('prusie','susieR'))
        cases.append(row)
    artifact=tracked(root/'reports/artifact_manifest.json'); layout=tracked(root/'reports/environment_and_layout.json')
    protocol=tracked(root/'shared/protocol.json');clock=tracked(root/'shared/protocol_clock_final.prefrozen.json')
    freeze=tracked(root/'shared/formal_freeze_receipt.json')
    # Preserve all measured identities, with the old binary explicitly identified.
    provenance=dict(measurement_date='2026-10-06',analysis='Reanalysis of retained fits and measurements; no real-panel refitting',
        prusie_version='0.2.3rc4',susieR_version='0.16.6',susieR_commit='8e56a8e038e989856d106d9ca5175cc664fea9d2',
        measured_native_sha256=artifact['measured_native_SHA256'],historical_source_sha256=next(x['source_sha256'] for x in artifact['artifacts'] if x['repository']=='prusie'),
        measured_wheel_sha256=freeze['measured_wheel']['sha256'],
        measured_sdist_sha256=freeze['corresponding_original_sdist']['sha256'],
        measured_source_files=freeze['current_numerical_sources_match_measured_sdist'],
        hardware_cpu=next(x.split(':',1)[1].strip() for x in layout['cpu'].splitlines() if x.startswith('Model name:')),
        timing='Complete susie_rss public call including validation, transformation, IBSS and posterior/CS summaries; imports, input reads, explicit GC and exports excluded',
        warmups=1,retained_repeats=7,numerical_threads=1,backends='Sequential per case; alternating order; exclusive two-slot measurement',
        memory='Separate fresh process for one case/backend; full-process peak RSS including interpreter, imports, inputs and retained result; Linux KiB divided by 1024',
        inclusion='Every successful, settings-matched pair with seven finite positive times; independent of PIP error, convergence, CS and speed',
        input_scope='Real-panel records are anonymized summaries, not redistributable association/LD inputs',
        real_panel=dict(regions=len(regions),GWAS_analyses=100,eQTL_analyses=100,distinct_genes=len(genes),overlapping_interval_pairs=len(overlaps),
                        dependencies='One GWAS study and one monocyte eQTL study; shared samples and LD reference panels across loci',
                        GWAS='FinnGen R10 K11_IBD_STRICT; n=412181',eQTL='BLUEPRINT/eQTL Catalogue QTD000021 monocytes; n=191',
                        LD='1000 Genomes GRCh38, FIN 99 for GWAS and EUR 525 for eQTL, recorded founders; external signed r'),
        raw_measurement_sources=source_receipts)
    write(args.output,dict(schema_version=1,provenance=provenance,environments=environments,cases=cases))
    write(args.private_receipt,dict(input_hashes=hashes,region_metadata=regions,overlapping_interval_pairs=overlaps,
        metadata_group_verified_count=len(metadata),summary=summarize(cases)))
    print(json.dumps({k:{x:v for x,v in val.items() if x not in ('errors',)} for k,val in summarize(cases).items()},indent=2))

if __name__=='__main__':main()
