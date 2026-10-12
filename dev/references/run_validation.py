"""Run the bounded independent panel inside the project's resource wrapper.

No benchmarking and no dependency installation. Arguments name the output path,
verified Rscript and pinned library. Outputs retain raw unpermuted components.
"""
import argparse
import copy
import hashlib
import json
import subprocess
import sys
import warnings
import shutil
from pathlib import Path

import numpy as np
import scipy
import prusie
from validation_panel import cases
from restricted_numpy_susie import susie_ss_ref, susie_rss_ref, credible_sets_ref

FIELDS = ('pip','alpha','mu','mu2','V','sigma2','elbo','KL','lbf','lbf_variable')


def serial(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,Path): return str(value)
    raise TypeError(type(value).__name__)


def write(path,obj):
    path.write_text(json.dumps(obj,default=serial,indent=2,allow_nan=False)+'\n')


def field_errors(a,b):
    output={}
    for key in FIELDS:
        left=np.asarray(a[key],float); right=np.asarray(b[key],float)
        if left.size != right.size:
            output[key]=dict(shape_equal=False,left_shape=left.shape,right_shape=right.shape,
                             max_abs_error=None,passed=False)
            continue
        right=right.reshape(left.shape)  # JSON unboxes length-one vectors only.
        error=float(np.max(np.abs(left-right)))
        output[key]=dict(shape_equal=True,max_abs_error=error,
            passed=bool(np.allclose(left,right,atol=1e-5 if key=='pip' else 1e-6,
                                   rtol=0 if key=='pip' else 1e-6)))
    return output


def native_record(f):
    names=('min_abs_corr','mean_abs_corr','median_abs_corr')
    cs=[dict(component=int(c),members=m.tolist(),coverage=float(v),
             purity=[float(f.sets['purity'][key][i]) for key in names])
        for i,(c,m,v) in enumerate(zip(f.sets['cs_index'],f.sets['cs'],f.sets['coverage']))]
    return dict(status='returned',fields={k:getattr(f,k) for k in FIELDS},
        niter=f.niter,converged=f.converged,stop_reason=f.convergence['stop_reason'],sets=cs)


def cs_compare(left,right):
    # Original component IDs, no optimized reassignment and no PIP sorting.
    a={x['component']:x for x in left}
    b={x['component']:x for x in right}
    equal=set(a)==set(b)
    coverage=[]; purity=[]
    for key in set(a)&set(b):
        equal &= sorted(np.atleast_1d(a[key]['members']).tolist())==sorted(np.atleast_1d(b[key]['members']).tolist())
        coverage.append(abs(a[key]['coverage']-b[key]['coverage']))
        purity.append(float(np.max(np.abs(np.asarray(a[key]['purity'])-b[key]['purity']))))
    return dict(members_and_components_equal=bool(equal),
        max_coverage_error=max(coverage,default=0.),max_purity_error=max(purity,default=0.),
        native_deduplicated=len({tuple(x['members']) for x in left})==len(left),
        R_deduplicated=len({tuple(np.atleast_1d(x['members'])) for x in right})==len(right))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rscript',required=True)
    parser.add_argument('--rlib',required=True)
    parser.add_argument('--rdeps',required=True)
    parser.add_argument('--reuse-r',type=Path,
        help='Reuse actual raw R results only when the serialized case panel is byte-identical')
    parser.add_argument('--summary-only',action='store_true',
                        help='Reuse complete raw files after an output-summary failure; never refit')
    a=parser.parse_args()
    if a.summary_only:
        summarize(a.output,Path(__file__).resolve().parents[2])
        return
    a.output.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[2]
    panel=cases(repo)
    from prusie.datasets import load_convergence_example
    ex=load_convergence_example()
    for refine in (False,True):
        panel.append(dict(name='official_refinement' if refine else 'official_baseline',
            kind='ss',args=dict(ex['inputs'],**ex['parameters'],refine=refine,n_purity=10000),
            numpy_supported=False,expected_error=False,refinement=True))
    write(a.output/'cases.json',panel)
    command=[a.rscript,'--vanilla',str(Path(__file__).with_name('validate_pinned_R.R')),
             str(a.output/'cases.json'),str(a.output/'raw_R.json'),a.rlib,a.rdeps]
    write(a.output/'R_command.json',command)
    if a.reuse_r:
        if (a.reuse_r/'cases.json').read_bytes() != (a.output/'cases.json').read_bytes():
            raise ValueError('Cannot reuse R output: input panel differs')
        shutil.copyfile(a.reuse_r/'raw_R.json',a.output/'raw_R.json')
        write(a.output/'R_reuse.json',dict(source=str(a.reuse_r.resolve()),
            cases_sha256=hashlib.sha256((a.output/'cases.json').read_bytes()).hexdigest(),
            R_sha256=hashlib.sha256((a.output/'raw_R.json').read_bytes()).hexdigest(),
            rationale='Native artifact changed; fixed reference and identical inputs do not require refitting R'))
    else:
        with (a.output/'R.log').open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(f'R failed, see {a.output}/R.log')
    reference=json.loads((a.output/'raw_R.json').read_text())
    raw_native={}; raw_numpy={}; comparisons=[]; refinement_fits={}
    for case in panel:
        name=case['name']; args=case['args']; r=reference['cases'][name]
        with warnings.catch_warnings(record=True) as caught:
            try:
                f=(prusie.susie_rss if case['kind']=='rss' else prusie.susie_suff_stat)(**args,retain_data=True)
                record=native_record(f)
                if case.get('refinement'):refinement_fits[name]=f
            except ValueError as exc:
                record=dict(status='error',error=str(exc))
        record['warnings']=[str(w.message) for w in caught]
        raw_native[name]=record
        row=dict(name=name,expected_error=case['expected_error'],
                 native_status=record['status'],R_status=r['status'])
        if case['expected_error']:
            row['passed']=(record['status']==r['status']=='error' and
                           'prior variance is unreasonably large' in record['error'] and
                           'prior variance is unreasonably large' in r['error'])
        elif record['status']==r['status']=='returned':
            row['fields']=field_errors(record['fields'],r['fields'])
            row['niter']=dict(native=record['niter'],R=r['niter'])
            row['converged']=dict(native=record['converged'],R=r['converged'])
            row['stop_reason']=dict(native=record['stop_reason'],R=r['stop_reason'])
            row['cs']=cs_compare(record['sets'],r['sets'])
            row['validity']=bool(all(np.isfinite(record['fields'][k]).all() for k in FIELDS)
                and np.allclose(f.alpha.sum(1),1,atol=1e-12,rtol=0)
                and np.all((f.pip>=0)&(f.pip<=1)) and np.all((f.alpha>=0)&(f.alpha<=1)))
            row['passed']=row['fields']['pip']['passed'] and row['validity']
            if case['numpy_supported']:
                kw=copy.deepcopy(args)
                for key in ('n_purity','check_prior'):kw.pop(key,None)
                if 'estimate_prior_method' in kw:kw['method']=kw.pop('estimate_prior_method')
                nf=(susie_rss_ref if case['kind']=='rss' else susie_ss_ref)(**kw)
                raw_numpy[name]=nf
                row['numpy_fields']=field_errors(record['fields'],nf)
                row['numpy_niter']=nf['niter']; row['numpy_converged']=nf['converged']
                row['numpy_pip_passed']=row['numpy_fields']['pip']['passed']
                corr=args.get('R')
                if corr is None:
                    q=args['XtX']; sd=np.sqrt(np.diag(q));corr=q/sd[:,None]/sd[None,:]
                ncs=credible_sets_ref(nf['alpha'],nf['V'],corr)
                row['numpy_cs_members_equal']={x[0]:x[1] for x in ncs}=={
                    x['component']:x['members'] for x in record['sets']}
                row['numpy_cs_coverage_error']=max((abs(x[2]-next(y['coverage'] for y in record['sets'] if y['component']==x[0]))
                    for x in ncs if any(y['component']==x[0] for y in record['sets'])),default=0.)
                row['numpy_cs_min_purity_error']=max((abs(x[3]-next(y['purity'][0] for y in record['sets'] if y['component']==x[0]))
                    for x in ncs if any(y['component']==x[0] for y in record['sets'])),default=0.)
        else:row['passed']=False
        comparisons.append(row)
    # Actual fresh R refinement is reported separately from standard fits.
    base=refinement_fits['official_baseline']; refined=refinement_fits['official_refinement']
    before=copy.deepcopy(base.as_dict())
    independent=prusie.refine_fit(base,prusie.RefinementConfig(inner_tol=.001))
    objectives={key:prusie.evaluate_objective(fit) for key,fit in
        [('baseline',base),('refine_true',refined),('refine_fit',independent)]}
    refinement=dict(objectives=objectives,
        objective_gain=objectives['refine_true']['elbo']-objectives['baseline']['elbo'],
        original_unchanged=all(np.array_equal(getattr(base,k),before[k]) for k in FIELDS),
        original_prior_restored=np.array_equal(base.params['prior_weights'],refined.params['prior_weights']),
        refine_true_refine_fit_pip_error=float(np.max(abs(refined.pip-independent.pip))),
        history=refined.refinement)
    write(a.output/'raw_native.json',raw_native)
    write(a.output/'raw_numpy.json',raw_numpy)
    write(a.output/'comparisons.json',comparisons)
    write(a.output/'refinement.json',refinement)
    summarize(a.output,repo)


def summarize(output,repo):
    comparisons=json.loads((output/'comparisons.json').read_text())
    reference=json.loads((output/'raw_R.json').read_text())
    raw_numpy=json.loads((output/'raw_numpy.json').read_text())
    compute_files=['rust/core.rs','bindings/lib.rs','src/prusie/api.py','src/prusie/refinement.py','src/prusie/diagnostics.py']
    native_extension=Path(prusie._native.__file__)
    summary=dict(prusie_version=prusie.__version__,prusie_import=prusie.__file__,
        Python=sys.version,NumPy=np.__version__,SciPy=scipy.__version__,
        worktree_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        native_extension=str(native_extension),
        native_extension_sha256=hashlib.sha256(native_extension.read_bytes()).hexdigest(),
        compute_sha256={key:hashlib.sha256((repo/key).read_bytes()).hexdigest() for key in compute_files},
        cases=len(comparisons),R_executed=True,R_reference=reference['reference'],
        passed=sum(x['passed'] for x in comparisons),
        failures=[x['name'] for x in comparisons if not x['passed']],
        numpy_cases=len(raw_numpy),numpy_failures=[x['name'] for x in comparisons if x.get('numpy_pip_passed') is False],
        niter_differences=[x['name'] for x in comparisons if 'niter' in x and x['niter']['native']!=x['niter']['R']],
        convergence_differences=[x['name'] for x in comparisons if 'converged' in x and x['converged']['native']!=x['converged']['R']],
        cs_members_differences=[x['name'] for x in comparisons if 'cs' in x and not x['cs']['members_and_components_equal']],
        cs_deduplication_failures=[x['name'] for x in comparisons if 'cs' in x and not(x['cs']['native_deduplicated'] and x['cs']['R_deduplicated'])],
        cs_max_coverage_error=max((x['cs']['max_coverage_error'] for x in comparisons if 'cs' in x),default=0),
        cs_max_purity_error=max((x['cs']['max_purity_error'] for x in comparisons if 'cs' in x),default=0),
        diagnostic_field_failures={x['name']:[k for k,v in x['fields'].items() if not v['passed']]
                                   for x in comparisons if 'fields' in x and not all(v['passed'] for v in x['fields'].values())},
        numpy_diagnostic_field_failures={x['name']:[k for k,v in x['numpy_fields'].items() if not v['passed']]
                                   for x in comparisons if 'numpy_fields' in x and not all(v['passed'] for v in x['numpy_fields'].values())},
        max_abs_error={key:max((x['fields'][key]['max_abs_error'] or 0 for x in comparisons if 'fields' in x),default=0) for key in FIELDS},
        numpy_max_abs_error={key:max((x['numpy_fields'][key]['max_abs_error'] or 0 for x in comparisons if 'numpy_fields' in x),default=0) for key in FIELDS},
        field_tolerances=dict(pip='atol=1e-5, rtol=0',diagnostic_fields='atol=1e-6, rtol=1e-6'),
        note='No permutations applied. R stop reason can be null; native reason has no identical R field. Refinement compared separately.')
    write(output/'summary.json',summary)
    print(json.dumps(summary,default=serial,indent=2))
    if summary['failures'] or summary['numpy_failures']:raise SystemExit(1)

if __name__=='__main__':main()
