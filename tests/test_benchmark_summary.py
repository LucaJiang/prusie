"""Scientific summary definitions and outcome-independent inclusion."""
import copy
import importlib.util
import math
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('summary',Path(__file__).resolve().parents[1]/'tools/benchmark/summary.py')
summary=importlib.util.module_from_spec(spec);spec.loader.exec_module(summary)


def case(name='region01/gwas', p=2., r=10.):
    return dict(case_id=name,stage='C',group='GWAS',max_iter=100,n_variants=500,settings_match=True,
        backends={k:dict(status='ok',memory_status='ok',repeat_seconds=[t*.5,t,t,t,t,t,t*2],
                         peak_memory_mib=m,niter=100,converged=False,cs_count=0)
                  for k,t,m in [('prusie',p,30.),('susieR',r,100.)]})


def test_grouping_requires_independent_metadata():
    assert summary.classify('r/gwas','real',{'trait':'gwas'})=='GWAS'
    assert summary.classify('r/eqtl','real',{'trait':'eqtl'})=='eQTL'
    assert summary.classify('D1','public',{})=='Synthetic regression'
    with pytest.raises(ValueError):summary.classify('r/gwas','real',{'trait':'eqtl'})


def test_iteration_cap_no_cs_and_numeric_difference_do_not_filter_timing():
    row=case();row['numerical']={'outputs_valid':False,'pip_within_1e5':False}
    result=summary.paired(row)
    assert result['timing_eligible'] and result['speedup']==5
    assert result['memory_ratio']==100/30
    assert result['memory_reduction']==.7
    group=summary.summarize([result])['GWAS']
    assert group['iteration_limit_cases']==['region01/gwas']
    assert group['no_cs_analyses']=={'prusie':1,'susieR':1}


@pytest.mark.parametrize('change',[{'stage':'A'},{'stage':'B'},{'stage':'D'}, {'settings_match':False}, {'measurement_invalid':True}])
def test_stage_and_settings_selection(change):
    row=case();row.update(change);assert not summary.paired(row)['timing_eligible']


@pytest.mark.parametrize('problem',['failed','missing','zero','nan','few_repeats'])
def test_failed_and_missing_measurements_are_retained(problem):
    row=case();b=row['backends']['prusie']
    if problem=='failed':b['status']='error'
    elif problem=='missing':row['backends'].pop('prusie')
    elif problem=='zero':b['repeat_seconds'][0]=0.
    elif problem=='nan':b['repeat_seconds'][0]=float('nan')
    else:b['repeat_seconds']=b['repeat_seconds'][:6]
    result=summary.paired(row)
    assert result['case_id']=='region01/gwas' and not result['timing_eligible'] and result['speedup'] is None


def test_paired_geomean_is_not_ratio_of_group_medians():
    rows=[summary.paired(case('a/gwas',1.,100.)),summary.paired(case('b/gwas',100.,200.)),summary.paired(case('c/gwas',10.,10.))]
    result=summary.summarize(rows)['GWAS']
    assert result['runtime_speedup_geometric_mean']==pytest.approx(200**(1/3))
    assert result['runtime_speedup_geometric_mean']!=result['runtime_seconds']['susieR']['median']/result['runtime_seconds']['prusie']['median']


def test_memory_missing_does_not_remove_valid_timing():
    row=case();row['backends']['susieR']['peak_memory_mib']=None
    result=summary.paired(row)
    assert result['timing_eligible'] and not result['memory_eligible'] and result['memory_ratio'] is None


def test_true_array_maximum_and_trace_includes_iteration_cap():
    rows=[]
    for i,delta in enumerate([1e-9,1e-6]):
        row=summary.paired(case(f'region{i}/gwas'))
        errors={k:summary.array_error([0.,delta,0.],[0.,0.,0.]) for k in summary.FIELDS}
        row['numerical']=dict(outputs_valid=True,errors=errors,sources={'prusie':'raw p','susieR':'raw r'})
        rows.append(row)
    error=summary.summarize(rows)['GWAS']['errors']['pip']
    assert error['max_abs_error']==1e-6 and error['case_id']=='region1/gwas'
    assert error['index_0based']==[1] and error['valid_case_count']==2
    assert summary.array_error([0],[0,1])['max_abs_error'] is None
    assert summary.array_error([0],[0])['max_abs_error']==0
