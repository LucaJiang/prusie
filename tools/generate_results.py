#!/usr/bin/env python3
"""Regenerate fine-mapping tables, records and vector figures from retained data."""
from __future__ import annotations
import argparse,csv,io,json,re,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator, NullFormatter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/benchmark'))
from summary import paired,summarize,FIELDS


def text_json(value):return json.dumps(value,indent=2,allow_nan=False)+'\n'
def table(headers,rows):
    # A vertical bar in a scientific label is text, not another table column.
    cell=lambda value:str(value).replace('|', '&#124;')
    return '\n| '+' | '.join(map(cell,headers))+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(cell,row))+' |\n' for row in rows)
def f(x):return f'{x:.3g}'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    evidence=json.loads((ROOT/'docs/assets/benchmark_records.json').read_text())
    cases=[paired(c) for c in evidence['cases'] if c['stage']=='C']
    assert len(cases)==len(evidence['cases']), 'Canonical records must contain fine-mapping only'
    groups=summarize(cases);real=['GWAS','eQTL'];outputs={}
    def out(name,value):outputs[ROOT/name]=value.encode() if isinstance(value,str) else value
    def fragment(file,key,value):
        path=ROOT/file;s=outputs.get(path,path.read_bytes()).decode()
        pattern=rf'<!-- generated:{key}:start -->.*?<!-- generated:{key}:end -->'
        assert re.search(pattern,s,re.S),f'Missing marker {file}:{key}'
        s=re.sub(pattern,lambda m:f'<!-- generated:{key}:start -->\n{value.rstrip()}\n<!-- generated:{key}:end -->',s,flags=re.S)
        out(file,s)
    summary=dict(schema_version=1,operators=dict(runtime_speedup='exp(mean(log(median_repeat_time_susieR / median_repeat_time_prusie)))',
        absolute_runtime='Across-case median of within-case retained-repeat medians, seconds',
        peak_memory='Across-case median of independent full-process peak resident set size, MiB',
        memory_ratio='Median of per-case peak_susieR / peak_prusie',
        memory_reduction='Median of per-case 1 - peak_prusie / peak_susieR',
        error='Maximum absolute difference over all valid case/array elements; zero-based index retained'),
        provenance=evidence['provenance'],groups=groups)
    out('docs/assets/r_comparison_summary.json',text_json(summary))
    out('docs/assets/r_comparison_cases.json',text_json(cases))
    out('docs/assets/r_comparison_numerical.json',text_json([dict(case_id=c['case_id'],stage='C',panel=c['panel'],group=c['group'],n_variants=c['n_variants'],**c['numerical']) for c in cases]))
    out('docs/assets/r_comparison_protocol.json',text_json(dict(provenance=evidence['provenance'],environments=evidence['environments'])))
    repeats=[dict(case_id=c['case_id'],stage='C',panel=c['panel'],group=c['group'],backend=k,repeat=i+1,time_seconds=t) for c in cases for k,b in c['backends'].items() for i,t in enumerate(b['repeat_seconds'])]
    out('docs/assets/r_comparison_repeats.json',text_json(repeats))
    buf=io.StringIO();columns=['case_id','panel','group','n_variants','max_iter','timing_eligible','prusie_niter','susieR_niter','prusie_converged','susieR_converged','prusie_seconds','susieR_seconds','speedup','prusie_peak_mib','susieR_peak_mib','memory_ratio','memory_reduction','max_abs_pip_error']
    writer=csv.DictWriter(buf,columns,delimiter='\t',lineterminator='\n');writer.writeheader()
    for c in cases:
        row={k:c[k] for k in ('case_id','panel','group','n_variants','max_iter','timing_eligible','speedup','memory_ratio','memory_reduction')}
        for k in ('prusie','susieR'):
            b=c['backends'][k];row.update({k+'_niter':b['niter'],k+'_converged':b['converged'],k+'_seconds':b.get('runtime_seconds',{}).get('median'),k+'_peak_mib':b['peak_memory_mib']})
        row['max_abs_pip_error']=c['numerical'].get('errors',{}).get('pip',{}).get('max_abs_error');writer.writerow(row)
    out('docs/assets/r_comparison_cases.tsv',buf.getvalue())
    highlights=table(['Analysis','Maximum absolute PIP difference','Paired runtime speedup¹','Median peak-memory ratio²'],[[g,f(groups[g]['errors']['pip']['max_abs_error']),f(groups[g]['runtime_speedup_geometric_mean'])+'×',f(groups[g]['memory_ratio_distribution']['median'])+'×'] for g in real])
    highlights+=f"\n{groups['GWAS']['analyses']} GWAS locus analyses and {groups['eQTL']['analyses']} eQTL analyses, each with {min(groups[g]['variant_count_range'][0] for g in real)}–{max(groups[g]['variant_count_range'][1] for g in real)} variants, compared with susieR {evidence['provenance']['susieR_version']}. "
    highlights+='¹ Geometric mean of per-case ratios of median runtimes (susieR/prusie). ² Median per-case ratio of full-process peak memory (susieR/prusie). '
    highlights+='Results reanalyze measurements of prusie 0.2.3rc4 made on 6 October 2026; these values identify that measured version.\n'
    for page in ('README.md','docs/index.md'):fragment(page,'highlights',highlights)
    absolute=table(['Analysis','prusie time (ms)','susieR time (ms)','prusie peak memory (MiB)','susieR peak memory (MiB)'],[[g,f(groups[g]['runtime_seconds']['prusie']['median']*1000),f(groups[g]['runtime_seconds']['susieR']['median']*1000),f(groups[g]['peak_memory_mib']['prusie']['median']),f(groups[g]['peak_memory_mib']['susieR']['median'])] for g in real])
    absolute+='\nRuntime columns are medians across cases of each case’s median retained-repeat time. Memory columns are separate across-case medians of fresh-process peaks.\n'
    absolute+=table(['Analysis','Runtime speedup, geometric mean','Per-case speedup median [Q1, Q3]','Speedup range','Median memory ratio','Median memory reduction'],[[g,f(groups[g]['runtime_speedup_geometric_mean'])+'×',f"{f(groups[g]['runtime_speedup_distribution']['median'])} [{f(groups[g]['runtime_speedup_distribution']['q1'])}, {f(groups[g]['runtime_speedup_distribution']['q3'])}]",f"{f(groups[g]['runtime_speedup_distribution']['min'])}–{f(groups[g]['runtime_speedup_distribution']['max'])}",f(groups[g]['memory_ratio_distribution']['median'])+'×',f"{100*groups[g]['memory_reduction_distribution']['median']:.1f}%"] for g in real])
    fragment('docs/performance.md','absolute',absolute)
    variability=table(['Analysis','prusie repeat IQR/median','susieR repeat IQR/median'],[[g,f(groups[g]['repeat_iqr_over_median']['prusie']['median']),f(groups[g]['repeat_iqr_over_median']['susieR']['median'])] for g in real])
    variability+='\nEach entry is the median across cases of its seven-repeat IQR/median. Individual repeat times and case ranges are retained in the downloadable records.\n'
    fragment('docs/performance.md','variability',variability)
    panel=evidence['provenance']['real_panel']
    panel_text=f"The fixed panel contains {panel['regions']} genomic windows: 100 analyses of one FinnGen R10 inflammatory bowel disease GWAS (K11_IBD_STRICT; n = 412181) and 100 BLUEPRINT monocyte eQTL analyses (QTD000021; n = 191), representing {panel['distinct_genes']} distinct genes. The interval metadata contain {panel['overlapping_interval_pairs']} overlapping window pairs. Analyses share study participants and reference samples; the 200 trait analyses are not 200 independent loci."
    fragment('docs/performance.md','panel',panel_text)
    maxima=table(['Analysis','Analyses','Variants per analysis','Maximum absolute PIP difference','Trace: case / variant index (zero-based)'],[[g,groups[g]['valid_numerical_analyses'],'–'.join(map(str,groups[g]['variant_count_range'])),f(groups[g]['errors']['pip']['max_abs_error']),f"{groups[g]['errors']['pip']['case_id']} / {groups[g]['errors']['pip']['index_0based'][0]}"] for g in real])
    maxima+='\n'+ ' '.join(f"Across the evaluated {g} analyses, the maximum absolute difference in variant PIPs relative to susieR 0.16.6 was {groups[g]['errors']['pip']['max_abs_error']:.8g}." for g in real)
    fragment('docs/numerical_accuracy.md','maxima',maxima)
    units={'alpha':'Probability','mu':'Standardized effect','mu2':'Squared standardized effect','lbf_variable':'Natural-log BF','lbf':'Natural-log BF','V':'Squared effect','sigma2':'Phenotype variance','elbo':'Natural-log objective','KL':'Natural-log divergence'}
    errors=table(['Quantity','Scale / units','GWAS maximum absolute difference','eQTL maximum absolute difference'],[[k,units[k],*[f(groups[g]['errors'][k]['max_abs_error']) for g in real]] for k in FIELDS if k!='pip'])
    errors+='\nMaxima are computed elementwise across every valid analysis, including iteration-limit outputs. The machine summary includes the case, array index, both values at the maximum and reference magnitude for each field. ELBO traces are compared only with matching shapes; per-case records retain any shape mismatch.\n'
    fragment('docs/numerical_accuracy.md','errors',errors)
    csrows=[];discrepancies=[]
    for g in real:
        rs=[c for c in cases if c['group']==g];cr=[d for c in rs for d in c['numerical'].get('credible_set_differences',[])]
        mismatch=[(c['case_id'],d['component_id_0based']) for c in rs for d in c['numerical'].get('credible_set_differences',[]) if not d.get('members_match',False)]
        discrepancies+=mismatch
        csrows.append([g,sum(len(c['numerical']['credible_sets']['prusie']) for c in rs),sum(len(c['numerical']['credible_sets']['susieR']) for c in rs),len(mismatch),f(max((d.get('posterior_mass_abs_error',0.) for d in cr),default=0.)),f(max((max(d.get('purity_abs_error',{}).values(),default=0.) for d in cr),default=0.))])
    cs=table(['Analysis','prusie sets','susieR sets','Component/member differences','Maximum mass difference','Maximum purity difference'],csrows)
    cs+='\nSets are matched by original component ID and compared as member sets; returned ordering does not affect the comparison. All actual masses, purity statistics and zero-based members are retained.\n'
    if discrepancies:cs+='\nComponent/member discrepancies: '+', '.join(f'{c}, component {k}' for c,k in discrepancies)+'.\n'
    fragment('docs/numerical_accuracy.md','cs',cs)
    limits=[c for c in cases if c['panel']=='real' and c['case_id'] in groups[c['group']]['iteration_limit_cases']]
    limit='; '.join(f"{c['case_id']}: prusie {c['backends']['prusie']['niter']} iterations, susieR {c['backends']['susieR']['niter']} iterations, both unconverged" for c in limits)
    fragment('docs/numerical_accuracy.md','limits',limit+'. Both calls completed and their outputs enter the numerical maxima and timing summaries.')
    # Deterministic scientific vector figures.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':10,'svg.hashsalt':'prusie-standalone-1','pdf.fonttype':42,'figure.facecolor':'white','savefig.facecolor':'white'})
    colors=['#0072B2','#D55E00']
    def save(fig,name):
        for ext in ('svg','pdf'):
            b=io.BytesIO();md={'Date':None} if ext=='svg' else {'CreationDate':None,'ModDate':None}
            fig.savefig(b,format=ext,bbox_inches='tight',metadata=md);out(f'docs/assets/{name}.{ext}',b.getvalue())
        plt.close(fig)
    rng=np.random.default_rng(20261007);fig,axes=plt.subplots(1,2,figsize=(8.6,3.4),layout='constrained')
    for ax,key,title,label in zip(axes,['speedup','memory_ratio'],['A  Runtime','B  Peak memory'],['Runtime ratio (susieR / prusie)','Peak-memory ratio (susieR / prusie)']):
        values=[[c[key] for c in cases if c['group']==g and c['timing_eligible'] and c[key] is not None] for g in real]
        bp=ax.boxplot(values,positions=[1,2],widths=.43,showfliers=False,patch_artist=True,medianprops={'color':'#111111','linewidth':1.6})
        for j,(a,color) in enumerate(zip(values,colors)):
            bp['boxes'][j].set(facecolor=color,alpha=.15)
            ax.scatter(j+1+rng.uniform(-.13,.13,len(a)),a,s=10,alpha=.45,color=color,linewidths=0)
        ax.axhline(1,color='#777777',ls='--',lw=.8);ax.set(xticks=[1,2],xticklabels=real,ylabel=label,title=title);ax.set_yscale('log');ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}×'));ax.grid(axis='y',alpha=.15)
        ax.yaxis.set_minor_formatter(NullFormatter())
        if key=='memory_ratio':ax.yaxis.set_major_locator(FixedLocator([1,2,4,6,8]))
    save(fig,'r_comparison')
    for measure,label,filename in [('runtime','Median complete-call time (ms)','runtime_by_variants'),('memory','Peak memory (MiB)','memory_by_variants')]:
        fig,axes=plt.subplots(1,2,figsize=(8.6,3.4),layout='constrained')
        for ax,g in zip(axes,real):
            rows=[c for c in cases if c['group']==g and c['timing_eligible']]
            for k,color,marker in [('prusie',colors[0],'o'),('susieR',colors[1],'^')]:
                vals=[c['backends'][k]['runtime_seconds']['median']*1000 if measure=='runtime' else c['backends'][k]['peak_memory_mib'] for c in rows]
                ax.scatter([c['n_variants'] for c in rows],vals,s=14,alpha=.65,label=k,color=color,marker=marker,linewidths=0)
            ax.set(xlabel='Variants in analysis',ylabel=label,title=g);ax.set_xscale('log');ax.set_yscale('log');ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=9)
        save(fig,filename)
    fig,axes=plt.subplots(1,2,figsize=(8.6,3.3),layout='constrained')
    for ax,g,color in zip(axes,real,colors):
        rows=[c for c in cases if c['group']==g and c['numerical']['outputs_valid']]
        ax.scatter([c['n_variants'] for c in rows],[c['numerical']['errors']['pip']['max_abs_error'] for c in rows],s=15,color=color,alpha=.65)
        ax.set(xlabel='Variants in analysis',ylabel='Maximum absolute PIP difference',title=g);ax.set_yscale('symlog',linthresh=1e-14);ax.set_xscale('log');ax.grid(alpha=.15)
    save(fig,'pip_accuracy')
    teaching=ROOT/'docs/assets/teaching_result.json'
    if teaching.exists():
        t=json.loads(teaching.read_text());ref=np.load(ROOT/'src/prusie/data/teaching/reference.npz');inp=np.load(ROOT/'src/prusie/data/teaching/inputs.npz')
        fig,axes=plt.subplots(2,1,figsize=(8.6,4.6),sharex=True,layout='constrained');x=np.arange(1,501)
        axes[0].scatter(x,inp['z'],s=8,color=colors[0]);axes[0].set(ylabel='Signed z')
        axes[1].scatter(x,t['pip'],s=12,color=colors[0],label='prusie');axes[1].plot(x,ref['pip'],color=colors[1],lw=.7,alpha=.7,label='susieR');axes[1].set(xlabel='Synthetic variant index (one-based)',ylabel='PIP',ylim=(-.03,1.05));axes[1].legend(frameon=False)
        for ax in axes:
            for pos in [100,350]:ax.axvline(pos,color='#777777',ls='--',lw=.7)
        save(fig,'teaching_fit')
        info=f"The independent 500-variant toy example fit converged in {t['niter']} iterations and returned {len(t['cs'])} credible sets. Its maximum absolute PIP difference from the separately generated susieR 0.16.6 reference was {t['comparison']['errors']['pip']['max_abs_error']:.3g}."
        for page in ['docs/example.md','docs/numerical_accuracy.md']:fragment(page,'teaching',info)
        cs=table(['Original component','Synthetic variants','Returned posterior mass','Minimum |r|'],[[c['component_id_0based'],', '.join(f'syn{j+1:04d}' for j in c['members_0based']),f"{c['coverage']:.6f}",f"{c['purity']['min_abs_corr']:.6f}"] for c in t['cs']]);fragment('docs/example.md','teaching-cs',cs)
    stale=[]
    for path,payload in outputs.items():
        if args.check:
            if not path.exists() or path.read_bytes()!=payload:stale.append(str(path.relative_to(ROOT)))
        else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
    if stale:raise SystemExit('Stale generated evidence: '+', '.join(stale))
    print(('Checked' if args.check else 'Generated'),len(outputs),'outputs;',len(cases),'fine-mapping cases;',len(repeats),'retained repeats')

if __name__=='__main__':main()
