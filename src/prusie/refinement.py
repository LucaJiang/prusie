"""Immutable prepared inputs, checked continuation, and two-step CS refinement.

The likelihood uses the same working sufficient statistics in every candidate.
Stage A's changed-prior objective is recorded, never used for model selection.
"""
from dataclasses import dataclass
from collections.abc import Mapping
import copy
import hashlib
import math
import time
from functools import cached_property
from types import MappingProxyType
import numpy as np
from . import _native
from .posterior import credible_sets, marginal_pip
from .diagnostics import (ConvergenceConfig, tracking_level, convergence_summary,
                          resolve_convergence, nonconvergence_message)


def _readonly(value):
    a=np.array(value,copy=True,order='K'); a.flags.writeable=False; return a


_HASH_BUFFER_ELEMENTS = 32768  # At most 256 KiB of canonical float64 data per chunk.


def _hash_float64_c_order(h, value):
    """Hash canonical native float64 C-order bytes with a bounded conversion buffer.

    The signature predates this iterator and canonicalizes numeric dtypes to
    float64. Preserve that identity contract, including shape, without copying
    an entire strided/F-order matrix or allocating its full bytes representation.
    """
    a = np.asarray(value)
    h.update(str(a.shape).encode())
    with np.nditer(a, flags=['external_loop', 'buffered', 'zerosize_ok'],
                   op_flags=[['readonly', 'contig']], op_dtypes=[np.dtype(np.float64)],
                   casting='unsafe', order='C', buffersize=_HASH_BUFFER_ELEMENTS) as chunks:
        for chunk in chunks:
            h.update(memoryview(chunk).cast('B'))


def data_signature(matrix,response,yty,n,scales,multiplier,ids,metadata):
    h=hashlib.sha256()
    for a in (matrix,response,np.asarray([yty,n,multiplier]),scales):
        _hash_float64_c_order(h, a)
    for value in ids: h.update(str(value).encode()); h.update(b'\0')
    for key in ('effect_allele','other_allele'):
        h.update(key.encode())
        for value in metadata.get(key,[]): h.update(str(value).encode()); h.update(b'\0')
    return h.hexdigest()


@dataclass(frozen=True)
class PreparedData:
    """Owned read-only input/scale snapshots; reused by all candidate states."""
    matrix: np.ndarray
    response: np.ndarray
    yty: float
    n: float
    options: dict
    cs_matrix: np.ndarray
    diagonal: np.ndarray
    scales: np.ndarray
    means: np.ndarray
    y_mean: float
    signature: str
    matrix_scale: float
    variant_ids: np.ndarray
    alleles: dict

    @cached_property
    def _working_matrix(self):
        """Materialize only for explicit objective evaluation (never a refit)."""
        if 'matrix_global_scale' in self.options:
            q=((self.matrix*self.options['matrix_global_scale'])*
               self.options['matrix_inverse_scales'][None,:])/self.options['matrix_scales'][:,None]
            q.flags.writeable=False
            return q
        return self.matrix

    def working_matrix(self):
        return self._working_matrix


def make_prepared(matrix,response,yty,n,options,cs_matrix,diagonal,scales,means,y_mean,signature,matrix_scale,ids,metadata):
    o={k:(_readonly(v) if isinstance(v,np.ndarray) else copy.deepcopy(v))
       for k,v in options.items() if not k.startswith('init_')}
    # One matrix copy if matrix is the transposed CS covariance view; preserve
    # its layout for native borrowing. No candidate copies this p by p buffer.
    cs=_readonly(cs_matrix)
    native=cs.T if np.shares_memory(matrix,cs_matrix) else _readonly(matrix)
    return PreparedData(native,_readonly(response),yty,n,MappingProxyType(o),cs,_readonly(diagonal),
                        _readonly(scales),_readonly(means),y_mean,signature,matrix_scale,_readonly(ids),
                        MappingProxyType({k:_readonly(metadata[k]) for k in ("effect_allele","other_allele") if k in metadata}))


def initial_options(s_init, *, ids,metadata,scales,null_index,signature,L,weights,
                    estimate_prior_variance,estimate_residual_variance,reset_variances=False,prior_variance=0.2):
    """Validate same-data/scale state, resize by retaining the first L components."""
    if hasattr(s_init,'alpha'):
        state=dict(alpha=s_init.alpha,mu=s_init.mu,mu2=s_init.mu2,V=s_init.V,sigma2=s_init.sigma2,
                   variant_ids=s_init.variant_ids,variant_metadata=s_init.variant_metadata,
                   scales=s_init.X_column_scale_factors,null_index=s_init.null_index,
                   data_signature=s_init.data_signature)
    elif isinstance(s_init,Mapping): state=dict(s_init)
    else: raise TypeError('s_init must be a SusieResult or an explicit posterior-state mapping')
    required=('alpha','mu','mu2','variant_ids','scales','null_index','data_signature')
    missing=[k for k in required if k not in state]
    if missing: raise ValueError('Initialization metadata required: '+', '.join(missing))
    if not np.array_equal(np.asarray(state['variant_ids']),ids): raise ValueError('Initialization variant IDs/order differ')
    if state['null_index']!=null_index: raise ValueError('Initialization null column differs')
    if not np.array_equal(np.asarray(state['scales']),scales): raise ValueError('Initialization working scale differs')
    if state['data_signature']!=signature: raise ValueError('Initialization data/allele/scale signature differs; no implicit transformation')
    for key in ('effect_allele','other_allele'):
        a=state.get('variant_metadata',{}).get(key); b=metadata.get(key)
        if (a is None)!=(b is None) or (a is not None and not np.array_equal(a,b)):
            raise ValueError('Initialization allele metadata differs or is missing')
    arrays=[np.asarray(state[k],dtype=float) for k in ('alpha','mu','mu2')]
    alpha,mu,mu2=arrays; p=len(weights)
    if alpha.ndim!=2 or alpha.shape[1]!=p or alpha.shape[0]<1 or any(a.shape!=alpha.shape for a in arrays):
        raise ValueError('Initialization posterior dimensions differ')
    if any(not np.isfinite(a).all() for a in arrays) or (alpha<0).any() or not np.allclose(alpha.sum(1),1,atol=1e-8,rtol=0):
        raise ValueError('Initialization alpha must be finite, nonnegative and normalized')
    if np.any(mu2<mu*mu): raise ValueError('Initialization mu2 must be >= mu squared')
    keep=min(L,len(alpha)); out={}
    for name,a in zip(('alpha','mu','mu2'),arrays):
        target=np.broadcast_to(weights,(L,p)).copy() if name=='alpha' else np.zeros((L,p))
        target[:keep]=a[:keep]; out['init_'+name]=target.ravel()
    if 'V' in state:
        v=np.asarray(state['V'],dtype=float)
        if v.shape!=(len(alpha),) or not np.isfinite(v).all() or (v<0).any(): raise ValueError('Initialization V invalid')
        if not reset_variances and estimate_prior_variance:
            resized=np.full(L,prior_variance);resized[:keep]=v[:keep];out['init_V']=resized
    if 'sigma2' in state:
        v=float(state['sigma2'])
        if not math.isfinite(v) or v<=0: raise ValueError('Initialization sigma2 invalid')
        if not reset_variances and estimate_residual_variance: out['init_sigma2']=v
    return out


def _validate_labels(fit,data):
    if not np.array_equal(fit.variant_ids,data.variant_ids):
        raise ValueError('Posterior variant IDs/order differ from prepared data')
    for key in ('effect_allele','other_allele'):
        a=fit.variant_metadata.get(key);b=data.alleles.get(key)
        if (a is None)!=(b is None) or (a is not None and not np.array_equal(a,b)):
            raise ValueError('Posterior allele direction differs from prepared data')


def evaluate_objective(fit, data=None, *, prior_weights=None, hard_exclusion=False):
    """Recompute the Gaussian objective for the complete saved post-trim state.

    Ordinary SER uses log(pi + sqrt(eps)) without renormalizing; KL follows
    that exact convention. Trimmed V=0 components contribute zero KL. Zero
    diagonal columns contribute no conditional-Gaussian KL. Caller caches,
    lbf and KL are ignored. This explicit audit allocates a working matrix.
    """
    data=data or getattr(fit,'_prepared',None)
    if data is None: raise ValueError('Objective evaluation needs retained data; fit with retain_data=True')
    _validate_labels(fit,data)
    if fit.data_signature != data.signature or not np.array_equal(fit.X_column_scale_factors,data.scales):
        raise ValueError('Objective input/scale identity differs from posterior state')
    expected_prior=np.asarray(data.options['prior_weights'] if prior_weights is None else prior_weights,dtype=float)
    if not np.array_equal(fit.params['prior_weights'],expected_prior):
        raise ValueError('Objective prior differs from immutable original model')
    for key in ('estimate_prior_variance','estimate_residual_variance','estimate_prior_method'):
        if fit.params[key] != data.options[key]: raise ValueError('Objective model setting differs: '+key)
    if fit.null_index!=data.options['null_index'] or fit.alpha.shape!=(data.options['l'],len(data.response)):
        raise ValueError('Objective dimension/null identity differs')
    if not data.options['estimate_residual_variance'] and fit.sigma2!=data.options['residual_variance']:
        raise ValueError('Fixed residual variance differs from original model')
    if not data.options['estimate_prior_variance']:
        expected_v=data.options['prior_variance']
        if expected_v<data.options['prior_tol']:expected_v=0.
        if np.any(fit.V!=expected_v):raise ValueError('Fixed prior variance differs from original model')
    a,m,m2=fit.alpha,fit.mu,fit.mu2
    if not all(np.isfinite(x).all() for x in (a,m,m2,fit.V)) or not math.isfinite(fit.sigma2) or fit.sigma2<=0:
        raise ValueError('Nonfinite or invalid posterior state')
    if (a<0).any() or not np.allclose(a.sum(1),1,atol=1e-8,rtol=0) or (m2<m*m).any():
        raise ValueError('Inconsistent posterior probabilities or moments')
    q=data.working_matrix(); d=np.diag(q); b=a*m; bbar=b.sum(0); qb=b@q.T
    er2=data.yty-2*float(bbar@data.response)+float(bbar@q@bbar)-float(np.sum(b*qb))+float(np.sum(a*m2*d))
    pi=expected_prior
    effective=pi+np.sqrt(np.finfo(float).eps)
    if hard_exclusion: effective=np.where(pi==0,0.,effective)
    kl=[]
    for l,v in enumerate(fit.V):
        if v==0:
            if np.any(m[l]!=0) or np.any(m2[l]!=0): raise ValueError('V=0 component has nonzero moments')
            kl.append(0.); continue
        use=a[l]>0
        if np.any(effective[use]<=0): raise ValueError('Posterior mass on hard-excluded variant')
        categorical=float(np.sum(a[l,use]*np.log(a[l,use]/effective[use])))
        physical=use & (d>0)
        s2=m2[l,physical]-m[l,physical]**2
        if np.any(s2<=0): raise ValueError('Non-positive conditional variance in active component')
        normal=.5*float(np.sum(a[l,physical]*(m2[l,physical]/v-1+np.log(v/s2))))
        kl.append(categorical+normal)
    objective=-data.n/2*np.log(2*np.pi*fit.sigma2)-er2/(2*fit.sigma2)-sum(kl)
    if not np.isfinite(objective): raise ValueError('Nonfinite objective')
    return dict(elbo=float(objective),logged_elbo=float(fit.elbo[-1]),
                difference_from_logged=float(objective-fit.elbo[-1]),KL=np.array(kl),
                expected_residual_sum_squares=float(er2),XtXr=q@bbar,
                phase='post_loop_post_sigma2_update_and_trim',
                prior_convention='log(pi + sqrt(eps)); zero KL for trimmed V=0')


@dataclass(frozen=True)
class RefinementConfig:
    max_rounds: int = 10
    absolute_tol: float = 1e-3
    relative_tol: float | None = None
    max_iter: int = 100
    inner_tol: float = 1e-4
    candidate_budget: int | None = None
    def validate(self):
        for key in ('max_rounds','max_iter'):
            v=getattr(self,key)
            if isinstance(v,bool) or int(v)!=v or v<1: raise ValueError(key+' must be a positive integer')
        if self.candidate_budget is not None and (isinstance(self.candidate_budget,bool) or int(self.candidate_budget)!=self.candidate_budget or self.candidate_budget<0):
            raise ValueError('candidate_budget must be a nonnegative integer or None')
        for key in ('absolute_tol','inner_tol','relative_tol'):
            v=getattr(self,key)
            if v is not None and (not math.isfinite(v) or v<0): raise ValueError(key+' must be finite and nonnegative')


def _copy_result(fit):
    # The only large shared buffer is immutable prepared data.
    new=copy.copy(fit)
    for key,value in vars(fit).items():
        if key!='_prepared': setattr(new,key,copy.deepcopy(value))
    return new


def _run(fit,data,options,initial=None,reset_variances=False,track_fit=False,convergence=None):
    _validate_labels(fit,data)
    o=dict(options)
    for k in list(o):
        if k.startswith('init_'): del o[k]
    control,resolution=resolve_convergence(fit,convergence)
    o.update(control.options(),track_level=tracking_level(track_fit))
    if initial is not None:
        o.update(initial_options(initial,ids=fit.variant_ids,metadata=fit.variant_metadata,
            scales=data.scales,null_index=fit.null_index,signature=data.signature,L=o['l'],
            weights=o['prior_weights'],estimate_prior_variance=o['estimate_prior_variance'],
            estimate_residual_variance=o['estimate_residual_variance'],reset_variances=reset_variances,prior_variance=o['prior_variance']))
    native=_native.fit(data.matrix,data.response,data.yty,data.n,o)
    result=_copy_result(fit)
    for name in ('alpha','mu','mu2','lbf_variable','lbf','V','sigma2','elbo','niter','converged','KL','XtXr'):
        setattr(result,name,native[name])
    result.params.update(max_iter=o['max_iter'],tol=o['tol'],prior_weights=np.asarray(o['prior_weights']).tolist(),
                         hard_exclusion=o.get('hard_exclusion',False),track_fit=track_fit,
                         convergence=control.options(),convergence_resolution=resolution,
                         s_init='posterior_state' if initial is not None else None)
    result.pip=marginal_pip(result.alpha,result.V,null_index=result.null_index,prior_tol=o['prior_tol'])
    result.cs_audit=[]
    result.sets=credible_sets(result.alpha,result.V,data.cs_matrix,coverage=result.params['coverage'],
          min_abs_corr=result.params['min_abs_corr'],null_index=result.null_index,n_purity=result.params['n_purity'],
          covariance_scales=np.sqrt(data.diagonal),covariance_multiplier=data.matrix_scale,audit=result.cs_audit)
    result.intercept=data.y_mean-float(np.sum(data.means*np.sum(result.alpha*result.mu,axis=0)/data.scales))
    result.warnings=list(native['warnings'])
    if not native['converged']:
        result.warnings.append(nonconvergence_message(native))
    result.convergence,result.trace=convergence_summary(native,o,control)
    result.convergence['config_resolution']=copy.deepcopy(resolution)
    result.refinement={'status':'not_executed','stop_reason':'not_requested','history':[],'total_fits':0}
    return result


def continue_fit(fit, *, max_iter=100,tol=None,track_fit=False,convergence=None):
    """Continue the same data, inheriting requested controls and ELBO tolerance.

    An explicit ``tol`` changes ELBO tolerance; ``convergence`` separately
    overrides the criterion and posterior thresholds. Caller state is unchanged.
    """
    data=getattr(fit,'_prepared',None)
    if data is None: raise ValueError('Continuation needs retained data; fit with retain_data=True')
    tol_source='explicit_override'
    if tol is None:
        tol=fit.params.get('tol',data.options.get('tol',1e-4))
        tol_source='inherited_params' if 'tol' in fit.params else 'recovered_prepared_options'
    if int(max_iter)!=max_iter or max_iter<1 or not math.isfinite(tol) or tol<0: raise ValueError('Invalid continuation budget/tolerance')
    o=dict(data.options);o.update(max_iter=int(max_iter),tol=float(tol))
    result=_run(fit,data,o,initial=fit,track_fit=track_fit,convergence=convergence)
    result.params['tol_resolution']=tol_source
    result.convergence['tol_resolution']=tol_source
    return result


def _brief(fit,objective=None):
    return dict(pip=fit.pip.copy(),cs=[c.copy() for c in fit.sets['cs']],
                cs_index=fit.sets['cs_index'].copy(),ibss_converged=fit.converged,
                elbo=objective,logged_elbo=float(fit.elbo[-1]),niter=fit.niter,
                convergence=copy.deepcopy(fit.convergence),trace=copy.deepcopy(fit.trace))


def refine_fit(fit, config=None, *, convergence=None):
    """Search CS exclusions, restoring the original prior before every comparison."""
    config=config or RefinementConfig();config.validate()
    control,resolution=resolve_convergence(fit,convergence)
    best=_copy_result(fit); data=getattr(fit,'_prepared',None)
    history=[]; record=dict(status='not_executed',stop_reason=None,history=history,total_fits=0,
                           before=_brief(fit),after=_brief(fit),config=vars(config).copy(),
                           requested_convergence=control.options(),convergence_resolution=resolution,
                           objective_definition='post_loop_post_sigma2_update_and_trim; original prior')
    best.refinement=record
    def finish(status,reason):
        record.update(status=status,stop_reason=reason,after=_brief(best,record.get('best_objective')))
        if record.get('best_objective') is not None:
            from .comparison import compare_fits
            record['signal_comparison']=compare_fits({'before':fit,'after':best})
        best.refinement=record
        return best
    if data is None: return finish('unsupported','prepared_data_unavailable')
    if not fit.converged: return finish('unsupported','baseline_ibss_not_converged')
    try: current=evaluate_objective(best,data)['elbo']
    except ValueError as e: record['error']=str(e); return finish('unsupported','invalid_baseline_state')
    record['best_objective']=current;record['before']=_brief(fit,current)
    requested_tracking=fit.params.get('track_fit',False)
    attempts=0;accepted_any=False
    original=np.array(data.options['prior_weights'],copy=True)
    for round_id in range(1,config.max_rounds+1):
        if not best.sets['cs']: return finish('no_cs','no_retained_credible_sets')
        incumbent=best; incumbent_obj=current; candidates=[]; failures=0; evaluated=0
        budget_exhausted=False
        # Freeze the complete incumbent and its CS list for this round.
        for component,members in zip(incumbent.sets['cs_index'],incumbent.sets['cs']):
            if config.candidate_budget is not None and attempts>=config.candidate_budget:
                budget_exhausted=True;break
            row=dict(round=round_id,component=int(component),excluded_indices=members.tolist(),
                     excluded_ids=incumbent.variant_ids[members].tolist(),accepted=False,reason=None,
                     incumbent_objective=incumbent_obj)
            history.append(row)
            weights=original.copy();weights[members]=0.
            if weights.sum()<=0:
                row['reason']='no_prior_support';failures+=1;continue
            weights/=math.fsum(weights)
            attempts+=1
            o=dict(data.options);o.update(max_iter=config.max_iter,tol=config.inner_tol,
                                          prior_weights=weights,hard_exclusion=True)
            stages={};proposal=None
            for phase in ('A','B'):
                started=time.perf_counter();record['total_fits']+=1
                try:
                    if phase=='A': candidate=_run(incumbent,data,o,convergence=control)
                    else:
                        restored=dict(data.options);restored.update(max_iter=config.max_iter,tol=config.inner_tol,
                                                                  prior_weights=original,hard_exclusion=False)
                        candidate=_run(incumbent,data,restored,initial=proposal,reset_variances=True,
                                       track_fit=requested_tracking,convergence=control)
                    # Both stages use one resolved request; internal forwarding is
                    # not a second user override. Keep its actual origin explicit.
                    candidate.params['convergence_resolution']=copy.deepcopy(resolution)
                    candidate.convergence['config_resolution']=copy.deepcopy(resolution)
                    stage=dict(niter=candidate.niter,stop_reason=candidate.convergence['stop_reason'],
                               ibss_converged=candidate.converged,seconds=time.perf_counter()-started,
                               logged_elbo=float(candidate.elbo[-1]),
                               solver_settings=dict(convergence=control.options(),tol=config.inner_tol,
                                                    max_iter=config.max_iter,track_fit=candidate.params['track_fit']),
                               convergence=copy.deepcopy(candidate.convergence))
                    stages[phase]=stage;row['stages']=stages
                    if phase=='A':
                        stage['excluded_max_alpha']=float(np.max(candidate.alpha[:,members]))
                        if stage['excluded_max_alpha']!=0: raise ValueError('Hard exclusion failed')
                    stage['objective']=evaluate_objective(candidate,data,prior_weights=weights if phase=='A' else original,hard_exclusion=phase=='A')['elbo']
                    if not candidate.converged: raise ValueError('inner_ibss_not_converged')
                    if phase=='A': proposal=candidate
                    else:
                        if not np.array_equal(candidate.params['prior_weights'],original): raise ValueError('Original prior not restored')
                        obj=stage['objective'];row.update(objective=obj,improvement=obj-incumbent_obj,
                                   relative_improvement=(obj-incumbent_obj)/max(1.,abs(incumbent_obj)))
                        candidates.append((obj,candidate,row));evaluated+=1
                except (ValueError,RuntimeError,FloatingPointError) as e:
                    if phase not in stages: stages[phase]=dict(seconds=time.perf_counter()-started,error=str(e))
                    row.update(stages=stages,reason=phase+': '+str(e));failures+=1;break
        if candidates:
            obj,candidate,row=max(candidates,key=lambda x:x[0])
            if obj-current>config.absolute_tol:
                best=candidate;current=obj;accepted_any=True;row['accepted']=True;row['reason']='accepted_best_original_objective'
                record['best_objective']=current
                for _,_,other in candidates:
                    if other is not row: other['reason']='not_best_candidate'
                if budget_exhausted: return finish('budget_exhausted','candidate_budget')
                if config.relative_tol is not None and (obj-incumbent_obj)/max(1.,abs(incumbent_obj))<config.relative_tol:
                    return finish('completed','relative_improvement_threshold')
                continue
            for _,_,row in candidates: row['reason']='no_improvement_above_absolute_tol'
        if budget_exhausted: return finish('budget_exhausted','candidate_budget')
        if not evaluated and failures: return finish('all_candidates_failed','no_valid_converged_candidate')
        return finish('completed' if accepted_any else 'no_improvement','no_improvement_above_absolute_tol')
    return finish('budget_exhausted','round_limit')
