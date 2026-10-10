"""Convergence controls and exact state summaries for Gaussian IBSS."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class ConvergenceConfig:
    """Strict distance thresholds; the first eligible check is iteration two.

    ``elbo`` requires 0 ≤ ΔELBO < fit tol. ``alpha_pip`` requires both
    maximum absolute distances below their tolerances. ``joint`` requires all
    three. Cycles retain the entire current state and are not convergence.
    """
    criterion: str = 'elbo'
    alpha_tol: float = 1e-4
    pip_tol: float = 1e-4
    min_iter: int = 2
    cycle_lag: int = 5

    def options(self):
        if self.criterion not in ('elbo','alpha_pip','joint'):
            raise ValueError('criterion must be elbo, alpha_pip, or joint')
        if any(not math.isfinite(x) or x < 0 for x in (self.alpha_tol,self.pip_tol)):
            raise ValueError('alpha_tol and pip_tol must be finite and nonnegative')
        if isinstance(self.min_iter,bool) or int(self.min_iter)!=self.min_iter or self.min_iter<2:
            raise ValueError('min_iter must be an integer >= 2')
        if isinstance(self.cycle_lag,bool) or int(self.cycle_lag)!=self.cycle_lag or not 1<=self.cycle_lag<=5:
            raise ValueError('cycle_lag must be an integer in [1, 5]')
        return dict(criterion=self.criterion,alpha_tol=self.alpha_tol,pip_tol=self.pip_tol,
                    min_iter=self.min_iter,cycle_lag=self.cycle_lag)


def tracking_level(value):
    if value is False or value is None: return 0
    if value is True or value == 'compact': return 1
    if value == 'dense': return 2
    raise ValueError("track_fit must be False, True, 'compact', or 'dense'")


def convergence_summary(native, options, control):
    d=dict(native['diagnostics']); trace=d.pop('trace')
    d.update(requested_criterion=control.criterion,tol=options['tol'],
             alpha_tol=control.alpha_tol,pip_tol=control.pip_tol,min_iter=control.min_iter,
             max_cycle_lag=control.cycle_lag,niter=native['niter'],
             ibss_converged=native['converged'],phase='post_sweep_pre_sigma2_update',
             posterior_phase='post_loop_post_sigma2_update_and_trim',
             alpha_distance_includes_null=True,pip_excludes_null=True,
             online_component_entropy_includes_null=True,online_top_variant_uses_internal_indices=True,
             cs_activity_threshold=1e-9,pip_activity_threshold=options["prior_tol"],
             bbar_scale='standardized working predictor scale',
             V_fixed=not options['estimate_prior_variance'],
             sigma2_fixed=not options['estimate_residual_variance'],
             relative_elbo_formula='delta_elbo / max(1, abs(previous_elbo))',
             elbo_valid=all(math.isfinite(x) for x in native['elbo']),
             warnings=list(native['warnings']),trace_available=bool(trace))
    return d,trace


def diagnose_convergence(fit):
    """Return available evidence without reconstructing an unrecorded trace."""
    import copy
    if not getattr(fit,'convergence',None):
        return dict(stop_reason='not_available',trace_available=False,
                    warnings=['Exact online convergence metrics were not recorded.'])
    return copy.deepcopy(fit.convergence)
