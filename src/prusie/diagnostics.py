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
        return dict(criterion=str(self.criterion),alpha_tol=float(self.alpha_tol),pip_tol=float(self.pip_tol),
                    min_iter=int(self.min_iter),cycle_lag=int(self.cycle_lag))

    @classmethod
    def from_dict(cls, value):
        """Restore the requested controls saved in result/cache metadata."""
        config = cls(**value)
        config.options()  # Validate persisted values before passing them to native code.
        return config


def resolve_convergence(fit=None, override=None):
    """Resolve requested controls, never a runtime nonfinite fallback.

    Older results can recover the complete request from their diagnostic fields
    or retained native options. If neither exists, defaults are an explicitly
    recorded compatibility choice, not claimed inheritance.
    """
    previous = None
    source = 'default' if fit is None else 'legacy_default'
    if fit is not None:
        saved = getattr(fit, 'params', {}).get('convergence')
        if saved is not None:
            previous = ConvergenceConfig.from_dict(saved)
            source = 'inherited_params'
        else:
            diagnostic = getattr(fit, 'convergence', {}) or {}
            fields = dict(criterion='requested_criterion', alpha_tol='alpha_tol',
                          pip_tol='pip_tol', min_iter='min_iter', cycle_lag='max_cycle_lag')
            if all(key in diagnostic for key in fields.values()):
                previous = ConvergenceConfig.from_dict({k: diagnostic[v] for k, v in fields.items()})
                source = 'recovered_legacy_diagnostics'
            else:
                prepared = getattr(fit, '_prepared', None)
                options = prepared.options if prepared is not None else {}
                if all(key in options for key in fields):
                    previous = ConvergenceConfig.from_dict({key: options[key] for key in fields})
                    source = 'recovered_prepared_options'
    if override is not None:
        if not isinstance(override, ConvergenceConfig):
            raise TypeError('convergence must be a ConvergenceConfig')
        override.options()
        return override, dict(source='explicit_override',
                              previous_requested=previous.options() if previous is not None else None)
    control = previous or ConvergenceConfig()
    provenance = dict(source=source)
    if source == 'legacy_default':
        provenance['compatibility_note'] = (
            'Original requested controls were not recorded; current default '
            'ELBO controls are used. No inheritance can be established.')
    return control, provenance


def nonconvergence_message(native):
    return (f"IBSS did not converge after {native['niter']} iterations "
            f"(stop_reason={native['diagnostics']['stop_reason']}); "
            "check summary/LD consistency")


def tracking_level(value):
    if value is False or value is None: return 0
    if value is True or value == 'compact': return 1
    if value == 'dense': return 2
    raise ValueError("track_fit must be False, True, 'compact', or 'dense'")


def convergence_summary(native, options, control):
    d=dict(native['diagnostics']); trace=d.pop('trace')
    d.update(requested_criterion=control.criterion,requested_config=control.options(),tol=options['tol'],
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
