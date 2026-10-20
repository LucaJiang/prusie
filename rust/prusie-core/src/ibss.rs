//! Sequential iterative Bayesian stepwise selection.
use crate::{
    diagnostics::{self, Diagnostics, FitControl, Snapshot, StopDecision},
    input_validation,
    numerical::{digamma, dot, sum},
    operator::{materialize_scaled, operator_entries_bounded, DenseGram, GramOperator},
    options::{ConvergenceMethod, PriorMethod, ResidualMethod},
    residual::{expected_residual_sum_squares, ResidualScratch},
    ser::{single_effect, single_effect_mixture, SingleEffect},
    FitOptions, FitOutput, MatrixScaling,
};
use ndarray::{ArrayView2, ShapeBuilder};
use std::f64::consts::PI;
/// Fit from centered sufficient statistics (test entry point). XtX must be
/// row-major and already include any standardization and null column.
#[cfg(test)]
pub fn fit(
    xtx: &[f64],
    xty: &[f64],
    yty: f64,
    n: f64,
    options: &FitOptions,
) -> Result<FitOutput, String> {
    fit_layout(xtx, xty, yty, n, options, false)
}

#[cfg(test)]
pub fn fit_layout(
    xtx: &[f64],
    xty: &[f64],
    yty: f64,
    n: f64,
    options: &FitOptions,
    column_major: bool,
) -> Result<FitOutput, String> {
    fit_layout_operator(xtx, xty, yty, n, options, column_major, None)
}

#[cfg(test)]
pub fn fit_layout_operator(
    xtx: &[f64],
    xty: &[f64],
    yty: f64,
    n: f64,
    options: &FitOptions,
    column_major: bool,
    scaling: Option<&MatrixScaling<'_>>,
) -> Result<FitOutput, String> {
    fit_controlled(
        xtx,
        xty,
        yty,
        n,
        options,
        column_major,
        scaling,
        &FitControl::default(),
    )
}

/// Run IBSS on centered sufficient statistics with explicit stopping controls.
/// `xtx` is row- or column-major (`column_major`) and already includes any
/// standardization and null column, unless `scaling` describes the
/// standardization to be applied implicitly.
#[allow(clippy::too_many_arguments)]
pub fn fit_controlled(
    xtx: &[f64],
    xty: &[f64],
    yty: f64,
    n: f64,
    options: &FitOptions,
    column_major: bool,
    scaling: Option<&MatrixScaling<'_>>,
    control: &FitControl,
) -> Result<FitOutput, String> {
    let p = xty.len();
    if p == 0
        || p.checked_mul(p) != Some(xtx.len())
        || options.l == 0
        || p.checked_mul(options.l).is_none()
    {
        return Err("Invalid sufficient-statistic dimensions or number of effects".into());
    }
    // Every raw entry is checked here. If the bounds hold, the input is finite
    // and the implicit operator is safe; otherwise the full finiteness check
    // and the materialized matrix are used. Each bound is stated next to the
    // corresponding scale check.
    #[allow(clippy::manual_range_contains)]
    let operator_bounded = scaling.is_some_and(|scale| {
        scale.global >= 1.
            && scale.global <= 1e9
            && scale.scales.len() == p
            && scale.inverse.len() == p
            && scale.scales.iter().all(|&x| x >= 0.5 && x <= 2.)
            && scale.inverse.iter().all(|&x| x >= 0.5 && x <= 2.)
            && operator_entries_bounded(xtx)
    });
    if !n.is_finite()
        || n <= 1.0
        || !yty.is_finite()
        || yty < 0.0
        || !input_validation::all_finite(xty)
        || (!operator_bounded && !input_validation::all_finite(xtx))
    {
        return Err("Sufficient statistics and n must be finite; n > 1 and yty >= 0".into());
    }
    if let Some(scale) = scaling {
        if !scale.global.is_finite()
            || scale.global <= 0.
            || scale.scales.len() != p
            || scale.inverse.len() != p
            || scale
                .scales
                .iter()
                .chain(scale.inverse)
                .any(|&x| !x.is_finite() || x <= 0.)
        {
            return Err("Matrix factors must have p finite positive entries and a finite positive global multiplier".into());
        }
    }
    // Either contiguous layout is accepted; column-major input is used in place,
    // avoiding a dense copy. The caller owns the borrowed matrix lifetime.
    let columns;
    let matrix = if column_major {
        ArrayView2::from_shape((p, p).f(), xtx).map_err(|error| error.to_string())?
    } else {
        let input = ArrayView2::from_shape((p, p), xtx).map_err(|error| error.to_string())?;
        columns = input.t().as_standard_layout().into_owned().reversed_axes();
        columns.view()
    };
    // The bounds choose only how the arithmetic is carried out: outside them the
    // scaled matrix is materialized. Within them every prepared entry is at most
    // 8 × 10⁹ in magnitude and cannot overflow.
    let operator = if operator_bounded { scaling } else { None };
    let prepared;
    let matrix = if let Some(scale) = scaling.filter(|_| operator.is_none()) {
        prepared = materialize_scaled(matrix, scale);
        if prepared.iter().any(|x| !x.is_finite()) {
            return Err("Scaled matrix entries must remain finite".into());
        }
        ArrayView2::from_shape((p, p).f(), &prepared).map_err(|error| error.to_string())?
    } else {
        matrix
    };
    let diagonal: Vec<f64> = (0..p)
        .map(|j| {
            operator.map_or(matrix[(j, j)], |scale| {
                ((matrix[(j, j)] * scale.global) * scale.inverse[j]) / scale.scales[j]
            })
        })
        .collect();
    if diagonal.iter().any(|&d| d < 0.0) {
        return Err("XtX diagonal cannot be negative".into());
    }
    let mut gram = DenseGram {
        matrix,
        scaling: operator,
        diagonal,
        scratch: vec![0.; p],
        compression: crate::signed_columns::SignedColumns::discover(matrix),
    };
    fit_operator(&mut gram, xty, yty, n, options, control)
}

/// Shared IBSS implementation; individual-level designs never form XᵀX.
pub(crate) fn fit_operator(
    gram: &mut impl GramOperator,
    xty: &[f64],
    yty: f64,
    n: f64,
    options: &FitOptions,
    control: &FitControl,
) -> Result<FitOutput, String> {
    let p = xty.len();
    let model_pip = control.model.slot_prior.is_some()
        || (control.model.residual_method == ResidualMethod::Nig && options.l > 1);
    let mut effective_control;
    let control = if model_pip || control.criterion == ConvergenceMethod::Pip {
        effective_control = control.clone();
        effective_control.criterion = ConvergenceMethod::Pip;
        effective_control.alpha_tol = options.tol;
        effective_control.pip_tol = options.tol;
        &effective_control
    } else {
        control
    };
    let diagonal = gram.diagonal().to_vec();
    if diagonal.len() != p || diagonal.iter().any(|&d| !d.is_finite() || d < 0.) {
        return Err("Gram diagonal must have p finite nonnegative entries".into());
    }
    validate_fit_options(p, xty, yty, n, options)?;
    validate_model(&control.model, n, options.l)?;
    let log_prior: Vec<f64> = options
        .prior_weights
        .iter()
        .map(|w| {
            if control.hard_exclusion && *w == 0. {
                f64::NEG_INFINITY
            } else {
                (w + f64::EPSILON.sqrt()).ln()
            }
        })
        .collect();
    let use_nig = control.model.residual_method == ResidualMethod::Nig;
    let alpha0 = control.model.alpha0.unwrap_or(1. / n.sqrt());
    let beta0 = control.model.beta0.unwrap_or(1. / n.sqrt());
    let method = if use_nig && options.estimate_prior_variance {
        PriorMethod::Em
    } else if options.estimate_prior_variance {
        options.estimate_prior_method
    } else {
        PriorMethod::None
    };
    // In working coordinates |Xty_j|/√XtX_jj does not depend on σ²; the check
    // below divides by the current σ, after each sweep and before the σ² update
    // (susieR validate_prior.ss). Zero-information columns have no z-score, and
    // an empty set gives a maximum of −∞, as in R. The guard and the division
    // are kept in one expression.
    #[allow(clippy::filter_map_bool_then)]
    let max_abs_score = diagonal
        .iter()
        .zip(xty)
        .filter_map(|(&d, &r)| (d > 0.).then(|| r.abs() / d.sqrt()))
        .fold(f64::NEG_INFINITY, f64::max);
    let size = options.l * p;
    let mut output = FitOutput {
        alpha: vec![1.0 / p as f64; size],
        mu: vec![0.0; size],
        mu2: vec![0.0; size],
        lbf_variable: vec![f64::NAN; size],
        lbf: vec![f64::NAN; options.l],
        v: vec![options.prior_variance; options.l],
        kl: vec![f64::NAN; options.l],
        elbo: Vec::with_capacity(options.max_iter),
        sigma2: options.residual_variance,
        niter: 0,
        converged: false,
        xtxr: vec![0.0; p],
        warnings: Vec::new(),
        lbf_grid: control
            .model
            .mixture
            .as_ref()
            .map(|m| vec![0.; size * m.variances.len()]),
        slot_weights: None,
        rv: use_nig.then(|| vec![1.; options.l]),
        marginal_loglik: use_nig.then(|| vec![f64::NAN; options.l]),
        diagnostics: Diagnostics {
            stop_reason: "iteration_limit".into(),
            effective_criterion: control.criterion.as_str().into(),
            ..Diagnostics::default()
        },
    };
    let mut slot_state = control.model.slot_prior.as_ref().map(|prior| {
        let (state, weights) = crate::slot::SlotState::new(prior, options.l);
        output.slot_weights = Some(weights);
        state
    });
    if control.min_iter < 2
        || control.cycle_lag > 5
        || !control.alpha_tol.is_finite()
        || control.alpha_tol < 0.
        || !control.pip_tol.is_finite()
        || control.pip_tol < 0.
    {
        return Err("Invalid convergence control".into());
    }
    if let Some(v) = &control.component_prior_variance {
        if v.len() != options.l || v.iter().any(|x| !x.is_finite() || *x < 0.0) {
            return Err("Component prior variances must have L finite nonnegative entries".into());
        }
        output.v.copy_from_slice(v);
    }
    if let Some(initial) = &control.initial {
        if initial.alpha.len() != size
            || initial.mu.len() != size
            || initial.mu2.len() != size
            || initial
                .alpha
                .iter()
                .chain(&initial.mu)
                .chain(&initial.mu2)
                .any(|x| !x.is_finite())
            || initial.alpha.iter().any(|&x| !(0.0..=1.0).contains(&x))
        {
            return Err("Invalid initial posterior dimensions or values".into());
        }
        for l in 0..options.l {
            let row = l * p..(l + 1) * p;
            if (sum(initial.alpha[row].iter().copied()) - 1.).abs() > 1e-8 {
                return Err("Initial alpha rows must sum to one".into());
            }
        }
        if initial
            .mu
            .iter()
            .zip(&initial.mu2)
            .any(|(&m, &m2)| m2 < m * m)
        {
            return Err("Initial mu2 must be at least mu squared".into());
        }
        output.alpha.copy_from_slice(&initial.alpha);
        output.mu.copy_from_slice(&initial.mu);
        output.mu2.copy_from_slice(&initial.mu2);
        if let Some(v) = &initial.v {
            if v.len() != options.l || v.iter().any(|x| !x.is_finite() || *x < 0.) {
                return Err("Invalid initial V".into());
            }
            output.v.copy_from_slice(v);
        }
        if let Some(v) = initial.sigma2 {
            if !v.is_finite() || v <= 0. {
                return Err("Invalid initial sigma2".into());
            }
            output.sigma2 = v;
        }
    }
    let mut component = vec![0.0; p];
    let mut product = vec![0.0; p];
    let mut residual = vec![0.0; p];
    let mut component_products = vec![0.0; size];
    let mut likelihood_scratch = vec![0.0; p];
    let mut ser_sampling = None;
    let mut ser_betahat = vec![0.0; p];
    let mut residual_scratch = ResidualScratch {
        mean: vec![0.0; p],
        fitted: vec![0.0; p],
        component_quadratics: Vec::with_capacity(options.l),
    };
    // With an initial state, every product Q m_ℓ is recomputed; no stored
    // fitted quantity is reused.
    if control.initial.is_some() {
        for l in 0..options.l {
            // Index all posterior arrays by the same effect and variant.
            #[allow(clippy::needless_range_loop)]
            for j in 0..p {
                component[j] = output.alpha[l * p + j] * output.mu[l * p + j];
            }
            gram.apply(&component, &mut product);
            component_products[l * p..(l + 1) * p].copy_from_slice(&product);
            #[allow(clippy::needless_range_loop)]
            for j in 0..p {
                output.xtxr[j] += output.slot_weights.as_ref().map_or(1., |w| w[l]) * product[j];
            }
        }
    }
    let mut previous_elbo = f64::NEG_INFINITY;
    let mut previous_alpha: Option<Vec<f64>> = None;
    let mut previous_pip: Option<Vec<f64>> = None;
    let mut previous_mean: Option<Vec<f64>> = None;
    let mut previous_v: Option<Vec<f64>> = None;
    let mut previous_sigma2: Option<f64> = None;
    let mut history: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    for iteration in 0..options.max_iter {
        for effect in 0..options.l {
            if slot_state.as_ref().is_some_and(|state| {
                output.slot_weights.as_ref().expect("slot weights")[effect] < state.threshold
            }) {
                continue;
            }
            let offset = effect * p;
            // component_products holds Q m_ℓ for each effect. Subtracting this
            // effect's product leaves the fitted crossproduct of the others.
            for j in 0..p {
                output.xtxr[j] -= output.slot_weights.as_ref().map_or(1., |w| w[effect])
                    * component_products[offset + j];
                residual[j] = xty[j] - output.xtxr[j];
            }
            let nig_yy = if use_nig {
                for (j, b) in component.iter_mut().enumerate() {
                    *b = sum((0..options.l).filter(|&l| l != effect).map(|l| {
                        output.slot_weights.as_ref().map_or(1., |w| w[l])
                            * output.alpha[l * p + j]
                            * output.mu[l * p + j]
                    }));
                }
                (yty - 2. * dot(&component, xty) + dot(&component, &output.xtxr)).max(f64::EPSILON)
            } else {
                0.
            };
            let prior_v = output.v[effect];
            let posterior_row = SingleEffect {
                alpha: &mut output.alpha[offset..offset + p],
                mu: &mut output.mu[offset..offset + p],
                mu2: &mut output.mu2[offset..offset + p],
                lbf: &mut output.lbf_variable[offset..offset + p],
                model_lbf: f64::NAN,
                v: prior_v,
            };
            let posterior = if let Some(mixture) = &control.model.mixture {
                let grid_size = p * mixture.variances.len();
                single_effect_mixture(
                    &residual,
                    &diagonal,
                    output.sigma2,
                    &log_prior,
                    mixture,
                    &mut output.lbf_grid.as_mut().expect("mixture grid allocated")
                        [effect * grid_size..(effect + 1) * grid_size],
                    posterior_row,
                )?
            } else if use_nig {
                let result = crate::nig::single_effect_nig(
                    &residual,
                    &diagonal,
                    None,
                    nig_yy,
                    n,
                    output.v[effect],
                    alpha0,
                    beta0,
                    &log_prior,
                    &options.prior_weights,
                    method,
                    posterior_row,
                )?;
                output.rv.as_mut().expect("NIG rv allocated")[effect] = result.residual_mean;
                output
                    .marginal_loglik
                    .as_mut()
                    .expect("NIG likelihood allocated")[effect] = result.marginal_loglik;
                output.kl[effect] = if options.l == 1 { result.kl } else { 0. };
                result.posterior
            } else {
                single_effect(
                    &residual,
                    &diagonal,
                    output.v[effect],
                    output.sigma2,
                    &log_prior,
                    method,
                    options.check_null_threshold,
                    &mut likelihood_scratch,
                    &mut ser_sampling,
                    &mut ser_betahat,
                    posterior_row,
                )?
            };
            output.v[effect] = posterior.v;
            output.lbf[effect] = posterior.model_lbf;
            for (j, value) in component.iter_mut().enumerate() {
                *value = posterior.alpha[j] * posterior.mu[j];
            }
            // Expected SER log likelihood; zero-information columns are excluded.
            let expected_loglik = -0.5
                * sum((0..p).filter_map(|j| {
                    let s2 = output.sigma2 / diagonal[j];
                    s2.is_finite().then(|| {
                        (-2.0 * component[j] * (residual[j] / diagonal[j])
                            + posterior.alpha[j] * posterior.mu2[j])
                            / s2
                    })
                }));
            if !use_nig {
                output.kl[effect] = -posterior.model_lbf + expected_loglik;
            } else if control.model.mixture.is_some() {
                // Mixture Gaussian moments enter NIG's KL dispatch before V
                // is replaced by the mixture's effective slab variance.
                output.kl[effect] = if options.l == 1 {
                    crate::nig::kl_from_posterior(
                        &residual,
                        &diagonal,
                        None,
                        nig_yy,
                        n,
                        prior_v,
                        alpha0,
                        beta0,
                        &options.prior_weights,
                        posterior.alpha,
                        posterior.mu,
                    )
                } else {
                    0.
                };
            }
            gram.apply(&component, &mut product);
            component_products[offset..offset + p].copy_from_slice(&product);
            if let Some(state) = &mut slot_state {
                state.update(
                    output.slot_weights.as_mut().expect("slot weights"),
                    effect,
                    output.lbf[effect],
                );
            }
            let weight = output.slot_weights.as_ref().map_or(1., |w| w[effect]);
            for (fitted, increment) in output.xtxr.iter_mut().zip(&product) {
                *fitted += weight * increment;
            }
        }
        if let Some(state) = &mut slot_state {
            state.finish_sweep(output.slot_weights.as_ref().expect("slot weights"));
        }
        let max_z = max_abs_score / output.sigma2.sqrt();
        if options.check_prior && output.v.iter().any(|&v| v > 100.0 * max_z * max_z) {
            return Err("The estimated prior variance is unreasonably large; check summary statistics and LD consistency".into());
        }
        let er2 = expected_residual_sum_squares(
            xty,
            yty,
            &diagonal,
            &output,
            &component_products,
            &mut residual_scratch,
        );
        let mut elbo = if use_nig && options.l == 1 {
            output.marginal_loglik.as_ref().expect("NIG likelihood")[0]
        } else if use_nig {
            let a = (alpha0 + n) / 2.;
            let b = (beta0 + er2) / 2.;
            let b_term = sum((0..options.l).flat_map(|l| {
                let alpha = &output.alpha;
                let v = output.v[l];
                let d = &diagonal;
                (0..p).map(move |j| alpha[l * p + j] * v / (v + 1. / d[j]))
            }));
            let a_term = er2 - b / (a - 1.) * b_term;
            -n / 2. * (2. * PI).ln()
                - n / 2. * (b.ln() - digamma(a))
                - 0.5 * (a_term * a / b + b_term)
        } else {
            -n / 2.0 * (2.0 * PI * output.sigma2).ln()
                - 1.0 / (2.0 * output.sigma2) * er2
                - sum(output.kl.iter().copied())
        };
        if let Some(state) = &slot_state {
            elbo += state.objective(output.slot_weights.as_ref().expect("slot weights"));
        }
        if elbo.is_infinite() {
            return Err("get_objective() produced an infinite ELBO value".into());
        }
        if !elbo.is_finite()
            && output
                .alpha
                .iter()
                .chain(&output.mu)
                .chain(&output.mu2)
                .chain(&output.v)
                .any(|x| !x.is_finite())
        {
            return Err("Numerical failure: nonfinite posterior state".into());
        }
        output.elbo.push(elbo);
        output.niter = iteration + 1;
        let improvement = elbo - previous_elbo;
        let current_pip = fit_pip(&output.alpha, &output, p, options);
        if control.criterion == ConvergenceMethod::Pip {
            if let Some(alpha) = &previous_alpha {
                previous_pip = Some(fit_pip(alpha, &output, p, options));
            }
            for (alpha, pip) in &mut history {
                *pip = fit_pip(alpha, &output, p, options);
            }
        }
        let mean = diagnostics::posterior_mean(&output, p);
        let mut snapshot = Snapshot {
            iteration: iteration + 1,
            elbo,
            delta_elbo: (iteration > 0).then_some(improvement),
            relative_delta_elbo: (iteration > 0)
                .then_some(improvement / previous_elbo.abs().max(1.)),
            delta_alpha: previous_alpha
                .as_ref()
                .map(|v| diagnostics::distance(&output.alpha, v)),
            delta_pip: previous_pip
                .as_ref()
                .map(|v| diagnostics::distance(&current_pip, v)),
            delta_mean: previous_mean
                .as_ref()
                .map(|v| diagnostics::distance(&mean, v)),
            delta_v: previous_v
                .as_ref()
                .map(|v| output.v.iter().zip(v).map(|(a, b)| a - b).collect()),
            delta_sigma2: previous_sigma2.map(|v| output.sigma2 - v),
            v: output.v.clone(),
            sigma2: output.sigma2,
            active: output.v.iter().map(|v| *v > options.prior_tol).collect(),
            ..Snapshot::default()
        };
        for l in 0..options.l {
            let row = &output.alpha[l * p..(l + 1) * p];
            let j = (0..p)
                .max_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap().then(b.cmp(&a)))
                .unwrap();
            snapshot.top_variant.push(j);
            snapshot.top_alpha.push(row[j]);
            snapshot
                .entropy
                .push(-sum(row.iter().filter(|&&a| a > 0.).map(|a| a * a.ln())));
        }
        if !elbo.is_finite() {
            snapshot.events.push("nonfinite_elbo".into());
            output.diagnostics.nonfinite_count += 1;
        }
        if iteration > 0 && improvement < -options.tol {
            output.warnings.push(format!(
                "ELBO decreased by {} at iteration {}",
                -improvement,
                iteration + 1
            ));
            snapshot.events.push("elbo_decreased".into());
            output.diagnostics.decrease_count += 1;
        }
        output.diagnostics.effective_criterion = control.criterion.as_str().into();
        if let Some((lag, da, dp)) = diagnostics::detect_cycle(
            &output.alpha,
            &current_pip,
            &history,
            control.cycle_lag,
            control.alpha_tol,
            control.pip_tol,
        ) {
            snapshot.cycle_lag = Some(lag);
            snapshot.cycle_alpha_distance = Some(da);
            snapshot.cycle_pip_distance = Some(dp);
            snapshot.events.push(
                if lag == 1 {
                    "fixed_point"
                } else {
                    "cycle_detected"
                }
                .into(),
            );
        }
        let (decision, fallback) = diagnostics::stopping_decision(control, &snapshot, options.tol);
        if control.criterion == ConvergenceMethod::Pip && decision == StopDecision::Converged {
            if let Some(lag) = snapshot.cycle_lag.filter(|&lag| lag > 1) {
                let from = history.len().saturating_sub(lag - 1);
                for (j, a) in output.alpha.iter_mut().enumerate() {
                    *a = (*a + sum(history[from..].iter().map(|(alpha, _)| alpha[j]))) / lag as f64;
                }
                snapshot.events.push("alpha_pip_cycle_averaged".into());
            }
        }
        if fallback {
            output.diagnostics.effective_criterion = "alpha_pip_nonfinite_fallback".into();
            output.diagnostics.fallback_iterations.push(iteration + 1);
            output.warnings.push(format!(
                "NaN ELBO at iteration {}; using alpha/PIP convergence",
                iteration + 1
            ));
        }
        output.converged = decision == StopDecision::Converged;
        output.diagnostics.stop_reason = match decision {
            StopDecision::Converged => "criterion_satisfied",
            StopDecision::Cycle => "cycle_detected",
            StopDecision::NumericalFailure => "numerical_failure",
            StopDecision::Continue => "iteration_limit",
        }
        .into();
        if decision == StopDecision::NumericalFailure {
            snapshot.events.push("numerical_failure".into());
        }
        if control.track > 0 {
            // Compact mode thresholds α below for display; only dense
            // mode records the moments. Distances never use these records.
            snapshot.pip = current_pip.clone();
            if control.track == 2 {
                snapshot.alpha = output.alpha.clone();
                snapshot.mu = output.mu.clone();
                snapshot.mu2 = output.mu2.clone();
            } else {
                for l in 0..options.l {
                    let row = &output.alpha[l * p..(l + 1) * p];
                    let show = snapshot.top_alpha[l] >= 2. / p as f64;
                    let mut retained = 0.;
                    for (j, &a) in row.iter().enumerate() {
                        if show && a >= 1. / p as f64 {
                            snapshot.alpha_entries.push((l, j, a));
                            retained += a;
                        }
                    }
                    snapshot
                        .omitted_mass
                        .push(sum(row.iter().copied()) - retained);
                }
            }
            output.diagnostics.trace.push(snapshot.clone());
        }
        snapshot.alpha.clear();
        snapshot.mu.clear();
        snapshot.mu2.clear();
        snapshot.pip.clear();
        snapshot.alpha_entries.clear();
        snapshot.omitted_mass.clear();
        output.diagnostics.last = Some(snapshot);
        if decision != StopDecision::Continue {
            break;
        }
        previous_alpha = Some(output.alpha.clone());
        previous_pip = Some(current_pip.clone());
        previous_mean = Some(mean);
        previous_v = Some(output.v.clone());
        previous_sigma2 = Some(output.sigma2);
        history.push((output.alpha.clone(), current_pip));
        if history.len() > control.cycle_lag.max(1) {
            history.remove(0);
        }
        previous_elbo = elbo;
        if options.estimate_residual_variance || use_nig {
            let estimate = if use_nig {
                (beta0 + er2) / (alpha0 + n - 2.)
            } else {
                er2 / n
            };
            if !estimate.is_finite() || estimate <= 0.0 {
                return Err("Estimating residual variance failed: non-positive estimate".into());
            }
            output.sigma2 = estimate.clamp(
                control.model.residual_lowerbound,
                control.model.residual_upperbound,
            );
        }
    }
    output.diagnostics.sigma2_after_loop = output.sigma2;
    if let Some(rv) = &output.rv {
        for (v, r) in output.v.iter_mut().zip(rv) {
            *v *= r;
        }
    }
    let pretrim_alpha = output.alpha.clone();
    let pretrim_pip = diagnostics::pip(
        &output.alpha,
        &output.v,
        p,
        options.prior_tol,
        options.null_index,
    );
    // The stored ELBO and XtXr refer to the completed sweeps, before trimming.
    // As in susieR, trimming also applies at max_iter and neither is recomputed.
    for effect in 0..options.l {
        if output.v[effect] < options.prior_tol {
            output.diagnostics.trimmed_components.push(effect);
            output.v[effect] = 0.0;
            output.lbf[effect] = 0.0;
            output.kl[effect] = 0.0;
            let row = effect * p..(effect + 1) * p;
            output.alpha[row.clone()].copy_from_slice(&options.prior_weights);
            output.mu[row.clone()].fill(0.0);
            output.mu2[row.clone()].fill(0.0);
            output.lbf_variable[row].fill(0.0);
        }
    }
    output.diagnostics.trim_alpha_distance = diagnostics::distance(&pretrim_alpha, &output.alpha);
    output.diagnostics.trim_pip_distance = diagnostics::distance(
        &pretrim_pip,
        &diagnostics::pip(
            &output.alpha,
            &output.v,
            p,
            options.prior_tol,
            options.null_index,
        ),
    );
    Ok(output)
}

fn fit_pip(alpha: &[f64], fit: &FitOutput, p: usize, options: &FitOptions) -> Vec<f64> {
    (0..p)
        .filter(|&j| Some(j) != options.null_index)
        .map(|j| {
            1. - (0..fit.v.len())
                .filter(|&l| fit.v[l] > options.prior_tol)
                .map(|l| 1. - alpha[l * p + j] * fit.slot_weights.as_ref().map_or(1., |w| w[l]))
                .product::<f64>()
        })
        .collect()
}

pub(crate) fn validate_model(
    model: &crate::model::ModelOptions,
    n: f64,
    l: usize,
) -> Result<(), String> {
    if model.residual_lowerbound.is_nan()
        || model.residual_lowerbound < 0.
        || model.residual_upperbound.is_nan()
        || model.residual_upperbound <= 0.
        || model.residual_lowerbound > model.residual_upperbound
    {
        return Err("Residual variance bounds must satisfy 0 ≤ lower ≤ upper and upper > 0".into());
    }
    if let Some(mixture) = &model.mixture {
        if mixture.variances.is_empty()
            || mixture.variances.len() != mixture.weights.len()
            || mixture.variances.iter().any(|&v| !v.is_finite() || v <= 0.)
            || mixture.weights.iter().any(|&w| !w.is_finite() || w < 0.)
            || (sum(mixture.weights.iter().copied()) - 1.).abs() >= 1e-8
        {
            return Err(
                "Mixture requires positive variances and nonnegative normalized weights".into(),
            );
        }
    }
    if model.residual_method == crate::options::ResidualMethod::Nig {
        let alpha0 = model.alpha0.unwrap_or(1. / n.sqrt());
        let beta0 = model.beta0.unwrap_or(1. / n.sqrt());
        if !alpha0.is_finite() || alpha0 <= 0. || !beta0.is_finite() || beta0 <= 0. {
            return Err("NIG requires finite alpha0 > 0 and beta0 > 0".into());
        }
    }
    if let Some(slot) = &model.slot_prior {
        let (a, b) = match slot.family {
            crate::model::SlotPriorFamily::BetaBinomial { a, b } => (a, b),
            crate::model::SlotPriorFamily::GammaPoisson { expected, shape } => (expected, shape),
        };
        if !a.is_finite()
            || a <= 0.
            || !b.is_finite()
            || b <= 0.
            || !slot.skip_threshold_multiplier.is_finite()
            || slot.skip_threshold_multiplier < 0.
            || slot.initial.as_ref().is_some_and(|v| {
                v.len() != l
                    || v.iter()
                        .any(|&x| !x.is_finite() || !(0.0..=1.0).contains(&x))
            })
        {
            return Err("Invalid slot prior parameters or initial probabilities".into());
        }
    }
    Ok(())
}

fn validate_fit_options(
    p: usize,
    xty: &[f64],
    yty: f64,
    n: f64,
    options: &FitOptions,
) -> Result<(), String> {
    if p == 0
        || options.l == 0
        || p.checked_mul(options.l).is_none()
        || !n.is_finite()
        || n <= 1.
        || !yty.is_finite()
        || yty < 0.
        || !input_validation::all_finite(xty)
    {
        return Err("Invalid sufficient statistics, sample size, or number of effects".into());
    }
    if options.max_iter == 0
        || !options.tol.is_finite()
        || options.tol < 0.0
        || options.prior_tol.is_nan()
        || options.prior_tol < 0.0
        || !options.prior_variance.is_finite()
        || options.prior_variance < 0.0
        || !options.residual_variance.is_finite()
        || options.residual_variance <= 0.0
        || !options.check_null_threshold.is_finite()
    {
        return Err("Invalid variance, convergence, or null-threshold parameter".into());
    }
    if options.prior_weights.len() != p
        || options
            .prior_weights
            .iter()
            .any(|w| !w.is_finite() || *w < 0.0)
        || sum(options.prior_weights.iter().copied()) <= 0.0
    {
        return Err("Prior weights must be finite, nonnegative, and nonzero".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::dot;
    fn options(p: usize) -> FitOptions {
        FitOptions {
            l: 1,
            prior_variance: 0.2,
            residual_variance: 1.0,
            prior_weights: vec![1.0 / p as f64; p],
            estimate_prior_variance: false,
            estimate_prior_method: PriorMethod::Optim,
            estimate_residual_variance: false,
            check_null_threshold: 0.0,
            max_iter: 100,
            tol: 1e-9,
            check_prior: false,
            prior_tol: 1e-9,
            null_index: None,
        }
    }

    #[test]
    fn fixed_mixture_nig_preserves_initialized_nig_fields() {
        let mut settings = options(2);
        settings.l = 2;
        let control = FitControl {
            model: crate::model::ModelOptions {
                mixture: Some(crate::model::MixturePrior {
                    variances: vec![0.02, 0.2, 1.],
                    weights: vec![0.2, 0.5, 0.3],
                }),
                residual_method: ResidualMethod::Nig,
                alpha0: Some(0.2),
                beta0: Some(0.3),
                ..crate::model::ModelOptions::default()
            },
            ..FitControl::default()
        };
        let result = fit_controlled(
            &[19., 2., 2., 19.],
            &[5., -3.],
            19.,
            20.,
            &settings,
            true,
            None,
            &control,
        )
        .unwrap();
        assert!(result.alpha.iter().all(|x| x.is_finite()));
        assert!(result.elbo.iter().all(|x| x.is_finite()));
        assert_eq!(result.kl, [0., 0.]);
        assert_eq!(result.rv.unwrap(), [1., 1.]);
        assert!(result.marginal_loglik.unwrap().iter().all(|x| x.is_nan()));
        for v in result.v {
            assert!((v - 0.404).abs() <= 1e-15);
        }
        settings.l = 1;
        let result = fit_controlled(
            &[19., 2., 2., 19.],
            &[5., -3.],
            19.,
            20.,
            &settings,
            true,
            None,
            &control,
        )
        .unwrap();
        assert!(result.alpha.iter().all(|x| x.is_finite()));
        assert!(result.elbo.iter().all(|x| x.is_nan()));
        assert!(result.kl[0].is_finite());
    }

    #[test]
    fn residual_estimation_respects_both_variance_bounds() {
        let mut settings = options(2);
        settings.estimate_residual_variance = true;
        settings.max_iter = 3;
        let control = FitControl {
            model: crate::model::ModelOptions {
                residual_lowerbound: 0.8,
                residual_upperbound: 1.2,
                ..crate::model::ModelOptions::default()
            },
            ..FitControl::default()
        };
        let high = fit_controlled(
            &[9., 0., 0., 9.],
            &[0., 0.],
            10000.,
            10.,
            &settings,
            false,
            None,
            &control,
        )
        .unwrap();
        assert_eq!(high.sigma2, 1.2);
        let low = fit_controlled(
            &[9., 0., 0., 9.],
            &[0., 0.],
            0.1,
            10.,
            &settings,
            false,
            None,
            &control,
        )
        .unwrap();
        assert_eq!(low.sigma2, 0.8);
    }

    #[test]
    fn prior_guard_uses_current_and_warm_start_residual_variance() {
        let mut settings = options(2);
        settings.prior_variance = 50.;
        settings.check_prior = true;
        settings.estimate_residual_variance = true;
        settings.max_iter = 1;
        let q = [9., 0., 0., 9.];
        for method in ["none", "EM"] {
            settings.estimate_prior_variance = method != "none";
            settings.estimate_prior_method = if method == "none" {
                PriorMethod::Optim
            } else {
                method.parse().unwrap()
            };
            let first = fit(&q, &[3., 0.], 100000., 10., &settings).unwrap();
            assert!(first.sigma2 > 1000.);
            assert!(first.v[0] < 100.);
            settings.max_iter = 2;
            let error = fit(&q, &[3., 0.], 100000., 10., &settings).err().unwrap();
            assert!(error.contains("prior variance is unreasonably large"));
            settings.estimate_residual_variance = false;
            assert!(fit(&q, &[3., 0.], 100000., 10., &settings).is_ok());
            settings.estimate_residual_variance = true;
            settings.max_iter = 1;
        }
        settings.estimate_prior_variance = false;
        settings.estimate_residual_variance = false;
        let control = FitControl {
            initial: Some(diagnostics::InitialState {
                alpha: vec![0.5, 0.5],
                mu: vec![0., 0.],
                mu2: vec![0., 0.],
                v: Some(vec![50.]),
                sigma2: Some(10000.),
            }),
            ..FitControl::default()
        };
        assert!(fit_controlled(
            &q,
            &[3., 0.],
            100000.,
            10.,
            &settings,
            false,
            None,
            &control
        )
        .err()
        .unwrap()
        .contains("prior variance is unreasonably large"));
    }

    #[test]
    fn no_information_prior_guard_preserves_empty_maximum() {
        let mut settings = options(2);
        settings.check_prior = true;
        let fit = fit(&[0.; 4], &[0.; 2], 99., 100., &settings).unwrap();
        assert!(fit.converged);
        assert_eq!(fit.v, vec![0.2]);
    }

    #[test]
    fn streamed_second_moments_match_materialized_identity() {
        // Compare with an independent implementation that materializes the
        // L × p second-moment terms. Fits include ordinary and strong signals,
        // fixed and estimated σ², and a zero-information column; the comparison
        // is checked against independently materialized moments.
        for strong in [false, true] {
            for estimate_residual in [false, true] {
                for degenerate in [false, true] {
                    let diagonal = [99.0, 99.0, if degenerate { 0.0 } else { 99.0 }];
                    let matrix = [
                        diagonal[0],
                        0.0,
                        0.0,
                        0.0,
                        diagonal[1],
                        0.0,
                        0.0,
                        0.0,
                        diagonal[2],
                    ];
                    let xty = [if strong { 50.0 } else { 3.0 }, -2.0, 0.0];
                    let mut settings = options(3);
                    settings.l = 2;
                    settings.estimate_residual_variance = estimate_residual;
                    let mut fitted = fit(&matrix, &xty, 99.0, 100.0, &settings).unwrap();
                    // ER2 must reconstruct the fitted sum, not reuse this
                    // incrementally maintained cache with different rounding.
                    fitted.xtxr.fill(1.0e20);
                    let mut products = vec![0.0; 6];
                    let mut means = vec![0.0; 3];
                    let mut prediction = vec![0.0; 3];
                    let mut quadratics = Vec::new();
                    let mut moments = Vec::new();
                    for l in 0..2 {
                        let offset = l * 3;
                        for j in 0..3 {
                            products[offset + j] =
                                diagonal[j] * (fitted.alpha[offset + j] * fitted.mu[offset + j]);
                        }
                        quadratics.push(sum((0..3).map(|j| {
                            fitted.alpha[offset + j] * fitted.mu[offset + j] * products[offset + j]
                        })));
                        for j in 0..3 {
                            let i = offset + j;
                            means[j] += fitted.alpha[i] * fitted.mu[i];
                            prediction[j] += products[i];
                            moments.push(diagonal[j] * (fitted.alpha[i] * fitted.mu2[i]));
                        }
                    }
                    let expected = 99.0 - 2.0 * dot(&means, &xty) + dot(&means, &prediction)
                        - sum(quadratics.into_iter())
                        + sum(moments.into_iter());
                    let mut scratch = ResidualScratch {
                        mean: vec![0.0; 3],
                        fitted: vec![0.0; 3],
                        component_quadratics: Vec::new(),
                    };
                    let actual = expected_residual_sum_squares(
                        &xty,
                        99.0,
                        &diagonal,
                        &fitted,
                        &products,
                        &mut scratch,
                    );
                    assert!((actual - expected).abs() <= 1e-12 * expected.abs().max(1.0));
                    assert!(actual.is_finite());
                }
            }
        }
    }

    #[test]
    fn fixed_single_effect_matches_closed_form() {
        let result = fit(
            &[10.0, 0.0, 0.0, 10.0],
            &[4.0, -1.0],
            20.0,
            21.0,
            &options(2),
        )
        .expect("fit");
        let variance = 1.0 / 15.0;
        assert!((result.mu[0] - 4.0 * variance).abs() < 1e-14);
        assert!((result.mu2[0] - variance - (4.0 * variance).powi(2)).abs() < 1e-14);
        let b0 = -0.5 * 3.0_f64.ln() + 0.5 * 1.6 * (2.0 / 3.0);
        let b1 = -0.5 * 3.0_f64.ln() + 0.5 * 0.1 * (2.0 / 3.0);
        assert!((result.lbf_variable[0] - b0).abs() < 1e-14);
        assert!((result.alpha[0] - 1.0 / (1.0 + (b1 - b0).exp())).abs() < 1e-14);
        assert_eq!(result.niter, 2);
        assert!(result.converged);
    }

    #[test]
    fn null_column_matches_upstream_probability_rule() {
        let mut settings = options(2);
        settings.prior_weights = vec![0.9, 0.1];
        let result =
            fit(&[10.0, 0.0, 0.0, 0.0], &[0.0, 0.0], 20.0, 21.0, &settings).expect("null fit");
        let real_weight = (0.9 + f64::EPSILON.sqrt()) / 3.0_f64.sqrt();
        let null_weight = 0.1 + f64::EPSILON.sqrt();
        assert!((result.alpha[1] - null_weight / (null_weight + real_weight)).abs() < 1e-14);
        assert_eq!(result.lbf_variable[1], 0.0);
        assert_eq!(result.mu[1], 0.0);
        assert_eq!(result.mu2[1], 0.0);
    }

    #[test]
    fn optim_single_variant_recovers_likelihood_optimum() {
        let mut settings = options(1);
        settings.estimate_prior_variance = true;
        let result = fit(&[10.0], &[8.0], 20.0, 21.0, &settings).expect("optimized fit");
        assert!((result.v[0] - (0.8_f64.powi(2) - 0.1)).abs() < 1e-7);
    }

    #[test]
    fn weak_effect_is_set_to_exact_zero() {
        let mut settings = options(2);
        settings.estimate_prior_variance = true;
        let result =
            fit(&[10.0, 1.0, 1.0, 10.0], &[0.0, 0.0], 20.0, 21.0, &settings).expect("weak fit");
        assert_eq!(result.v, vec![0.0]);
        assert_eq!(result.mu, vec![0.0, 0.0]);
        assert_eq!(result.alpha, vec![0.5, 0.5]);
    }

    #[test]
    fn final_trim_at_cap_restores_exact_prior_and_zeros_all_fields() {
        let mut settings = options(2);
        settings.max_iter = 1;
        settings.prior_variance = 1e-12;
        settings.prior_weights = vec![0.8, 0.2];
        let result = fit(&[10.0, 0.0, 0.0, 10.0], &[1.0, -2.0], 20.0, 21.0, &settings).unwrap();
        assert!(!result.converged);
        assert_eq!(result.alpha, settings.prior_weights);
        for values in [
            &result.v,
            &result.mu,
            &result.mu2,
            &result.lbf_variable,
            &result.lbf,
            &result.kl,
        ] {
            assert!(values.iter().all(|&x| x == 0.0));
        }
        assert_eq!(result.elbo.len(), 1);
        // Trace and fitted products refer to the state before trimming, as in susieR.
        assert!(result.xtxr.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn gaussian_em_does_not_apply_optim_null_test() {
        let mut settings = options(1);
        settings.estimate_prior_variance = true;
        settings.estimate_prior_method = PriorMethod::Em;
        settings.check_null_threshold = 1e6;
        settings.max_iter = 1;
        let result = fit(&[10.0], &[1.0], 20.0, 21.0, &settings).unwrap();
        assert!(result.v[0] > settings.prior_tol);
        assert_eq!(result.v[0], result.mu2[0]);
    }

    #[test]
    fn em_updates_variance_after_posterior() {
        let mut settings = options(1);
        settings.estimate_prior_variance = true;
        settings.estimate_prior_method = PriorMethod::Em;
        settings.max_iter = 1;
        let result = fit(&[10.0], &[8.0], 20.0, 21.0, &settings).expect("EM fit");
        let old_posterior_variance = 1.0 / 15.0;
        assert!((result.mu[0] - 8.0 * old_posterior_variance).abs() < 1e-14);
        assert_eq!(result.v[0], result.mu2[0]);
        assert!((result.mu[0] - 8.0 / (1.0 / result.v[0] + 10.0)).abs() > 0.01);
    }

    #[test]
    fn simple_only_compares_current_variance_with_zero() {
        let mut settings = options(1);
        settings.estimate_prior_variance = true;
        settings.estimate_prior_method = PriorMethod::Simple;
        let signal = fit(&[10.0], &[8.0], 20.0, 21.0, &settings).expect("simple signal");
        assert_eq!(signal.v[0], 0.2);
        settings.check_null_threshold = 100.0;
        let null = fit(&[10.0], &[8.0], 20.0, 21.0, &settings).expect("simple null");
        assert_eq!(null.v[0], 0.0);
    }

    #[test]
    fn zero_prior_weight_retains_upstream_epsilon_floor() {
        let mut settings = options(2);
        settings.prior_weights = vec![1.0, 0.0];
        settings.prior_variance = 0.0;
        settings.prior_tol = 0.0; // Inspect untrimmed SER epsilon handling.
        let result = fit(&[10.0, 0.0, 0.0, 10.0], &[0.0, 0.0], 20.0, 21.0, &settings)
            .expect("zero prior fit");
        let eps = f64::EPSILON.sqrt();
        assert!(result.alpha[1] > 0.0);
        assert!((result.alpha[1] - eps / (1.0 + 2.0 * eps)).abs() < 1e-22);
    }

    #[test]
    fn sequential_updates_use_new_effects() {
        let mut settings = options(2);
        settings.l = 2;
        settings.max_iter = 1;
        let matrix = [10.0, 4.0, 4.0, 10.0];
        let result = fit(&matrix, &[8.0, -4.0], 50.0, 51.0, &settings).expect("sequential fit");
        let b0 = result.alpha[0] * result.mu[0];
        let b1 = result.alpha[1] * result.mu[1];
        assert!((result.mu[2] - (8.0 - 10.0 * b0 - 4.0 * b1) / 15.0).abs() < 1e-14);
        assert_ne!(result.mu[0], result.mu[2]);
        assert!(!result.converged);
    }

    #[test]
    fn singular_design_and_large_bf_remain_finite() {
        let result = fit(
            &[100.0, 100.0, 100.0, 100.0],
            &[500.0, 500.0],
            10000.0,
            101.0,
            &options(2),
        )
        .expect("singular fit");
        assert_eq!(result.alpha, vec![0.5, 0.5]);
        assert!(result.lbf[0] > 1000.0);
        assert!(result.elbo.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn residual_variance_updates_after_objective() {
        let mut settings = options(1);
        settings.max_iter = 1;
        settings.estimate_residual_variance = true;
        let result = fit(&[10.0], &[4.0], 20.0, 21.0, &settings).expect("residual fit");
        let expected = (20.0 - 8.0 * result.mu[0] + 10.0 * result.mu2[0]) / 21.0;
        assert!((result.sigma2 - expected).abs() < 1e-14);
        let expected_elbo = -21.0 / 2.0 * (2.0 * PI).ln() - 21.0 * expected / 2.0 - result.kl[0];
        assert!((result.elbo[0] - expected_elbo).abs() < 1e-13);
    }
}
