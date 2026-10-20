//! Gaussian single-effect regression (susieR BSD-3-Clause; see THIRD_PARTY_NOTICES).
use crate::{
    numerical::{dot, sum},
    optimizer,
    options::PriorMethod,
};
/// Gaussian SER log Bayes factors and unnormalized log posterior weights.
/// Uninformative variants have log BF zero; finite sampling variances are floored
/// at machine epsilon, following the pinned Gaussian SER model.
fn log_probabilities_into(
    prior_variance: f64,
    betahat: &[f64],
    shat2: &[f64],
    log_prior: &[f64],
    lbf: &mut [f64],
    lpo: &mut [f64],
) {
    for ((((&beta, &s2), &prior), bf_out), po_out) in
        betahat.iter().zip(shat2).zip(log_prior).zip(lbf).zip(lpo)
    {
        let bf = if !beta.is_finite() || !s2.is_finite() {
            0.0
        } else {
            let safe = s2.max(f64::EPSILON);
            -0.5 * (1.0 + prior_variance / safe).ln()
                + 0.5 * (beta * beta) * prior_variance / (safe * (prior_variance + safe))
        };
        *bf_out = bf;
        *po_out = bf + prior;
    }
}

fn log_sum_exp(x: &mut [f64]) -> f64 {
    log_sum_exp_at(x, None)
}

fn log_sum_exp_at(x: &mut [f64], known_maximum: Option<f64>) -> f64 {
    let largest =
        known_maximum.unwrap_or_else(|| x.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    crate::vector_math::exp_shift(x, largest);
    largest + sum(x.iter().copied()).ln()
}

fn log_likelihood(
    v: f64,
    beta: &[f64],
    shat2: &[f64],
    log_prior: &[f64],
    scratch: &mut [f64],
) -> f64 {
    // The same per-variant Bayes factor arithmetic as log_probabilities, followed
    // by log_sum_exp; the prior-variance search needs only this marginal value.
    for (((&beta, &s2), &prior), value) in beta
        .iter()
        .zip(shat2)
        .zip(log_prior)
        .zip(scratch.iter_mut())
    {
        let bf = if !beta.is_finite() || !s2.is_finite() {
            0.0
        } else {
            let safe = s2.max(f64::EPSILON);
            -0.5 * (1.0 + v / safe).ln() + 0.5 * (beta * beta) * v / (safe * (v + safe))
        };
        *value = bf + prior;
    }
    log_sum_exp(scratch)
}

#[cfg(feature = "ser-grouped")]
struct VarianceGroups {
    indices: Vec<usize>,
    variances: Vec<f64>,
}
#[cfg(feature = "ser-grouped")]
impl VarianceGroups {
    fn new(sampling: &[f64]) -> Option<Self> {
        let mut result = Self {
            indices: Vec::with_capacity(sampling.len()),
            variances: Vec::new(),
        };
        for &s in sampling {
            if !s.is_finite() {
                result.indices.push(usize::MAX);
                continue;
            }
            let safe = s.max(f64::EPSILON);
            let index = if let Some(index) = result.variances.iter().position(|&v| v == safe) {
                index
            } else {
                if result.variances.len() == 16 {
                    return None;
                }
                result.variances.push(safe);
                result.variances.len() - 1
            };
            result.indices.push(index);
        }
        Some(result)
    }
    fn terms(&self, v: f64) -> ([f64; 16], [f64; 16]) {
        let mut constant = [0.; 16];
        let mut scale = [0.; 16];
        for (k, &s) in self.variances.iter().enumerate() {
            constant[k] = -0.5 * (1. + v / s).ln();
            scale[k] = v / (s * (s + v));
        }
        (constant, scale)
    }
}
#[cfg(feature = "ser-grouped")]
fn grouped_probabilities(
    v: f64,
    beta: &[f64],
    log_prior: &[f64],
    groups: &VarianceGroups,
    lbf: &mut [f64],
    lpo: &mut [f64],
) {
    let (constant, scale) = groups.terms(v);
    for j in 0..beta.len() {
        let k = groups.indices[j];
        let bf = if k == usize::MAX || !beta[j].is_finite() {
            0.
        } else {
            constant[k] + 0.5 * beta[j].powi(2) * scale[k]
        };
        lbf[j] = bf;
        lpo[j] = bf + log_prior[j];
    }
}
#[cfg(feature = "ser-grouped")]
fn grouped_likelihood(
    v: f64,
    beta: &[f64],
    log_prior: &[f64],
    groups: &VarianceGroups,
    scratch: &mut [f64],
) -> f64 {
    let (constant, scale) = groups.terms(v);
    for j in 0..beta.len() {
        let k = groups.indices[j];
        let bf = if k == usize::MAX || !beta[j].is_finite() {
            0.
        } else {
            constant[k] + 0.5 * beta[j].powi(2) * scale[k]
        };
        scratch[j] = bf + log_prior[j];
    }
    log_sum_exp(scratch)
}

/// Sampling variances are invariant between residual-variance updates.
pub(crate) struct SerSampling {
    sigma2: f64,
    pub(crate) shat2: Vec<f64>,
    #[cfg(feature = "ser-grouped")]
    groups: Option<VarianceGroups>,
}
impl SerSampling {
    fn new(diagonal: &[f64], sigma2: f64) -> Self {
        let shat2: Vec<f64> = diagonal.iter().map(|d| sigma2 / d).collect();
        #[cfg(feature = "ser-grouped")]
        let groups = VarianceGroups::new(&shat2);
        Self {
            sigma2,
            shat2,
            #[cfg(feature = "ser-grouped")]
            groups,
        }
    }
}
/// Mutable rows of the fit arrays for one single effect. A row is overwritten
/// only after the effect's previous matrix product has been subtracted from the
/// fitted crossproduct.
pub(crate) struct SingleEffect<'a> {
    pub(crate) alpha: &'a mut [f64],
    pub(crate) mu: &'a mut [f64],
    pub(crate) mu2: &'a mut [f64],
    pub(crate) lbf: &'a mut [f64],
    pub(crate) model_lbf: f64,
    pub(crate) v: f64,
}

/// One SER update of a single effect from its crossproduct residual `xtr`:
/// choose V by `method` ("optim", "simple", "EM" or "none"), then write the
/// posterior inclusion probabilities, conditional moments and log Bayes factors.
// Statistical inputs and caller-owned scratch and output buffers are passed explicitly.
#[allow(clippy::too_many_arguments)]
pub(crate) fn single_effect<'a>(
    xtr: &[f64],
    diagonal: &[f64],
    v_init: f64,
    sigma2: f64,
    log_prior: &[f64],
    method: PriorMethod,
    null_threshold: f64,
    likelihood_scratch: &mut [f64],
    sampling_cache: &mut Option<SerSampling>,
    betahat: &mut [f64],
    mut posterior: SingleEffect<'a>,
) -> Result<SingleEffect<'a>, String> {
    for ((&d, &r), beta) in diagonal.iter().zip(xtr).zip(betahat.iter_mut()) {
        *beta = (1.0 / d) * r;
    }
    // Sampling variances depend on σ² and are rebuilt after it is updated;
    // the matrix diagonal is fixed for the whole fit.
    if sampling_cache
        .as_ref()
        .is_none_or(|sampling| sampling.sigma2 != sigma2)
    {
        *sampling_cache = Some(SerSampling::new(diagonal, sigma2));
    }
    let sampling = sampling_cache.as_ref().expect("sampling initialized");
    let shat2 = &sampling.shat2;
    let mut likelihood = |v| {
        #[cfg(feature = "ser-grouped")]
        if let Some(groups) = &sampling.groups {
            return grouped_likelihood(v, betahat, log_prior, groups, likelihood_scratch);
        }
        log_likelihood(v, betahat, shat2, log_prior, likelihood_scratch)
    };
    let mut v = v_init;
    let mut selected_likelihood = None;
    if method == PriorMethod::Optim {
        let (log_v, objective) = optimizer::minimize(|lv| -likelihood(lv.exp()), -30.0, 15.0)?;
        let candidate = log_v.exp();
        let old_likelihood = likelihood(v_init);
        if objective <= -old_likelihood {
            v = candidate;
            selected_likelihood = Some(-objective);
        } else {
            selected_likelihood = Some(old_likelihood);
        }
    }
    if method != PriorMethod::None && method != PriorMethod::Em {
        let null_likelihood = likelihood(0.0);
        let current_likelihood = selected_likelihood.unwrap_or_else(|| likelihood(v));
        if null_likelihood + null_threshold >= current_likelihood {
            v = 0.0;
        }
    }
    #[cfg(feature = "ser-grouped")]
    if let Some(groups) = &sampling.groups {
        grouped_probabilities(
            v,
            betahat,
            log_prior,
            groups,
            posterior.lbf,
            posterior.alpha,
        );
    } else {
        log_probabilities_into(v, betahat, shat2, log_prior, posterior.lbf, posterior.alpha);
    }
    #[cfg(not(feature = "ser-grouped"))]
    log_probabilities_into(v, betahat, shat2, log_prior, posterior.lbf, posterior.alpha);
    let largest = posterior
        .alpha
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    // The row holds log weights, then weights, then normalized α; the rows of
    // the other effects are not touched during this update.
    crate::vector_math::exp_shift(posterior.alpha, largest);
    let total = sum(posterior.alpha.iter().copied());
    for value in posterior.alpha.iter_mut() {
        *value /= total;
    }
    for ((((&d, &r), &s2), mean), second) in diagonal
        .iter()
        .zip(xtr)
        .zip(shat2)
        .zip(posterior.mu.iter_mut())
        .zip(posterior.mu2.iter_mut())
    {
        let beta = r / d;
        let (post_variance, post_mean) = if beta.is_finite() && s2.is_finite() {
            let variance = v * s2 / (v + s2);
            (variance, variance / s2 * beta)
        } else {
            (0.0, 0.0)
        };
        *mean = post_mean;
        // mu2 = E[a_ℓ² | γ_ℓ = j], the conditional second moment.
        *second = post_variance + post_mean * post_mean;
    }
    if method == PriorMethod::Em {
        // EM sets V to the posterior second moment after the posterior has been
        // computed; KL uses that posterior. EM has no comparison with V = 0.
        v = dot(posterior.alpha, posterior.mu2);
    }
    let model_lbf = largest + total.ln();
    if !model_lbf.is_finite()
        || posterior
            .mu
            .iter()
            .chain(posterior.mu2.iter())
            .any(|x| !x.is_finite())
    {
        return Err("Non-finite single-effect posterior; check sufficient statistics".into());
    }
    posterior.model_lbf = model_lbf;
    posterior.v = v;
    Ok(posterior)
}

/// Fixed-mixture SER; the grid is expressed in working coefficient units.
/// The p × K grid matrix is stored by variant then mixture component.
#[allow(clippy::too_many_arguments)]
pub(crate) fn single_effect_mixture<'a>(
    xtr: &[f64],
    diagonal: &[f64],
    sigma2: f64,
    log_prior: &[f64],
    mixture: &crate::model::MixturePrior,
    grid_lbf: &mut [f64],
    mut posterior: SingleEffect<'a>,
) -> Result<SingleEffect<'a>, String> {
    let k_count = mixture.variances.len();
    let mut responsibilities = vec![0.; k_count];
    for j in 0..xtr.len() {
        let beta = xtr[j] / diagonal[j];
        let sampling = sigma2 / diagonal[j];
        let safe = sampling.max(f64::EPSILON);
        let informative = beta.is_finite() && sampling.is_finite();
        for (k, (&variance, &weight)) in mixture.variances.iter().zip(&mixture.weights).enumerate()
        {
            let bf = if informative {
                -0.5 * (1. + variance / safe).ln()
                    + 0.5 * beta.powi(2) * variance / (safe * (variance + safe))
            } else {
                0.
            };
            grid_lbf[j * k_count + k] = bf;
            responsibilities[k] = if weight == 0. {
                f64::NEG_INFINITY
            } else {
                weight.ln() + bf
            };
        }
        let bf = if k_count == 1 {
            grid_lbf[j * k_count]
        } else {
            log_sum_exp(&mut responsibilities)
        };
        posterior.lbf[j] = bf;
        posterior.alpha[j] = bf + log_prior[j];
        // Responsibilities use the same epsilon stabilization as the pinned
        // mixture moment calculation. It is numerical stabilization, not a
        // change to the fitted mixture weights.
        for k in 0..k_count {
            responsibilities[k] =
                grid_lbf[j * k_count + k] + (mixture.weights[k] + f64::EPSILON).ln();
        }
        let maximum = responsibilities
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        crate::vector_math::exp_shift(&mut responsibilities, maximum);
        let total = sum(responsibilities.iter().copied());
        let mut mean = 0.;
        let mut second = 0.;
        if informative {
            for (k, &variance) in mixture.variances.iter().enumerate() {
                let post_var = variance * safe / (variance + safe);
                let post_mean = post_var / safe * beta;
                mean += responsibilities[k] / total * post_mean;
                second += responsibilities[k] / total * (post_var + post_mean.powi(2));
            }
        }
        posterior.mu[j] = mean;
        posterior.mu2[j] = second;
    }
    let maximum = posterior
        .alpha
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    crate::vector_math::exp_shift(posterior.alpha, maximum);
    let total = sum(posterior.alpha.iter().copied());
    for a in posterior.alpha.iter_mut() {
        *a /= total;
    }
    posterior.model_lbf = maximum + total.ln();
    posterior.v = dot(&mixture.weights, &mixture.variances);
    if !posterior.model_lbf.is_finite() {
        return Err("Non-finite mixture posterior".into());
    }
    Ok(posterior)
}

/// Perform one untrimmed SER update for model-specific outer algorithms.
///
/// `inflation[j]` multiplies the sampling variance of variant j; centering,
/// fitting residuals and changes to an LD operator belong to the outer model.
/// `yy` is the residual sum of squares excluding this effect, used only by NIG.
#[allow(clippy::too_many_arguments)]
pub fn ser_update(
    xtr: &[f64],
    diagonal: &[f64],
    inflation: Option<&[f64]>,
    yy: f64,
    n: f64,
    options: &crate::FitOptions,
    model: &crate::model::ModelOptions,
    hard_exclusion: bool,
) -> Result<crate::SerOutput, String> {
    let p = xtr.len();
    if p == 0
        || diagonal.len() != p
        || options.prior_weights.len() != p
        || xtr.iter().any(|x| !x.is_finite())
        || diagonal.iter().any(|&x| !x.is_finite() || x < 0.)
        || options
            .prior_weights
            .iter()
            .any(|&x| !x.is_finite() || x < 0.)
        || sum(options.prior_weights.iter().copied()) <= 0.
        || !options.prior_variance.is_finite()
        || options.prior_variance < 0.
        || !options.residual_variance.is_finite()
        || options.residual_variance <= 0.
        || inflation.is_some_and(|v| v.len() != p || v.iter().any(|&x| !x.is_finite() || x <= 0.))
    {
        return Err("Invalid single-effect statistics, priors, variance, or inflation".into());
    }
    crate::ibss::validate_model(model, n, 1)?;
    let log_prior: Vec<f64> = options
        .prior_weights
        .iter()
        .map(|&w| {
            if hard_exclusion && w == 0. {
                f64::NEG_INFINITY
            } else {
                (w + f64::EPSILON.sqrt()).ln()
            }
        })
        .collect();
    let mut result = crate::SerOutput {
        alpha: vec![0.; p],
        mu: vec![0.; p],
        mu2: vec![0.; p],
        lbf_variable: vec![0.; p],
        lbf: 0.,
        v: options.prior_variance,
        lbf_grid: model
            .mixture
            .as_ref()
            .map(|m| vec![0.; p * m.variances.len()]),
        rv: None,
        marginal_loglik: None,
        kl: None,
    };
    let posterior = SingleEffect {
        alpha: &mut result.alpha,
        mu: &mut result.mu,
        mu2: &mut result.mu2,
        lbf: &mut result.lbf_variable,
        model_lbf: 0.,
        v: options.prior_variance,
    };
    let nig = model.residual_method == crate::options::ResidualMethod::Nig;
    let method = if options.estimate_prior_variance {
        if nig {
            PriorMethod::Em
        } else {
            options.estimate_prior_method
        }
    } else {
        PriorMethod::None
    };
    let adjusted_diagonal;
    let adjusted_xtr;
    let (working_diagonal, working_xtr) = if let Some(tau) = inflation {
        adjusted_diagonal = diagonal
            .iter()
            .zip(tau)
            .map(|(d, t)| d / t)
            .collect::<Vec<f64>>();
        adjusted_xtr = xtr
            .iter()
            .zip(tau)
            .map(|(r, t)| r / t)
            .collect::<Vec<f64>>();
        (adjusted_diagonal.as_slice(), adjusted_xtr.as_slice())
    } else {
        (diagonal, xtr)
    };
    let posterior = if let Some(mixture) = &model.mixture {
        single_effect_mixture(
            working_xtr,
            working_diagonal,
            options.residual_variance,
            &log_prior,
            mixture,
            result.lbf_grid.as_mut().expect("mixture grid"),
            posterior,
        )?
    } else if nig {
        if !n.is_finite() || n <= 1. || !yy.is_finite() || yy <= 0. {
            return Err("NIG SER requires n > 1 and positive residual sum of squares".into());
        }
        let updated = crate::nig::single_effect_nig(
            xtr,
            diagonal,
            inflation,
            yy,
            n,
            options.prior_variance,
            model.alpha0.unwrap_or(1. / n.sqrt()),
            model.beta0.unwrap_or(1. / n.sqrt()),
            &log_prior,
            &options.prior_weights,
            method,
            posterior,
        )?;
        result.rv = Some(updated.residual_mean);
        result.marginal_loglik = Some(updated.marginal_loglik);
        result.kl = Some(updated.kl);
        updated.posterior
    } else {
        let mut scratch = vec![0.; p];
        let mut sampling = None;
        let mut beta = vec![0.; p];
        single_effect(
            working_xtr,
            working_diagonal,
            options.prior_variance,
            options.residual_variance,
            &log_prior,
            method,
            options.check_null_threshold,
            &mut scratch,
            &mut sampling,
            &mut beta,
            posterior,
        )?
    };
    result.lbf = posterior.model_lbf;
    result.v = posterior.v;
    if nig && model.mixture.is_some() {
        // Fixed-mixture SER precedes NIG conjugacy in susieR's dispatch and
        // retains these initialized NIG fields for the outer algorithm.
        result.rv = Some(1.);
        result.marginal_loglik = Some(f64::NAN);
        result.kl = Some(if options.l > 1 {
            0.
        } else {
            crate::nig::kl_from_posterior(
                xtr,
                diagonal,
                inflation,
                yy,
                n,
                options.prior_variance,
                model.alpha0.unwrap_or(1. / n.sqrt()),
                model.beta0.unwrap_or(1. / n.sqrt()),
                &options.prior_weights,
                &result.alpha,
                &result.mu,
            )
        });
    }
    Ok(result)
}

#[cfg(test)]
fn log_probabilities(v: f64, beta: &[f64], shat2: &[f64], prior: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut bf = vec![0.0; beta.len()];
    let mut po = vec![0.0; beta.len()];
    log_probabilities_into(v, beta, shat2, prior, &mut bf, &mut po);
    (bf, po)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bayes_factor_matches_normal_density_ratio() {
        let variance: f64 = 0.2;
        let beta: [f64; 3] = [0.0, 0.7, -2.0];
        let sampling: [f64; 3] = [0.4, 0.01, 0.25];
        let (actual, _) = log_probabilities(variance, &beta, &sampling, &[0.0; 3]);
        for j in 0..beta.len() {
            let expected = 0.5 * (sampling[j].ln() - (sampling[j] + variance).ln())
                + 0.5 * beta[j].powi(2) * (1.0 / sampling[j] - 1.0 / (sampling[j] + variance));
            assert!((actual[j] - expected).abs() <= 2e-12);
        }
    }
}

#[cfg(test)]
mod mixture_tests {
    use super::*;
    use crate::model::MixturePrior;
    #[test]
    fn one_point_mixture_reduces_to_gaussian_oracle() {
        let xtr = [2., -3., 0.];
        let diagonal = [4., 5., 2.];
        let log_prior = [0.2_f64.ln(), 0.3_f64.ln(), 0.5_f64.ln()];
        let (mut a, mut mu, mut mu2, mut lbf) = ([0.; 3], [0.; 3], [0.; 3], [0.; 3]);
        let mut grid = [0.; 3];
        let posterior = single_effect_mixture(
            &xtr,
            &diagonal,
            0.8,
            &log_prior,
            &MixturePrior {
                variances: vec![0.7],
                weights: vec![1.],
            },
            &mut grid,
            SingleEffect {
                alpha: &mut a,
                mu: &mut mu,
                mu2: &mut mu2,
                lbf: &mut lbf,
                model_lbf: 0.,
                v: 0.,
            },
        )
        .unwrap();
        let raw: Vec<f64> = (0..3)
            .map(|j| {
                let s2 = 0.8 / diagonal[j];
                let beta = xtr[j] / diagonal[j];
                let bf =
                    0.5 * (s2 / (s2 + 0.7)).ln() + beta.powi(2) / 2. * (1. / s2 - 1. / (s2 + 0.7));
                (bf + log_prior[j]).exp()
            })
            .collect();
        let total: f64 = raw.iter().sum();
        for j in 0..3 {
            assert!((posterior.alpha[j] - raw[j] / total).abs() < 2e-14);
            assert!((posterior.mu[j] - 0.7 * xtr[j] / (0.8 + 0.7 * diagonal[j])).abs() < 2e-14);
            assert_eq!(posterior.lbf[j], grid[j]);
        }
    }
}
