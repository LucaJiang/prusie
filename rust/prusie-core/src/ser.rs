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

fn log_sum_exp(x: &mut [f64]) -> f64 {
    log_sum_exp_at(x, None)
}

fn log_sum_exp_at(x: &mut [f64], known_maximum: Option<f64>) -> f64 {
    let largest =
        known_maximum.unwrap_or_else(|| x.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    for value in x.iter_mut() {
        *value = (*value - largest).exp();
    }
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

/// Sampling variances are invariant between residual-variance updates.
pub(crate) struct SerSampling {
    sigma2: f64,
    pub(crate) shat2: Vec<f64>,
}
impl SerSampling {
    fn new(diagonal: &[f64], sigma2: f64) -> Self {
        Self {
            sigma2,
            shat2: diagonal.iter().map(|d| sigma2 / d).collect(),
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
    let shat2 = &sampling_cache.as_ref().expect("sampling initialized").shat2;
    let mut likelihood = |v| log_likelihood(v, betahat, shat2, log_prior, likelihood_scratch);
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
    log_probabilities_into(v, betahat, shat2, log_prior, posterior.lbf, posterior.alpha);
    let largest = posterior
        .alpha
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    // The row holds log weights, then weights, then normalized α; the rows of
    // the other effects are not touched during this update.
    for value in posterior.alpha.iter_mut() {
        *value = (*value - largest).exp();
    }
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
