//! Normal-inverse-gamma single-effect regression (susieR BSD-3-Clause).
//! See THIRD_PARTY_NOTICES.md for the upstream attribution.
use crate::{
    numerical::{digamma, sum},
    options::PriorMethod,
    ser::SingleEffect,
};
use std::f64::consts::PI;

pub(crate) struct NigPosterior<'a> {
    pub posterior: SingleEffect<'a>,
    pub residual_mean: f64,
    pub marginal_loglik: f64,
    pub kl: f64,
}

/// Null integrated Gaussian likelihood under IG(alpha0/2,beta0/2).
pub(crate) fn null_loglik(n: f64, yy: f64, alpha0: f64, beta0: f64) -> f64 {
    let a0 = alpha0 / 2.;
    let b0 = beta0 / 2.;
    let a1 = (alpha0 + n) / 2.;
    let b1 = (beta0 + yy) / 2.;
    -n / 2. * (2. * PI).ln() + a0 * b0.ln() - libm::lgamma(a0) - a1 * b1.ln() + libm::lgamma(a1)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn single_effect_nig<'a>(
    xtr: &[f64],
    diagonal: &[f64],
    inflation: Option<&[f64]>,
    yy: f64,
    n: f64,
    v: f64,
    alpha0: f64,
    beta0: f64,
    log_prior: &[f64],
    prior_weights: &[f64],
    method: PriorMethod,
    mut posterior: SingleEffect<'a>,
) -> Result<NigPosterior<'a>, String> {
    let p = xtr.len();
    let a1 = n + alpha0;
    let mut r0 = vec![0.; p];
    let mut rss = vec![yy; p];
    let mut residual_means = vec![1.; p];
    for j in 0..p {
        if diagonal[j] > 0. {
            let tau = inflation.map_or(1., |t| t[j]);
            r0[j] = v / (v + tau / diagonal[j]);
            let correlation = (xtr[j] / (diagonal[j] * yy).sqrt()).clamp(-1., 1.);
            rss[j] = yy * (1. - r0[j] * correlation.powi(2));
            posterior.lbf[j] = -0.5
                * ((1. + v * diagonal[j] / tau).ln() + a1 * ((beta0 + rss[j]) / (beta0 + yy)).ln());
        } else {
            posterior.lbf[j] = 0.;
        }
        posterior.alpha[j] = posterior.lbf[j] + log_prior[j];
    }
    let maximum = posterior
        .alpha
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    for a in posterior.alpha.iter_mut() {
        *a = (*a - maximum).exp();
    }
    let total = sum(posterior.alpha.iter().copied());
    for a in posterior.alpha.iter_mut() {
        *a /= total;
    }
    for j in 0..p {
        if diagonal[j] > 0. && v > 0. {
            let rv = (beta0 + rss[j]) / (a1 - 2.);
            let mean = r0[j] * xtr[j] / diagonal[j];
            posterior.mu[j] = mean;
            posterior.mu2[j] =
                rv * r0[j] * inflation.map_or(1., |t| t[j]) / diagonal[j] + mean.powi(2);
            residual_means[j] = rv;
        } else {
            posterior.mu[j] = 0.;
            posterior.mu2[j] = 0.;
        }
    }
    posterior.model_lbf = maximum + total.ln();
    let residual_mean = if v <= 0. {
        1.
    } else {
        sum(posterior
            .alpha
            .iter()
            .zip(&residual_means)
            .map(|(a, r)| a * r))
    };
    let kl = if v > 0. {
        let apost = a1 / 2.;
        let bpost = (beta0 + sum(posterior.alpha.iter().zip(&rss).map(|(a, r)| a * r))) / 2.;
        let kg = sum(posterior
            .alpha
            .iter()
            .zip(prior_weights)
            .map(|(&a, &prior)| a * (a.max(f64::EPSILON).ln() - prior.max(f64::EPSILON).ln())));
        let kb = 0.5
            * sum((0..p).map(|j| {
                let sj = if diagonal[j] > 0. {
                    r0[j] * inflation.map_or(1., |t| t[j]) / diagonal[j]
                } else {
                    0.
                };
                posterior.alpha[j]
                    * (v.ln() - sj.max(f64::EPSILON).ln()
                        + sj / v
                        + posterior.mu[j].powi(2) * apost / bpost / v
                        - 1.)
            }));
        let a0 = alpha0 / 2.;
        let b0 = beta0 / 2.;
        kg + kb + libm::lgamma(a0) - libm::lgamma(apost)
            + a0 * (bpost / b0).ln()
            + (apost - a0) * digamma(apost)
            - apost
            + apost * b0 / bpost
    } else {
        0.
    };
    if method == PriorMethod::Em {
        // Algebraically simplify the upstream gamma/beta expression:
        // Var(b/sigma)+E(b/sigma)^2 = r0/xx + mean²*(n+alpha0)/(beta0+RSS).
        // This avoids overflow of gamma ratios for large sample sizes.
        posterior.v = sum((0..p).filter(|&j| diagonal[j] > 0.).map(|j| {
            posterior.alpha[j]
                * (r0[j] * inflation.map_or(1., |t| t[j]) / diagonal[j]
                    + posterior.mu[j].powi(2) * a1 / (beta0 + rss[j]))
        }));
    } else {
        posterior.v = v;
    }
    let marginal_loglik = posterior.model_lbf + null_loglik(n, yy, alpha0, beta0);
    if !marginal_loglik.is_finite()
        || posterior
            .alpha
            .iter()
            .chain(posterior.mu.iter())
            .chain(posterior.mu2.iter())
            .any(|x| !x.is_finite())
    {
        return Err("Non-finite NIG posterior; check sample size and sufficient statistics".into());
    }
    Ok(NigPosterior {
        posterior,
        residual_mean,
        marginal_loglik,
        kl,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nig_moments_and_em_match_conjugate_normal_inverse_gamma() {
        let (mut alpha, mut mu, mut mu2, mut bf) = ([0.], [0.], [0.], [0.]);
        let result = single_effect_nig(
            &[3.],
            &[5.],
            None,
            12.,
            10.,
            0.2,
            0.3,
            0.4,
            &[0.],
            &[1.],
            PriorMethod::Em,
            SingleEffect {
                alpha: &mut alpha,
                mu: &mut mu,
                mu2: &mut mu2,
                lbf: &mut bf,
                model_lbf: 0.,
                v: 0.2,
            },
        )
        .unwrap();
        let shrink = 0.5;
        let mean = shrink * 3. / 5.;
        let rss = 12. - shrink * 9. / 5.;
        let rv = (0.4 + rss) / 8.3;
        assert_eq!(result.posterior.alpha, [1.]);
        assert!((result.posterior.mu[0] - mean).abs() < 1e-14);
        assert!((result.posterior.mu2[0] - rv * shrink / 5. - mean * mean).abs() < 1e-14);
        assert!((result.residual_mean - rv).abs() < 1e-14);
        assert!(
            (result.posterior.v - shrink / 5. - mean * mean * 10.3 / (0.4 + rss)).abs() < 1e-14
        );
    }
}
