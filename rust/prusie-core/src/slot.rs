//! Slot activity updates (susieR BSD-3-Clause; see THIRD_PARTY_NOTICES.md).
use crate::{
    model::{SlotPrior, SlotPriorFamily, UpdateSchedule},
    numerical::{digamma, sum},
};

pub(crate) struct SlotState {
    prior: SlotPrior,
    gamma_shape: f64,
    gamma_rate: f64,
    pub threshold: f64,
}

impl SlotState {
    pub fn new(prior: &SlotPrior, l: usize) -> (Self, Vec<f64>) {
        let mean = match prior.family {
            SlotPriorFamily::BetaBinomial { a, b } => a / (a + b),
            SlotPriorFamily::GammaPoisson { expected, .. } => expected / l as f64,
        };
        let weights = prior
            .initial
            .clone()
            .unwrap_or_else(|| vec![mean.min(1. - 1e-10); l]);
        let (shape, rate) = match prior.family {
            SlotPriorFamily::BetaBinomial { .. } => (0., 0.),
            SlotPriorFamily::GammaPoisson { expected, shape } => (
                shape
                    + if prior.initial.is_some() {
                        sum(weights.iter().copied())
                    } else {
                        expected
                    },
                shape / expected.max(1e-6) + 1.,
            ),
        };
        (
            Self {
                prior: prior.clone(),
                gamma_shape: shape,
                gamma_rate: rate,
                threshold: 0.,
            },
            weights,
        )
    }
    pub fn update(&mut self, weights: &mut [f64], effect: usize, lbf: f64) {
        let l = weights.len() as f64;
        let lbf = if lbf.is_finite() { lbf } else { 0. };
        let logodds = match self.prior.family {
            SlotPriorFamily::BetaBinomial { a, b } => {
                let others = sum(weights
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != effect)
                    .map(|(_, x)| *x));
                (a + others).ln() - (b + l - 1. - others).ln() + lbf
            }
            SlotPriorFamily::GammaPoisson { .. } => {
                digamma(self.gamma_shape) - self.gamma_rate.ln() - l.ln() + lbf
            }
        }
        .clamp(-20., 20.);
        weights[effect] = 1. / (1. + (-logodds).exp());
        if let SlotPriorFamily::GammaPoisson { shape, .. } = self.prior.family {
            if self.prior.schedule == UpdateSchedule::Sequential {
                self.gamma_shape = shape + sum(weights.iter().copied());
            }
        }
    }
    pub fn finish_sweep(&mut self, weights: &[f64]) {
        if let SlotPriorFamily::GammaPoisson { shape, .. } = self.prior.family {
            if self.prior.schedule == UpdateSchedule::Batch {
                self.gamma_shape = shape + sum(weights.iter().copied());
            }
        }
        if self.prior.skip_threshold_multiplier > 0. {
            let l = weights.len() as f64;
            let logodds = match self.prior.family {
                SlotPriorFamily::BetaBinomial { a, b } => {
                    let total = sum(weights.iter().copied());
                    let approx = (a + total).ln() - (b + l - 1. - total).ln();
                    let others = total - 1. / (1. + (-approx).exp());
                    (a + others).ln() - (b + l - 1. - others).ln()
                }
                SlotPriorFamily::GammaPoisson { .. } => {
                    digamma(self.gamma_shape) - self.gamma_rate.ln() - l.ln()
                }
            };
            self.threshold = self.prior.skip_threshold_multiplier / (1. + (-logodds).exp());
        }
    }
    pub fn objective(&self, weights: &[f64]) -> f64 {
        let l = weights.len() as f64;
        let k = sum(weights.iter().copied());
        let entropy = -sum(weights.iter().map(|&c| {
            let c = c.clamp(f64::EPSILON, 1. - f64::EPSILON);
            c * c.ln() + (1. - c) * (1. - c).ln()
        }));
        let prior = match self.prior.family {
            SlotPriorFamily::BetaBinomial { a, b } => {
                libm::lgamma(a + k) + libm::lgamma(b + l - k)
                    - libm::lgamma(a + b + l)
                    - libm::lgamma(a)
                    - libm::lgamma(b)
                    + libm::lgamma(a + b)
            }
            SlotPriorFamily::GammaPoisson { expected, shape } => {
                let a = self.gamma_shape;
                let b = self.gamma_rate;
                let logmean = digamma(a) - b.ln();
                let mean = a / b;
                let rate = shape / expected.max(1e-10);
                (shape - 1.) * logmean - rate * mean + shape * rate.ln() - libm::lgamma(shape) + a
                    - b.ln()
                    + libm::lgamma(a)
                    + (1. - a) * digamma(a)
                    + k * (logmean - l.ln())
            }
        };
        prior + entropy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn beta_binomial_update_matches_conditional_odds() {
        let prior = SlotPrior {
            family: SlotPriorFamily::BetaBinomial { a: 1., b: 2. },
            schedule: UpdateSchedule::Sequential,
            initial: Some(vec![0.2, 0.3, 0.4]),
            skip_threshold_multiplier: 0.,
        };
        let (mut state, mut weights) = SlotState::new(&prior, 3);
        state.update(&mut weights, 0, 2.);
        let odds = (1. + 0.7) / (2. + 2. - 0.7) * 2_f64.exp();
        assert!((weights[0] - odds / (1. + odds)).abs() < 1e-14);
    }
}
