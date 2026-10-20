//! Model extensions of the Gaussian single-effect prior.
//! susieR BSD-3-Clause; see THIRD_PARTY_NOTICES.md for the upstream attribution.
use crate::options::ResidualMethod;

/// A fixed mixture of positive Gaussian slab variances.
#[derive(Clone, Debug)]
pub struct MixturePrior {
    pub variances: Vec<f64>,
    pub weights: Vec<f64>,
}

/// Optional model features; the default is ordinary Gaussian SuSiE.
#[derive(Clone, Debug)]
pub struct ModelOptions {
    pub mixture: Option<MixturePrior>,
    pub residual_method: ResidualMethod,
    pub residual_lowerbound: f64,
    pub residual_upperbound: f64,
    pub alpha0: Option<f64>,
    pub beta0: Option<f64>,
    pub slot_prior: Option<SlotPrior>,
}

impl Default for ModelOptions {
    fn default() -> Self {
        Self {
            mixture: None,
            residual_method: ResidualMethod::Mom,
            residual_lowerbound: 0.,
            residual_upperbound: f64::INFINITY,
            alpha0: None,
            beta0: None,
            slot_prior: None,
        }
    }
}

/// Prior family for the probability that a single-effect slot is active.
#[derive(Clone, Debug)]
pub enum SlotPriorFamily {
    BetaBinomial { a: f64, b: f64 },
    GammaPoisson { expected: f64, shape: f64 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UpdateSchedule {
    Sequential,
    Batch,
}

/// Slot activity prior, including its initialization and update schedule.
#[derive(Clone, Debug)]
pub struct SlotPrior {
    pub family: SlotPriorFamily,
    pub schedule: UpdateSchedule,
    pub initial: Option<Vec<f64>>,
    pub skip_threshold_multiplier: f64,
}
