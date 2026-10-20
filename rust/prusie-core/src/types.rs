use crate::{diagnostics::Diagnostics, options::PriorMethod};

/// Validated and standardized fitting options received from Python.
pub struct FitOptions {
    pub l: usize,
    pub prior_variance: f64,
    pub residual_variance: f64,
    pub prior_weights: Vec<f64>,
    pub estimate_prior_variance: bool,
    pub estimate_prior_method: PriorMethod,
    pub estimate_residual_variance: bool,
    pub check_null_threshold: f64,
    pub max_iter: usize,
    pub tol: f64,
    pub check_prior: bool,
    pub prior_tol: f64,
    pub null_index: Option<usize>,
}

/// Fit results; L × p arrays are stored row by row (one row per single effect).
pub struct FitOutput {
    pub alpha: Vec<f64>,
    pub mu: Vec<f64>,
    pub mu2: Vec<f64>,
    pub lbf_variable: Vec<f64>,
    pub lbf: Vec<f64>,
    pub v: Vec<f64>,
    pub kl: Vec<f64>,
    pub elbo: Vec<f64>,
    pub sigma2: f64,
    pub niter: usize,
    pub converged: bool,
    pub xtxr: Vec<f64>,
    pub warnings: Vec<String>,
    pub diagnostics: Diagnostics,
    /// Fixed-mixture log Bayes factors, shape L × p × K, row-major.
    pub lbf_grid: Option<Vec<f64>>,
    /// Posterior slot activity probabilities (one per effect), when requested.
    pub slot_weights: Option<Vec<f64>>,
    /// NIG per-effect posterior residual variance means.
    pub rv: Option<Vec<f64>>,
    /// Per-effect marginal log likelihood under the NIG prior.
    pub marginal_loglik: Option<Vec<f64>>,
}

/// One untrimmed single-effect posterior, for model-specific outer algorithms.
pub struct SerOutput {
    pub alpha: Vec<f64>,
    pub mu: Vec<f64>,
    pub mu2: Vec<f64>,
    pub lbf_variable: Vec<f64>,
    pub lbf: f64,
    pub v: f64,
    pub lbf_grid: Option<Vec<f64>>,
    pub rv: Option<f64>,
    pub marginal_loglik: Option<f64>,
    pub kl: Option<f64>,
}
