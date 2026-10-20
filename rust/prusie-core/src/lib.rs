//! Gaussian SuSiE solver, independent of Python and its memory model.
pub mod diagnostics;
mod ibss;
pub mod input_validation;
mod numerical;
mod operator;
mod optimizer;
pub mod options;
mod residual;
mod ser;
mod types;

pub use ibss::fit_controlled;
pub use operator::MatrixScaling;
pub use types::{FitOptions, FitOutput};

/// Requested native thread count, sampled once for the process.
pub fn num_threads() -> usize {
    static THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *THREADS.get_or_init(|| {
        std::env::var("PRUSIE_NUM_THREADS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1)
            .clamp(1, 32)
    })
}
