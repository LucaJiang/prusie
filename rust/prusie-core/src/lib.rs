//! Gaussian SuSiE solver, independent of Python and its memory model.
mod design;
pub mod diagnostics;
mod exact_columns;
mod ibss;
pub mod input_validation;
pub mod model;
mod nig;
mod numerical;
mod operator;
mod optimizer;
pub mod options;
mod residual;
mod ser;
mod slot;
mod types;
mod vector_math;

pub use design::{fit_design_csc, fit_design_dense, fit_design_trend};
pub use ibss::fit_controlled;
pub use operator::MatrixScaling;
pub use ser::ser_update;
pub use types::{FitOptions, FitOutput, SerOutput};

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

/// Exponential implementation actually selected on the calling thread.
pub fn ser_math_backend() -> &'static str {
    vector_math::backend_name()
}

mod activation;
