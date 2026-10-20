//! Expected Gaussian residual sum of squares.
use crate::{
    numerical::{dot, sum},
    FitOutput,
};
pub(crate) struct ResidualScratch {
    pub(crate) mean: Vec<f64>,
    pub(crate) fitted: Vec<f64>,
    pub(crate) component_quadratics: Vec<f64>,
}

/// Expected residual sum of squares E‖y − Xb‖² under the variational posterior:
/// yᵀy − 2 b̄ᵀXᵀy + b̄ᵀXᵀX b̄ − ∑_ℓ m_ℓᵀXᵀX m_ℓ + ∑_ℓ ∑_j Q_jj α_ℓj mu2_ℓj,
/// with m_ℓ = α_ℓ ⊙ μ_ℓ and b̄ = ∑_ℓ m_ℓ, using the stored products Q m_ℓ.
pub(crate) fn expected_residual_sum_squares(
    xty: &[f64],
    yty: f64,
    diagonal: &[f64],
    fit: &FitOutput,
    component_products: &[f64],
    scratch: &mut ResidualScratch,
) -> f64 {
    let p = xty.len();
    let ResidualScratch {
        mean,
        fitted,
        component_quadratics,
    } = scratch;
    mean.fill(0.0);
    fitted.fill(0.0);
    component_quadratics.clear();
    for effect in 0..fit.v.len() {
        let offset = effect * p;
        component_quadratics.push(sum((0..p).map(|j| {
            fit.alpha[offset + j] * fit.mu[offset + j] * component_products[offset + j]
        })));
        for j in 0..p {
            let idx = offset + j;
            mean[j] += fit.alpha[idx] * fit.mu[idx];
            fitted[j] += component_products[idx];
        }
    }
    // Compensated summation reduces cancellation without an L × p temporary. By linearity, ∑_ℓ Q m_ℓ = Q b̄; every product
    // was formed after its effect's last update.
    yty - 2.0 * dot(mean, xty) + dot(mean, fitted) - sum(component_quadratics.iter().copied())
        + sum((0..fit.v.len()).flat_map(|effect| {
            let offset = effect * p;
            (0..p).map(move |j| diagonal[j] * (fit.alpha[offset + j] * fit.mu2[offset + j]))
        }))
}
