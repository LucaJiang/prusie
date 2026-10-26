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

/// Sum nonnegative exponentiated likelihood weights by a balanced tree.
/// Scratch is overwritten before every objective evaluation; signed KL and
/// residual sums and posterior normalization retain compensated accumulation.
fn positive_pairwise_sum(values: &mut [f64]) -> f64 {
    let mut length = values.len();
    while length > 1 {
        let pairs = length / 2;
        for j in 0..pairs {
            values[j] = values[2 * j] + values[2 * j + 1];
        }
        // Keep the parity test paired with the remainder in the unchanged tree update.
        #[allow(clippy::manual_is_multiple_of)]
        if length % 2 != 0 {
            values[pairs] = values[length - 1];
        }
        length = pairs + length % 2;
    }
    values.first().copied().unwrap_or(0.0)
}

fn log_sum_exp(x: &mut [f64]) -> f64 {
    log_sum_exp_at(x, None)
}

fn log_sum_exp_at(x: &mut [f64], known_maximum: Option<f64>) -> f64 {
    let largest =
        known_maximum.unwrap_or_else(|| x.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    crate::vector_math::exp_shift(x, largest);
    largest + positive_pairwise_sum(x).ln()
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

struct VarianceGroups {
    indices: Vec<usize>,
    variances: Vec<f64>,
}
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
// Four-wide objective arithmetic uses separate multiplies/adds (no FMA).
// SAFETY: AVX2 must be available. Slices have equal length, terms.len() <= 16,
// and each group is either usize::MAX or a valid terms index. Storage is Rust-owned.
#[cfg(all(has_vector_math, target_arch = "x86_64"))]
#[inline(never)]
#[target_feature(enable = "avx2")]
unsafe fn ser_group_fill_avx2(
    half_beta_squared: &[f64],
    groups: &[usize],
    terms: &[(f64, f64, f64)],
    prior: f64,
    output: &mut [f64],
) {
    use std::arch::x86_64::*;
    if terms.len() > 4 {
        for ((&half, &group), value) in half_beta_squared.iter().zip(groups).zip(output) {
            let bf = if group == usize::MAX {
                0.0
            } else {
                let (constant, _, scale) = terms[group];
                constant + half * scale
            };
            *value = bf + prior;
        }
        return;
    }
    let mut constants = [0.0_f64; 4];
    let mut scales = [0.0_f64; 4];
    for (j, &(constant, _, scale)) in terms.iter().enumerate() {
        constants[j] = constant;
        scales[j] = scale;
    }
    let constant_table = _mm256_castpd_si256(_mm256_loadu_pd(constants.as_ptr()));
    let scale_table = _mm256_castpd_si256(_mm256_loadu_pd(scales.as_ptr()));
    let prior4 = _mm256_set1_pd(prior);
    let invalid_index = _mm256_set1_epi64x(-1);
    let mut j = 0;
    while j + 4 <= output.len() {
        let group4 = _mm256_loadu_si256(groups.as_ptr().add(j).cast());
        let invalid = _mm256_cmpeq_epi64(group4, invalid_index);
        let indices = _mm256_andnot_si256(invalid, group4);
        // A double occupies two adjacent 32-bit permutation entries.
        let even = _mm256_slli_epi64::<1>(indices);
        let odd = _mm256_add_epi64(even, _mm256_set1_epi64x(1));
        let lanes = _mm256_or_si256(even, _mm256_slli_epi64::<32>(odd));
        let constant = _mm256_castsi256_pd(_mm256_permutevar8x32_epi32(constant_table, lanes));
        let scale = _mm256_castsi256_pd(_mm256_permutevar8x32_epi32(scale_table, lanes));
        let half = _mm256_andnot_pd(
            _mm256_castsi256_pd(invalid),
            _mm256_loadu_pd(half_beta_squared.as_ptr().add(j)),
        );
        let bf = _mm256_add_pd(constant, _mm256_mul_pd(half, scale));
        let bf = _mm256_andnot_pd(_mm256_castsi256_pd(invalid), bf);
        _mm256_storeu_pd(output.as_mut_ptr().add(j), _mm256_add_pd(bf, prior4));
        j += 4;
    }
    for ((&half, &group), value) in half_beta_squared[j..]
        .iter()
        .zip(&groups[j..])
        .zip(&mut output[j..])
    {
        let bf = if group == usize::MAX {
            0.0
        } else {
            let (constant, _, scale) = terms[group];
            constant + half * scale
        };
        *value = bf + prior;
    }
}

/// One scalar variance search reuses beta-dependent invariants. The storage is
/// owned by this public fit and is refreshed before each component's search.
/// Exact variance groups are unchanged; heterogeneous/exceptional inputs retain
/// the scalar objective, without rounded or approximate grouping.
struct SerObjective<'a> {
    beta: &'a [f64],
    shat2: &'a [f64],
    log_prior: &'a [f64],
    groups: Option<&'a VarianceGroups>,
    half_beta_squared: &'a mut Vec<f64>,
    group_maximum: [f64; 16],
    uniform_log_prior: Option<f64>,
    has_no_information: bool,
    #[cfg(all(has_vector_math, target_arch = "x86_64"))]
    group_fill_avx2: bool,
}

impl<'a> SerObjective<'a> {
    fn new(beta: &'a [f64], sampling: &'a mut SerSampling, log_prior: &'a [f64]) -> Self {
        let groups = sampling
            .groups
            .as_ref()
            .filter(|_| beta.iter().all(|x| x.is_finite()));
        let mut result = Self {
            beta,
            shat2: &sampling.shat2,
            log_prior,
            groups,
            half_beta_squared: &mut sampling.half_beta_squared,
            group_maximum: [0.; 16],
            uniform_log_prior: None,
            has_no_information: false,
            #[cfg(all(has_vector_math, target_arch = "x86_64"))]
            group_fill_avx2: crate::vector_math::group_fill_avx2_allowed(),
        };
        let Some(groups) = groups else {
            return result;
        };
        result.half_beta_squared.clear();
        result
            .half_beta_squared
            .extend(beta.iter().map(|&b| 0.5 * (b * b)));
        result.uniform_log_prior = log_prior
            .first()
            .copied()
            .filter(|&prior| log_prior.iter().all(|&x| x == prior));
        if result.uniform_log_prior.is_some() {
            for (&half, &group) in result.half_beta_squared.iter().zip(&groups.indices) {
                if group == usize::MAX {
                    result.has_no_information = true;
                } else if !half.is_finite() {
                    result.uniform_log_prior = None;
                    break;
                } else if half > result.group_maximum[group] {
                    result.group_maximum[group] = half;
                }
            }
        }
        result
    }

    fn evaluate(&mut self, v: f64, scratch: &mut [f64]) -> f64 {
        let Some(groups) = self.groups else {
            return log_likelihood(v, self.beta, self.shat2, self.log_prior, scratch);
        };
        let mut terms = [(0., 0., 0.); 16];
        for (&safe, term) in groups.variances.iter().zip(&mut terms) {
            let denominator = safe * (v + safe);
            *term = (-0.5 * (1. + v / safe).ln(), denominator, v / denominator);
            // Preserve the original multiply-then-divide expression when
            // distributing the division could lose a representable value.
            if !denominator.is_finite() || !(term.2.is_normal() || (v == 0. && term.2 == 0.)) {
                return log_likelihood(v, self.beta, self.shat2, self.log_prior, scratch);
            }
        }
        let terms = &terms[..groups.variances.len()];
        if v >= 0. {
            if let Some(prior) = self.uniform_log_prior {
                if v == 0. && prior.is_finite() {
                    // Preserve the prior normalization's actual rounding.
                    return prior + (self.beta.len() as f64).ln();
                }
                let mut largest = if self.has_no_information {
                    prior
                } else {
                    f64::NEG_INFINITY
                };
                for (&half, &(constant, _, scale)) in self.group_maximum.iter().zip(terms) {
                    largest = largest.max(constant + half * scale + prior);
                }
                #[cfg(all(has_vector_math, target_arch = "x86_64"))]
                if terms.len() <= 4 && self.group_fill_avx2 {
                    // SAFETY: runtime AVX2 and caller FP controls are checked
                    // per search. Exact group indices and all slice lengths
                    // come from this fit's validated sampling layout.
                    unsafe {
                        ser_group_fill_avx2(
                            self.half_beta_squared,
                            &groups.indices,
                            terms,
                            prior,
                            scratch,
                        );
                    }
                    return log_sum_exp_at(scratch, Some(largest));
                }
                for ((&half, &group), value) in self
                    .half_beta_squared
                    .iter()
                    .zip(&groups.indices)
                    .zip(scratch.iter_mut())
                {
                    let bf = if group == usize::MAX {
                        0.
                    } else {
                        let (constant, _, scale) = terms[group];
                        constant + half * scale
                    };
                    *value = bf + prior;
                }
                return log_sum_exp_at(scratch, Some(largest));
            }
        }
        for (((&half, &group), &prior), value) in self
            .half_beta_squared
            .iter()
            .zip(&groups.indices)
            .zip(self.log_prior)
            .zip(scratch.iter_mut())
        {
            let bf = if group == usize::MAX {
                0.
            } else {
                let (constant, _, scale) = terms[group];
                constant + half * scale
            };
            *value = bf + prior;
        }
        log_sum_exp(scratch)
    }
}

/// Sampling variances are invariant between residual-variance updates.
pub(crate) struct SerSampling {
    sigma2: f64,
    pub(crate) shat2: Vec<f64>,
    groups: Option<VarianceGroups>,
    half_beta_squared: Vec<f64>,
}
impl SerSampling {
    fn new(diagonal: &[f64], sigma2: f64) -> Self {
        let shat2: Vec<f64> = diagonal.iter().map(|d| sigma2 / d).collect();
        let groups = VarianceGroups::new(&shat2);
        Self {
            sigma2,
            shat2,
            groups,
            half_beta_squared: Vec::new(),
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
    let sampling = sampling_cache.as_mut().expect("sampling initialized");
    // Fixed-V and EM never evaluate the scalar objective and pay no invariant
    // preparation cost. No cache or fitted state survives this public call.
    let mut context = if method != PriorMethod::None && method != PriorMethod::Em {
        Some(SerObjective::new(betahat, sampling, log_prior))
    } else {
        None
    };
    let mut likelihood = |v| {
        context
            .as_mut()
            .expect("objective requested")
            .evaluate(v, likelihood_scratch)
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
    // End the objective's mutable storage borrow before posterior construction.
    #[allow(clippy::drop_non_drop)]
    drop(likelihood);
    #[allow(clippy::drop_non_drop)]
    drop(context);
    let shat2 = &sampling.shat2;
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

    fn compare_objective(beta: &[f64], diagonal: &[f64], prior: &[f64]) {
        let mut sampling = SerSampling::new(diagonal, 0.8);
        let shat2 = sampling.shat2.clone();
        let mut objective = SerObjective::new(beta, &mut sampling, prior);
        for v in [0., 1e-12, 0.03, 0.7, 1000.] {
            let mut scratch = vec![0.; beta.len()];
            let actual = objective.evaluate(v, &mut scratch);
            let expected = log_likelihood(v, beta, &shat2, prior, &mut scratch);
            assert!(
                (actual - expected).abs() <= 4e-12 * (1. + expected.abs()),
                "V={v}, actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn objective_exact_groups_cover_uniform_weighted_and_scalar_paths() {
        for groups in [1, 2, 4, 7, 17] {
            let beta: Vec<f64> = (0..35).map(|j| (j as f64 - 15.) / 9.).collect();
            let diagonal: Vec<f64> = (0..35).map(|j| 1. + (j % groups) as f64).collect();
            let uniform = vec![-(35_f64).ln(); 35];
            let weighted: Vec<f64> = (0..35)
                .map(|j| {
                    if j % 5 == 0 {
                        f64::NEG_INFINITY
                    } else {
                        -3. - j as f64 / 50.
                    }
                })
                .collect();
            compare_objective(&beta, &diagonal, &uniform);
            compare_objective(&beta, &diagonal, &weighted);
        }
    }

    #[test]
    fn objective_handles_no_information_and_resets_reused_beta_maxima() {
        let diagonal = [2., 2., 0., 3., 3.];
        let prior = [-2.; 5];
        let strong = [10., -20., 0., -8., 4.];
        let weak = [0., 0., 0., 0., 0.];
        let mut sampling = SerSampling::new(&diagonal, 0.8);
        for beta in [&strong, &weak, &strong, &weak] {
            let shat2 = sampling.shat2.clone();
            let mut objective = SerObjective::new(beta, &mut sampling, &prior);
            assert!(objective.has_no_information);
            let mut scratch = [0.; 5];
            let actual = objective.evaluate(0.2, &mut scratch);
            let expected = log_likelihood(0.2, beta, &shat2, &prior, &mut scratch);
            assert!((actual - expected).abs() <= 2e-12);
            assert_eq!(objective.evaluate(0., &mut scratch), -2. + 5_f64.ln());
        }
        compare_objective(&[1., f64::INFINITY, 0., -0.5, 0.7], &diagonal, &prior);
    }

    #[test]
    fn objective_exceptional_coefficients_retain_scalar_expression() {
        for diagonal in [[1e-200; 3], [1e200; 3]] {
            let beta = [0., 0.5, 1.];
            let prior = [-3.; 3];
            compare_objective(&beta, &diagonal, &prior);
        }
        let beta = [1e155, 0., 1.];
        let mut sampling = SerSampling::new(&[2.; 3], 0.8);
        let mut objective = SerObjective::new(&beta, &mut sampling, &[-3.; 3]);
        assert!(objective.uniform_log_prior.is_none());
        // Overflowing squares must not take the uniform-null shortcut.
        let mut scratch = [0.; 3];
        assert!(objective.evaluate(0., &mut scratch).is_nan());
    }

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
