//! Native sufficient-statistic SuSiE solver following pinned susieR 0.16.6 (Gaussian scope).
//!
//! Rust owns the complete sequential IBSS loop, single-effect posterior,
//! variance updates, expected residual sum of squares, and ELBO. Python owns
//! RSS transformations, standardization, metadata, and credible sets.

#[path = "optimizer.rs"]
mod optimizer;
pub(crate) mod input_validation;

#[path = "vector_math.rs"]
mod vector_math;

pub fn ser_math_backend() -> &'static str { vector_math::backend_name() }

use ndarray::{ArrayView2, ShapeBuilder};
use std::borrow::Cow;
use std::f64::consts::PI;

/// Already validated and standardized parameters at the Python/Rust boundary.
pub struct FitOptions {
    pub l: usize,
    pub prior_variance: f64,
    pub residual_variance: f64,
    pub prior_weights: Vec<f64>,
    pub estimate_prior_variance: bool,
    pub estimate_prior_method: String,
    pub estimate_residual_variance: bool,
    pub check_null_threshold: f64,
    pub max_iter: usize,
    pub tol: f64,
    pub check_prior: bool,
    pub prior_tol: f64,
    pub null_index: Option<usize>,
}

/// Native output arrays use component-major (row-major L by p) storage.
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
}

/// Compensated scalar reductions protect cancellation in likelihood/objective
/// sums. Dense matrix products use the library's contiguous reduction freely.
fn sum(values: impl Iterator<Item = f64>) -> f64 {
    let mut total: f64 = 0.0;
    let mut correction = 0.0;
    for value in values {
        let next = total + value;
        correction += if total.abs() >= value.abs() {
            (total - next) + value
        } else {
            (value - next) + total
        };
        total = next;
    }
    total + correction
}

fn dot(x: &[f64], y: &[f64]) -> f64 {
    sum(x.iter().zip(y).map(|(x, y)| x * y))
}

/// Current alpha/PIP fallback: fixed point or cycle of at most five states.
/// History is populated only when the ELBO difference is nonfinite.
fn fallback_convergence(
    alpha: &mut [f64], previous: &[f64], v: &[f64], p: usize,
    prior_tol: f64, null_index: Option<usize>, tol: f64, history: &mut Vec<(Vec<f64>, Vec<f64>)>,
) -> bool {
    let pip = |a: &[f64]| -> Vec<f64> {
        (0..p).filter(|&j| Some(j) != null_index).map(|j| 1.0 - (0..v.len()).filter(|&l| v[l] > prior_tol)
            .map(|l| 1.0 - a[l*p+j]).product::<f64>()).collect()
    };
    if history.is_empty() { history.push((previous.to_vec(), pip(previous))); }
    let current_pip = pip(alpha);
    let mut converged = false;
    for lag in 1..=history.len().min(5) {
        let (old_alpha, old_pip) = &history[history.len()-lag];
        let difference = alpha.iter().zip(old_alpha).chain(current_pip.iter().zip(old_pip))
            .map(|(a,b)| (a-b).abs()).fold(0.0_f64, f64::max);
        if difference < tol {
            if lag > 1 {
                for j in 0..alpha.len() {
                    alpha[j] = (alpha[j] + history.iter().rev().take(lag-1)
                        .map(|(a,_)| a[j]).sum::<f64>()) / lag as f64;
                }
            }
            converged = true;
            break;
        }
    }
    history.push((alpha.to_vec(), pip(alpha)));
    if history.len() > 5 { history.remove(0); }
    converged
}

/// Gaussian SER BF, with epsilon flooring and no-information handling.
fn log_probabilities_into(
    prior_variance: f64,
    betahat: &[f64],
    shat2: &[f64],
    log_prior: &[f64],
    lbf: &mut [f64],
    lpo: &mut [f64],
) {
    for ((((&beta, &s2), &prior), bf_out), po_out) in betahat.iter().zip(shat2)
        .zip(log_prior).zip(lbf).zip(lpo) {
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

/// The existing B07 balanced reduction for nonnegative objective weights.
/// This composition leaves posterior and signed KL/ER2 sums compensated.
fn positive_pairwise_sum(values: &mut [f64]) -> f64 {
    let mut length = values.len();
    while length > 1 {
        let pairs = length / 2;
        for j in 0..pairs { values[j] = values[2*j] + values[2*j+1]; }
        if length % 2 != 0 { values[pairs] = values[length-1]; }
        length = pairs + length % 2;
    }
    values.first().copied().unwrap_or(0.0)
}

fn log_sum_exp(x: &mut [f64]) -> f64 {
    log_sum_exp_at(x, None)
}

fn log_sum_exp_at(x: &mut [f64], known_maximum: Option<f64>) -> f64 {
    let largest = known_maximum.unwrap_or_else(|| x.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    // Apply the optional-vector/scalar exponential first, then the existing
    // B07 balanced positive reduction. Posterior and signed sums are unchanged.
    vector_math::exp_shift_inplace(x, largest);
    largest + positive_pairwise_sum(x).ln()
}

fn log_likelihood(
    v: f64, beta: &[f64], shat2: &[f64], log_prior: &[f64], scratch: &mut [f64],
) -> f64 {
    // Same per-SNP BF arithmetic as log_probabilities, then log_sum_exp.
    // Only the posterior log weights are needed during the V search.
    for (((&beta, &s2), &prior), value) in beta.iter().zip(shat2).zip(log_prior).zip(scratch.iter_mut()) {
        let bf = if !beta.is_finite() || !s2.is_finite() { 0.0 } else {
            let safe = s2.max(f64::EPSILON);
            -0.5 * (1.0 + v / safe).ln()
                + 0.5 * (beta * beta) * v / (safe * (v + safe))
        };
        *value = bf + prior;
    }
    log_sum_exp(scratch)
}

/// Fit-local sampling variances and exact group layout. The residual variance
/// is part of its identity; all storage belongs to this one public fit.
// Exact four-wide objective arithmetic. This prototype uses no FMA or reordering.
// SAFETY: AVX2 must be available. Slices have equal length, terms.len() <= 16,
// and each group is either usize::MAX or a valid terms index. Storage is Rust-owned.
#[cfg(all(has_vector_math, target_arch = "x86_64"))]
#[inline(never)]
#[target_feature(enable = "avx2")]
unsafe fn ser_group_fill_avx2(half_beta_squared: &[f64], groups: &[usize],
    terms: &[(f64,f64,f64)], prior: f64, output: &mut [f64]) {
    use std::arch::x86_64::*;
    if terms.len() > 4 {
        for ((&half, &group), value) in half_beta_squared.iter().zip(groups).zip(output) {
            let bf = if group == usize::MAX { 0.0 } else {
                let (constant, _, scale) = terms[group]; constant + half * scale
            };
            *value = bf + prior;
        }
        return;
    }
    let mut constants = [0.0_f64; 4];
    let mut scales = [0.0_f64; 4];
    for (j, &(constant, _, scale)) in terms.iter().enumerate() {
        constants[j] = constant; scales[j] = scale;
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
        let half = _mm256_andnot_pd(_mm256_castsi256_pd(invalid),
                                   _mm256_loadu_pd(half_beta_squared.as_ptr().add(j)));
        let bf = _mm256_add_pd(constant, _mm256_mul_pd(half, scale));
        let bf = _mm256_andnot_pd(_mm256_castsi256_pd(invalid), bf);
        _mm256_storeu_pd(output.as_mut_ptr().add(j), _mm256_add_pd(bf, prior4));
        j += 4;
    }
    for ((&half, &group), value) in half_beta_squared[j..].iter().zip(&groups[j..]).zip(&mut output[j..]) {
        let bf = if group == usize::MAX { 0.0 } else {
            let (constant, _, scale) = terms[group]; constant + half * scale
        };
        *value = bf + prior;
    }
}

struct SerSampling {
    sigma2: f64,
    shat2: Vec<f64>,
    group_index: Vec<usize>,
    safe_groups: Vec<f64>,
    grouped: bool,
}

impl SerSampling {
    fn new(diagonal: &[f64], sigma2: f64, prepare_groups: bool) -> Self {
        let shat2: Vec<f64> = diagonal.iter().map(|&d| sigma2 / d).collect();
        let mut result = Self { sigma2, shat2, group_index: Vec::new(),
                                safe_groups: Vec::new(), grouped: prepare_groups };
        if !prepare_groups { return result; }
        result.group_index.reserve(diagonal.len());
        for &s2 in &result.shat2 {
            if !s2.is_finite() {
                result.group_index.push(usize::MAX);
                continue;
            }
            let safe = s2.max(f64::EPSILON);
            let group = match result.safe_groups.iter().position(|&s| s == safe) {
                Some(group) => group,
                None => {
                    if result.safe_groups.len() == 16 {
                        result.grouped = false;
                        break;
                    }
                    result.safe_groups.push(safe);
                    result.safe_groups.len() - 1
                }
            };
            result.group_index.push(group);
        }
        result
    }
}

/// Reusable search storage owned by one fit. Only capacities persist between
/// components: every beta-dependent value and group maximum is rebuilt.
#[derive(Default)]
struct SerBuffers {
    half_beta_squared: Vec<f64>,
    group_index: Vec<usize>,
    safe_groups: Vec<f64>,
    terms: Vec<(f64, f64, f64)>,
    group_max_half_beta_squared: Vec<f64>,
}

/// A SER search changes V while beta, s² and prior weights stay fixed. Exact
/// equal s² values share logarithms/denominators; no rounding or binning occurs.
/// Standardized RSS inputs often have a few distinct floating s² values.
struct SerLikelihood<'a> {
    beta: &'a [f64],
    shat2: &'a [f64],
    log_prior: &'a [f64],
    half_beta_squared: Vec<f64>,
    group_index: Cow<'a, [usize]>,
    safe_groups: Cow<'a, [f64]>,
    terms: Vec<(f64, f64, f64)>,
    grouped: bool,
    uniform_log_prior: Option<f64>,
    group_max_half_beta_squared: Vec<f64>,
    has_no_information: bool,
}

impl<'a> SerLikelihood<'a> {
    #[cfg(test)]
    fn from_sampling(beta: &'a [f64], sampling: &'a SerSampling, log_prior: &'a [f64]) -> Self {
        Self::from_sampling_reusing(beta, sampling, log_prior, &mut SerBuffers::default())
    }

    fn from_sampling_reusing(beta: &'a [f64], sampling: &'a SerSampling,
        log_prior: &'a [f64], buffers: &mut SerBuffers) -> Self {
        // Nonfinite beta can remove a variance group in the original path.
        // Retain that exact handling, including zero-information columns.
        if !sampling.grouped || beta.iter().any(|b| !b.is_finite()) {
            return Self::new_reusing(beta, &sampling.shat2, log_prior, buffers);
        }
        let mut result = Self { beta, shat2: &sampling.shat2, log_prior,
            half_beta_squared: std::mem::take(&mut buffers.half_beta_squared),
            group_index: Cow::Borrowed(&sampling.group_index),
            safe_groups: Cow::Borrowed(&sampling.safe_groups),
            terms: std::mem::take(&mut buffers.terms), grouped: true,
            uniform_log_prior: None,
            group_max_half_beta_squared: std::mem::take(&mut buffers.group_max_half_beta_squared),
            has_no_information: false };
        result.half_beta_squared.clear();
        result.half_beta_squared.extend(beta.iter().map(|&b| 0.5 * (b * b)));
        result.terms.resize(sampling.safe_groups.len(), (0.0, 0.0, 0.0));
        result.group_max_half_beta_squared.clear();
        result.prepare_maximum();
        result
    }

    #[cfg(test)]
    fn new(beta: &'a [f64], shat2: &'a [f64], log_prior: &'a [f64]) -> Self {
        Self::new_reusing(beta, shat2, log_prior, &mut SerBuffers::default())
    }

    fn new_reusing(beta: &'a [f64], shat2: &'a [f64], log_prior: &'a [f64],
        buffers: &mut SerBuffers) -> Self {
        let mut result = Self { beta, shat2, log_prior,
            half_beta_squared: std::mem::take(&mut buffers.half_beta_squared),
            group_index: Cow::Owned(std::mem::take(&mut buffers.group_index)),
            safe_groups: Cow::Owned(std::mem::take(&mut buffers.safe_groups)),
            terms: std::mem::take(&mut buffers.terms), grouped: true,
            uniform_log_prior: None,
            group_max_half_beta_squared: std::mem::take(&mut buffers.group_max_half_beta_squared),
            has_no_information: false };
        result.half_beta_squared.clear();
        result.half_beta_squared.reserve(beta.len());
        result.group_index.to_mut().clear();
        result.group_index.to_mut().reserve(beta.len());
        result.safe_groups.to_mut().clear();
        result.safe_groups.to_mut().reserve(16);
        result.group_max_half_beta_squared.clear();
        for (&b, &s2) in beta.iter().zip(shat2) {
            if !b.is_finite() || !s2.is_finite() {
                result.half_beta_squared.push(0.0);
                result.group_index.to_mut().push(usize::MAX);
                continue;
            }
            let safe = s2.max(f64::EPSILON);
            let group = match result.safe_groups.iter().position(|&s| s == safe) {
                Some(group) => group,
                None => {
                    // Bound setup cost for unstandardized, heterogeneous inputs.
                    if result.safe_groups.len() == 16 {
                        result.grouped = false;
                        break;
                    }
                    result.safe_groups.to_mut().push(safe);
                    result.safe_groups.len() - 1
                }
            };
            result.half_beta_squared.push(0.5 * (b * b));
            result.group_index.to_mut().push(group);
        }
        result.terms.resize(result.safe_groups.len(), (0.0, 0.0, 0.0));
        result.prepare_maximum();
        result
    }

    fn return_buffers(self, buffers: &mut SerBuffers) {
        buffers.half_beta_squared = self.half_beta_squared;
        buffers.terms = self.terms;
        buffers.group_max_half_beta_squared = self.group_max_half_beta_squared;
        if let Cow::Owned(indices) = self.group_index { buffers.group_index = indices; }
        if let Cow::Owned(groups) = self.safe_groups { buffers.safe_groups = groups; }
    }

    fn prepare_maximum(&mut self) {
        if !self.grouped { return; }
        self.uniform_log_prior = self.log_prior.first().copied()
            .filter(|&prior| self.log_prior.iter().all(|&x| x == prior));
        if self.uniform_log_prior.is_none() { return; }
        self.group_max_half_beta_squared.resize(self.safe_groups.len(), 0.0);
        for (&half_beta_squared, &group) in self.half_beta_squared.iter().zip(self.group_index.iter()) {
            if group == usize::MAX { self.has_no_information = true; continue; }
            if !half_beta_squared.is_finite() {
                self.uniform_log_prior = None;
                return;
            }
            // Finite beta² is nonnegative (including +0). A group maximum
            // changes only on a new record, so avoid rewriting it otherwise.
            if half_beta_squared > self.group_max_half_beta_squared[group] {
                self.group_max_half_beta_squared[group] = half_beta_squared;
            }
        }
    }

    fn prepare_group_terms(&mut self, v: f64) {
        for (&safe, term) in self.safe_groups.iter().zip(&mut self.terms) {
            let denominator = safe * (v + safe);
            *term = (-0.5 * (1.0 + v / safe).ln(), denominator, v / denominator);
        }
    }

    fn evaluate(&mut self, v: f64, scratch: &mut [f64]) -> f64 {
        if !self.grouped {
            return log_likelihood(v, self.beta, self.shat2, self.log_prior, scratch);
        }
        self.prepare_group_terms(v);
        // Share the exact group coefficient across objective SNPs. Preserve
        // the original division path at exceptional scales, where distributing
        // the division could lose a representable intermediate.
        if self.terms.iter().any(|&(_, denominator, scale)|
            !denominator.is_finite() || !(scale.is_normal() || (v == 0.0 && scale == 0.0))) {
            return log_likelihood(v, self.beta, self.shat2, self.log_prior, scratch);
        }
        if v >= 0.0 {
            if let Some(prior) = self.uniform_log_prior {
                // At V=0 every valid BF is exactly zero. Equal finite priors
                // therefore give p copies of exp(0)=1 in the same reduction.
                // Keep prior+ln(p), including its rounding, instead of assuming
                // a normalized prior gives an exactly zero marginal likelihood.
                // prepare_maximum already excluded overflowing beta squares;
                // exceptional group coefficients took the original path above.
                if v == 0.0 && prior.is_finite() {
                    return prior + (self.beta.len() as f64).ln();
                }
                // The current nonnegative coefficient makes BF monotone in
                // beta² within each exact variance group. All SNP weights are
                // still evaluated in original order; only the maximum scan is
                // replaced by a preselected exact group maximum.
                let mut largest = if self.has_no_information { prior } else { f64::NEG_INFINITY };
                for (&half_beta_squared, &(constant, _, scale)) in
                    self.group_max_half_beta_squared.iter().zip(&self.terms) {
                    largest = largest.max(constant + half_beta_squared * scale + prior);
                }
                #[cfg(all(has_vector_math, target_arch = "x86_64"))]
                if self.terms.len() <= 4 && vector_math::backend_name() == "optional_glibc_libmvec_avx2_masked_exp" {
                    // SAFETY: the selected backend proves runtime AVX2 and
                    // default caller controls. Group construction bounds every
                    // index or uses usize::MAX; lengths equal the owned scratch.
                    // The kernel changes neither operation nor SNP order.
                    unsafe { ser_group_fill_avx2(&self.half_beta_squared, &self.group_index,
                                                &self.terms, prior, scratch); }
                    return log_sum_exp_at(scratch, Some(largest));
                }
                for ((&half_beta_squared, &group), value) in self.half_beta_squared.iter()
                    .zip(self.group_index.iter()).zip(scratch.iter_mut()) {
                    let bf = if group == usize::MAX { 0.0 } else {
                        let (constant, _, scale) = self.terms[group];
                        constant + half_beta_squared * scale
                    };
                    *value = bf + prior;
                }
                return log_sum_exp_at(scratch, Some(largest));
            }
        }
        for (((&half_beta_squared, &group), &prior), value) in self.half_beta_squared.iter()
            .zip(self.group_index.iter()).zip(self.log_prior).zip(scratch.iter_mut()) {
            let bf = if group == usize::MAX { 0.0 } else {
                let (constant, _, scale) = self.terms[group];
                constant + half_beta_squared * scale
            };
            *value = bf + prior;
        }
        log_sum_exp(scratch)
    }

    fn probabilities_into(&mut self, v: f64, lbf: &mut [f64], lpo: &mut [f64]) {
        if !self.grouped {
            log_probabilities_into(v, self.beta, self.shat2, self.log_prior, lbf, lpo);
            return;
        }
        // Rebuild the terms for the selected V, which may differ from the
        // last objective argument. The destination is this component's row.
        self.prepare_group_terms(v);
        for ((((&half_beta_squared, &group), &prior), bf_out), po_out) in self.half_beta_squared.iter()
            .zip(self.group_index.iter()).zip(self.log_prior).zip(lbf).zip(lpo) {
            let bf = if group == usize::MAX { 0.0 } else {
                let (constant, denominator, _) = self.terms[group];
                constant + half_beta_squared * v / denominator
            };
            *bf_out = bf;
            *po_out = bf + prior;
        }
    }

    #[cfg(test)]
    fn probabilities(&mut self, v: f64) -> (Vec<f64>, Vec<f64>) {
        let mut bf = vec![0.0; self.beta.len()];
        let mut po = vec![0.0; self.beta.len()];
        self.probabilities_into(v, &mut bf, &mut po);
        (bf, po)
    }
}

/// Disjoint rows of the final fit arrays. A row is overwritten only after its
/// previous matrix product has been removed from the fitted sufficient statistic.
struct SingleEffect<'a> {
    alpha: &'a mut [f64],
    mu: &'a mut [f64],
    mu2: &'a mut [f64],
    lbf: &'a mut [f64],
    model_lbf: f64,
    v: f64,
}

fn single_effect<'a>(
    xtr: &[f64],
    diagonal: &[f64],
    v_init: f64,
    sigma2: f64,
    log_prior: &[f64],
    method: &str,
    null_threshold: f64,
    likelihood_scratch: &mut [f64],
    sampling_cache: &mut Option<SerSampling>,
    betahat: &mut [f64],
    buffers: &mut SerBuffers,
    mut posterior: SingleEffect<'a>,
) -> Result<SingleEffect<'a>, String> {
    for ((&d, &r), beta) in diagonal.iter().zip(xtr).zip(betahat.iter_mut()) {
        *beta = (1.0 / d) * r;
    }
    if sampling_cache.as_ref().map_or(true, |sampling| sampling.sigma2 != sigma2) {
        *sampling_cache = Some(SerSampling::new(diagonal, sigma2, method != "none" && method != "EM"));
    }
    let sampling = sampling_cache.as_ref().expect("sampling initialized");
    let shat2 = &sampling.shat2;
    // Fixed-V and EM paths never evaluate this objective. Prepare its
    // invariants only when optim/simple actually request the first value.
    let mut context = None;
    let mut likelihood = |v| {
        let context = context.get_or_insert_with(|| SerLikelihood::from_sampling_reusing(&betahat, sampling, log_prior, buffers));
        context.evaluate(v, likelihood_scratch)
    };
    let mut v = v_init;
    let mut selected_likelihood = None;
    if method == "optim" {
        let (log_v, objective) = optimizer::minimize(|lv| -likelihood(lv.exp()), -30.0, 15.0)?;
        let candidate = log_v.exp();
        // The winning objective is already evaluated by Brent. Retain the
        // exact old-V exp(log(V_init)) comparison required by the R semantics.
        let old_v = v_init.ln().exp();
        let old_likelihood = likelihood(old_v);
        if objective <= -old_likelihood {
            v = candidate;
            selected_likelihood = Some(-objective);
        } else {
            v = v_init;
            if old_v == v_init { selected_likelihood = Some(old_likelihood); }
        }
    }
    if method != "none" && method != "EM" {
        let null_likelihood = likelihood(0.0);
        let current_likelihood = selected_likelihood.unwrap_or_else(|| likelihood(v));
        if null_likelihood + null_threshold >= current_likelihood { v = 0.0; }
    }
    if let Some(context) = &mut context {
        context.probabilities_into(v, posterior.lbf, posterior.alpha);
    } else {
        // Fixed-V/EM retain lazy setup: do not construct unused invariants.
        log_probabilities_into(v, betahat, &shat2, log_prior, posterior.lbf, posterior.alpha);
    }
    if let Some(context) = context.take() { context.return_buffers(buffers); }
    let largest = posterior.alpha.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    // This row first holds log weights, then weights, then normalized alpha.
    // Other component rows remain untouched throughout the sequential update.
    vector_math::exp_shift_inplace(posterior.alpha, largest);
    let total = sum(posterior.alpha.iter().copied());
    for value in posterior.alpha.iter_mut() { *value /= total; }
    for ((((&d, &r), &s2), mean), second) in diagonal.iter().zip(xtr).zip(shat2)
        .zip(posterior.mu.iter_mut()).zip(posterior.mu2.iter_mut()) {
        let beta = r / d;
        let (post_variance, post_mean) = if beta.is_finite() && s2.is_finite() {
            let variance = v * s2 / (v + s2);
            (variance, variance / s2 * beta)
        } else {
            (0.0, 0.0)
        };
        *mean = post_mean;
        *second = post_variance + post_mean * post_mean;
    }
    if method == "EM" {
        // Current Gaussian EM updates V after posterior/KL, without a null BF test.
        v = dot(posterior.alpha, posterior.mu2);
    }
    let model_lbf = largest + total.ln();
    if !model_lbf.is_finite() || posterior.mu.iter().chain(posterior.mu2.iter()).any(|x| !x.is_finite()) {
        return Err("Non-finite single-effect posterior; check sufficient statistics".into());
    }
    posterior.model_lbf = model_lbf;
    posterior.v = v;
    Ok(posterior)
}

/// Accumulate four nonzero columns while each output vector is in registers.
/// The four multiply/add pairs retain ascending-column order and ordinary f64
/// rounding. This reduces output loads/stores, not the matrix or its rank.
#[inline(always)]
fn multiply_columns_grouped(values: &[f64], vector: &[f64], output: &mut [f64]) {
    output.fill(0.0);
    let rows = output.len();
    let mut columns = [0usize;4];
    let mut weights = [0.0;4];
    let mut count = 0;
    for (j, &weight) in vector.iter().enumerate() {
        if weight == 0.0 { continue; }
        columns[count] = j; weights[count] = weight; count += 1;
        if count == 4 {
            let a = &values[columns[0]*rows..(columns[0]+1)*rows];
            let b = &values[columns[1]*rows..(columns[1]+1)*rows];
            let c = &values[columns[2]*rows..(columns[2]+1)*rows];
            let d = &values[columns[3]*rows..(columns[3]+1)*rows];
            for (i, result) in output.iter_mut().enumerate() {
                *result += a[i] * weights[0];
                *result += b[i] * weights[1];
                *result += c[i] * weights[2];
                *result += d[i] * weights[3];
            }
            count = 0;
        }
    }
    for k in 0..count {
        let column = &values[columns[k]*rows..(columns[k]+1)*rows];
        for (result, &value) in output.iter_mut().zip(column) {
            *result += value * weights[k];
        }
    }
}

fn multiply_columns(values: &[f64], vector: &[f64], output: &mut [f64]) {
    multiply_columns_grouped(values, vector, output);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn multiply_columns_avx2(values: &[f64], vector: &[f64], output: &mut [f64]) {
    multiply_columns_grouped(values, vector, output);
}

fn multiply(matrix: ArrayView2<'_, f64>, vector: &[f64], output: &mut [f64]) {
    if output.is_empty() { return; }
    if matrix.t().is_standard_layout() {
        let values = matrix.as_slice_memory_order().expect("contiguous columns");
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: feature detection above guards this target-feature call;
            // all reads/writes inside use bounds-checked, disjoint Rust slices.
            unsafe { multiply_columns_avx2(values, vector, output); }
            return;
        }
        multiply_columns(values, vector, output);
    } else {
        output.fill(0.0);
        for (column, &coefficient) in matrix.columns().into_iter().zip(vector) {
            if coefficient != 0.0 {
                for (result, &value) in output.iter_mut().zip(column) {
                    *result += value * coefficient;
                }
            }
        }
    }
}

/// Compare every finite entry; Boolean AND allows a full SIMD comparison per
/// chunk without weakening equality or treating a signature as proof.
#[inline(always)]
fn exact_column_chunks(column:&[f64],other:&[f64],sign:f64)->bool {
    assert_eq!(column.len(),other.len());
    let mut xs=column.chunks_exact(4);let mut ys=other.chunks_exact(4);
    for (x,y) in xs.by_ref().zip(ys.by_ref()) {
        if !((x[0]==sign*y[0]) & (x[1]==sign*y[1]) &
             (x[2]==sign*y[2]) & (x[3]==sign*y[3])) { return false; }
    }
    xs.remainder().iter().zip(ys.remainder()).all(|(&x,&y)| x==sign*y)
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
unsafe fn exact_column_avx2(column:&[f64],other:&[f64],sign:f64)->bool {
    exact_column_chunks(column,other,sign)
}

fn exact_column(column:&[f64],other:&[f64],sign:f64)->bool {
    #[cfg(target_arch="x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: CPU feature detection guards the call. The implementation
        // uses only immutable bounds-checked slices and reads complete chunks.
        return unsafe { exact_column_avx2(column,other,sign) };
    }
    exact_column_chunks(column,other,sign)
}

fn hash_exact_column(column:&[f64],sign:f64,sampled:bool)->u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;
    let mut hasher=DefaultHasher::new();
    if sampled {
        for k in 0..16 {
            let x=column[k*(column.len()-1)/15];
            hasher.write_u64(if x==0. { 0 } else { (x*sign).to_bits() });
        }
    } else {
        for &x in column { hasher.write_u64(if x==0. { 0 } else { (x*sign).to_bits() }); }
    }
    hasher.finish()
}

/// Independent candidate-signature buckets. Hashes propose comparisons only;
/// every returned signed relation is established by full float64 equality.
fn exact_bucket_batch(values:&[f64],p:usize,signs:&[f64],buckets:&[Vec<usize>])
    ->Vec<(usize,usize,f64)> {
    use std::collections::HashMap;
    enum Bucket { Few(Vec<usize>), Many(HashMap<u64,Vec<usize>>) }
    let mut records=Vec::new();
    for columns in buckets {
        let mut bucket=Bucket::Few(Vec::new());
        for &j in columns {
            let column=&values[j*p..(j+1)*p];
            let candidates=match &mut bucket {
                Bucket::Few(ids)=>ids,
                Bucket::Many(map)=>map.entry(hash_exact_column(column,signs[j],false)).or_default(),
            };
            let mut found=None;
            for &rep in candidates.iter() {
                let ratio=signs[j]*signs[rep];
                if exact_column(column,&values[rep*p..(rep+1)*p],ratio) {
                    found=Some((rep,ratio));break;
                }
            }
            let (rep,ratio)=found.unwrap_or_else(|| { candidates.push(j);(j,1.) });
            records.push((j,rep,ratio));
            if let Bucket::Few(ids)=&bucket {
                if ids.len()>8 {
                    let mut map:HashMap<u64,Vec<usize>>=HashMap::new();
                    for &rep in ids {
                        let other=&values[rep*p..(rep+1)*p];
                        map.entry(hash_exact_column(other,signs[rep],false)).or_default().push(rep);
                    }
                    bucket=Bucket::Many(map);
                }
            }
        }
    }
    records
}

fn verify_copy_columns(values:&[f64],p:usize,assignments:&[(usize,f64)],
    representatives:&[usize],range:std::ops::Range<usize>)->Option<Vec<f64>> {
    let mut small=Vec::with_capacity(representatives.len()*range.len());
    for index in range {
        let j=representatives[index];let column=&values[j*p..(j+1)*p];
        if !assignments.iter().enumerate()
            .all(|(i,&(g,sign))|column[i]==sign*column[representatives[g]]) { return None; }
        for &row in representatives { small.push(column[row]); }
    }
    Some(small)
}

fn exact_rows_scalar(values:&[f64],p:usize,mapped:&[i64],negative:&[f64],columns:&[usize])->bool {
    columns.iter().all(|&j| { let column=&values[j*p..(j+1)*p]; (0..p).all(|i| {
        let expected=column[mapped[i] as usize];
        column[i]==if negative[i].is_sign_negative() { -expected } else { expected }
    })})
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
unsafe fn exact_rows_avx2(values:&[f64],p:usize,mapped:&[i64],negative:&[f64],columns:&[usize])->bool {
    use std::arch::x86_64::*;
    let vector_end=p-p%4;
    for &j in columns {
        let column=&values[j*p..(j+1)*p];
        let mut i=0;
        while i<vector_end {
            let indices=_mm256_loadu_si256(mapped.as_ptr().add(i).cast());
            let representative=_mm256_i64gather_pd::<8>(column.as_ptr(),indices);
            // +0/-0 masks flip only the sign bit, exactly implementing the
            // previously verified +/-1 row relation, including signed zeros.
            let expected=_mm256_xor_pd(representative,_mm256_loadu_pd(negative.as_ptr().add(i)));
            let actual=_mm256_loadu_pd(column.as_ptr().add(i));
            let equal=_mm256_cmp_pd::<_CMP_EQ_OQ>(actual,expected);
            if _mm256_movemask_pd(equal)!=15 { return false; }
            i+=4;
        }
        for i in vector_end..p {
            let expected=column[mapped[i] as usize];
            if column[i] != if negative[i].is_sign_negative() { -expected } else { expected } { return false; }
        }
    }
    true
}

fn exact_rows_block(values:&[f64],p:usize,mapped:&[i64],negative:&[f64],columns:&[usize])->bool {
    debug_assert_eq!(mapped.len(),p);debug_assert_eq!(negative.len(),p);
    debug_assert!(mapped.iter().all(|&i|i>=0 && (i as usize)<p));
    debug_assert!(columns.iter().all(|&j|j<values.len()/p));
    #[cfg(target_arch="x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: mapped rows and selected columns are construction-bounded;
        // metadata has p entries, full vector chunks stay in bounds, and the
        // runtime feature check protects all AVX2 instructions. Reads are immutable.
        return unsafe { exact_rows_avx2(values,p,mapped,negative,columns) };
    }
    exact_rows_scalar(values,p,mapped,negative,columns)
}

/// Exact equality classes of full standardized matrix columns, including sign.
/// Every SNP remains in the posterior; only its matrix contribution is grouped.
struct ExactColumns {
    assignments: Vec<(usize,f64)>,
    representatives: Vec<usize>,
    sums: Vec<f64>,
    corrections: Vec<f64>,
    coefficients: Vec<f64>,
    compact: Option<Vec<f64>>,
    compact_product: Vec<f64>,
}

impl ExactColumns {
    /// Distinct canonical signatures are a lower bound on exact classes. Too
    /// many signatures prove that the desired redundancy cannot be present.
    fn may_reduce(matrix:ArrayView2<'_,f64>)->bool {
        use std::collections::{HashSet,hash_map::DefaultHasher};
        use std::hash::Hasher;
        let p=matrix.nrows();let mut signatures=HashSet::new();
        for column in matrix.as_slice_memory_order().unwrap().chunks_exact(p) {
            let sign=if (0..16).map(|k| column[k*(p-1)/15]).find(|&x| x!=0.).is_some_and(|x| x<0.) { -1. } else { 1. };
            let mut hasher=DefaultHasher::new();
            for k in 0..16 {
                let x=column[k*(p-1)/15];
                hasher.write_u64(if x==0. { 0 } else { (x*sign).to_bits() });
            }
            signatures.insert(hasher.finish());
            if signatures.len()*8>matrix.ncols()*7 { return false; }
        }
        true
    }

    fn new(matrix:ArrayView2<'_,f64>)->Self {
        let threads=std::env::var("PRUSIE_NUM_THREADS").ok()
            .and_then(|value|value.parse::<usize>().ok())
            .filter(|&value|value>=1 && value<=32).unwrap_or(1);
        if threads>1 && matrix.nrows()>=1024 {
            if let Some(plan)=Self::parallel(matrix,threads) { return plan; }
        }
        Self::serial(matrix)
    }

    fn parallel(matrix:ArrayView2<'_,f64>,threads:usize)->Option<Self> {
        use std::collections::HashMap;
        let p=matrix.nrows();
        let values=matrix.as_slice_memory_order()?;
        let mut signatures:HashMap<u64,Vec<usize>>=HashMap::new();
        let mut signs=Vec::with_capacity(matrix.ncols());
        for (j,column) in values.chunks_exact(p).enumerate() {
            let sign=if column.iter().find(|&&x|x!=0.).is_some_and(|&x|x<0.) { -1. } else { 1. };
            signs.push(sign);
            signatures.entry(hash_exact_column(column,sign,true)).or_default().push(j);
        }
        let mut buckets:Vec<Vec<usize>>=signatures.into_values().collect();
        // Largest buckets first, with original index as a deterministic tie-break.
        buckets.sort_by(|a,b|b.len().cmp(&a.len()).then(a[0].cmp(&b[0])));
        let workers=threads.min(buckets.len()).min(32);
        if workers<=1 { return None; }
        let mut bins:Vec<Vec<Vec<usize>>>=(0..workers).map(|_|Vec::new()).collect();
        let mut loads=vec![0usize;workers];
        for bucket in buckets {
            let mut target=0;
            for i in 1..workers { if loads[i]<loads[target] { target=i; } }
            loads[target]+=bucket.len();bins[target].push(bucket);
        }
        // Scoped workers read immutable borrowed matrix memory while the caller
        // holds the GIL. Each worker returns owned records; no shared mutation.
        let records=std::thread::scope(|scope| {
            let mut handles=Vec::new();let mut failed=false;
            for worker in 1..workers {
                let jobs=&bins[worker];let signs=&signs;
                match std::thread::Builder::new()
                    .name(format!("prusie-discovery-{worker}"))
                    .spawn_scoped(scope,move||exact_bucket_batch(values,p,signs,jobs)) {
                    Ok(handle)=>handles.push(handle),
                    Err(_)=>{ failed=true;break; }
                }
            }
            let mut records=exact_bucket_batch(values,p,&signs,&bins[0]);
            for handle in handles {
                match handle.join() {
                    Ok(part)=>records.extend(part),
                    Err(_)=>failed=true,
                }
            }
            if failed { None } else { Some(records) }
        })?;
        if records.len()!=matrix.ncols() { return None; }
        let mut representatives:Vec<usize>=records.iter()
            .filter_map(|&(j,rep,_)| if j==rep { Some(rep) } else { None }).collect();
        representatives.sort_unstable();
        let mut group_of=vec![usize::MAX;matrix.ncols()];
        for (g,&rep) in representatives.iter().enumerate() { group_of[rep]=g; }
        let mut assignments=vec![(usize::MAX,0.);matrix.ncols()];
        for (j,rep,sign) in records { assignments[j]=(group_of[rep],sign); }
        if assignments.iter().any(|&(g,_)|g==usize::MAX) { return None; }
        // Restoring original representative and SNP order preserves the serial
        // class summation order and compact matrix accumulation order.
        Some(Self::finish_parallel(matrix,assignments,representatives,threads))
    }

    fn serial(matrix: ArrayView2<'_,f64>) -> Self {
        use std::collections::HashMap;
        enum Bucket { Few(Vec<usize>), Many(HashMap<u64,Vec<usize>>) }
        let p=matrix.nrows();
        let values=matrix.as_slice_memory_order().expect("contiguous columns");
        let mut buckets:HashMap<u64,Bucket>=HashMap::new();
        let mut representatives:Vec<usize>=Vec::new();
        let mut signs:Vec<f64>=Vec::new();
        let mut assignments=Vec::with_capacity(matrix.ncols());
        for (j,column) in values.chunks_exact(p).enumerate() {
            let sign=if column.iter().find(|&&x| x!=0.).is_some_and(|&x| x<0.) { -1. } else { 1. };
            let bucket=buckets.entry(hash_exact_column(column,sign,true)).or_insert_with(|| Bucket::Few(Vec::new()));
            let candidates=match bucket {
                Bucket::Few(ids)=>ids,
                Bucket::Many(map)=>map.entry(hash_exact_column(column,sign,false)).or_default(),
            };
            let mut found=None;
            for &g in candidates.iter() {
                let rep=representatives[g];let ratio=sign*signs[g];
                let other=&values[rep*p..(rep+1)*p];
                if exact_column(column,other,ratio) { found=Some((g,ratio));break; }
            }
            let assignment=found.unwrap_or_else(|| {
                let g=representatives.len();representatives.push(j);signs.push(sign);
                candidates.push(g);(g,1.)
            });
            assignments.push(assignment);
            // Bound candidate scans on uninformative signatures (e.g. an identity
            // matrix whose sampled rows are mostly zero). Large buckets switch
            // to full-column hashes and still verify every proposed match.
            if let Bucket::Few(ids)=bucket {
                if ids.len()>8 {
                    let mut map:HashMap<u64,Vec<usize>>=HashMap::new();
                    for &g in ids.iter() {
                        let rep=representatives[g];let other=&values[rep*p..(rep+1)*p];
                        map.entry(hash_exact_column(other,signs[g],false)).or_default().push(g);
                    }
                    *bucket=Bucket::Many(map);
                }
            }
        }
        Self::finish(matrix,assignments,representatives)
    }

    fn finish_parallel(matrix:ArrayView2<'_,f64>,assignments:Vec<(usize,f64)>,
        representatives:Vec<usize>,threads:usize)->Self {
        let p=matrix.nrows();let groups=representatives.len();
        let workers=threads.min(groups).min(32);
        if workers<=1 || matrix.ncols()!=p || groups*8>p*7 {
            return Self::finish(matrix,assignments,representatives);
        }
        let values=matrix.as_slice_memory_order().expect("contiguous columns");
        // This second scope starts after discovery workers have joined, so the
        // maximum active numerical thread count remains the requested count.
        let partials=std::thread::scope(|scope| {
            let mut handles=Vec::new();let mut failed=false;
            for worker in 1..workers {
                let assignments=&assignments;let representatives=&representatives;
                let range=groups*worker/workers..groups*(worker+1)/workers;
                match std::thread::Builder::new()
                    .name(format!("prusie-rows-{worker}"))
                    .spawn_scoped(scope,move||verify_copy_columns(values,p,assignments,representatives,range)) {
                    Ok(handle)=>handles.push(handle),
                    Err(_)=>{ failed=true;break; }
                }
            }
            let first=verify_copy_columns(values,p,&assignments,&representatives,0..groups/workers);
            let mut parts=vec![first];
            for handle in handles {
                match handle.join() { Ok(part)=>parts.push(part),Err(_)=>failed=true }
            }
            if failed { None } else { Some(parts) }
        });
        let Some(partials)=partials else { return Self::finish(matrix,assignments,representatives); };
        let compact=if partials.iter().all(Option::is_some) {
            let mut small=Vec::with_capacity(groups*groups);
            // Join handles were kept in original column-range order. This copy
            // restores exactly the serial compact layout without arithmetic.
            for part in partials { small.extend(part.expect("verified column block")); }
            Some(small)
        } else { None };
        Self { assignments,representatives,sums:vec![0.;groups],corrections:vec![0.;groups],
            coefficients:vec![0.;matrix.ncols()],compact,compact_product:vec![0.;groups] }
    }

    fn finish(matrix:ArrayView2<'_,f64>,assignments:Vec<(usize,f64)>,representatives:Vec<usize>)->Self {
        let p=matrix.nrows();
        let values=matrix.as_slice_memory_order().expect("contiguous columns");
        let groups=representatives.len();
        // Keep a bounded block of verified representative columns in cache
        // before gathering their compact entries. The 512KiB data target leaves
        // space for mappings/output on the measured host's 1MiB private L2.
        // This is portable blocking, not a CPU instruction requirement.
        let compact=if matrix.ncols()==p && groups*8<=p*7 {
            let block=(512*1024/(p*std::mem::size_of::<f64>())).max(1);
            let mapped:Vec<i64>=assignments.iter().map(|&(g,_)|representatives[g] as i64).collect();
            let negative:Vec<f64>=assignments.iter().map(|&(_,sign)|if sign<0. {-0.} else {0.}).collect();
            let mut small=Vec::with_capacity(groups*groups);let mut valid=true;
            for columns in representatives.chunks(block) {
                if !exact_rows_block(values,p,&mapped,&negative,columns) { valid=false;break; }
                for &j in columns {
                    let column=&values[j*p..(j+1)*p];
                    for &row in &representatives { small.push(column[row]); }
                }
            }
            if valid { Some(small) } else { None }
        } else { None };
        Self { assignments,representatives,sums:vec![0.;groups],corrections:vec![0.;groups],
            coefficients:vec![0.;matrix.ncols()],compact,compact_product:vec![0.;groups] }
    }

    fn multiply(&mut self,matrix:ArrayView2<'_,f64>,vector:&[f64],output:&mut[f64]) {
        self.sums.fill(0.);self.corrections.fill(0.);
        for (&value,&(g,sign)) in vector.iter().zip(&self.assignments) {
            let value=value*sign;let old=self.sums[g];let next=old+value;
            self.corrections[g]+=if old.abs()>=value.abs() { (old-next)+value } else { (value-next)+old };
            self.sums[g]=next;
        }
        if let Some(small)=&self.compact {
            let groups=self.representatives.len();
            for (total,&correction) in self.sums.iter_mut().zip(&self.corrections) { *total+=correction; }
            let reduced=ArrayView2::from_shape((groups,groups).f(),small).expect("compact dimensions");
            multiply(reduced,&self.sums,&mut self.compact_product);
            for (value,&(g,sign)) in output.iter_mut().zip(&self.assignments) {
                *value=sign*self.compact_product[g];
            }
        } else {
            for (g,&j) in self.representatives.iter().enumerate() {
                self.coefficients[j]=self.sums[g]+self.corrections[g];
            }
            multiply(matrix,&self.coefficients,output);
        }
    }
}

// Complete bounded-entry proof also proves finiteness: NaN and infinities
// fail ordered abs<=2 comparison. Boolean AND groups permit portable SIMD.
#[inline(never)]
fn operator_entries_bounded(values:&[f64])->bool {
    let mut chunks=values.chunks_exact(8);
    for c in &mut chunks {
        if !((c[0].abs()<=2.) & (c[1].abs()<=2.) & (c[2].abs()<=2.) & (c[3].abs()<=2.) &
             (c[4].abs()<=2.) & (c[5].abs()<=2.) & (c[6].abs()<=2.) & (c[7].abs()<=2.)) { return false; }
    }
    chunks.remainder().iter().all(|x|x.abs()<=2.)
}

/// Exact real-arithmetic factorization of the prepared matrix. Floating-point
/// reassociation is intentional and validated separately from the default path.
pub struct MatrixScaling<'a> {
    pub global: f64,
    pub inverse: &'a [f64],
    pub scales: &'a [f64],
}

fn materialize_scaled(matrix:ArrayView2<'_,f64>,scaling:&MatrixScaling<'_>)->Vec<f64> {
    let p=matrix.nrows();let mut values=Vec::with_capacity(p*p);
    for j in 0..p {
        for i in 0..p {
            values.push(((matrix[(i,j)]*scaling.global)*scaling.inverse[j])/scaling.scales[i]);
        }
    }
    values
}

fn scaled_multiply(matrix:ArrayView2<'_,f64>,vector:&[f64],output:&mut[f64],
    groups:Option<&mut ExactColumns>,scaling:&MatrixScaling<'_>,scratch:&mut[f64])->bool {
    for ((x,&v),&factor) in scratch.iter_mut().zip(vector).zip(scaling.inverse) { *x=v*factor; }
    if scratch.iter().all(|x|x.is_finite()) {
        if let Some(groups)=groups { groups.multiply(matrix,scratch,output); }
        else { multiply(matrix,scratch,output); }
        for (x,&scale) in output.iter_mut().zip(scaling.scales) { *x=(*x*scaling.global)/scale; }
        if output.iter().all(|x|x.is_finite()) { return false; }
    }
    // Preserve the full matrix route when factorization creates an intermediate
    // overflow that the original ordered matrix preparation would avoid.
    let values=materialize_scaled(matrix,scaling);let p=matrix.nrows();
    let prepared=ArrayView2::from_shape((p,p).f(),&values).expect("prepared dimensions");
    multiply(prepared,vector,output);true
}

struct ResidualScratch {
    mean: Vec<f64>, fitted: Vec<f64>, component_quadratics: Vec<f64>, second_moments: Vec<f64>,
}

fn expected_residual_sum_squares(
    xty: &[f64], yty: f64, diagonal: &[f64], fit: &FitOutput,
    component_products: &[f64],
    scratch: &mut ResidualScratch,
) -> f64 {
    let p = xty.len();
    let ResidualScratch { mean, fitted, component_quadratics, second_moments } = scratch;
    mean.fill(0.0); fitted.fill(0.0);
    component_quadratics.clear(); second_moments.clear();
    for effect in 0..fit.v.len() {
        let offset = effect*p;
        component_quadratics.push(sum((0..p).map(|j|
            fit.alpha[offset+j] * fit.mu[offset+j] * component_products[offset+j])));
        for j in 0..p {
            let idx = offset+j;
            mean[j] += fit.alpha[idx] * fit.mu[idx];
            fitted[j] += component_products[idx];
            second_moments.push(diagonal[j] * (fit.alpha[idx] * fit.mu2[idx]));
        }
    }
    // Linearity: sum_l XtX*b_l = XtX*sum_l b_l. All products were computed
    // after the corresponding component update, so no stale coefficients enter.
    yty - 2.0 * dot(&mean, xty) + dot(&mean, &fitted)
        - sum(component_quadratics.iter().copied()) + sum(second_moments.iter().copied())
}

/// Fit from centered sufficient statistics. XtX must be row-major and must
/// already include any standardization and explicit null column.
#[cfg(test)]
pub fn fit(xtx: &[f64], xty: &[f64], yty: f64, n: f64, options: &FitOptions) -> Result<FitOutput, String> {
    fit_layout(xtx, xty, yty, n, options, false)
}

pub fn fit_layout(
    xtx: &[f64],
    xty: &[f64],
    yty: f64,
    n: f64,
    options: &FitOptions,
    column_major: bool,
) -> Result<FitOutput, String> {
    fit_layout_operator(xtx,xty,yty,n,options,column_major,None)
}

pub fn fit_layout_operator(
    xtx:&[f64],xty:&[f64],yty:f64,n:f64,options:&FitOptions,column_major:bool,
    scaling:Option<&MatrixScaling<'_>>,
)->Result<FitOutput,String> {
    let p = xty.len();
    if p == 0 || p.checked_mul(p) != Some(xtx.len()) || options.l == 0 || options.l > p {
        return Err("Invalid sufficient-statistic dimensions or number of effects".into());
    }
    // This certificate checks every raw entry inside this call. Its success
    // establishes finite input and the safe operator range together; failure
    // merely retains full finiteness validation and matrix materialization.
    let operator_bounded=scaling.is_some_and(|scale|scale.global>=1. && scale.global<=1e9 &&
        scale.scales.len()==p && scale.inverse.len()==p &&
        scale.scales.iter().all(|&x|x>=0.5 && x<=2.) &&
        scale.inverse.iter().all(|&x|x>=0.5 && x<=2.) && operator_entries_bounded(xtx));
    if !n.is_finite()
        || n <= 1.0
        || !yty.is_finite()
        || yty < 0.0
        || !input_validation::all_finite(xty)
        || (!operator_bounded && !input_validation::all_finite(xtx))
    {
        return Err("Sufficient statistics and n must be finite; n > 1 and yty >= 0".into());
    }
    if let Some(scale)=scaling {
        if !scale.global.is_finite() || scale.global<=0. || scale.scales.len()!=p || scale.inverse.len()!=p ||
            scale.scales.iter().chain(scale.inverse).any(|&x|!x.is_finite() || x<=0.) {
            return Err("Matrix factors must have p finite positive entries and a finite positive global multiplier".into());
        }
    }
    if options.max_iter == 0
        || !options.tol.is_finite()
        || options.tol < 0.0
        || !options.prior_tol.is_finite()
        || options.prior_tol < 0.0
        || !options.prior_variance.is_finite()
        || options.prior_variance < 0.0
        || !options.residual_variance.is_finite()
        || options.residual_variance <= 0.0
        || !options.check_null_threshold.is_finite()
    {
        return Err("Invalid variance, convergence, or null-threshold parameter".into());
    }
    if options.prior_weights.len() != p
        || options
            .prior_weights
            .iter()
            .any(|w| !w.is_finite() || *w < 0.0)
        || sum(options.prior_weights.iter().copied()) <= 0.0
    {
        return Err("Prior weights must be finite, nonnegative, and nonzero".into());
    }
    if !matches!(
        options.estimate_prior_method.as_str(),
        "optim" | "EM" | "simple"
    ) {
        return Err("Unsupported prior variance estimation method".into());
    }
    // Python may supply either contiguous layout. Column-major preparation
    // avoids an extra dense copy while borrowed memory remains under the GIL.
    let columns;
    let matrix = if column_major {
        ArrayView2::from_shape((p, p).f(), xtx).map_err(|error| error.to_string())?
    } else {
        let input = ArrayView2::from_shape((p, p), xtx).map_err(|error| error.to_string())?;
        columns = input.t().as_standard_layout().into_owned().reversed_axes();
        columns.view()
    };
    // These bounds select only an arithmetic execution route. Every value
    // outside them retains the original fully materialized matrix semantics.
    // Within them, every prepared entry is bounded by8e9 and cannot overflow.
    let operator=if operator_bounded {scaling} else {None};
    let prepared;
    let matrix=if let Some(scale)=scaling.filter(|_|operator.is_none()) {
        prepared=materialize_scaled(matrix,scale);
        if prepared.iter().any(|x|!x.is_finite()) { return Err("Scaled matrix entries must remain finite".into()); }
        ArrayView2::from_shape((p,p).f(),&prepared).map_err(|error|error.to_string())?
    } else { matrix };
    let diagonal: Vec<f64> = (0..p).map(|j|operator.map_or(matrix[(j,j)],|scale|
        ((matrix[(j,j)]*scale.global)*scale.inverse[j])/scale.scales[j])).collect();
    if diagonal.iter().any(|&d| d < 0.0) {
        return Err("XtX diagonal cannot be negative".into());
    }
    let log_prior: Vec<f64> = options
        .prior_weights
        .iter()
        .map(|w| (w + f64::EPSILON.sqrt()).ln())
        .collect();
    let method = if options.estimate_prior_variance {
        options.estimate_prior_method.as_str()
    } else {
        "none"
    };
    let max_z = diagonal
        .iter()
        .zip(xty)
        .filter_map(|(&d, &r)| {
            let z = ((1.0 / d) * r) / (options.residual_variance / d).sqrt();
            (!z.is_nan()).then_some(z.abs())
        })
        .fold(f64::NEG_INFINITY, f64::max);
    let size = options.l * p;
    let mut output = FitOutput {
        alpha: vec![1.0 / p as f64; size],
        mu: vec![0.0; size],
        mu2: vec![0.0; size],
        lbf_variable: vec![f64::NAN; size],
        lbf: vec![f64::NAN; options.l],
        v: vec![options.prior_variance; options.l],
        kl: vec![f64::NAN; options.l],
        elbo: Vec::with_capacity(options.max_iter),
        sigma2: options.residual_variance,
        niter: 0,
        converged: false,
        xtxr: vec![0.0; p],
        warnings: Vec::new(),
    };
    // Discovery is lazy; small or currently sparse products retain the full
    // matrix path. Exact redundancy alone authorizes the compact representation.
    let mut exact_columns:Option<ExactColumns>=None;
    let mut grouping_checked=false;
    let mut component = vec![0.0; p];
    let mut product = vec![0.0; p];
    let mut scaled_component=if operator.is_some() {vec![0.;p]} else {Vec::new()};
    let mut residual = vec![0.0; p];
    let mut component_products = vec![0.0; size];
    let mut likelihood_scratch = vec![0.0; p];
    let mut ser_sampling = None;
    let mut ser_betahat = vec![0.0; p];
    let mut ser_buffers = SerBuffers::default();
    let mut residual_scratch = ResidualScratch { mean: vec![0.0;p], fitted: vec![0.0;p],
        component_quadratics: Vec::with_capacity(options.l), second_moments: Vec::with_capacity(size) };
    let mut previous_alpha = vec![0.0;size];
    let mut previous_elbo = f64::NEG_INFINITY;
    let mut fallback_history: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    for iteration in 0..options.max_iter {
        previous_alpha.copy_from_slice(&output.alpha);
        for effect in 0..options.l {
            let offset = effect * p;
            // Reuse this effect's exact previous product; initial moments are zero.
            for j in 0..p {
                output.xtxr[j] -= component_products[offset+j];
                residual[j] = xty[j] - output.xtxr[j];
            }
            let posterior = single_effect(
                &residual,
                &diagonal,
                output.v[effect],
                output.sigma2,
                &log_prior,
                method,
                options.check_null_threshold,
                &mut likelihood_scratch,
                &mut ser_sampling,
                &mut ser_betahat,
                &mut ser_buffers,
                SingleEffect {
                    alpha: &mut output.alpha[offset..offset+p],
                    mu: &mut output.mu[offset..offset+p],
                    mu2: &mut output.mu2[offset..offset+p],
                    lbf: &mut output.lbf_variable[offset..offset+p],
                    model_lbf: f64::NAN,
                    v: output.v[effect],
                },
            )?;
            output.v[effect] = posterior.v;
            output.lbf[effect] = posterior.model_lbf;
            for (j, value) in component.iter_mut().enumerate() {
                *value = posterior.alpha[j] * posterior.mu[j];
            }
            // Gaussian SER expected likelihood excludes no-information columns.
            let expected_loglik = -0.5 * sum((0..p).filter_map(|j| {
                let s2 = output.sigma2 / diagonal[j];
                s2.is_finite().then(||
                    (-2.0 * component[j] * (residual[j] / diagonal[j])
                     + posterior.alpha[j] * posterior.mu2[j]) / s2)
            }));
            output.kl[effect] = -posterior.model_lbf + expected_loglik;
            if !grouping_checked && p>=1024 &&
                component.iter().filter(|&&x| x!=0.).take(p.div_ceil(8)).count()>=p.div_ceil(8) {
                grouping_checked=true;
                if ExactColumns::may_reduce(matrix) {
                    let groups=ExactColumns::new(matrix);
                    if groups.representatives.len()*8 <= p*7 { exact_columns=Some(groups); }
                }
            }
            if let Some(scale)=operator {
                scaled_multiply(matrix,&component,&mut product,exact_columns.as_mut(),scale,&mut scaled_component);
            } else if let Some(groups)=&mut exact_columns {
                groups.multiply(matrix,&component,&mut product);
            } else {
                multiply(matrix, &component, &mut product);
            }
            component_products[offset..offset+p].copy_from_slice(&product);
            for (fitted, increment) in output.xtxr.iter_mut().zip(&product) {
                *fitted += increment;
            }
        }
        if options.check_prior && output.v.iter().any(|&v| v > 100.0 * max_z * max_z) {
            return Err("The estimated prior variance is unreasonably large; check summary statistics and LD consistency".into());
        }
        let er2 = expected_residual_sum_squares(xty, yty, &diagonal, &output, &component_products, &mut residual_scratch);
        let elbo = -n / 2.0 * (2.0 * PI * output.sigma2).ln()
            - 1.0 / (2.0 * output.sigma2) * er2
            - sum(output.kl.iter().copied());
        output.elbo.push(elbo);
        output.niter = iteration + 1;
        let improvement = elbo - previous_elbo;
        if iteration > 0 {
            if improvement.is_finite() {
                if improvement < -options.tol {
                    output.warnings.push(format!("ELBO decreased by {} at iteration {}", -improvement, iteration + 1));
                }
                output.converged = improvement >= 0.0 && improvement < options.tol;
            } else {
                output.warnings.push(format!("Nonfinite ELBO improvement at iteration {}; using alpha/PIP convergence", iteration + 1));
                output.converged = fallback_convergence(&mut output.alpha, &previous_alpha,
                    &output.v, p, options.prior_tol, options.null_index, options.tol, &mut fallback_history);
            }
        }
        if output.converged { break; }
        previous_elbo = elbo;
        if options.estimate_residual_variance {
            let estimate = (1.0 / n) * er2;
            if !estimate.is_finite() || estimate <= 0.0 {
                return Err("Estimating residual variance failed: non-positive estimate".into());
            }
            output.sigma2 = estimate;
        }
    }
    // The stored ELBO and XtXr describe the completed iterations before trim.
    // Upstream trims even at max_iter and does not recompute either cache.
    for effect in 0..options.l {
        if output.v[effect] < options.prior_tol {
            output.v[effect] = 0.0;
            output.lbf[effect] = 0.0;
            output.kl[effect] = 0.0;
            let row = effect*p..(effect+1)*p;
            output.alpha[row.clone()].copy_from_slice(&options.prior_weights);
            output.mu[row.clone()].fill(0.0);
            output.mu2[row.clone()].fill(0.0);
            output.lbf_variable[row].fill(0.0);
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_pairwise_sum_matches_binary_geometric_series() {
        for n in [0,1,2,3,5,8,9,17,31,32,33,52] {
            let mut values: Vec<f64> = (1..=n).map(|j| 2.0_f64.powi(-j)).collect();
            let expected = 1.0 - 2.0_f64.powi(-n);
            assert_eq!(positive_pairwise_sum(&mut values), expected);
        }
        let mut values = [0.0, 0.0, 1.0, 0.0, 0.0];
        assert_eq!(positive_pairwise_sum(&mut values), 1.0);
    }

    #[test]
    fn grouped_final_bf_matches_general_arithmetic_and_fallback() {
        let beta = [0.0, 2.0, -2.0, f64::NAN, 1.0, 0.5];
        let shat2 = [0.5, 0.5, 0.25, f64::INFINITY, 0.0, 0.25];
        let prior = [0.5_f64.ln(), f64::EPSILON.sqrt().ln(), 0.25_f64.ln(),
                     0.125_f64.ln(), 0.0625_f64.ln(), 0.0625_f64.ln()];
        let mut context = SerLikelihood::new(&beta, &shat2, &prior);
        let mut scratch = [0.0; 6];
        for v in [0.0, 1.0e-12, 0.2, 1.0e6] {
            // Deliberately leave the objective context at a different V.
            context.evaluate(v + 0.5, &mut scratch);
            let (actual_bf, actual_po) = context.probabilities(v);
            let (expected_bf, expected_po) = log_probabilities(v, &beta, &shat2, &prior);
            for (actual, expected) in actual_bf.iter().chain(&actual_po)
                .zip(expected_bf.iter().chain(&expected_po)) {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
        let beta = [0.5; 17];
        let shat2: Vec<f64> = (1..=17).map(|x| x as f64).collect();
        let prior = [0.0625_f64.ln(); 17];
        let mut context = SerLikelihood::new(&beta, &shat2, &prior);
        assert!(!context.grouped);
        assert_eq!(context.probabilities(0.2), log_probabilities(0.2, &beta, &shat2, &prior));
    }

    #[test]
    fn objective_group_scale_preserves_zero_information_and_extreme_fallback() {
        let beta = [0.3, -1.7, 2.3, f64::NAN, 0.0];
        let shat2 = [0.25, 0.25, 0.5, f64::INFINITY, 1.0];
        let prior = [0.4_f64.ln(), f64::EPSILON.sqrt().ln(), 0.3_f64.ln(),
                     0.2_f64.ln(), 0.1_f64.ln()];
        let mut context = SerLikelihood::new(&beta, &shat2, &prior);
        for v in [0.0, 1.0e-310, 1.0e-12, 0.2, 1.0e6] {
            let mut actual_scratch = [0.0; 5];
            let mut expected_scratch = [0.0; 5];
            let actual = context.evaluate(v, &mut actual_scratch);
            let expected = log_likelihood(v, &beta, &shat2, &prior, &mut expected_scratch);
            assert!((actual - expected).abs() <= 8.0 * f64::EPSILON * expected.abs().max(1.0));
        }
        let extreme_shat2 = [1.0e300; 5];
        let mut context = SerLikelihood::new(&beta, &extreme_shat2, &prior);
        let mut actual_scratch = [0.0; 5];
        let mut expected_scratch = [0.0; 5];
        let actual = context.evaluate(0.2, &mut actual_scratch);
        let expected = log_likelihood(0.2, &beta, &extreme_shat2, &prior, &mut expected_scratch);
        assert_eq!(actual.to_bits(), expected.to_bits());
    }

    #[test]
    fn sampling_reuse_invalidates_on_residual_variance_change() {
        let mut buffers = SerBuffers::default();
        let mut sampling = None;
        let mut scratch = [0.0];
        let (mut alpha, mut mu, mut mu2, mut lbf, mut beta) = ([0.0], [0.0], [0.0], [0.0], [0.0]);
        for sigma2 in [1.0, 1.0, 2.0, 0.75, 1.0] {
            let posterior = single_effect(&[4.0], &[10.0], 0.2, sigma2,
                &[0.0], "none", 0.0, &mut scratch, &mut sampling, &mut beta, &mut buffers,
                SingleEffect { alpha: &mut alpha, mu: &mut mu, mu2: &mut mu2,
                    lbf: &mut lbf, model_lbf: f64::NAN, v: 0.2 }).unwrap();
            let s2 = sigma2 / 10.0;
            let variance = 0.2 * s2 / (0.2 + s2);
            let mean = variance / s2 * 0.4;
            assert_eq!(posterior.mu[0].to_bits(), mean.to_bits());
            assert_eq!(posterior.mu2[0].to_bits(), (variance + mean * mean).to_bits());
            assert_eq!(sampling.as_ref().unwrap().shat2, [s2]);
        }
        let design = SerSampling::new(&[10.0, 0.0], 1.0, true);
        let beta = [0.4, f64::NAN];
        let prior = [0.9_f64.ln(), f64::EPSILON.sqrt().ln()];
        let mut reused = SerLikelihood::from_sampling(&beta, &design, &prior);
        let mut fresh = SerLikelihood::new(&beta, &design.shat2, &prior);
        let mut a = [0.0; 2];
        let mut b = [0.0; 2];
        for v in [0.0, 0.2, 2.0] {
            assert_eq!(reused.evaluate(v, &mut a).to_bits(), fresh.evaluate(v, &mut b).to_bits());
            assert_eq!(reused.probabilities(v), fresh.probabilities(v));
        }
    }

    #[test]
    fn reused_posterior_rows_preserve_components_and_em_null_semantics() {
        let mut buffers = SerBuffers::default();
        let diagonal = [10.0, 10.0, 0.0];
        let xtr = [4.0, -2.0, 0.0];
        let prior = [(0.75 + f64::EPSILON.sqrt()).ln(),
                     f64::EPSILON.sqrt().ln(), (0.25 + f64::EPSILON.sqrt()).ln()];
        let mut alpha = [17.0; 9]; let mut mu = [19.0; 9];
        let mut mu2 = [23.0; 9]; let mut lbf = [29.0; 9];
        let mut beta = [f64::NAN; 3]; let mut scratch = [f64::NAN; 3];
        let mut sampling = None;
        for method in ["none", "EM", "simple", "none"] {
            let posterior = single_effect(&xtr, &diagonal, 0.2, 1.0, &prior,
                method, 1e6, &mut scratch, &mut sampling, &mut beta, &mut buffers,
                SingleEffect { alpha: &mut alpha[3..6], mu: &mut mu[3..6],
                    mu2: &mut mu2[3..6], lbf: &mut lbf[3..6], model_lbf: f64::NAN, v: 0.2 }).unwrap();
            let variance = if method == "simple" { 0.0 } else { 0.2 * 0.1 / (0.2 + 0.1) };
            for j in 0..2 {
                let mean = variance / 0.1 * (xtr[j] / 10.0);
                assert_eq!(posterior.mu[j].to_bits(), mean.to_bits());
                assert_eq!(posterior.mu2[j].to_bits(), (variance + mean * mean).to_bits());
            }
            assert_eq!(posterior.mu[2], 0.0); assert_eq!(posterior.mu2[2], 0.0);
            assert_eq!(posterior.lbf[2], 0.0);
            assert!((sum(posterior.alpha.iter().copied()) - 1.0).abs() < 1e-15);
            if method == "EM" {
                assert_eq!(posterior.v, dot(posterior.alpha, posterior.mu2));
            } else if method == "simple" {
                assert_eq!(posterior.v, 0.0);
                assert!(posterior.lbf.iter().all(|x| *x == 0.0));
            } else { assert_eq!(posterior.v, 0.2); }
            for (values, sentinel) in [(&alpha, 17.0), (&mu, 19.0), (&mu2, 23.0), (&lbf, 29.0)] {
                assert!(values[..3].iter().chain(&values[6..]).all(|x| *x == sentinel));
            }
        }
    }

    #[test]
    fn recycled_search_storage_resets_group_maxima_and_fallback_state() {
        let mut buffers = SerBuffers::default();
        let uniform = [0.125_f64.ln(); 6];
        let partial = [0.5_f64.ln(), f64::EPSILON.sqrt().ln(), 0.25_f64.ln(),
                       0.125_f64.ln(), 0.0625_f64.ln(), 0.0625_f64.ln()];
        let strong = [10.0, -8.0, 2.0, 0.0, 4.0, 0.5];
        let weak = [0.1, -0.2, 0.0, 0.3, 0.0, 0.1];
        let no_info = [0.1, f64::NAN, 0.0, -0.2, 0.0, 0.1];
        for (beta, prior, diagonal) in [
            (&strong, &uniform, [10.0; 6]),
            (&weak, &uniform, [10.0; 6]),
            (&no_info, &partial, [10.0, 0.0, 20.0, 20.0, 0.0, 10.0]),
            (&strong, &partial, [10.0; 6]),
            (&weak, &uniform, [20.0; 6]),
        ] {
            let sampling = SerSampling::new(&diagonal, 1.0, true);
            let mut reused = SerLikelihood::from_sampling_reusing(beta, &sampling, prior, &mut buffers);
            let mut fresh = SerLikelihood::new(beta, &sampling.shat2, prior);
            for v in [0.0, 0.2, 1e6] {
                assert_eq!(reused.evaluate(v, &mut [0.0; 6]).to_bits(),
                           fresh.evaluate(v, &mut [0.0; 6]).to_bits());
                assert_eq!(reused.probabilities(v), log_probabilities(v, beta, &sampling.shat2, prior));
            }
            reused.return_buffers(&mut buffers);
        }
        // More than sixteen distinct sampling variances use the complete
        // general path; a following small grouped search must reset that state.
        let heterogeneous: Vec<f64> = (1..=18).map(|x| x as f64).collect();
        let beta = [0.5; 18]; let prior = [0.0625_f64.ln(); 18];
        let mut context = SerLikelihood::new_reusing(&beta, &heterogeneous, &prior, &mut buffers);
        assert!(!context.grouped);
        assert_eq!(context.probabilities(0.2), log_probabilities(0.2, &beta, &heterogeneous, &prior));
        context.return_buffers(&mut buffers);
        let sampling = SerSampling::new(&[10.0; 6], 1.0, true);
        let mut context = SerLikelihood::from_sampling_reusing(&weak, &sampling, &uniform, &mut buffers);
        assert_eq!(context.probabilities(0.2), log_probabilities(0.2, &weak, &sampling.shat2, &uniform));
    }

    #[test]
    fn invariant_group_max_preserves_current_normalization_and_fallbacks() {
        let beta = [0.0, 2.0, -1.7, f64::NAN, 0.3, 1.0];
        let shat2 = [0.5, 0.5, 0.25, f64::INFINITY, 0.0, 0.25];
        let uniform = [0.125_f64.ln(); 6];
        let mut partial_zero = uniform;
        partial_zero[2] = f64::EPSILON.sqrt().ln();
        for prior in [&uniform, &partial_zero] {
            let mut fast = SerLikelihood::new(&beta, &shat2, prior);
            let mut scan = SerLikelihood::new(&beta, &shat2, prior);
            scan.uniform_log_prior = None;
            for v in [0.0, 1.0e-310, 1.0e-12, 0.2, 1.0e6, 1.0e308] {
                let mut a = [0.0; 6]; let mut b = [0.0; 6];
                let x = fast.evaluate(v, &mut a); let y = scan.evaluate(v, &mut b);
                assert!(x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()));
            }
        }
        let overflow_beta = [1.0e308, 0.5];
        let shat2 = [0.5; 2]; let prior = [0.5_f64.ln(); 2];
        let context = SerLikelihood::new(&overflow_beta, &shat2, &prior);
        assert!(context.uniform_log_prior.is_none());
    }

    #[test]
    fn uniform_null_keeps_prior_normalization_and_exceptional_fallbacks() {
        let beta = [0.0, 2.0, -1.7, f64::NAN, 0.3];
        let shat2 = [0.5, 0.5, 0.25, f64::INFINITY, 0.0];
        let prior = [(0.2 + f64::EPSILON.sqrt()).ln(); 5];
        let mut context = SerLikelihood::new(&beta, &shat2, &prior);
        let mut scratch = [0.0; 5];
        assert_eq!(context.evaluate(0.0, &mut scratch).to_bits(),
                   (prior[0] + 5.0_f64.ln()).to_bits());
        let huge = [1.0e308, 1.0];
        let finite_shat2 = [0.5; 2];
        let uniform = [0.5_f64.ln(); 2];
        let mut overflow = SerLikelihood::new(&huge, &finite_shat2, &uniform);
        assert!(overflow.uniform_log_prior.is_none());
        assert!(overflow.evaluate(0.0, &mut [0.0; 2]).is_nan());
        let beta = [0.5; 2]; let invalid_prior = [f64::NEG_INFINITY; 2];
        let mut invalid = SerLikelihood::new(&beta, &finite_shat2, &invalid_prior);
        assert!(invalid.evaluate(0.0, &mut [0.0; 2]).is_nan());
    }

    #[test]
    #[cfg(all(has_vector_math, target_arch = "x86_64"))]
    fn simd_group_arithmetic_keeps_groups_nulls_and_every_tail() {
        if !std::is_x86_feature_detected!("avx2") { return; }
        let half = [0.0, 1.0, 2.0, 3.0, 0.5, 7.0, 0.0, 4.0, 2.0, 8.0, 0.125, 1.0, 9.0];
        let groups = [0, 1, 2, usize::MAX, 3, 1, 2, 0, usize::MAX, 3, 2, 1, 0];
        let terms = [(-0.25, 1.0, 0.5), (-1.0, 2.0, 2.0), (0.0, 4.0, 0.25), (-0.125, 1.0, 1.0)];
        // All coefficients are binary-exact; these are independent known
        // answers for constant + half_beta_squared*scale + log_prior(=-2).
        let expected = [-2.25, -1.0, -1.5, -2.0, -1.625, 11.0, -2.0,
                        -0.25, -2.0, 5.875, -1.96875, -1.0, 2.25];
        for length in 0..=half.len() {
            let mut actual = vec![0.0; length];
            unsafe { ser_group_fill_avx2(&half[..length], &groups[..length], &terms, -2.0, &mut actual); }
            assert_eq!(actual, expected[..length]);
        }
        // The wider-table fallback remains safe and has identical answers.
        let mut wider = terms.to_vec(); wider.push((0.0, 1.0, 0.0));
        let mut actual = [0.0; 13];
        unsafe { ser_group_fill_avx2(&half, &groups, &wider, -2.0, &mut actual); }
        assert_eq!(actual, expected);
    }

    fn options(p: usize) -> FitOptions {
        FitOptions {
            l: 1,
            prior_variance: 0.2,
            residual_variance: 1.0,
            prior_weights: vec![1.0 / p as f64; p],
            estimate_prior_variance: false,
            estimate_prior_method: "optim".into(),
            estimate_residual_variance: false,
            check_null_threshold: 0.0,
            max_iter: 100,
            tol: 1e-9,
            check_prior: false,
            prior_tol: 1e-9,
            null_index: None,
        }
    }

    #[test]
    fn fixed_single_effect_matches_closed_form() {
        let result = fit(
            &[10.0, 0.0, 0.0, 10.0],
            &[4.0, -1.0],
            20.0,
            21.0,
            &options(2),
        )
        .expect("fit");
        let variance = 1.0 / 15.0;
        assert!((result.mu[0] - 4.0 * variance).abs() < 1e-14);
        assert!((result.mu2[0] - variance - (4.0 * variance).powi(2)).abs() < 1e-14);
        let b0 = -0.5 * 3.0_f64.ln() + 0.5 * 1.6 * (2.0 / 3.0);
        let b1 = -0.5 * 3.0_f64.ln() + 0.5 * 0.1 * (2.0 / 3.0);
        assert!((result.lbf_variable[0] - b0).abs() < 1e-14);
        assert!((result.alpha[0] - 1.0 / (1.0 + (b1 - b0).exp())).abs() < 1e-14);
        assert_eq!(result.niter, 2);
        assert!(result.converged);
    }

    #[test]
    fn null_column_matches_upstream_probability_rule() {
        let mut settings = options(2);
        settings.prior_weights = vec![0.9, 0.1];
        let result =
            fit(&[10.0, 0.0, 0.0, 0.0], &[0.0, 0.0], 20.0, 21.0, &settings).expect("null fit");
        let real_weight = (0.9 + f64::EPSILON.sqrt()) / 3.0_f64.sqrt();
        let null_weight = 0.1 + f64::EPSILON.sqrt();
        assert!((result.alpha[1] - null_weight / (null_weight + real_weight)).abs() < 1e-14);
        assert_eq!(result.lbf_variable[1], 0.0);
        assert_eq!(result.mu[1], 0.0);
        assert_eq!(result.mu2[1], 0.0);
    }

    #[test]
    fn optim_single_variant_recovers_likelihood_optimum() {
        let mut settings = options(1);
        settings.estimate_prior_variance = true;
        let result = fit(&[10.0], &[8.0], 20.0, 21.0, &settings).expect("optimized fit");
        assert!((result.v[0] - (0.8_f64.powi(2) - 0.1)).abs() < 1e-7);
    }

    #[test]
    fn weak_effect_is_set_to_exact_zero() {
        let mut settings = options(2);
        settings.estimate_prior_variance = true;
        let result =
            fit(&[10.0, 1.0, 1.0, 10.0], &[0.0, 0.0], 20.0, 21.0, &settings).expect("weak fit");
        assert_eq!(result.v, vec![0.0]);
        assert_eq!(result.mu, vec![0.0, 0.0]);
        assert_eq!(result.alpha, vec![0.5, 0.5]);
    }

    #[test]
    fn final_trim_at_cap_restores_exact_prior_and_zeros_all_fields() {
        let mut settings = options(2);
        settings.max_iter = 1;
        settings.prior_variance = 1e-12;
        settings.prior_weights = vec![0.8, 0.2];
        let result = fit(&[10.0, 0.0, 0.0, 10.0], &[1.0, -2.0], 20.0, 21.0, &settings).unwrap();
        assert!(!result.converged);
        assert_eq!(result.alpha, settings.prior_weights);
        for values in [&result.v, &result.mu, &result.mu2, &result.lbf_variable, &result.lbf, &result.kl] {
            assert!(values.iter().all(|&x| x == 0.0));
        }
        assert_eq!(result.elbo.len(), 1);
        // Trace/cache stay pre-trim, per the pinned finalization contract.
        assert!(result.xtxr.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn gaussian_em_does_not_apply_optim_null_test() {
        let mut settings = options(1);
        settings.estimate_prior_variance = true;
        settings.estimate_prior_method = "EM".into();
        settings.check_null_threshold = 1e6;
        settings.max_iter = 1;
        let result = fit(&[10.0], &[1.0], 20.0, 21.0, &settings).unwrap();
        assert!(result.v[0] > settings.prior_tol);
        assert_eq!(result.v[0], result.mu2[0]);
    }

    #[test]
    fn alpha_pip_fallback_fixed_point_and_two_cycle() {
        let mut history = Vec::new();
        let mut alpha = vec![0.8, 0.2];
        assert!(fallback_convergence(&mut alpha, &[0.8,0.2], &[1.0], 2, 1e-9, None, 1e-4, &mut history));
        let mut history = vec![(vec![0.8,0.2],vec![0.8,0.2]),(vec![0.2,0.8],vec![0.2,0.8])];
        assert!(fallback_convergence(&mut alpha, &[0.2,0.8], &[1.0], 2, 1e-9, None, 1e-4, &mut history));
        assert_eq!(alpha, vec![0.5,0.5]);
    }

    #[test]
    fn em_updates_variance_after_posterior() {
        let mut settings = options(1);
        settings.estimate_prior_variance = true;
        settings.estimate_prior_method = "EM".into();
        settings.max_iter = 1;
        let result = fit(&[10.0], &[8.0], 20.0, 21.0, &settings).expect("EM fit");
        let old_posterior_variance = 1.0 / 15.0;
        assert!((result.mu[0] - 8.0 * old_posterior_variance).abs() < 1e-14);
        assert_eq!(result.v[0], result.mu2[0]);
        assert!((result.mu[0] - 8.0 / (1.0 / result.v[0] + 10.0)).abs() > 0.01);
    }

    #[test]
    fn simple_only_compares_current_variance_with_zero() {
        let mut settings = options(1);
        settings.estimate_prior_variance = true;
        settings.estimate_prior_method = "simple".into();
        let signal = fit(&[10.0], &[8.0], 20.0, 21.0, &settings).expect("simple signal");
        assert_eq!(signal.v[0], 0.2);
        settings.check_null_threshold = 100.0;
        let null = fit(&[10.0], &[8.0], 20.0, 21.0, &settings).expect("simple null");
        assert_eq!(null.v[0], 0.0);
    }

    #[test]
    fn zero_prior_weight_retains_upstream_epsilon_floor() {
        let mut settings = options(2);
        settings.prior_weights = vec![1.0, 0.0];
        settings.prior_variance = 0.0;
        settings.prior_tol = 0.0; // Inspect untrimmed SER epsilon handling.
        let result = fit(&[10.0, 0.0, 0.0, 10.0], &[0.0, 0.0], 20.0, 21.0, &settings)
            .expect("zero prior fit");
        let eps = f64::EPSILON.sqrt();
        assert!(result.alpha[1] > 0.0);
        assert!((result.alpha[1] - eps / (1.0 + 2.0 * eps)).abs() < 1e-22);
    }

    #[test]
    fn sequential_updates_use_new_effects() {
        let mut settings = options(2);
        settings.l = 2;
        settings.max_iter = 1;
        let matrix = [10.0, 4.0, 4.0, 10.0];
        let result = fit(&matrix, &[8.0, -4.0], 50.0, 51.0, &settings).expect("sequential fit");
        let b0 = result.alpha[0] * result.mu[0];
        let b1 = result.alpha[1] * result.mu[1];
        assert!((result.mu[2] - (8.0 - 10.0 * b0 - 4.0 * b1) / 15.0).abs() < 1e-14);
        assert_ne!(result.mu[0], result.mu[2]);
        assert!(!result.converged);
    }

    #[test]
    fn singular_design_and_large_bf_remain_finite() {
        let result = fit(
            &[100.0, 100.0, 100.0, 100.0],
            &[500.0, 500.0],
            10000.0,
            101.0,
            &options(2),
        )
        .expect("singular fit");
        assert_eq!(result.alpha, vec![0.5, 0.5]);
        assert!(result.lbf[0] > 1000.0);
        assert!(result.elbo.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn residual_variance_updates_after_objective() {
        let mut settings = options(1);
        settings.max_iter = 1;
        settings.estimate_residual_variance = true;
        let result = fit(&[10.0], &[4.0], 20.0, 21.0, &settings).expect("residual fit");
        let expected = (20.0 - 8.0 * result.mu[0] + 10.0 * result.mu2[0]) / 21.0;
        assert!((result.sigma2 - expected).abs() < 1e-14);
        let expected_elbo = -21.0 / 2.0 * (2.0 * PI).ln() - 21.0 * expected / 2.0 - result.kl[0];
        assert!((result.elbo[0] - expected_elbo).abs() < 1e-13);
    }

    #[test]
    fn invalid_inputs_return_errors() {
        assert!(fit(&[], &[], 1.0, 10.0, &options(1)).is_err());
        assert!(fit(&[-1.0], &[0.0], 1.0, 10.0, &options(1)).is_err());
        let mut settings = options(1);
        settings.estimate_prior_method = "uniroot".into();
        assert!(fit(&[1.0], &[0.0], 1.0, 10.0, &settings).is_err());
    }

    #[test]
    fn contiguous_matrix_product_matches_independent_integer_arithmetic() {
        // Exactly representable integer arithmetic supplies an independent answer.
        let values: Vec<f64> = (0..32).map(|j| (j as i64 % 7 - 3) as f64).collect();
        let vector = [2.,-1.,3.,0.,-2.,1.,4.,-3.];
        let matrix = ArrayView2::from_shape((4,8), &values).unwrap();
        let mut output = [0.;4];
        multiply(matrix, &vector, &mut output);
        for i in 0..4 {
            let expected: i64 = (0..8).map(|j| values[i*8+j] as i64 * vector[j] as i64).sum();
            assert!((output[i] - expected as f64).abs() <= 1e-12);
        }
    }
    #[test]
    fn column_major_dispatch_tail_and_zero_columns() {
        let values = [1.,2.,3.,4.,5., 7.,8.,9.,10.,11., -1.,-2.,-3.,-4.,-5.];
        let matrix = ArrayView2::from_shape((5,3).f(), &values).unwrap();
        let mut output = [99.;5];
        multiply(matrix, &[2.,0.,-1.], &mut output);
        assert_eq!(output,[3.,6.,9.,12.,15.]);
        let mut portable = [99.;5];
        multiply_columns(&values, &[2.,0.,-1.], &mut portable);
        assert_eq!(output,portable);
        multiply(matrix, &[0.,0.,0.], &mut output);
        assert_eq!(output,[0.;5]);
    }

    #[test]
    fn grouped_columns_cross_group_and_tail() {
        // Seven columns, one exactly zero coefficient and an odd row count.
        let values: Vec<f64> = (0..35).map(|i| (i as i64 % 11 - 5) as f64).collect();
        let vector = [2.,0.,-3.,4.,1.,-2.,5.];
        let mut output = [0.;5];
        let matrix = ArrayView2::from_shape((5,7).f(), &values).unwrap();
        multiply(matrix, &vector, &mut output);
        for i in 0..5 {
            let expected: i64 = (0..7).map(|j| values[j*5+i] as i64 * vector[j] as i64).sum();
            assert_eq!(output[i],expected as f64);
        }
    }

    #[test]
    fn exact_column_groups_preserve_signed_and_zero_contributions() {
        let values=[1.,2.,3., -1.,-2.,-3., 0.,0.,0., 1.,2.,3., 4.,5.,6.];
        let matrix=ArrayView2::from_shape((3,5).f(),&values).unwrap();
        let mut groups=ExactColumns::new(matrix);
        assert_eq!(groups.representatives.len(),3);
        let vector=[2.,3.,17.,4.,-2.];let mut output=[0.;3];
        groups.multiply(matrix,&vector,&mut output);
        assert_eq!(output,[-5.,-4.,-3.]);
        groups.multiply(matrix,&[0.;5],&mut output);
        assert_eq!(output,[0.;3]);
    }

    #[test]
    fn close_columns_are_not_grouped() {
        let values=[1.,2.,3., 1.,2.,3.+1e-12];
        let matrix=ArrayView2::from_shape((3,2).f(),&values).unwrap();
        let groups=ExactColumns::new(matrix);
        assert_eq!(groups.representatives.len(),2);
    }

    #[test]
    fn exact_rows_and_columns_expand_every_original_output() {
        // Signed incidence times [[2,1],[1,3]] times incidence transpose.
        let values=[2.,1.,-2.,2., 1.,3.,-1.,1., -2.,-1.,2.,-2., 2.,1.,-2.,2.];
        let matrix=ArrayView2::from_shape((4,4).f(),&values).unwrap();
        let mut plan=ExactColumns::new(matrix);
        assert!(plan.compact.is_some());
        let mut result=[0.;4];
        plan.multiply(matrix,&[1.,2.,-3.,4.],&mut result);
        assert_eq!(result,[18.,14.,-18.,18.]);
    }

    #[test]
    fn row_mismatch_retains_full_row_fallback() {
        // Equal columns do not establish equal rows for a general matrix.
        let values=[1.,2.,3., 1.,2.,3., 2.,3.,4.];
        let matrix=ArrayView2::from_shape((3,3).f(),&values).unwrap();
        let mut plan=ExactColumns::new(matrix);
        assert!(plan.compact.is_none());
        let mut result=[0.;3];
        plan.multiply(matrix,&[2.,3.,4.],&mut result);
        assert_eq!(result,[13.,22.,31.]);
    }

    #[test]
    fn sampled_signature_collisions_use_exact_full_column_checks() {
        // All 32 columns agree at the 16 signature rows. Sixteen distinct row-1
        // values, repeated twice, force the bounded full-hash bucket fallback.
        let p=65;let columns=32;let mut values=vec![0.;p*columns];
        for j in 0..columns { values[j*p]=1.;values[j*p+1]=(j%16+1) as f64; }
        let matrix=ArrayView2::from_shape((p,columns).f(),&values).unwrap();
        let mut plan=ExactColumns::new(matrix);
        assert_eq!(plan.representatives.len(),16);
        let vector:Vec<f64>=(0..columns).map(|j| (j%7) as f64-3.).collect();
        let mut output=vec![0.;p];plan.multiply(matrix,&vector,&mut output);
        let expected0:i64=vector.iter().map(|&x| x as i64).sum();
        let expected1:i64=vector.iter().enumerate().map(|(j,&x)| x as i64*(j%16+1) as i64).sum();
        assert_eq!(output[0],expected0 as f64);assert_eq!(output[1],expected1 as f64);
        assert!(output[2..].iter().all(|&x| x==0.));
    }

    #[test]
    fn distinct_signatures_prove_no_useful_exact_redundancy() {
        let rows=17;let columns=32;let mut values=vec![0.;rows*columns];
        for j in 0..columns { values[j*rows]=(j+1) as f64;values[j*rows+1]=1.; }
        let matrix=ArrayView2::from_shape((rows,columns).f(),&values).unwrap();
        assert!(!ExactColumns::may_reduce(matrix));
        assert_eq!(ExactColumns::new(matrix).representatives.len(),columns);
    }

    #[test]
    fn large_repeated_design_retains_zero_prior_and_no_information_columns() {
        // An analytical three-type design: 512 identical A columns, 512 B
        // columns and two no-information columns. One fixed-V effect has a
        // closed-form posterior for every original SNP, including zero priors.
        let p=1026;let class=|i:usize| if i<512 { 0 } else if i<1024 { 1 } else { 2 };
        let cross=[[10.,2.,0.],[2.,8.,0.],[0.,0.,0.]];
        let xtx:Vec<f64>=(0..p*p).map(|k| cross[class(k%p)][class(k/p)]).collect();
        let response:Vec<f64>=(0..p).map(|i| [4.,-1.,0.][class(i)]).collect();
        let mut settings=options(p);settings.prior_weights=vec![1./(p-8) as f64;p];
        settings.prior_weights[..8].fill(0.);settings.null_index=Some(p-1);
        let result=fit_layout(&xtx,&response,20.,21.,&settings,true).unwrap();
        assert_eq!(result.alpha.len(),p);assert_eq!(result.mu.len(),p);
        let lbf_a=-0.5*(1.0_f64+0.2/0.1).ln()+0.5*0.4*0.4*0.2/(0.1*(0.2+0.1));
        let lbf_b=-0.5*(1.0_f64+0.2/0.125).ln()+0.5*0.125*0.125*0.2/(0.125*(0.2+0.125));
        let bf=[lbf_a.exp(),lbf_b.exp(),1.];
        let weights:Vec<f64>=(0..p).map(|i| (settings.prior_weights[i]+f64::EPSILON.sqrt())*bf[class(i)]).collect();
        let total:f64=weights.iter().sum();
        for i in 0..p { assert!((result.alpha[i]-weights[i]/total).abs()<1e-12); }
        assert!(result.alpha[..8].iter().all(|&x| x>0.));
        assert_eq!(&result.mu[1024..],&[0.,0.]);
        assert_eq!(&result.mu2[1024..],&[0.,0.]);
    }

    #[test]
    fn representative_rows_prove_nonsymmetric_signed_expansion() {
        // Signed incidence times the non-symmetric [[2,3],[5,7]] matrix.
        // This independently checks that no symmetry assumption enters.
        let values=[2.,5.,-2.,2., 3.,7.,-3.,3., -2.,-5.,2.,-2., 2.,5.,-2.,2.];
        let matrix=ArrayView2::from_shape((4,4).f(),&values).unwrap();
        let mut plan=ExactColumns::new(matrix);
        assert!(plan.compact.is_some());
        assert_eq!(plan.representatives.len(),2);
        let mut output=[0.;4];
        plan.multiply(matrix,&[1.,2.,-3.,4.],&mut output);
        assert_eq!(output,[22.,54.,-22.,22.]);
    }

    #[test]
    fn column_comparison_vector_chunks_and_tails_are_exact() {
        let positive=[0.,1.,-2.,3.,4.,-5.,6.,7.,-8.,9.,10.,-11.,12.];
        for length in 1..=positive.len() {
            for sign in [-1.,1.] {
                let original=&positive[..length];
                let mut other:Vec<f64>=original.iter().map(|&x| sign*x).collect();
                other[0]=-0.;
                assert!(exact_column(original,&other,sign));
                assert!(exact_column_chunks(original,&other,sign));
                for index in 0..length {
                    let old=other[index];other[index]+=1e-10;
                    assert!(!exact_column(original,&other,sign));
                    assert!(!exact_column_chunks(original,&other,sign));
                    other[index]=old;
                }
            }
        }
    }

    #[test]
    fn parallel_discovery_restores_original_signed_class_order() {
        let values=[2.,5.,-2.,2., 3.,7.,-3.,3., -2.,-5.,2.,-2., 2.,5.,-2.,2.];
        let matrix=ArrayView2::from_shape((4,4).f(),&values).unwrap();
        let mut plan=ExactColumns::parallel(matrix,4).unwrap();
        assert_eq!(plan.representatives,vec![0,1]);
        assert_eq!(plan.assignments,vec![(0,1.),(1,1.),(0,-1.),(0,1.)]);
        let mut output=[0.;4];plan.multiply(matrix,&[1.,2.,-3.,4.],&mut output);
        assert_eq!(output,[22.,54.,-22.,22.]);
    }

    #[test]
    fn single_signature_bucket_retains_serial_fallback() {
        let values=[2.,2.,2.,2.];
        let matrix=ArrayView2::from_shape((2,2).f(),&values).unwrap();
        assert!(ExactColumns::parallel(matrix,4).is_none());
        let mut plan=ExactColumns::serial(matrix);let mut output=[0.;2];
        plan.multiply(matrix,&[3.,-1.],&mut output);
        assert_eq!(output,[4.,4.]);
    }

    #[test]
    fn parallel_row_failure_retains_original_output_rows() {
        // First representative has valid row relations; the second violates
        // the negative row relation. No partial compact result may be used.
        let values=[2.,1.,-2.,2., 3.,7.,-4.,3., -2.,-1.,2.,-2., 2.,1.,-2.,2.];
        let matrix=ArrayView2::from_shape((4,4).f(),&values).unwrap();
        let mut plan=ExactColumns::parallel(matrix,4).unwrap();
        assert!(plan.compact.is_none());
        let mut output=[0.;4];plan.multiply(matrix,&[1.,2.,-3.,4.],&mut output);
        assert_eq!(output,[22.,22.,-24.,22.]);
    }

    #[test]
    fn blocked_serial_finish_discards_partial_output_after_later_block_failure() {
        // Closed-form incidence matrix with 65 exact column classes. For
        // p=1024 the512KiB block contains64 columns, so failure in class64
        // occurs after the first64 compact columns have already been copied.
        let p=1024;let groups=65;let mut values=vec![0.;p*p];
        for j in 0..p {
            for i in 0..p {
                if i%groups==j%groups { values[j*p+i]=(i%groups+1) as f64; }
            }
            if j%groups==64 { values[j*p+65]=7.; }
        }
        let matrix=ArrayView2::from_shape((p,p).f(),&values).unwrap();
        let mut plan=ExactColumns::serial(matrix);
        assert_eq!(plan.representatives.len(),groups);
        assert!(plan.compact.is_none());
        let mut product=vec![0.;p];plan.multiply(matrix,&vec![1.;p],&mut product);
        assert_eq!(product[0],16.);assert_eq!(product[65],121.);assert_eq!(product[64],975.);
    }
    #[test]
    fn representative_row_simd_checks_signed_zeros_vector_and_tail() {
        let mapping=[0_i64,1,0,1,0];let signs=[0.,0.,-0.,0.,0.];
        let original=[2.,1.,-2.,1.,2., 1.,3.,-1.,3.,1.];
        for broken in [None,Some(2usize),Some(4usize),Some(9usize)] {
            let mut values=original;
            if let Some(i)=broken { values[i]+=1.; }
            let expected=broken.is_none();
            assert_eq!(exact_rows_scalar(&values,5,&mapping,&signs,&[0,1]),expected);
            assert_eq!(exact_rows_block(&values,5,&mapping,&signs,&[0,1]),expected);
        }
        let zeros=[0.,-0.,0.,0.,-0.];
        assert!(exact_rows_scalar(&zeros,5,&mapping,&signs,&[0]));
        assert!(exact_rows_block(&zeros,5,&mapping,&signs,&[0]));
        // Only selected representatives are inspected; the full column proof
        // elsewhere is responsible for deriving all nonrepresentative columns.
        let values=[2.,1.,-2.,1.,2., 1.,3.,-1.,3.,99.];
        assert!(exact_rows_block(&values,5,&mapping,&signs,&[0]));
        assert!(!exact_rows_block(&values,5,&mapping,&signs,&[1]));
    }

    #[test]
    fn scaled_operator_preserves_fixed_model_null_and_partial_zero_prior() {
        let raw=[2.,1.,0., 1.,3.,0., 0.,0.,0.];
        let matrix=ArrayView2::from_shape((3,3).f(),&raw).unwrap();
        let scale=MatrixScaling {global:8.,inverse:&[0.5,2.,1.],scales:&[2.,0.5,1.]};
        let dense=materialize_scaled(matrix,&scale);let mut settings=options(3);
        settings.prior_weights=vec![0.2,0.,0.8];settings.null_index=Some(2);
        let expected=fit_layout(&dense,&[3.,4.,0.],20.,21.,&settings,true).unwrap();
        let observed=fit_layout_operator(&raw,&[3.,4.,0.],20.,21.,&settings,true,Some(&scale)).unwrap();
        for (a,b) in expected.alpha.iter().zip(&observed.alpha) { assert!((a-b).abs()<1e-12); }
        assert_eq!(expected.niter,observed.niter);assert_eq!(observed.mu[2],0.);
    }

    #[test]
    fn scaled_operator_materializes_extreme_factors_without_clipping() {
        let raw=[1e-308];let scale=MatrixScaling {global:1e308,inverse:&[1.],scales:&[1.]};
        let mut settings=options(1);settings.estimate_prior_variance=false;
        let expected=fit_layout(&[1e-308*1e308],&[0.1],20.,21.,&settings,true).unwrap();
        let observed=fit_layout_operator(&raw,&[0.1],20.,21.,&settings,true,Some(&scale)).unwrap();
        assert_eq!(expected.mu,observed.mu);assert_eq!(expected.alpha,observed.alpha);
    }

    #[test]
    fn scaled_product_intermediate_overflow_uses_full_matrix_fallback() {
        let raw=[0.25];let matrix=ArrayView2::from_shape((1,1).f(),&raw).unwrap();
        let scale=MatrixScaling {global:1.,inverse:&[2.],scales:&[0.5]};
        let mut output=[0.];let mut scratch=[0.];
        assert!(scaled_multiply(matrix,&[1e308],&mut output,None,&scale,&mut scratch));
        assert_eq!(output,[1e308]);
    }

    #[test]
    fn scaled_operator_rejects_invalid_factors_and_preparation_overflow() {
        let settings=options(1);
        for scale in [MatrixScaling{global:-1.,inverse:&[1.],scales:&[1.]},
            MatrixScaling{global:1.,inverse:&[],scales:&[1.]},
            MatrixScaling{global:1.,inverse:&[1.],scales:&[f64::NAN]},
            MatrixScaling{global:f64::MAX,inverse:&[2.],scales:&[1.]}] {
            assert!(fit_layout_operator(&[2.],&[0.1],20.,21.,&settings,true,Some(&scale)).is_err());
        }
    }

    #[test]
    fn operator_entry_certificate_checks_every_chunk_tail_and_nonfinite() {
        let valid=[-2.,-0.,0.,2.,1.,-1.,0.5,-0.5,1.,1.,1.,1.,1.,1.,1.,1.,1.];
        assert!(operator_entries_bounded(&valid));
        for bad in [f64::NAN,f64::INFINITY,f64::NEG_INFINITY,f64::from_bits(2.0_f64.to_bits()+1)] {
            for i in 0..valid.len() {
                let mut values=valid;values[i]=bad;assert!(!operator_entries_bounded(&values));
            }
        }
    }

}
