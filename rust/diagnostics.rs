//! Exact online summaries at the post-sweep, pre-residual-variance-update phase.
use super::FitOutput;

#[derive(Default)]
pub struct InitialState {
    pub alpha: Vec<f64>,
    pub mu: Vec<f64>,
    pub mu2: Vec<f64>,
    pub v: Option<Vec<f64>>,
    pub sigma2: Option<f64>,
}
pub struct FitControl {
    pub initial: Option<InitialState>,
    pub track: u8,
    pub criterion: String,
    pub alpha_tol: f64,
    pub pip_tol: f64,
    pub min_iter: usize,
    pub cycle_lag: usize,
    pub hard_exclusion: bool,
}
impl Default for FitControl {
    fn default() -> Self {
        Self {
            initial: None,
            track: 0,
            criterion: "elbo".into(),
            alpha_tol: 1e-4,
            pip_tol: 1e-4,
            min_iter: 2,
            cycle_lag: 5,
            hard_exclusion: false,
        }
    }
}
#[derive(Clone, Default)]
pub struct Snapshot {
    pub iteration: usize,
    pub elbo: f64,
    pub delta_elbo: Option<f64>,
    pub relative_delta_elbo: Option<f64>,
    pub delta_alpha: Option<f64>,
    pub delta_pip: Option<f64>,
    pub delta_mean: Option<f64>,
    pub delta_v: Option<Vec<f64>>,
    pub delta_sigma2: Option<f64>,
    pub v: Vec<f64>,
    pub sigma2: f64,
    pub top_variant: Vec<usize>,
    pub top_alpha: Vec<f64>,
    pub entropy: Vec<f64>,
    pub active: Vec<bool>,
    pub alpha: Vec<f64>,
    pub mu: Vec<f64>,
    pub mu2: Vec<f64>,
    pub pip: Vec<f64>,
    pub alpha_entries: Vec<(usize, usize, f64)>,
    pub omitted_mass: Vec<f64>,
    pub cycle_lag: Option<usize>,
    pub cycle_alpha_distance: Option<f64>,
    pub cycle_pip_distance: Option<f64>,
    pub events: Vec<String>,
}
#[derive(Default)]
pub struct Diagnostics {
    pub trace: Vec<Snapshot>,
    pub last: Option<Snapshot>,
    pub stop_reason: String,
    pub effective_criterion: String,
    pub decrease_count: usize,
    pub nonfinite_count: usize,
    pub fallback_iterations: Vec<usize>,
    pub trimmed_components: Vec<usize>,
    pub trim_alpha_distance: f64,
    pub trim_pip_distance: f64,
    pub sigma2_after_loop: f64,
}
pub fn pip(alpha: &[f64], v: &[f64], p: usize, prior_tol: f64, null: Option<usize>) -> Vec<f64> {
    (0..p)
        .filter(|&j| Some(j) != null)
        .map(|j| {
            1. - (0..v.len())
                .filter(|&l| v[l] > prior_tol)
                .map(|l| 1. - alpha[l * p + j])
                .product::<f64>()
        })
        .collect()
}
pub fn elbo_stable(delta: f64, tol: f64) -> bool {
    delta.is_finite() && delta >= 0. && delta < tol
}

pub fn distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0., f64::max)
}
pub fn posterior_mean(fit: &FitOutput, p: usize) -> Vec<f64> {
    (0..p)
        .map(|j| {
            (0..fit.v.len())
                .map(|l| fit.alpha[l * p + j] * fit.mu[l * p + j])
                .sum()
        })
        .collect()
}
/// Returns a fixed point (lag 1) separately from a cycle; never mutates posterior.
pub fn detect_cycle(
    alpha: &[f64],
    pip: &[f64],
    history: &[(Vec<f64>, Vec<f64>)],
    max_lag: usize,
    alpha_tol: f64,
    pip_tol: f64,
) -> Option<(usize, f64, f64)> {
    for lag in 1..=history.len().min(max_lag) {
        let (a, b) = &history[history.len() - lag];
        let da = distance(alpha, a);
        let dp = distance(pip, b);
        if da < alpha_tol && dp < pip_tol {
            return Some((lag, da, dp));
        }
    }
    None
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_and_nonfinite_elbo_rules() {
        assert!(elbo_stable(0., 1e-4));
        assert!(elbo_stable(1e-5, 1e-4));
        for delta in [
            -1e-15,
            -1.,
            1e-4,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            assert!(!elbo_stable(delta, 1e-4));
        }
        assert!(!elbo_stable(0., 0.));
    }
    #[test]
    fn active_pip_and_cycles() {
        assert_eq!(pip(&[0.8, 0.2], &[0.], 2, 1e-9, None), vec![0., 0.]);
        assert_eq!(pip(&[0.8, 0.2], &[1.], 2, 1e-9, Some(1)), vec![0.8]);
        let a = vec![0.8, 0.2];
        let b = vec![0.2, 0.8];
        let c = vec![0.5, 0.5];
        assert_eq!(
            detect_cycle(&a, &a, &[(a.clone(), a.clone())], 5, 1e-8, 1e-8)
                .unwrap()
                .0,
            1
        );
        assert_eq!(
            detect_cycle(
                &a,
                &a,
                &[(a.clone(), a.clone()), (b.clone(), b.clone())],
                5,
                1e-8,
                1e-8
            )
            .unwrap()
            .0,
            2
        );
        assert_eq!(
            detect_cycle(
                &a,
                &a,
                &[
                    (a.clone(), a.clone()),
                    (b.clone(), b.clone()),
                    (c.clone(), c)
                ],
                5,
                1e-8,
                1e-8
            )
            .unwrap()
            .0,
            3
        );
        assert_eq!(a, vec![0.8, 0.2]);
    }
}
