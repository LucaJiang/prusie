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

#[derive(Debug, PartialEq)]
pub enum StopDecision {
    Continue,
    Converged,
    Cycle,
    NumericalFailure,
}

/// A finite current objective always uses the requested signed ELBO rule.
/// NaN objectives may use the documented posterior fallback in ELBO mode;
/// infinities are numerical failures, as in the pinned R objective evaluator.
/// Joint mode never drops its objective condition. Lagged cycles only stop
/// explicit posterior/joint modes or the NaN fallback, and are not convergence.
pub fn stopping_decision(
    control: &FitControl,
    snapshot: &Snapshot,
    tol: f64,
) -> (StopDecision, bool) {
    if snapshot.elbo.is_infinite() || (snapshot.elbo.is_nan() && control.criterion == "joint") {
        return (StopDecision::NumericalFailure, false);
    }
    if snapshot.iteration < control.min_iter {
        return (StopDecision::Continue, false);
    }
    let fallback = control.criterion == "elbo" && snapshot.elbo.is_nan();
    let posterior_stable = snapshot.delta_alpha.is_some_and(|d| d < control.alpha_tol)
        && snapshot.delta_pip.is_some_and(|d| d < control.pip_tol);
    let objective_stable =
        snapshot.elbo.is_finite() && snapshot.delta_elbo.is_some_and(|d| elbo_stable(d, tol));
    let converged = match control.criterion.as_str() {
        "alpha_pip" => posterior_stable,
        "joint" => objective_stable && posterior_stable,
        _ if fallback => posterior_stable,
        _ => objective_stable,
    };
    // A diagnostic cannot override an already satisfied convergence criterion.
    if converged {
        return (StopDecision::Converged, fallback);
    }
    if (control.criterion != "elbo" || fallback) && snapshot.cycle_lag.is_some_and(|lag| lag > 1) {
        return (StopDecision::Cycle, fallback);
    }
    (StopDecision::Continue, fallback)
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
    fn snapshot(elbo: f64, delta: Option<f64>) -> Snapshot {
        Snapshot {
            iteration: 3,
            elbo,
            delta_elbo: delta,
            delta_alpha: Some(0.1),
            delta_pip: Some(0.1),
            cycle_lag: Some(2),
            ..Snapshot::default()
        }
    }
    #[test]
    fn repeated_posterior_does_not_stop_rising_finite_elbo() {
        let control = FitControl::default();
        let a = vec![0.8, 0.2];
        let b = vec![0.2, 0.8];
        let history = [(a.clone(), a.clone()), (b.clone(), b)];
        let lag = detect_cycle(&a, &a, &history, 5, 1e-4, 1e-4).unwrap().0;
        let mut state = snapshot(-50., Some(1350.));
        state.cycle_lag = Some(lag);
        assert_eq!(
            stopping_decision(&control, &state, 1e-4),
            (StopDecision::Continue, false)
        );
        state.delta_elbo = Some(1e-5);
        assert_eq!(
            stopping_decision(&control, &state, 1e-4),
            (StopDecision::Converged, false)
        );
        state.delta_elbo = Some(-1e-15);
        assert_eq!(
            stopping_decision(&control, &state, 1e-4),
            (StopDecision::Continue, false)
        );
    }
    #[test]
    fn minimum_applies_to_success_and_cycle_in_every_mode() {
        for criterion in ["elbo", "alpha_pip", "joint"] {
            let mut control = FitControl {
                criterion: criterion.into(),
                min_iter: 4,
                ..FitControl::default()
            };
            let mut state = snapshot(1., Some(0.));
            assert_eq!(
                stopping_decision(&control, &state, 1e-4).0,
                StopDecision::Continue
            );
            state.delta_alpha = Some(0.);
            state.delta_pip = Some(0.);
            assert_eq!(
                stopping_decision(&control, &state, 1e-4).0,
                StopDecision::Continue
            );
            control.min_iter = 3;
            assert_eq!(
                stopping_decision(&control, &state, 1e-4).0,
                StopDecision::Converged
            );
        }
    }
    #[test]
    fn posterior_joint_and_fallback_have_distinct_success_rules() {
        let mut control = FitControl::default();
        let mut state = snapshot(10., Some(1.));
        control.criterion = "alpha_pip".into();
        assert_eq!(
            stopping_decision(&control, &state, 1e-4).0,
            StopDecision::Cycle
        );
        state.delta_alpha = Some(0.);
        state.delta_pip = Some(0.);
        assert_eq!(
            stopping_decision(&control, &state, 1e-4).0,
            StopDecision::Converged
        );
        control.criterion = "joint".into();
        state.cycle_lag = Some(1);
        assert_eq!(
            stopping_decision(&control, &state, 1e-4).0,
            StopDecision::Continue
        );
        state.delta_elbo = Some(-1e-15);
        assert_eq!(
            stopping_decision(&control, &state, 1e-4).0,
            StopDecision::Continue
        );
        state.delta_elbo = Some(0.);
        assert_eq!(
            stopping_decision(&control, &state, 1e-4).0,
            StopDecision::Converged
        );
        state.delta_pip = Some(1.);
        assert_eq!(
            stopping_decision(&control, &state, 1e-4).0,
            StopDecision::Continue
        );
        state.elbo = f64::NAN;
        assert_eq!(
            stopping_decision(&control, &state, 1e-4).0,
            StopDecision::NumericalFailure
        );
        control.criterion = "elbo".into();
        state.delta_pip = Some(0.);
        assert_eq!(
            stopping_decision(&control, &state, 1e-4),
            (StopDecision::Converged, true)
        );
        state.delta_pip = Some(1.);
        state.cycle_lag = Some(2);
        assert_eq!(
            stopping_decision(&control, &state, 1e-4),
            (StopDecision::Cycle, true)
        );
        control.min_iter = 4;
        assert_eq!(
            stopping_decision(&control, &state, 1e-4),
            (StopDecision::Continue, false)
        );
    }
    #[test]
    fn initial_infinity_and_actual_numerical_failure_are_distinct() {
        let control = FitControl::default();
        let mut state = snapshot(1., None);
        state.iteration = 1;
        assert_eq!(
            stopping_decision(&control, &state, 1e-4),
            (StopDecision::Continue, false)
        );
        state.iteration = 2;
        state.delta_elbo = Some(f64::INFINITY);
        // Even an overflowing difference between finite objectives is not
        // a posterior fallback or a successful signed ELBO check.
        assert_eq!(
            stopping_decision(&control, &state, 1e-4),
            (StopDecision::Continue, false)
        );
        for elbo in [f64::INFINITY, f64::NEG_INFINITY] {
            state.elbo = elbo;
            state.iteration = 1;
            assert_eq!(
                stopping_decision(&control, &state, 1e-4),
                (StopDecision::NumericalFailure, false)
            );
        }
    }
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
