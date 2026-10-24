//! Fit-local deterministic rent-or-buy accounting for exact-class discovery.
//! Constants are frozen in the run's activation_rule.json, calibrated on disjoint
//! real LD inputs. These are operation-count weights, never in-call timers.
#[derive(Default)]
pub(crate) struct ExactActivation {
    pub screened: bool,
    pub estimate: Option<usize>,
    pub rent: f64,
    pub dense_products: usize,
    pub sweep: usize,
    pub attempt_sweep: Option<usize>,
}
impl ExactActivation {
    pub fn purchase_cost(p: usize, g: usize) -> f64 {
        let p = p as f64;
        let g = g as f64;
        5.058382503803307 * p * p + 0.0 * p * g + 18.146224292032198 * g * g + 0.0 * p
    }
    pub fn ready(&self, p: usize) -> bool {
        self.estimate
            .is_some_and(|g| self.rent >= Self::purchase_cost(p, g))
    }
    pub fn accrue(&mut self, p: usize, nonzero: usize) {
        self.dense_products = self.dense_products.saturating_add(1);
        if let Some(g) = self.estimate {
            let p = p as f64;
            let g = g as f64;
            let nonzero = nonzero as f64;
            let full = 1.4532339697391639 * p * nonzero + 7160.611495595075 * p;
            let compact = 2.225971001931629 * g * g + 419.80247665380097 * p;
            self.rent += (full - compact).max(0.);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn purchase_boundary_is_inclusive_and_requires_an_estimate() {
        let p = 1000;
        let g = 250;
        let threshold = ExactActivation::purchase_cost(p, g);
        let mut state = ExactActivation {
            estimate: Some(g),
            ..Default::default()
        };
        state.rent = f64::from_bits(threshold.to_bits() - 1);
        assert!(!state.ready(p));
        state.rent = threshold;
        assert!(state.ready(p));
        state.estimate = None;
        assert!(!state.ready(p));
    }
    #[test]
    fn rent_uses_only_observed_products_and_exact_nonzero_counts() {
        let mut state = ExactActivation {
            estimate: Some(1),
            ..Default::default()
        };
        state.accrue(1000, 1000);
        assert_eq!(state.dense_products, 1);
        assert!(state.rent >= 0.);
        let previous = state.rent;
        state.accrue(1000, 125);
        assert_eq!(state.dense_products, 2);
        assert!(state.rent >= previous);
    }
}
