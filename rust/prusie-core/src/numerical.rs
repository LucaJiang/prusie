/// Compensated (Neumaier) summation, used for likelihood and objective sums
/// that are prone to cancellation. Dense matrix products do not use it.
pub(crate) fn sum(values: impl Iterator<Item = f64>) -> f64 {
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

pub(crate) fn dot(x: &[f64], y: &[f64]) -> f64 {
    sum(x.iter().zip(y).map(|(x, y)| x * y))
}

/// Digamma for positive arguments, using recurrence to an asymptotic expansion.
/// Terms through x⁻¹⁶ give better than 2e-13 absolute accuracy for x ≥ 8.
pub(crate) fn digamma(mut x: f64) -> f64 {
    let mut result = 0.;
    while x < 8. {
        result -= 1. / x;
        x += 1.;
    }
    let inverse = 1. / x;
    let inverse2 = inverse * inverse;
    result + x.ln()
        - 0.5 * inverse
        - inverse2
            * (1. / 12.
                - inverse2
                    * (1. / 120.
                        - inverse2
                            * (1. / 252.
                                - inverse2
                                    * (1. / 240.
                                        - inverse2
                                            * (1. / 132.
                                                - inverse2
                                                    * (691. / 32760.
                                                        - inverse2
                                                            * (1. / 12.
                                                                - inverse2 * (3617. / 8160.))))))))
}

#[cfg(test)]
mod special_tests {
    use super::*;
    #[test]
    fn digamma_matches_known_values_and_recurrence() {
        let euler = 0.577_215_664_901_532_9;
        assert!((digamma(1.) + euler).abs() < 2e-13);
        assert!((digamma(0.5) + euler + 2. * 2_f64.ln()).abs() < 2e-13);
        assert!((digamma(19.25) - digamma(18.25) - 1. / 18.25).abs() < 2e-13);
    }
}
