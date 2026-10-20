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
