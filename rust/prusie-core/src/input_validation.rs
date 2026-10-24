//! Finiteness checks of prepared inputs; values are neither changed nor skipped.

/// Whether every entry is finite. Combining the predicates with `&` instead of
/// short-circuiting allows vectorization; the whole slice is always scanned.
/// No particular CPU feature is required.
#[inline]
pub fn all_finite(values: &[f64]) -> bool {
    values
        .iter()
        .fold(true, |valid, value| valid & value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::all_finite;

    #[test]
    fn handles_every_nonfinite_position_and_tail() {
        for size in [0, 1, 2, 3, 4, 7, 8, 9, 31, 32, 33, 129] {
            let mut values = vec![-0.0; size];
            assert!(all_finite(&values));
            for index in 0..size {
                for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                    values[index] = bad;
                    assert!(!all_finite(&values));
                    values[index] = -0.0;
                }
            }
        }
    }

    #[test]
    fn finite_extremes_subnormals_and_signed_zeros_are_valid() {
        assert!(all_finite(&[
            f64::MAX,
            f64::MIN,
            f64::MIN_POSITIVE,
            f64::from_bits(1),
            -f64::from_bits(1),
            0.0,
            -0.0
        ]));
    }
}

mod rss_pairs_owned;
pub use rss_pairs_owned::{rss_pair_validation_owned, RssPairLayout, RssPairValidation};
