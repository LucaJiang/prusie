//! Exact signed column classes, with an independent proof before reducing rows.
//!
//! Hashes propose comparisons only. Every retained relation uses full f64
//! equality; no LD threshold, rounding, predictor removal or posterior merging
//! is involved. This restores the mathematical construction used before the
//! workspace split, with fit-local lazy discovery controlled by DenseGram.
use ndarray::{ArrayView2, ShapeBuilder};
use std::collections::{hash_map::DefaultHasher, HashMap, HashSet};
use std::hash::Hasher;

fn signature(column: &[f64], sign: f64, sampled: bool) -> u64 {
    let mut hash = DefaultHasher::new();
    let mut add = |value: f64| {
        hash.write_u64(if value == 0. {
            0
        } else {
            (value * sign).to_bits()
        });
    };
    if sampled {
        for k in 0..16 {
            add(column[k * (column.len() - 1) / 15]);
        }
    } else {
        for &value in column {
            add(value);
        }
    }
    hash.finish()
}

#[inline(always)]
fn equal_signed(column: &[f64], reference: &[f64], sign: f64) -> bool {
    let mut left = column.chunks_exact(4);
    let mut right = reference.chunks_exact(4);
    for (a, b) in left.by_ref().zip(right.by_ref()) {
        if !((a[0] == sign * b[0])
            & (a[1] == sign * b[1])
            & (a[2] == sign * b[2])
            & (a[3] == sign * b[3]))
        {
            return false;
        }
    }
    left.remainder()
        .iter()
        .zip(right.remainder())
        .all(|(&a, &b)| a == sign * b)
}

pub(crate) struct ExactColumns {
    assignments: Vec<(usize, f64)>,
    representatives: Vec<usize>,
    sums: Vec<f64>,
    corrections: Vec<f64>,
    coefficients: Vec<f64>,
    compact: Option<Vec<f64>>,
    compact_product: Vec<f64>,
}

impl ExactColumns {
    /// The sampled signatures give a lower bound on exact class count. This
    /// cheap rejection avoids a full proof when even 1/8 column savings cannot
    /// occur. Collisions can cause extra proof work, never a false relation.
    fn may_reduce(matrix: ArrayView2<'_, f64>) -> bool {
        let rows = matrix.nrows();
        let mut unique = HashSet::new();
        for column in matrix
            .as_slice_memory_order()
            .expect("contiguous columns")
            .chunks_exact(rows)
        {
            let sign = (0..16)
                .map(|k| column[k * (rows - 1) / 15])
                .find(|&x| x != 0.)
                .map_or(1., |x| if x < 0. { -1. } else { 1. });
            unique.insert(signature(column, sign, true));
            if unique.len() * 8 > matrix.ncols() * 7 {
                return false;
            }
        }
        true
    }

    pub(crate) fn discover(matrix: ArrayView2<'_, f64>) -> Option<Self> {
        let rows = matrix.nrows();
        let columns = matrix.ncols();
        if rows == 0 || columns < 2 || !matrix.t().is_standard_layout() || !Self::may_reduce(matrix)
        {
            return None;
        }
        enum Bucket {
            Few(Vec<usize>),
            Many(HashMap<u64, Vec<usize>>),
        }
        let values = matrix.as_slice_memory_order()?;
        let mut buckets: HashMap<u64, Bucket> = HashMap::new();
        let mut assignments = Vec::with_capacity(columns);
        let mut representatives: Vec<usize> = Vec::new();
        let mut polarities = Vec::new();
        for (j, column) in values.chunks_exact(rows).enumerate() {
            let sign = column
                .iter()
                .find(|&&x| x != 0.)
                .map_or(1., |&x| if x < 0. { -1. } else { 1. });
            let bucket = buckets
                .entry(signature(column, sign, true))
                .or_insert_with(|| Bucket::Few(Vec::new()));
            let candidates = match bucket {
                Bucket::Few(ids) => ids,
                Bucket::Many(map) => map.entry(signature(column, sign, false)).or_default(),
            };
            let relation = candidates.iter().find_map(|&group| {
                let representative = representatives[group];
                let ratio = sign * polarities[group];
                equal_signed(
                    column,
                    &values[representative * rows..(representative + 1) * rows],
                    ratio,
                )
                .then_some((group, ratio))
            });
            assignments.push(relation.unwrap_or_else(|| {
                let group = representatives.len();
                representatives.push(j);
                polarities.push(sign);
                candidates.push(group);
                (group, 1.)
            }));
            // Uninformative sampled rows (e.g. an identity matrix) must not
            // create unbounded all-pairs full-column comparisons.
            if let Bucket::Few(ids) = bucket {
                if ids.len() > 8 {
                    let mut full: HashMap<u64, Vec<usize>> = HashMap::new();
                    for &group in ids.iter() {
                        let j = representatives[group];
                        full.entry(signature(
                            &values[j * rows..(j + 1) * rows],
                            polarities[group],
                            false,
                        ))
                        .or_default()
                        .push(group);
                    }
                    *bucket = Bucket::Many(full);
                }
            }
        }
        let groups = representatives.len();
        if groups * 8 > columns * 7 {
            return None;
        }
        // Column relations alone do not prove row relations for an asymmetric
        // matrix. Prove every row on every representative column separately.
        // All remaining columns already have a proved relation to one of these.
        let compact = if rows == columns
            && representatives.iter().all(|&j| {
                let column = &values[j * rows..(j + 1) * rows];
                assignments
                    .iter()
                    .enumerate()
                    .all(|(i, &(g, sign))| column[i] == sign * column[representatives[g]])
            }) {
            let mut compact = Vec::with_capacity(groups * groups);
            for &j in &representatives {
                for &i in &representatives {
                    compact.push(values[j * rows + i]);
                }
            }
            Some(compact)
        } else {
            None
        };
        Some(Self {
            assignments,
            representatives,
            sums: vec![0.; groups],
            corrections: vec![0.; groups],
            coefficients: vec![0.; columns],
            compact,
            compact_product: vec![0.; groups],
        })
    }

    /// Return false at exceptional arithmetic scales so the original dense
    /// operator can be used. In particular, regrouped sums must not introduce
    /// an overflowing intermediate that was absent from the original product.
    pub(crate) fn apply(
        &mut self,
        matrix: ArrayView2<'_, f64>,
        vector: &[f64],
        output: &mut [f64],
    ) -> bool {
        self.sums.fill(0.);
        self.corrections.fill(0.);
        for (&value, &(g, sign)) in vector.iter().zip(&self.assignments) {
            let value = value * sign;
            let old = self.sums[g];
            let next = old + value;
            self.corrections[g] += if old.abs() >= value.abs() {
                (old - next) + value
            } else {
                (value - next) + old
            };
            self.sums[g] = next;
        }
        for (sum, &correction) in self.sums.iter_mut().zip(&self.corrections) {
            *sum += correction;
        }
        if self.sums.iter().any(|x| !x.is_finite()) {
            return false;
        }
        if let Some(compact) = &self.compact {
            let groups = self.representatives.len();
            let reduced =
                ArrayView2::from_shape((groups, groups).f(), compact).expect("compact dimensions");
            super::operator::multiply(reduced, &self.sums, &mut self.compact_product);
            for (out, &(g, sign)) in output.iter_mut().zip(&self.assignments) {
                *out = sign * self.compact_product[g];
            }
        } else {
            for (g, &j) in self.representatives.iter().enumerate() {
                self.coefficients[j] = self.sums[g];
            }
            super::operator::multiply(matrix, &self.coefficients, output);
        }
        output.iter().all(|x| x.is_finite())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compare(matrix: ArrayView2<'_, f64>, vector: &[f64], expected_two_way: bool) {
        let mut plan = ExactColumns::discover(matrix).expect("exact redundancy");
        assert_eq!(plan.compact.is_some(), expected_two_way);
        let mut result = vec![0.; matrix.nrows()];
        assert!(plan.apply(matrix, vector, &mut result));
        for i in 0..matrix.nrows() {
            let expected: f64 = (0..matrix.ncols())
                .map(|j| matrix[(i, j)] * vector[j])
                .sum();
            assert_eq!(result[i], expected);
        }
    }

    #[test]
    fn signed_columns_do_not_imply_signed_rows() {
        let values = [1., 2., 3., -1., -2., -3., 1., 2., 3.];
        let matrix = ArrayView2::from_shape((3, 3).f(), &values).unwrap();
        compare(matrix, &[2., -3., 5.], false);
    }

    #[test]
    fn two_way_signed_relations_preserve_original_outputs() {
        let x = [1., -1., 2., 1., -2., 0.];
        let values: Vec<f64> = x
            .iter()
            .flat_map(|&b| x.iter().map(move |&a| a * b))
            .collect();
        let matrix = ArrayView2::from_shape((6, 6).f(), &values).unwrap();
        compare(matrix, &[2., -3., 4., 5., -1., 7.], true);
    }

    #[test]
    fn a_single_unsampled_difference_is_never_merged() {
        let mut matrix = ndarray::Array2::zeros((64, 4).f());
        matrix[(1, 0)] = 1.;
        matrix[(1, 1)] = 1.;
        matrix[(1, 2)] = 1.;
        matrix[(1, 3)] = 1.;
        matrix[(2, 3)] = f64::from_bits(1);
        let plan = ExactColumns::discover(matrix.view()).unwrap();
        assert_eq!(plan.representatives.len(), 2);
        assert_ne!(plan.assignments[0].0, plan.assignments[3].0);
    }

    #[test]
    fn degenerate_samples_without_duplicates_are_rejected() {
        let matrix = ndarray::Array2::<f64>::eye(128).reversed_axes();
        assert!(ExactColumns::discover(matrix.view()).is_none());
    }

    #[test]
    fn signed_zero_and_smallest_subnormal_are_not_thresholded() {
        let tiny = f64::from_bits(1);
        let values = [tiny, -0., -tiny, 0., tiny, 0.];
        let matrix = ArrayView2::from_shape((2, 3).f(), &values).unwrap();
        compare(matrix, &[1., -1., 1.], false);
    }

    #[test]
    fn overflowing_group_sum_requests_the_dense_fallback() {
        let values = [1., 1., 1., 1.];
        let matrix = ArrayView2::from_shape((2, 2).f(), &values).unwrap();
        let mut plan = ExactColumns::discover(matrix).unwrap();
        assert!(!plan.apply(matrix, &[f64::MAX, f64::MAX], &mut [0.; 2]));
    }
}
