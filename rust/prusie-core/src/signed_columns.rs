//! Exact signed-column compression. Hashes only propose matches; every relation
//! is proved by comparing all rows, without LD rounding or thresholds.
use ndarray::ArrayView2;
use std::collections::{hash_map::DefaultHasher, hash_map::Entry, HashMap};
use std::hash::Hasher;

enum SampleBucket {
    Single(usize),
    Hashed(HashMap<u64, Vec<usize>>),
}

fn canonical_hash(column: &[f64]) -> u64 {
    let mut hasher = DefaultHasher::new();
    hasher.write_usize(column.len());
    let mut polarity = None;
    for &value in column {
        let bits = if value == 0. {
            0
        } else {
            let bits = value.to_bits();
            let sign = polarity.get_or_insert(bits & (1 << 63));
            bits ^ *sign
        };
        hasher.write_u64(bits);
    }
    hasher.finish()
}

fn exact_sign(column: &[f64], reference: &[f64]) -> Option<f64> {
    if column.iter().zip(reference).all(|(a, b)| a == b) {
        Some(1.)
    } else if column.iter().zip(reference).all(|(a, b)| *a == -*b) {
        Some(-1.)
    } else {
        None
    }
}

pub(crate) struct SignedColumns {
    representatives: Vec<usize>,
    groups: Vec<usize>,
    signs: Vec<f64>,
    coefficients: Vec<f64>,
    correction: Vec<f64>,
}
impl SignedColumns {
    pub fn discover(matrix: ArrayView2<'_, f64>) -> Option<Self> {
        if !matrix.t().is_standard_layout() {
            return None;
        }
        let rows = matrix.nrows();
        let columns = matrix.ncols();
        let values = matrix.as_slice_memory_order()?;
        let mut buckets: HashMap<[u64; 8], SampleBucket> = HashMap::new();
        let mut representatives = Vec::new();
        let mut groups = Vec::with_capacity(columns);
        let mut signs = Vec::with_capacity(columns);
        for j in 0..columns {
            let column = &values[j * rows..(j + 1) * rows];
            let samples: [f64; 8] = std::array::from_fn(|k| column[k * (rows - 1) / 7]);
            let polarity =
                samples
                    .iter()
                    .find(|&&x| x != 0.)
                    .map_or(1., |&x| if x < 0. { -1. } else { 1. });
            let key = samples.map(|x| if x == 0. { 0 } else { (x * polarity).to_bits() });
            let (group, sign) = match buckets.entry(key) {
                Entry::Vacant(entry) => {
                    let group = representatives.len();
                    representatives.push(j);
                    entry.insert(SampleBucket::Single(group));
                    (group, 1.)
                }
                Entry::Occupied(mut entry) => match entry.get_mut() {
                    SampleBucket::Single(group) => {
                        let original_group = *group;
                        let rep = representatives[original_group];
                        let reference = &values[rep * rows..(rep + 1) * rows];
                        if let Some(sign) = exact_sign(column, reference) {
                            (original_group, sign)
                        } else {
                            // Underflowed dense columns can have identical zero
                            // samples. Index the entire column after the first
                            // proven nonmatch to avoid all-pairs prefix scans.
                            let mut hashes: HashMap<u64, Vec<usize>> = HashMap::new();
                            hashes
                                .entry(canonical_hash(reference))
                                .or_default()
                                .push(original_group);
                            let group = representatives.len();
                            representatives.push(j);
                            hashes
                                .entry(canonical_hash(column))
                                .or_default()
                                .push(group);
                            *entry.get_mut() = SampleBucket::Hashed(hashes);
                            (group, 1.)
                        }
                    }
                    SampleBucket::Hashed(hashes) => {
                        let candidates = hashes.entry(canonical_hash(column)).or_default();
                        let matched = candidates.iter().find_map(|&group| {
                            let rep = representatives[group];
                            let reference = &values[rep * rows..(rep + 1) * rows];
                            exact_sign(column, reference).map(|sign| (group, sign))
                        });
                        matched.unwrap_or_else(|| {
                            let group = representatives.len();
                            representatives.push(j);
                            candidates.push(group);
                            (group, 1.)
                        })
                    }
                },
            };
            groups.push(group);
            signs.push(sign);
        }
        if representatives.len() == columns {
            return None;
        }
        let count = representatives.len();
        Some(Self {
            representatives,
            groups,
            signs,
            coefficients: vec![0.; count],
            correction: vec![0.; count],
        })
    }
    pub fn apply(&mut self, matrix: ArrayView2<'_, f64>, vector: &[f64], output: &mut [f64]) {
        self.coefficients.fill(0.);
        self.correction.fill(0.);
        for (j, &v) in vector.iter().enumerate() {
            let group = self.groups[j];
            let value = v * self.signs[j];
            let old = self.coefficients[group];
            let next = old + value;
            self.correction[group] += if old.abs() >= value.abs() {
                (old - next) + value
            } else {
                (value - next) + old
            };
            self.coefficients[group] = next;
        }
        output.fill(0.);
        let rows = matrix.nrows();
        let values = matrix.as_slice_memory_order().expect("column-major matrix");
        for (k, &j) in self.representatives.iter().enumerate() {
            let weight = self.coefficients[k] + self.correction[k];
            if weight == 0. {
                continue;
            }
            for (result, &x) in output.iter_mut().zip(&values[j * rows..(j + 1) * rows]) {
                *result += x * weight;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ShapeBuilder;
    #[test]
    fn exact_relations_include_negative_columns_and_signed_zero() {
        let values = [1., 2., -0., -1., -2., 0., 3., 5., 7., 1., 2., 0.];
        let matrix = ArrayView2::from_shape((3, 4).f(), &values).unwrap();
        let mut plan = SignedColumns::discover(matrix).unwrap();
        let mut actual = [0.; 3];
        let vector = [2., -3., 4., 5.];
        plan.apply(matrix, &vector, &mut actual);
        for i in 0..3 {
            let expected: f64 = (0..4).map(|j| values[j * 3 + i] * vector[j]).sum();
            assert_eq!(actual[i], expected);
        }
        let mut almost = values;
        almost[10] += f64::EPSILON * 2.;
        let matrix = ArrayView2::from_shape((3, 4).f(), &almost).unwrap();
        let plan = SignedColumns::discover(matrix).unwrap();
        assert_eq!(plan.representatives.len(), 3);
    }

    #[test]
    fn zero_samples_keep_distinct_columns_and_find_signed_duplicates() {
        let mut matrix = ndarray::Array2::zeros((64, 10).f());
        for j in 0..8 {
            matrix[(j + 1, j)] = (j + 1) as f64;
        }
        for i in 0..64 {
            matrix[(i, 8)] = -matrix[(i, 2)];
            matrix[(i, 9)] = if matrix[(i, 4)] == 0. {
                -0.
            } else {
                matrix[(i, 4)]
            };
        }
        // All eight discovery samples are zero, including for the two
        // duplicates whose first nonzero lies between sampled rows.
        for j in 0..10 {
            assert!((0..8).all(|k| matrix[(k * 63 / 7, j)] == 0.));
        }
        let mut plan = SignedColumns::discover(matrix.view()).unwrap();
        assert_eq!(plan.representatives.len(), 8);
        let vector: Vec<_> = (1..=10).map(f64::from).collect();
        let mut actual = vec![0.; 64];
        plan.apply(matrix.view(), &vector, &mut actual);
        for i in 0..64 {
            let expected: f64 = (0..10).map(|j| matrix[(i, j)] * vector[j]).sum();
            assert_eq!(actual[i], expected);
        }
    }
}
