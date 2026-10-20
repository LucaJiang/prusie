//! Exact signed-column compression. Sample hashes only propose matches; every
//! relation is proved by comparing all rows, without LD rounding or thresholds.
use ndarray::ArrayView2;
use std::collections::HashMap;

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
        let mut buckets: HashMap<[u64; 8], Vec<usize>> = HashMap::new();
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
            let candidates = buckets.entry(key).or_default();
            let mut matched = None;
            for &group in candidates.iter() {
                let rep = representatives[group];
                let reference = &values[rep * rows..(rep + 1) * rows];
                if column.iter().zip(reference).all(|(a, b)| a == b) {
                    matched = Some((group, 1.));
                    break;
                }
                if column.iter().zip(reference).all(|(a, b)| *a == -*b) {
                    matched = Some((group, -1.));
                    break;
                }
            }
            let (group, sign) = matched.unwrap_or_else(|| {
                let group = representatives.len();
                representatives.push(j);
                candidates.push(group);
                (group, 1.)
            });
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
}
