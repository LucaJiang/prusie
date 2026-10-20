//! Dense matrix products and exact diagonal scaling.
use ndarray::{ArrayView2, ShapeBuilder};
/// Accumulate four nonzero columns at a time, keeping each output element in a
/// register. Columns are added in ascending order using f64 arithmetic;
/// only output loads and stores are reduced.
#[inline(always)]
fn multiply_columns_grouped(values: &[f64], vector: &[f64], output: &mut [f64]) {
    output.fill(0.0);
    let rows = output.len();
    let mut columns = [0usize; 4];
    let mut weights = [0.0; 4];
    let mut count = 0;
    for (j, &weight) in vector.iter().enumerate() {
        if weight == 0.0 {
            continue;
        }
        columns[count] = j;
        weights[count] = weight;
        count += 1;
        if count == 4 {
            let a = &values[columns[0] * rows..(columns[0] + 1) * rows];
            let b = &values[columns[1] * rows..(columns[1] + 1) * rows];
            let c = &values[columns[2] * rows..(columns[2] + 1) * rows];
            let d = &values[columns[3] * rows..(columns[3] + 1) * rows];
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
        let column = &values[columns[k] * rows..(columns[k] + 1) * rows];
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

pub(crate) fn multiply(matrix: ArrayView2<'_, f64>, vector: &[f64], output: &mut [f64]) {
    if output.is_empty() {
        return;
    }
    if matrix.t().is_standard_layout() {
        let values = matrix.as_slice_memory_order().expect("contiguous columns");
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: feature detection above guards this target-feature call;
            // all reads/writes inside use bounds-checked, disjoint Rust slices.
            unsafe {
                multiply_columns_avx2(values, vector, output);
            }
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

// Checking |x| ≤ 2 for every entry also establishes finiteness, because NaN and
// infinities fail the ordered comparison. Combining eight checks with `&`
// permits a SIMD evaluation.
#[inline(never)]
// The explicit eight-element groups and the separately checked remainder are intended.
#[allow(clippy::chunks_exact_to_as_chunks)]
pub(crate) fn operator_entries_bounded(values: &[f64]) -> bool {
    let mut chunks = values.chunks_exact(8);
    for c in &mut chunks {
        if !((c[0].abs() <= 2.)
            & (c[1].abs() <= 2.)
            & (c[2].abs() <= 2.)
            & (c[3].abs() <= 2.)
            & (c[4].abs() <= 2.)
            & (c[5].abs() <= 2.)
            & (c[6].abs() <= 2.)
            & (c[7].abs() <= 2.))
        {
            return false;
        }
    }
    chunks.remainder().iter().all(|x| x.abs() <= 2.)
}

/// Implicit working matrix Q = a D⁻¹ M D⁻¹, applied as a D⁻¹ (M (D⁻¹ v)). The
/// factorization is exact in real arithmetic; its different floating-point
/// association is intentional and tested against the materialized matrix.
pub struct MatrixScaling<'a> {
    pub global: f64,
    pub inverse: &'a [f64],
    pub scales: &'a [f64],
}

pub(crate) fn materialize_scaled(
    matrix: ArrayView2<'_, f64>,
    scaling: &MatrixScaling<'_>,
) -> Vec<f64> {
    let p = matrix.nrows();
    let mut values = Vec::with_capacity(p * p);
    for j in 0..p {
        for i in 0..p {
            values
                .push(((matrix[(i, j)] * scaling.global) * scaling.inverse[j]) / scaling.scales[i]);
        }
    }
    values
}

pub(crate) fn scaled_multiply(
    matrix: ArrayView2<'_, f64>,
    vector: &[f64],
    output: &mut [f64],
    scaling: &MatrixScaling<'_>,
    scratch: &mut [f64],
) -> bool {
    for ((x, &v), &factor) in scratch.iter_mut().zip(vector).zip(scaling.inverse) {
        *x = v * factor;
    }
    if scratch.iter().all(|x| x.is_finite()) {
        multiply(matrix, scratch, output);
        for (x, &scale) in output.iter_mut().zip(scaling.scales) {
            *x = (*x * scaling.global) / scale;
        }
        if output.iter().all(|x| x.is_finite()) {
            return false;
        }
    }
    // If the factored product overflows in an intermediate step, multiply by
    // the materialized matrix instead, whose entry-wise preparation avoids it.
    let values = materialize_scaled(matrix, scaling);
    let p = matrix.nrows();
    let prepared = ArrayView2::from_shape((p, p).f(), &values).expect("prepared dimensions");
    multiply(prepared, vector, output);
    true
}

/// Matrix-free positive-semidefinite Gram operator used by IBSS.
pub(crate) trait GramOperator {
    fn diagonal(&self) -> &[f64];
    fn apply(&mut self, vector: &[f64], output: &mut [f64]);
}

pub(crate) struct DenseGram<'a> {
    pub matrix: ArrayView2<'a, f64>,
    pub scaling: Option<&'a MatrixScaling<'a>>,
    pub diagonal: Vec<f64>,
    pub scratch: Vec<f64>,
}

impl GramOperator for DenseGram<'_> {
    fn diagonal(&self) -> &[f64] {
        &self.diagonal
    }
    fn apply(&mut self, vector: &[f64], output: &mut [f64]) {
        if let Some(scaling) = self.scaling {
            scaled_multiply(self.matrix, vector, output, scaling, &mut self.scratch);
        } else {
            multiply(self.matrix, vector, output);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn contiguous_matrix_product_matches_independent_integer_arithmetic() {
        // Exactly representable integer arithmetic supplies an independent answer.
        let values: Vec<f64> = (0..32).map(|j| (j as i64 % 7 - 3) as f64).collect();
        let vector = [2., -1., 3., 0., -2., 1., 4., -3.];
        let matrix = ArrayView2::from_shape((4, 8), &values).unwrap();
        let mut output = [0.; 4];
        multiply(matrix, &vector, &mut output);
        for i in 0..4 {
            let expected: i64 = (0..8)
                .map(|j| values[i * 8 + j] as i64 * vector[j] as i64)
                .sum();
            assert!((output[i] - expected as f64).abs() <= 1e-12);
        }
    }
    #[test]
    fn column_major_dispatch_tail_and_zero_columns() {
        let values = [
            1., 2., 3., 4., 5., 7., 8., 9., 10., 11., -1., -2., -3., -4., -5.,
        ];
        let matrix = ArrayView2::from_shape((5, 3).f(), &values).unwrap();
        let mut output = [99.; 5];
        multiply(matrix, &[2., 0., -1.], &mut output);
        assert_eq!(output, [3., 6., 9., 12., 15.]);
        let mut portable = [99.; 5];
        multiply_columns(&values, &[2., 0., -1.], &mut portable);
        assert_eq!(output, portable);
        multiply(matrix, &[0., 0., 0.], &mut output);
        assert_eq!(output, [0.; 5]);
    }

    #[test]
    fn grouped_columns_cross_group_and_tail() {
        // Seven columns, one exactly zero coefficient and an odd row count.
        let values: Vec<f64> = (0..35).map(|i| (i as i64 % 11 - 5) as f64).collect();
        let vector = [2., 0., -3., 4., 1., -2., 5.];
        let mut output = [0.; 5];
        let matrix = ArrayView2::from_shape((5, 7).f(), &values).unwrap();
        multiply(matrix, &vector, &mut output);
        for i in 0..5 {
            let expected: i64 = (0..7)
                .map(|j| values[j * 5 + i] as i64 * vector[j] as i64)
                .sum();
            assert_eq!(output[i], expected as f64);
        }
    }

    #[test]
    fn operator_entry_certificate_checks_every_chunk_tail_and_nonfinite() {
        let valid = [
            -2., -0., 0., 2., 1., -1., 0.5, -0.5, 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        ];
        assert!(operator_entries_bounded(&valid));
        for bad in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::from_bits(2.0_f64.to_bits() + 1),
        ] {
            for i in 0..valid.len() {
                let mut values = valid;
                values[i] = bad;
                assert!(!operator_entries_bounded(&values));
            }
        }
    }
}
