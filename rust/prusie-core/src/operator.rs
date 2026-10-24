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

// SAFETY: callers must establish AVX2 support before entering this function.
// Slice accesses remain bounds checked and inputs do not alias mutable output.
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

/// The existing ordered four-column arithmetic on one disjoint output range.
#[inline(always)]
fn multiply_columns_range(
    values: &[f64],
    rows: usize,
    vector: &[f64],
    start: usize,
    output: &mut [f64],
) {
    output.fill(0.);
    let end = start + output.len();
    let mut columns = [0usize; 4];
    let mut weights = [0.; 4];
    let mut count = 0;
    for (j, &weight) in vector.iter().enumerate() {
        if weight == 0. {
            continue;
        }
        columns[count] = j;
        weights[count] = weight;
        count += 1;
        if count == 4 {
            let a = &values[columns[0] * rows + start..columns[0] * rows + end];
            let b = &values[columns[1] * rows + start..columns[1] * rows + end];
            let c = &values[columns[2] * rows + start..columns[2] * rows + end];
            let d = &values[columns[3] * rows + start..columns[3] * rows + end];
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
        let column = &values[columns[k] * rows + start..columns[k] * rows + end];
        for (result, &value) in output.iter_mut().zip(column) {
            *result += value * weights[k];
        }
    }
}

// SAFETY: callers establish AVX2 before using this otherwise bounds-checked
// kernel on an immutable input slice and a disjoint output chunk.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn multiply_columns_range_avx2(
    values: &[f64],
    rows: usize,
    vector: &[f64],
    start: usize,
    output: &mut [f64],
) {
    multiply_columns_range(values, rows, vector, start, output);
}

pub(crate) fn multiply_with_pool(
    matrix: ArrayView2<'_, f64>,
    vector: &[f64],
    output: &mut [f64],
    pool: Option<&rayon::ThreadPool>,
) {
    let Some(pool) = pool.filter(|_| crate::parallel::applicable(matrix.nrows(), matrix.ncols()))
    else {
        multiply(matrix, vector, output);
        return;
    };
    if matrix.t().is_standard_layout() {
        let values = matrix.as_slice_memory_order().expect("contiguous columns");
        let rows = matrix.nrows();
        crate::parallel::rows(pool, output, |start, chunk| {
            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: runtime detection guards this target-feature call;
                // static chunking supplies nonoverlapping writable output.
                unsafe {
                    multiply_columns_range_avx2(values, rows, vector, start, chunk);
                }
                return;
            }
            multiply_columns_range(values, rows, vector, start, chunk);
        });
    } else {
        crate::parallel::rows(pool, output, |start, chunk| {
            chunk.fill(0.);
            for (j, &coefficient) in vector.iter().enumerate() {
                if coefficient != 0. {
                    for (i, value) in chunk.iter_mut().enumerate() {
                        *value += matrix[(start + i, j)] * coefficient;
                    }
                }
            }
        });
    }
}

// The former raw-correlation certificate admitted |R| ≤ 2 and a ≤ 10⁹.
// Parity preparation may already move a into the supplied matrix. Admit that
// representation too: |M| ≤ 2 × 10⁹. With the factor bounds below, ordered
// prepared-entry intermediates are bounded by 2 × 10¹⁸, 4 × 10¹⁸ and 8 × 10¹⁸.
// Ordered comparisons also establish finiteness; `&` permits SIMD evaluation.
#[inline(never)]
// The explicit eight-element groups and the separately checked remainder are intended.
#[allow(clippy::chunks_exact_to_as_chunks)]
pub(crate) fn operator_entries_bounded(values: &[f64]) -> bool {
    let mut chunks = values.chunks_exact(8);
    for c in &mut chunks {
        if !((c[0].abs() <= 2e9)
            & (c[1].abs() <= 2e9)
            & (c[2].abs() <= 2e9)
            & (c[3].abs() <= 2e9)
            & (c[4].abs() <= 2e9)
            & (c[5].abs() <= 2e9)
            & (c[6].abs() <= 2e9)
            & (c[7].abs() <= 2e9))
        {
            return false;
        }
    }
    chunks.remainder().iter().all(|x| x.abs() <= 2e9)
}

/// Implicit working matrix Q = a D⁻¹ M D⁻¹, applied as a D⁻¹ (M (D⁻¹ v)). The
/// factorization is exact in real arithmetic; its different floating-point
/// association is intentional and tested against the materialized matrix.
pub struct MatrixScaling<'a> {
    pub global: f64,
    pub inverse: &'a [f64],
    pub scales: &'a [f64],
}

/// A sufficient dispatch certificate, not an input restriction. It proves
/// finite ordered prepared entries, not arbitrary dot-product intermediates.
#[allow(clippy::manual_range_contains)]
pub(crate) fn operator_scaling_bounded(
    values: &[f64],
    p: usize,
    scaling: Option<&MatrixScaling<'_>>,
) -> bool {
    scaling.is_some_and(|scale| {
        scale.global >= 1.
            && scale.global <= 1e9
            && scale.scales.len() == p
            && scale.inverse.len() == p
            && scale.scales.iter().all(|&x| x >= 0.5 && x <= 2.)
            && scale.inverse.iter().all(|&x| x >= 0.5 && x <= 2.)
            && operator_entries_bounded(values)
    })
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

#[cfg(test)]
pub(crate) fn scaled_multiply(
    matrix: ArrayView2<'_, f64>,
    vector: &[f64],
    output: &mut [f64],
    scaling: &MatrixScaling<'_>,
    scratch: &mut [f64],
) -> bool {
    scaled_multiply_with_pool(matrix, vector, output, scaling, scratch, None)
}

pub(crate) fn scaled_multiply_with_pool(
    matrix: ArrayView2<'_, f64>,
    vector: &[f64],
    output: &mut [f64],
    scaling: &MatrixScaling<'_>,
    scratch: &mut [f64],
    pool: Option<&rayon::ThreadPool>,
) -> bool {
    for ((x, &v), &factor) in scratch.iter_mut().zip(vector).zip(scaling.inverse) {
        *x = v * factor;
    }
    if scratch.iter().all(|x| x.is_finite()) {
        multiply_with_pool(matrix, scratch, output, pool);
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
    multiply_with_pool(prepared, vector, output, pool);
    true
}

/// Matrix-free positive-semidefinite Gram operator used by IBSS.
pub(crate) trait GramOperator {
    fn diagonal(&self) -> &[f64];
    fn begin_sweep(&mut self, _iteration: usize) {}
    fn exact_diagnostics(&self) -> Option<(Option<usize>, Option<usize>, Option<usize>, usize)> {
        None
    }
    fn apply(&mut self, vector: &[f64], output: &mut [f64]);
}

pub(crate) struct DenseGram<'a> {
    pub pool: Option<&'a rayon::ThreadPool>,
    pub matrix: ArrayView2<'a, f64>,
    pub scaling: Option<&'a MatrixScaling<'a>>,
    pub diagonal: Vec<f64>,
    pub scratch: Vec<f64>,
    pub exact: Option<crate::exact_columns::ExactColumns>,
    pub discovery_checked: bool,
    pub activation: crate::activation::ExactActivation,
}

impl GramOperator for DenseGram<'_> {
    fn diagonal(&self) -> &[f64] {
        &self.diagonal
    }
    fn begin_sweep(&mut self, iteration: usize) {
        self.activation.sweep = iteration + 1;
    }
    fn exact_diagnostics(&self) -> Option<(Option<usize>, Option<usize>, Option<usize>, usize)> {
        Some((
            self.activation.attempt_sweep,
            self.exact.as_ref().map(|x| x.groups()),
            self.activation.estimate,
            self.activation.dense_products,
        ))
    }

    fn apply(&mut self, vector: &[f64], output: &mut [f64]) {
        let p = vector.len();
        // Exact zero counts describe work already skipped by the sparse kernel.
        let nonzero = vector.iter().filter(|&&x| x != 0.).count();
        let dense = nonzero >= p.div_ceil(8);
        if dense && !self.discovery_checked {
            if !self.activation.screened {
                self.activation.screened = true;
                self.activation.estimate =
                    crate::exact_columns::ExactColumns::sampled_groups(self.matrix);
                if self.activation.estimate.is_none() {
                    // A failed initial screen is terminal; never resample.
                    self.discovery_checked = true;
                }
            }
            if !self.discovery_checked && self.activation.ready(p) {
                self.discovery_checked = true;
                self.activation.attempt_sweep = Some(self.activation.sweep);
                self.exact = crate::exact_columns::ExactColumns::discover(self.matrix);
            }
        }
        if dense {
            if let Some(exact) = &mut self.exact {
                if let Some(scale) = self.scaling {
                    for ((value, &coefficient), &factor) in
                        self.scratch.iter_mut().zip(vector).zip(scale.inverse)
                    {
                        *value = coefficient * factor;
                    }
                    if self.scratch.iter().all(|x| x.is_finite())
                        && exact.apply(self.matrix, &self.scratch, output, self.pool)
                    {
                        for (value, &denominator) in output.iter_mut().zip(scale.scales) {
                            *value = (*value * scale.global) / denominator;
                        }
                        if output.iter().all(|x| x.is_finite()) {
                            return;
                        }
                    }
                } else if exact.apply(self.matrix, vector, output, self.pool) {
                    return;
                }
            }
        }
        if let Some(scaling) = self.scaling {
            scaled_multiply_with_pool(
                self.matrix,
                vector,
                output,
                scaling,
                &mut self.scratch,
                self.pool,
            );
        } else {
            multiply_with_pool(self.matrix, vector, output, self.pool);
        }
        if dense && !self.discovery_checked {
            // Rent is paid only after a real dense product. The threshold is
            // checked before the next such product, without predicting stopping.
            self.activation.accrue(p, nonzero);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_factored_overflow_retains_materialized_fallback_bits() {
        let p = 1025;
        let values = vec![0.25; p * p];
        let matrix = ArrayView2::from_shape((p, p).f(), &values).unwrap();
        let factors = vec![2.; p];
        let scaling = MatrixScaling {
            global: 1.,
            inverse: &factors,
            scales: &factors,
        };
        let mut vector = vec![-0.; p];
        vector[0] = f64::MAX;
        vector[1] = -f64::MAX;
        vector[2] = f64::from_bits(1);
        let mut expected = vec![0.; p];
        let mut scratch = vec![0.; p];
        scaled_multiply(matrix, &vector, &mut expected, &scaling, &mut scratch);
        assert!(!scratch[0].is_finite());
        assert!(expected.iter().all(|x| x.is_finite()));
        let expected: Vec<_> = expected.iter().map(|x| x.to_bits()).collect();
        for threads in [2, 4] {
            let pool = crate::parallel::pool(threads, p).unwrap().unwrap();
            let mut actual = vec![f64::NAN; p];
            scaled_multiply_with_pool(
                matrix,
                &vector,
                &mut actual,
                &scaling,
                &mut scratch,
                Some(&pool),
            );
            assert_eq!(
                actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                expected
            );
        }
    }

    #[test]
    fn static_parallel_row_chunks_match_serial_bits_in_both_layouts() {
        let p = 1025;
        let values: Vec<f64> = (0..p * p)
            .map(|k| ((k * 13 + k / p * 7) % 103) as f64 / 128. - 0.25)
            .collect();
        let vector: Vec<f64> = (0..p)
            .map(|j| {
                if j % 7 == 0 {
                    -0.
                } else {
                    (j % 31) as f64 / 32. - 0.5
                }
            })
            .collect();
        for column_major in [false, true] {
            let matrix = if column_major {
                ArrayView2::from_shape((p, p).f(), &values).unwrap()
            } else {
                ArrayView2::from_shape((p, p), &values).unwrap()
            };
            let mut expected = vec![0.; p];
            multiply(matrix, &vector, &mut expected);
            let expected: Vec<_> = expected.iter().map(|x| x.to_bits()).collect();
            for threads in [2, 4] {
                let pool = crate::parallel::pool(threads, p).unwrap().unwrap();
                assert_eq!(pool.current_num_threads(), threads);
                for _ in 0..2 {
                    let mut actual = vec![f64::NAN; p];
                    multiply_with_pool(matrix, &vector, &mut actual, Some(&pool));
                    assert_eq!(
                        actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        expected
                    );
                }
            }
        }
    }
    #[test]
    fn failed_screen_and_failed_proof_never_retry() {
        for p in [16usize, 64] {
            let mut values = vec![0.; p * p];
            for i in 0..p {
                values[i * p + i] = 1.;
            }
            let ones = vec![1.; p * p];
            let mut gram = DenseGram {
                pool: None,
                matrix: ArrayView2::from_shape((p, p).f(), &values).unwrap(),
                scaling: None,
                diagonal: vec![1.; p],
                scratch: vec![0.; p],
                exact: None,
                discovery_checked: false,
                activation: Default::default(),
            };
            gram.activation.rent = f64::MAX;
            let mut out = vec![0.; p];
            gram.apply(&vec![1.; p], &mut out);
            assert!(gram.discovery_checked && gram.exact.is_none());
            if p == 16 {
                assert!(gram.activation.estimate.is_none());
                assert!(gram.activation.attempt_sweep.is_none());
            } else {
                assert!(gram.activation.estimate.is_some());
                assert!(gram.activation.attempt_sweep.is_some());
            }
            gram.matrix = ArrayView2::from_shape((p, p).f(), &ones).unwrap();
            gram.apply(&vec![1.; p], &mut out);
            assert!(gram.exact.is_none());
            assert_eq!(out, vec![p as f64; p]);
        }
    }
    #[test]
    fn sparse_calls_do_not_sample_or_accrue_rent() {
        let values = [1.; 256];
        let mut gram = DenseGram {
            pool: None,
            matrix: ArrayView2::from_shape((16, 16).f(), &values).unwrap(),
            scaling: None,
            diagonal: vec![1.; 16],
            scratch: vec![0.; 16],
            exact: None,
            discovery_checked: false,
            activation: Default::default(),
        };
        let mut vector = [0.; 16];
        vector[3] = 2.;
        let mut out = [0.; 16];
        for i in 0..20 {
            gram.begin_sweep(i);
            gram.apply(&vector, &mut out);
        }
        assert!(!gram.activation.screened);
        assert_eq!(gram.activation.dense_products, 0);
        assert_eq!(gram.activation.rent, 0.);
        assert_eq!(out, [2.; 16]);
    }
    #[test]
    fn sparse_product_defers_discovery_until_a_later_dense_product() {
        let values = [1.; 16 * 16];
        let matrix = ArrayView2::from_shape((16, 16).f(), &values).unwrap();
        let mut gram = DenseGram {
            pool: None,
            matrix,
            scaling: None,
            diagonal: vec![1.; 16],
            scratch: vec![0.; 16],
            exact: None,
            discovery_checked: false,
            activation: Default::default(),
        };
        gram.begin_sweep(5);
        let mut vector = [0.; 16];
        vector[3] = 2.;
        let mut output = [0.; 16];
        gram.apply(&vector, &mut output);
        assert_eq!(output, [2.; 16]);
        assert!(!gram.discovery_checked);
        gram.activation.rent = f64::MAX;
        gram.apply(&[1.; 16], &mut output);
        assert_eq!(output, [16.; 16]);
        assert!(gram.exact.is_some());
    }

    #[test]
    fn overflowing_group_uses_a_finite_original_dense_result() {
        let values = [0.25; 4];
        let matrix = ArrayView2::from_shape((2, 2).f(), &values).unwrap();
        let mut gram = DenseGram {
            pool: None,
            matrix,
            scaling: None,
            diagonal: vec![0.25; 2],
            scratch: vec![0.; 2],
            exact: None,
            discovery_checked: false,
            activation: Default::default(),
        };
        gram.begin_sweep(5);
        let mut output = [0.; 2];
        gram.activation.rent = f64::MAX;
        gram.apply(&[f64::MAX, f64::MAX], &mut output);
        assert!(gram.exact.is_some());
        assert_eq!(output, [f64::MAX * 0.5; 2]);
    }

    #[test]
    fn lazy_exact_product_preserves_independent_scaled_matrix_arithmetic() {
        let x = [1., -1., 2., 1., -2., 0.];
        let values: Vec<f64> = x
            .iter()
            .flat_map(|&b| x.iter().map(move |&a| a * b))
            .collect();
        let matrix = ArrayView2::from_shape((6, 6).f(), &values).unwrap();
        let inverse = [1., 0.5, 2., 1., 0.5, 2.];
        let scales = [1., 2., 0.5, 1., 2., 0.5];
        let scaling = MatrixScaling {
            global: 8.,
            inverse: &inverse,
            scales: &scales,
        };
        let vector = [2., -4., 8., 1., -2., 4.];
        let expected: Vec<f64> = (0..6)
            .map(|i| {
                (0..6)
                    .map(|j| ((matrix[(i, j)] * 8.) * inverse[j]) / scales[i] * vector[j])
                    .sum()
            })
            .collect();
        let mut gram = DenseGram {
            pool: None,
            matrix,
            scaling: Some(&scaling),
            diagonal: vec![0.; 6],
            scratch: vec![0.; 6],
            exact: None,
            discovery_checked: false,
            activation: Default::default(),
        };
        let mut output = [0.; 6];
        gram.begin_sweep(4);
        gram.apply(&vector, &mut output);
        assert!(!gram.discovery_checked);
        assert_eq!(output.as_slice(), expected.as_slice());
        gram.begin_sweep(5);
        gram.activation.rent = f64::MAX;
        gram.apply(&vector, &mut output);
        assert!(gram.discovery_checked && gram.exact.is_some());
        assert_eq!(output.as_slice(), expected.as_slice());
    }

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
            -2e9,
            -0.,
            0.,
            2e9,
            799.,
            -486018.,
            0.5,
            -0.5,
            f64::from_bits(2e9_f64.to_bits() - 1),
            -f64::from_bits(2e9_f64.to_bits() - 1),
            f64::from_bits(2.0_f64.to_bits() + 1),
            f64::from_bits(1),
            -f64::from_bits(1),
            1.,
            1.,
            1.,
            1.,
        ];
        assert!(operator_entries_bounded(&valid));
        for bad in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::from_bits(2e9_f64.to_bits() + 1),
            -f64::from_bits(2e9_f64.to_bits() + 1),
        ] {
            for i in 0..valid.len() {
                let mut values = valid;
                values[i] = bad;
                assert!(!operator_entries_bounded(&values));
            }
        }
    }

    #[test]
    fn scaling_certificate_preserves_factor_boundaries_and_dispatch_misses() {
        let values = [486018.; 4];
        let mut factors = MatrixScaling {
            global: 1.,
            inverse: &[0.5, 2.],
            scales: &[2., 0.5],
        };
        assert!(operator_scaling_bounded(&values, 2, Some(&factors)));
        factors.global = 1e9;
        assert!(operator_scaling_bounded(&values, 2, Some(&factors)));
        for global in [
            0.,
            -1.,
            f64::NAN,
            f64::INFINITY,
            f64::from_bits(1.0_f64.to_bits() - 1),
            f64::from_bits(1e9_f64.to_bits() + 1),
        ] {
            factors.global = global;
            assert!(!operator_scaling_bounded(&values, 2, Some(&factors)));
        }
        factors.global = 1.;
        for bad in [
            0.,
            -1.,
            f64::NAN,
            f64::INFINITY,
            f64::from_bits(0.5_f64.to_bits() - 1),
            f64::from_bits(2.0_f64.to_bits() + 1),
        ] {
            let changed = [bad, 1.];
            assert!(!operator_scaling_bounded(
                &values,
                2,
                Some(&MatrixScaling {
                    inverse: &changed,
                    ..factors
                })
            ));
            assert!(!operator_scaling_bounded(
                &values,
                2,
                Some(&MatrixScaling {
                    scales: &changed,
                    ..factors
                })
            ));
        }
        assert!(!operator_scaling_bounded(&values, 3, Some(&factors)));
        assert!(!operator_scaling_bounded(&[2e9 + 1.; 4], 2, Some(&factors)));
        assert!(!operator_scaling_bounded(&values, 2, None));
    }

    #[test]
    fn newly_bounded_product_matches_an_independent_power_of_two_oracle() {
        let values = [1073741824., 268435456., 536870912., 1073741824.];
        let matrix = ArrayView2::from_shape((2, 2).f(), &values).unwrap();
        let scaling = MatrixScaling {
            global: 8.,
            inverse: &[0.5, 2.],
            scales: &[2., 0.5],
        };
        assert!(operator_scaling_bounded(&values, 2, Some(&scaling)));
        let vector = [2., -3.];
        let mut output = [0.; 2];
        assert!(!scaled_multiply(
            matrix,
            &vector,
            &mut output,
            &scaling,
            &mut [0.; 2]
        ));
        for i in 0..2 {
            let expected: f64 = (0..2)
                .map(|j| {
                    (((values[j * 2 + i] * 8.) * scaling.inverse[j]) / scaling.scales[i])
                        * vector[j]
                })
                .sum();
            assert_eq!(output[i], expected);
        }
    }

    #[test]
    fn newly_bounded_product_retains_finite_materialized_fallback() {
        let values = [1073741824.; 4]; // 2³⁰
        let matrix = ArrayView2::from_shape((2, 2).f(), &values).unwrap();
        let scaling = MatrixScaling {
            global: 1.,
            inverse: &[0.5; 2],
            scales: &[2.; 2],
        };
        assert!(operator_scaling_bounded(&values, 2, Some(&scaling)));
        let vector = [f64::MAX * 2.0_f64.powi(-29); 2];
        let mut output = [0.; 2];
        assert!(scaled_multiply(
            matrix,
            &vector,
            &mut output,
            &scaling,
            &mut [0.; 2]
        ));
        assert_eq!(output, [f64::MAX; 2]);
    }
}
