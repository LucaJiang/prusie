//! Independently built dense-product candidates. Each public-call artifact
//! enables one feature, so runtime selection cannot contaminate gate timings.
use ndarray::ArrayView2;

#[cfg(all(feature = "kernel-openblas", not(feature = "kernel-faer")))]
use openblas_src as _;
#[cfg(all(feature = "kernel-openblas", not(feature = "kernel-faer")))]
unsafe extern "C" {
    fn cblas_dgemv(
        order: i32,
        transpose: i32,
        m: i32,
        n: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        x: *const f64,
        incx: i32,
        beta: f64,
        y: *mut f64,
        incy: i32,
    );
}

pub(crate) fn candidate_multiply(
    matrix: ArrayView2<'_, f64>,
    vector: &[f64],
    output: &mut [f64],
) -> bool {
    if vector.len() != matrix.ncols() || output.len() != matrix.nrows() {
        return false;
    }
    #[cfg(feature = "kernel-faer")]
    {
        let Some(values) = matrix.as_slice_memory_order() else {
            return false;
        };
        let view = if matrix.t().is_standard_layout() {
            faer::MatRef::from_column_major_slice(values, matrix.nrows(), matrix.ncols())
        } else if matrix.is_standard_layout() {
            faer::MatRef::from_row_major_slice(values, matrix.nrows(), matrix.ncols())
        } else {
            return false;
        };
        let x = faer::ColRef::from_slice(vector).as_mat();
        let y = faer::ColMut::from_slice_mut(output).as_mat_mut();
        faer::linalg::matmul::matmul(y, faer::Accum::Replace, view, x, 1., faer::Par::Seq);
        true
    }
    #[cfg(all(feature = "kernel-openblas", not(feature = "kernel-faer")))]
    {
        let Ok(rows) = i32::try_from(matrix.nrows()) else {
            return false;
        };
        let Ok(columns) = i32::try_from(matrix.ncols()) else {
            return false;
        };
        let Some(values) = matrix.as_slice_memory_order() else {
            return false;
        };
        let (layout, leading) = if matrix.t().is_standard_layout() {
            (102, rows)
        } else if matrix.is_standard_layout() {
            (101, columns)
        } else {
            return false;
        };
        // SAFETY: checked contiguous matrix, matching slice dimensions and
        // disjoint borrowed input/output slices satisfy the CBLAS ABI. LP64
        // dimensions are checked above; dgemv needs no matrix symmetry claim.
        unsafe {
            cblas_dgemv(
                layout,
                111,
                rows,
                columns,
                1.,
                values.as_ptr(),
                leading,
                vector.as_ptr(),
                1,
                0.,
                output.as_mut_ptr(),
                1,
            );
        }
        true
    }
    #[cfg(all(
        feature = "kernel-ndarray",
        not(any(feature = "kernel-faer", feature = "kernel-openblas"))
    ))]
    {
        ndarray::linalg::general_mat_vec_mul(
            1.,
            &matrix,
            &ndarray::ArrayView1::from(vector),
            0.,
            &mut ndarray::ArrayViewMut1::from(output),
        );
        true
    }
    #[cfg(not(any(
        feature = "kernel-faer",
        feature = "kernel-openblas",
        feature = "kernel-ndarray"
    )))]
    {
        let _ = (matrix, vector, output);
        false
    }
}

#[cfg(all(
    test,
    any(
        feature = "kernel-faer",
        feature = "kernel-openblas",
        feature = "kernel-ndarray"
    )
))]
mod tests {
    use super::*;
    use ndarray::{Array2, ShapeBuilder};

    #[test]
    fn rectangular_products_match_both_storage_orders() {
        for (rows, columns) in [(3, 5), (13, 7), (32, 19)] {
            let values = Array2::from_shape_fn((rows, columns), |(i, j)| {
                ((i * 7 + j * 13) % 23) as f64 / 11. - 1.
            });
            let column_major = Array2::from_shape_fn((rows, columns).f(), |index| values[index]);
            let vector: Vec<_> = (0..columns).map(|j| (j % 7) as f64 / 5. - 0.6).collect();
            let expected: Vec<f64> = (0..rows)
                .map(|i| (0..columns).map(|j| values[(i, j)] * vector[j]).sum())
                .collect();
            for matrix in [values.view(), column_major.view()] {
                let mut actual = vec![f64::NAN; rows];
                assert!(candidate_multiply(matrix, &vector, &mut actual));
                for (actual, expected) in actual.iter().zip(&expected) {
                    assert!((actual - expected).abs() <= 1e-13);
                }
                assert!(!candidate_multiply(
                    matrix,
                    &vector[..columns - 1],
                    &mut actual
                ));
            }
        }
    }
}
