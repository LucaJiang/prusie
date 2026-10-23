//! Preparation of the working crossproduct with AVX-512 when available;
//! otherwise Python uses the equivalent NumPy calculation.
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
fn preparation_simd_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

// SAFETY: callers must detect AVX512F, supply input length n*n and scale
// lengths n, and provide nonaliasing writable output for n*n f64 entries.
// The caller holds the NumPy borrows and GIL until this function returns.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn prepare_avx512(
    input: &[f64],
    inverse: &[f64],
    scales: &[f64],
    multiplier: f64,
    n: usize,
    output: *mut f64,
) {
    use std::arch::x86_64::*;
    let mult = _mm512_set1_pd(multiplier);
    for j in 0..n {
        let inv = _mm512_set1_pd(inverse[j]);
        let mut i = 0;
        while i + 8 <= n {
            let mut x = _mm512_loadu_pd(input.as_ptr().add(j * n + i));
            if multiplier != 1.0 {
                x = _mm512_mul_pd(x, mult);
            }
            x = _mm512_mul_pd(x, inv);
            x = _mm512_div_pd(x, _mm512_loadu_pd(scales.as_ptr().add(i)));
            _mm512_storeu_pd(output.add(j * n + i), x);
            i += 8;
        }
        while i < n {
            let mut x = input[j * n + i];
            if multiplier != 1.0 {
                x *= multiplier;
            }
            output.add(j * n + i).write(x * inverse[j] / scales[i]);
            i += 1;
        }
    }
}

#[pyfunction]
#[pyo3(signature = (crossproduct, inverse_scales, scales, multiplier=1.0))]
fn prepare_crossproduct<'py>(
    py: Python<'py>,
    crossproduct: Bound<'py, PyArray2<f64>>,
    inverse_scales: Bound<'py, PyArray1<f64>>,
    scales: Bound<'py, PyArray1<f64>>,
    multiplier: f64,
) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
    crate::array_layout::ensure_layout(&crossproduct)?;
    crate::array_layout::ensure_layout(&inverse_scales)?;
    crate::array_layout::ensure_layout(&scales)?;
    let crossproduct = crossproduct
        .try_readonly()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let inverse_scales = inverse_scales
        .try_readonly()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let scales = scales
        .try_readonly()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let a = crossproduct.as_array();
    let inverse = inverse_scales.as_array();
    let scales = scales.as_array();
    let n = a.nrows();
    if a.ncols() != n || inverse.len() != n || scales.len() != n {
        return Err(PyValueError::new_err(
            "Crossproduct dimensions must match scale lengths",
        ));
    }
    if !preparation_simd_supported() {
        return Ok(None);
    }
    let (Some(input), Some(inverse), Some(scales)) =
        (a.as_slice(), inverse.as_slice(), scales.as_slice())
    else {
        return Ok(None);
    };
    let _size = n
        .checked_mul(n)
        .ok_or_else(|| PyValueError::new_err("Crossproduct is too large"))?;
    #[cfg(target_arch = "x86_64")]
    {
        // Allocate through NumPy, matching np.empty's allocator and page policy.
        // SAFETY: shape multiplication was checked. The uninitialized array is
        // private and never read or exposed before every entry is written below.
        let output = unsafe { PyArray2::<f64>::new(py, [n, n], true) };
        // SAFETY: runtime AVX512F detection precedes the call. Contiguous input
        // slices have lengths n*n/n. The kernel writes each output entry once,
        // has no early return, and does not read uninitialized output. This
        // newly allocated output cannot alias the borrowed Python input. The
        // GIL remains held throughout; borrowed memory is never mutated.
        unsafe {
            prepare_avx512(input, inverse, scales, multiplier, n, output.data());
        }
        Ok(Some(output))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (input, inverse, scales, multiplier, _size);
        Ok(None)
    }
}

/// Required parity averaging in tiles, with one output and no working p² copies.
/// Separate ordered multiplies/add/divide preserve the Python preparation value.
#[pyfunction]
#[pyo3(signature = (matrix, scale=1.0, name=None))]
fn symmetrize_matrix<'py>(
    py: Python<'py>,
    matrix: Bound<'py, PyArray2<f64>>,
    scale: f64,
    name: Option<&str>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    crate::array_layout::ensure_layout(&matrix)?;
    let matrix = matrix
        .try_readonly()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let a = matrix.as_array();
    let n = a.nrows();
    if a.ncols() != n
        || n.checked_mul(n)
            .and_then(|v| v.checked_mul(8))
            .filter(|&v| v <= isize::MAX as usize)
            .is_none()
    {
        return Err(PyValueError::new_err(
            "Matrix must be square with addressable storage",
        ));
    }
    // SAFETY: dimension bytes were checked. Every output entry is written below
    // before this private allocation is exposed; it cannot alias the input.
    let output = unsafe { PyArray2::<f64>::new(py, [n, n], false) };
    // Public callers request the same validation that previously followed repair.
    // Keep the private unchecked utility for ordered-arithmetic diagnostics.
    let mut negative_diagonal = false;
    let contiguous = a.as_slice_memory_order();
    let row_major = a.strides() == [n as isize, 1];
    let column_major = a.strides() == [1, n as isize];
    for i0 in (0..n).step_by(32) {
        for j0 in (i0..n).step_by(32) {
            for i in i0..(i0 + 32).min(n) {
                for j in j0.max(i)..(j0 + 32).min(n) {
                    let (x, y) = match contiguous {
                        Some(values) if row_major => (values[i * n + j], values[j * n + i]),
                        Some(values) if column_major => (values[j * n + i], values[i * n + j]),
                        _ => (a[(i, j)], a[(j, i)]),
                    };
                    let left = if scale == 1. { x } else { x * scale };
                    let right = if scale == 1. { y } else { y * scale };
                    let value = (left + right) / 2.;
                    if let Some(name) = name {
                        if !value.is_finite() {
                            return Err(PyValueError::new_err(format!(
                                "{name} must contain only finite values"
                            )));
                        }
                        // Report a negative diagonal only after every entry has
                        // passed finiteness, matching validate_matrix priority.
                        negative_diagonal |= i == j && value < 0.;
                    }
                    // SAFETY: both indices are in range, output is C ordered,
                    // the GIL/input borrow stay live, and no entry is read.
                    unsafe {
                        output.data().add(i * n + j).write(value);
                        if i != j {
                            output.data().add(j * n + i).write(value);
                        }
                    }
                }
            }
        }
    }
    if negative_diagonal {
        return Err(PyValueError::new_err(format!(
            "{} diagonal must be nonnegative",
            name.unwrap_or("XtX")
        )));
    }
    Ok(output)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(preparation_simd_supported, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_crossproduct, module)?)?;
    module.add_function(wrap_pyfunction!(symmetrize_matrix, module)?)?;
    Ok(())
}
