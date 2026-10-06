//! Exact contiguous preparation with runtime SIMD and portable NumPy fallback.
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
fn preparation_simd_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    { std::is_x86_feature_detected!("avx512f") }
    #[cfg(not(target_arch = "x86_64"))]
    { false }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn prepare_avx512(input: &[f64], inverse: &[f64], scales: &[f64],
                         multiplier: f64, n: usize, output: *mut f64) {
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
    let crossproduct = crossproduct.try_readonly().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let inverse_scales = inverse_scales.try_readonly().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let scales = scales.try_readonly().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let a = crossproduct.as_array();
    let inverse = inverse_scales.as_array();
    let scales = scales.as_array();
    let n = a.nrows();
    if a.ncols() != n || inverse.len() != n || scales.len() != n {
        return Err(PyValueError::new_err("Crossproduct dimensions must match scale lengths"));
    }
    if !preparation_simd_supported() {
        return Ok(None);
    }
    let (Some(input), Some(inverse), Some(scales)) =
        (a.as_slice(), inverse.as_slice(), scales.as_slice()) else {
            return Ok(None);
        };
    let _size = n.checked_mul(n)
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

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(preparation_simd_supported, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_crossproduct, module)?)?;
    Ok(())
}
