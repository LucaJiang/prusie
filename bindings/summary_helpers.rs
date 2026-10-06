//! Allocation-bounded helpers for complete public matrix validation and CS purity.
//!
//! These helpers borrow NumPy arrays only while holding the GIL. They never
//! mutate input memory and accept C, F, and arbitrary strided layouts.
use crate::array_layout::ensure_layout;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (array, name, correlation=false, scale=1.0))]
fn validate_matrix(array: Bound<'_, PyArray2<f64>>, name: &str, correlation: bool, scale: f64) -> PyResult<()> {
    ensure_layout(&array)?;
    let array = array.try_readonly().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let a = array.as_array();
    let n = a.nrows();
    if a.ncols() != n {
        return Err(PyValueError::new_err(format!("{name} must be square")));
    }
    let mut negative_diagonal = false;
    let mut valid_correlation = true;
    for i in 0..n {
        let value = a[[i, i]] * scale;
        if !value.is_finite() {
            return Err(PyValueError::new_err(format!("{name} must contain only finite values")));
        }
        negative_diagonal |= value < 0.0;
        if correlation {
            valid_correlation &= (value - 1.0).abs() <= 1e-8;
        }
    }
    // Each off-diagonal pair is checked once, in both tolerance directions.
    // Small tiles keep both sides of a transposed comparison cache-resident.
    let mut symmetric = true;
    const BLOCK: usize = 32;
    for ii in (0..n).step_by(BLOCK) {
        for jj in (ii..n).step_by(BLOCK) {
            for i in ii..(ii + BLOCK).min(n) {
                for j in jj.max(i + 1)..(jj + BLOCK).min(n) {
                    let (x, y) = if scale == 1.0 {
                        (a[[i, j]], a[[j, i]])
                    } else {
                        (a[[i, j]] * scale, a[[j, i]] * scale)
                    };
                    // np.allclose's atol + rtol * abs(reference), evaluated
                    // in both directions since the full matrix was checked.
                    if x == y {
                        // Equality proves both entries share finiteness and
                        // absolute bounds, including the two signed zeros.
                        // Equal infinities still fail this finite check.
                        if !x.is_finite() {
                            return Err(PyValueError::new_err(format!("{name} must contain only finite values")));
                        }
                        if correlation {
                            valid_correlation &= x.abs() <= 1.0 + 1e-8;
                        }
                    } else {
                        if !x.is_finite() || !y.is_finite() {
                            return Err(PyValueError::new_err(format!("{name} must contain only finite values")));
                        }
                        let delta = (x - y).abs();
                        symmetric &= delta <= 1e-12 + 1e-12 * y.abs()
                            && delta <= 1e-12 + 1e-12 * x.abs();
                        if correlation {
                            valid_correlation &= x.abs() <= 1.0 + 1e-8 && y.abs() <= 1.0 + 1e-8;
                        }
                    }
                }
            }
        }
    }
    if !symmetric {
        return Err(PyValueError::new_err(format!(
            "{name} must be symmetric; no automatic symmetrization is performed")));
    }
    if negative_diagonal {
        return Err(PyValueError::new_err(format!("{name} diagonal must be nonnegative")));
    }
    if !valid_correlation {
        return Err(PyValueError::new_err(
            "R must be a signed correlation matrix with unit diagonal and values in [-1,1]"));
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (correlation, members, inverse_scales=None, min_abs_corr=None, covariance_multiplier=1.0))]
fn cs_pairwise_abs<'py>(
    py: Python<'py>,
    correlation: Bound<'py, PyArray2<f64>>,
    members: Bound<'py, PyArray1<i64>>,
    inverse_scales: Option<Bound<'py, PyArray1<f64>>>,
    min_abs_corr: Option<f64>,
    covariance_multiplier: f64,
) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
    ensure_layout(&correlation)?;
    ensure_layout(&members)?;
    if let Some(ref inverse) = inverse_scales { ensure_layout(inverse)?; }
    let correlation = correlation.try_readonly().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let members = members.try_readonly().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let inverse_scales = inverse_scales.as_ref().map(|s| s.try_readonly())
        .transpose().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let a = correlation.as_array();
    let indices = members.as_array();
    let n = indices.len();
    if a.nrows() != a.ncols()
        || indices.iter().any(|&i| i < 0 || i as usize >= a.nrows())
    {
        return Err(PyValueError::new_err("CS members must index a square matrix"));
    }
    let inv = inverse_scales.as_ref().map(|s| s.as_array());
    if inv.as_ref().is_some_and(|s| s.len() != n) {
        return Err(PyValueError::new_err("CS inverse scales must match members"));
    }
    let size = n.checked_mul(n.saturating_sub(1)).and_then(|x| x.checked_div(2))
        .ok_or_else(|| PyValueError::new_err("CS is too large"))?;
    if let Some(threshold) = min_abs_corr {
        // A failing pair proves the complete minimum is below threshold (or
        // NaN). Rejected sets return no purity values; later pairs cannot
        // reverse rejection. Retained sets still visit every pair here and
        // gather every pair below for the complete original summaries.
        // Public matrix validation is separate and always remains complete.
        for i in 0..n {
            let row = indices[i] as usize;
            for j in (i + 1)..n {
                let mut value = a[[row, indices[j] as usize]];
                if covariance_multiplier != 1.0 {
                    value *= covariance_multiplier;
                }
                if let Some(ref scales) = inv {
                    value = value * scales[i] * scales[j];
                }
                if !(value.abs() >= threshold) {
                    return Ok(None);
                }
            }
        }
    }
    let mut values = Vec::with_capacity(size);
    // Match np.triu_indices(n, 1) order and the original two multiplies.
    // Mean/median remain NumPy operations on exactly this ordered vector.
    for i in 0..n {
        let row = indices[i] as usize;
        for j in (i + 1)..n {
            let mut value = a[[row, indices[j] as usize]];
            if covariance_multiplier != 1.0 {
                value *= covariance_multiplier;
            }
            if let Some(ref scales) = inv {
                value = value * scales[i] * scales[j];
            }
            values.push(value.abs());
        }
    }
    Ok(Some(values.into_pyarray(py)))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(validate_matrix, module)?)?;
    module.add_function(wrap_pyfunction!(validate_rss_matrix, module)?)?;
    module.add_function(wrap_pyfunction!(cs_pairwise_abs, module)?)?;
    module.add_function(wrap_pyfunction!(complete_finite_scan, module)?)?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (array, reduction=true))]
fn complete_finite_scan(array: Bound<'_, PyArray1<f64>>, reduction: bool) -> PyResult<bool> {
    ensure_layout(&array)?;
    let array = array.try_readonly().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let values = array.as_slice()?;
    Ok(if reduction {
        crate::core::input_validation::all_finite(values)
    } else {
        !values.iter().any(|value| !value.is_finite())
    })
}

fn validate_rss_matrix_impl<const SAFE_SCALE: bool>(a: ndarray::ArrayView2<'_, f64>,
                                                    scale: f64) -> PyResult<()> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(PyValueError::new_err("R must be square"));
    }
    let mut raw_nonnegative = true;
    let mut raw_correlation = true;
    let mut raw_symmetric = true;
    let mut scaled_finite = true;
    let mut scaled_nonnegative = true;
    let mut scaled_symmetric = true;
    for i in 0..n {
        let raw = a[[i, i]];
        if !raw.is_finite() {
            return Err(PyValueError::new_err("R must contain only finite values"));
        }
        raw_nonnegative &= raw >= 0.0;
        raw_correlation &= (raw - 1.0).abs() <= 1e-8 && raw.abs() <= 1.0 + 1e-8;
        let scaled = raw * scale;
        scaled_finite &= scaled.is_finite();
        scaled_nonnegative &= scaled >= 0.0;
    }
    const BLOCK: usize = 32;
    for ii in (0..n).step_by(BLOCK) {
        for jj in (ii..n).step_by(BLOCK) {
            for i in ii..(ii + BLOCK).min(n) {
                for j in jj.max(i + 1)..(jj + BLOCK).min(n) {
                    let x = a[[i, j]];
                    let y = a[[j, i]];
                    if x == y && x.abs() <= 1.0 + 1e-8 {
                        // Equality plus the range bound proves both raw
                        // entries finite and symmetric. For SAFE_SCALE,
                        // |scale| <= MAX/2 and |x| < 2 also prove both scaled
                        // entries finite. Scaled equality then proves BOTH
                        // symmetry orientations. No input entry is omitted.
                        if !SAFE_SCALE {
                            scaled_finite &= (x * scale).is_finite();
                        }
                        continue;
                    }
                    // The exceptional path still evaluates every original
                    // condition, including the scaled tolerance expressions.
                    if !x.is_finite() || !y.is_finite() {
                        return Err(PyValueError::new_err("R must contain only finite values"));
                    }
                    let difference = (x - y).abs();
                    raw_symmetric &= difference <= 1e-12 + 1e-12 * x.abs()
                        && difference <= 1e-12 + 1e-12 * y.abs();
                    raw_correlation &= x.abs() <= 1.0 + 1e-8 && y.abs() <= 1.0 + 1e-8;
                    let sx = x * scale;
                    let sy = y * scale;
                    scaled_finite &= sx.is_finite() && sy.is_finite();
                    let scaled_difference = (sx - sy).abs();
                    scaled_symmetric &= scaled_difference <= 1e-12 + 1e-12 * sx.abs()
                        && scaled_difference <= 1e-12 + 1e-12 * sy.abs();
                }
            }
        }
    }
    if !raw_symmetric {
        return Err(PyValueError::new_err("R must be symmetric; no automatic symmetrization is performed"));
    }
    if !raw_nonnegative {
        return Err(PyValueError::new_err("R diagonal must be nonnegative"));
    }
    if !raw_correlation {
        return Err(PyValueError::new_err(
            "R must be a signed correlation matrix with unit diagonal and values in [-1,1]"));
    }
    if !scaled_finite {
        return Err(PyValueError::new_err("XtX must contain only finite values"));
    }
    if !scaled_symmetric {
        return Err(PyValueError::new_err("XtX must be symmetric; no automatic symmetrization is performed"));
    }
    if !scaled_nonnegative {
        return Err(PyValueError::new_err("XtX diagonal must be nonnegative"));
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (array, scale, use_simd=true, proof_threads=None))]
fn validate_rss_matrix(array: Bound<'_, PyArray2<f64>>, scale: f64, use_simd: bool, proof_threads: Option<usize>) -> PyResult<()> {
    ensure_layout(&array)?;
    let array = array.try_readonly().map_err(|e| PyValueError::new_err(e.to_string()))?;
    // Match the existing numerical thread environment. Keep small matrices
    // serial because scoped thread startup exceeds their validation work.
    let proof_threads = proof_threads.unwrap_or_else(|| {
        if array.shape()[0] < 1024 { return 1; }
        std::env::var("PRUSIE_NUM_THREADS").ok()
            .and_then(|value| value.parse::<usize>().ok()).unwrap_or(1).clamp(1, 32)
    });
    #[cfg(target_arch = "x86_64")]
    if use_simd && scale.abs() <= f64::MAX * 0.5
        && exact_contiguous_rss_proof(array.as_array(), scale, proof_threads) {
        return Ok(());
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = (use_simd, proof_threads);
    // A conservative sufficient bound avoids introducing an overflow in the
    // proof itself. Larger finite scales are checked explicitly, not rejected.
    if scale.abs() <= f64::MAX * 0.5 {
        validate_rss_matrix_impl::<true>(array.as_array(), scale)
    } else {
        validate_rss_matrix_impl::<false>(array.as_array(), scale)
    }
}

/// Complete sufficient proof for ordinary exact-symmetric contiguous input.
/// Failure is not rejection: the caller evaluates all general validation rules.
#[cfg(target_arch = "x86_64")]
fn exact_contiguous_rss_proof(a: ndarray::ArrayView2<'_, f64>, scale: f64, proof_threads: usize) -> bool {
    let n = a.nrows();
    if a.ncols() != n {
        return false;
    }
    // Physical contiguity alone also admits negative strides. The memory
    // interpretation below is equivalent only to positive C/F orientations.
    if !a.is_standard_layout() && !a.t().is_standard_layout() {
        return false;
    }
    let Some(values) = a.as_slice_memory_order() else {
        return false;
    };
    // Memory-order interpretation transposes F input. The proof checks both
    // members of each pair, so this does not change any validation condition.
    for i in 0..n {
        let raw = values[i * n + i];
        let scaled = raw * scale;
        if !(raw.is_finite() && (raw - 1.0).abs() <= 1e-8
            && raw.abs() <= 1.0 + 1e-8 && raw >= 0.0
            && scaled.is_finite() && scaled >= 0.0) {
            return false;
        }
    }
    let threads = proof_threads.clamp(1, 32).min(n.div_ceil(32).max(1));
    if threads == 1 {
        return exact_contiguous_rss_pairs(values, n, 0, 1);
    }
    // Scoped threads borrow only an immutable slice while the GIL and NumPy
    // read borrow remain held. The main thread performs one share of the work.
    // Creation/join costs are inside the native public helper call.
    std::thread::scope(|scope| {
        let mut workers = Vec::with_capacity(threads - 1);
        let mut complete = true;
        for rank in 1..threads {
            match std::thread::Builder::new().name("prusie-validate".into())
                .spawn_scoped(scope, move || exact_contiguous_rss_pairs(values, n, rank, threads)) {
                Ok(worker) => workers.push(worker),
                Err(_) => { complete = false; break; }
            }
        }
        complete &= exact_contiguous_rss_pairs(values, n, 0, threads);
        for worker in workers {
            complete &= worker.join().unwrap_or(false);
        }
        complete
    })
}

#[cfg(target_arch = "x86_64")]
fn exact_contiguous_rss_pairs(values: &[f64], n: usize, rank: usize, threads: usize) -> bool {
    // Cyclic rows balance triangular work; within each row the original
    // 32×32/4×4 proof and complete checks remain unchanged.
    const BLOCK: usize = 32;
    for ii in (rank * BLOCK..n).step_by(BLOCK * threads) {
        for jj in (ii..n).step_by(BLOCK) {
            for i in (ii..(ii + BLOCK).min(n)).step_by(4) {
                for j in (jj.max(i)..(jj + BLOCK).min(n)).step_by(4) {
                    if j > i && i + 4 <= n && j + 4 <= n {
                        if !unsafe { exact_block4(values.as_ptr(), n, i, j) } {
                            return false;
                        }
                    } else {
                        for row in i..(i + 4).min(n) {
                            for col in j.max(row + 1)..(j + 4).min(n) {
                                let x = values[row * n + col];
                                let y = values[col * n + row];
                                if !(x == y && x.abs() <= 1.0 + 1e-8) {
                                    return false;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    true
}

/// SSE2 is part of the x86-64 target baseline. The caller proves both complete
/// four-by-four blocks are in bounds and keeps the NumPy read borrow/GIL alive.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn exact_block4(base: *const f64, n: usize, i: usize, j: usize) -> bool {
    use std::arch::x86_64::*;
    let abs_mask = _mm_castsi128_pd(_mm_set1_epi64x(0x7fff_ffff_ffff_ffff));
    let bound = _mm_set1_pd(1.0 + 1e-8);
    let mut valid = _mm_castsi128_pd(_mm_set1_epi64x(-1));
    for half in [0, 2] {
        let b0 = _mm_loadu_pd(base.add(j * n + i + half));
        let b1 = _mm_loadu_pd(base.add((j + 1) * n + i + half));
        let b2 = _mm_loadu_pd(base.add((j + 2) * n + i + half));
        let b3 = _mm_loadu_pd(base.add((j + 3) * n + i + half));
        let columns = [(_mm_unpacklo_pd(b0, b1), _mm_unpacklo_pd(b2, b3)),
                       (_mm_unpackhi_pd(b0, b1), _mm_unpackhi_pd(b2, b3))];
        for (offset, (low, high)) in columns.into_iter().enumerate() {
            let row = i + half + offset;
            let x0 = _mm_loadu_pd(base.add(row * n + j));
            let x1 = _mm_loadu_pd(base.add(row * n + j + 2));
            valid = _mm_and_pd(valid, _mm_cmpeq_pd(x0, low));
            valid = _mm_and_pd(valid, _mm_cmpeq_pd(x1, high));
            valid = _mm_and_pd(valid, _mm_cmple_pd(_mm_and_pd(x0, abs_mask), bound));
            valid = _mm_and_pd(valid, _mm_cmple_pd(_mm_and_pd(x1, abs_mask), bound));
        }
    }
    _mm_movemask_pd(valid) == 3
}
