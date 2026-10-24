//! Matrix validation and credible-set purity with bounded additional memory.
//!
//! NumPy arrays are borrowed only while the GIL is held and are never
//! modified by default. The explicit overwrite helper instead holds one write
//! borrow for its destructive traversal. C/F/strided layouts remain accepted
//! by the ordinary nonmutating helpers.
use crate::array_layout::ensure_layout;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (array, name, correlation=false, scale=1.0))]
fn validate_matrix(
    array: Bound<'_, PyArray2<f64>>,
    name: &str,
    correlation: bool,
    scale: f64,
) -> PyResult<bool> {
    ensure_layout(&array)?;
    let array = array
        .try_readonly()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
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
            return Err(PyValueError::new_err(format!(
                "{name} must contain only finite values"
            )));
        }
        negative_diagonal |= value < 0.0;
        if correlation {
            valid_correlation &= (value - 1.0).abs() <= 1e-8;
        }
    }
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
                    if x == y {
                        // Equality proves both entries share finiteness and
                        // absolute bounds, including the two signed zeros.
                        // Equal infinities still fail this finite check.
                        if !x.is_finite() {
                            return Err(PyValueError::new_err(format!(
                                "{name} must contain only finite values"
                            )));
                        }
                        if correlation {
                            valid_correlation &= x.abs() <= 1.0 + 1e-8;
                        }
                    } else {
                        if !x.is_finite() || !y.is_finite() {
                            return Err(PyValueError::new_err(format!(
                                "{name} must contain only finite values"
                            )));
                        }
                        symmetric = false;
                        if correlation {
                            valid_correlation &= x.abs() <= 1.0 + 1e-8 && y.abs() <= 1.0 + 1e-8;
                        }
                    }
                }
            }
        }
    }
    if negative_diagonal {
        return Err(PyValueError::new_err(format!(
            "{name} diagonal must be nonnegative"
        )));
    }
    if !valid_correlation {
        return Err(PyValueError::new_err(
            "R must be a signed correlation matrix with unit diagonal and values in [-1,1]",
        ));
    }
    Ok(symmetric)
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
    if let Some(ref inverse) = inverse_scales {
        ensure_layout(inverse)?;
    }
    let correlation = correlation
        .try_readonly()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let members = members
        .try_readonly()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let inverse_scales = inverse_scales
        .as_ref()
        .map(|s| s.try_readonly())
        .transpose()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let a = correlation.as_array();
    let indices = members.as_array();
    let n = indices.len();
    if a.nrows() != a.ncols() || indices.iter().any(|&i| i < 0 || i as usize >= a.nrows()) {
        return Err(PyValueError::new_err(
            "CS members must index a square matrix",
        ));
    }
    let inv = inverse_scales.as_ref().map(|s| s.as_array());
    if inv.as_ref().is_some_and(|s| s.len() != n) {
        return Err(PyValueError::new_err(
            "CS inverse scales must match members",
        ));
    }
    let size = n
        .checked_mul(n.saturating_sub(1))
        .and_then(|x| x.checked_div(2))
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
                // Negation deliberately rejects NaN as well as below-threshold pairs.
                #[allow(clippy::neg_cmp_op_on_partial_ord)]
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
    module.add_function(wrap_pyfunction!(validate_and_repair_rss, module)?)?;
    module.add_function(wrap_pyfunction!(
        _validate_and_repair_rss_with_executor,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(validate_and_overwrite_rss, module)?)?;
    module.add_function(wrap_pyfunction!(cs_pairwise_abs, module)?)?;
    module.add_function(wrap_pyfunction!(complete_finite_scan, module)?)?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (array, reduction=true))]
fn complete_finite_scan(array: Bound<'_, PyArray1<f64>>, reduction: bool) -> PyResult<bool> {
    ensure_layout(&array)?;
    let array = array
        .try_readonly()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let values = array.as_slice()?;
    Ok(if reduction {
        prusie_core::input_validation::all_finite(values)
    } else {
        !values.iter().any(|value| !value.is_finite())
    })
}

fn validate_rss_matrix_impl<const SAFE_SCALE: bool>(
    a: ndarray::ArrayView2<'_, f64>,
    scale: f64,
) -> PyResult<bool> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(PyValueError::new_err("R must be square"));
    }
    let mut raw_nonnegative = true;
    let mut scaled_symmetric = true;
    let mut scaled_finite = true;
    let mut scaled_nonnegative = true;
    for i in 0..n {
        let raw = a[[i, i]];
        if !raw.is_finite() {
            return Err(PyValueError::new_err("R must contain only finite values"));
        }
        raw_nonnegative &= raw >= 0.0;
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
                        // entries finite.
                        if !SAFE_SCALE {
                            scaled_finite &= (x * scale).is_finite();
                        }
                        continue;
                    }
                    if !x.is_finite() || !y.is_finite() {
                        return Err(PyValueError::new_err("R must contain only finite values"));
                    }
                    let sx = x * scale;
                    let sy = y * scale;
                    scaled_symmetric &= sx == sy;
                    scaled_finite &= sx.is_finite() && sy.is_finite();
                }
            }
        }
    }
    if !raw_nonnegative {
        return Err(PyValueError::new_err("R diagonal must be nonnegative"));
    }
    if !scaled_finite {
        return Err(PyValueError::new_err("XtX must contain only finite values"));
    }
    if !scaled_nonnegative {
        return Err(PyValueError::new_err("XtX diagonal must be nonnegative"));
    }
    Ok(scaled_symmetric)
}

#[pyfunction]
#[pyo3(signature = (array, scale, use_simd=true, proof_threads=None))]
fn validate_rss_matrix(
    array: Bound<'_, PyArray2<f64>>,
    scale: f64,
    use_simd: bool,
    proof_threads: Option<usize>,
) -> PyResult<bool> {
    ensure_layout(&array)?;
    let array = array
        .try_readonly()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    // Match the existing numerical thread environment. Keep small matrices
    // serial because scoped thread startup exceeds their validation work.
    let proof_threads = proof_threads.unwrap_or_else(|| {
        if array.shape()[0] < 1024 {
            return 1;
        }
        prusie_core::num_threads()
    });
    #[cfg(target_arch = "x86_64")]
    if use_simd
        && scale.abs() <= f64::MAX * 0.5
        && exact_contiguous_rss_proof(array.as_array(), scale, proof_threads)
    {
        return Ok(true);
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
fn exact_contiguous_rss_proof(
    a: ndarray::ArrayView2<'_, f64>,
    scale: f64,
    proof_threads: usize,
) -> bool {
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
        if !(raw.is_finite()
            && (raw - 1.0).abs() <= 1e-8
            && raw.abs() <= 1.0 + 1e-8
            && raw >= 0.0
            && scaled.is_finite()
            && scaled >= 0.0)
        {
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
            match std::thread::Builder::new()
                .name("prusie-validate".into())
                .spawn_scoped(scope, move || {
                    exact_contiguous_rss_pairs(values, n, rank, threads)
                }) {
                Ok(worker) => workers.push(worker),
                Err(_) => {
                    complete = false;
                    break;
                }
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
                        // SAFETY: both 4 × 4 blocks are inside the checked n × n
                        // contiguous slice; SSE2 is an x86-64 baseline feature.
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
// SAFETY: base must address a live contiguous n*n f64 slice, and both
// transposed four-by-four blocks starting at (i,j) must lie within that slice.
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
        let columns = [
            (_mm_unpacklo_pd(b0, b1), _mm_unpacklo_pd(b2, b3)),
            (_mm_unpackhi_pd(b0, b1), _mm_unpackhi_pd(b2, b3)),
        ];
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

/// Validate raw/scaled RSS input and prepare an owned repair in the same general
/// traversal. Return repair errors to Python so its warning still precedes them.
#[pyfunction]
#[pyo3(signature = (array, scale, proof_threads=1))]
fn validate_and_repair_rss<'py>(
    py: Python<'py>,
    array: Bound<'py, PyArray2<f64>>,
    scale: f64,
    proof_threads: usize,
) -> PyResult<(bool, Option<Bound<'py, PyArray2<f64>>>, Option<String>)> {
    ensure_layout(&array)?;
    let input = array
        .try_readonly()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let a = input.as_array();
    let n = a.nrows();
    if a.ncols() != n {
        return Err(PyValueError::new_err("R must be square"));
    }
    #[cfg(target_arch = "x86_64")]
    if scale.abs() <= f64::MAX * 0.5
        && exact_contiguous_rss_proof(a, scale, if n < 1024 { 1 } else { proof_threads })
    {
        return Ok((true, None, None));
    }
    if n.checked_mul(n)
        .and_then(|v| v.checked_mul(8))
        .filter(|&v| v <= isize::MAX as usize)
        .is_none()
    {
        return Err(PyValueError::new_err(
            "Matrix must be square with addressable storage",
        ));
    }
    // SAFETY: shape/bytes are checked; the private output is completely written
    // before exposure, cannot alias the caller, and is dropped on any error.
    let repaired = unsafe { PyArray2::<f64>::new(py, [n, n], false) };
    let values = a.as_slice_memory_order();
    let row_major = a.strides() == [n as isize, 1];
    let column_major = a.strides() == [1, n as isize];
    let mut raw_finite = true;
    let mut raw_nonnegative = true;
    let mut scaled_finite = true;
    let mut scaled_nonnegative = true;
    let mut symmetric = true;
    let mut repair_finite = true;
    let mut repair_nonnegative = true;
    let paired = values
        .filter(|_| row_major || column_major)
        .and_then(|values| {
            let layout = if row_major {
                prusie_core::input_validation::RssPairLayout::C
            } else {
                prusie_core::input_validation::RssPairLayout::F
            };
            // SAFETY: the checked private NumPy allocation has exactly n*n cells,
            // cannot alias the held input borrow, and is not exposed. MaybeUninit
            // permits writes without forming f64 references over unwritten cells.
            let output = unsafe {
                std::slice::from_raw_parts_mut(
                    repaired.data().cast::<std::mem::MaybeUninit<f64>>(),
                    n * n,
                )
            };
            prusie_core::input_validation::rss_pair_validation_owned(
                values, n, scale, layout, output,
            )
        });
    if let Some(flags) = paired {
        raw_finite = flags.raw_finite;
        raw_nonnegative = flags.raw_nonnegative;
        scaled_finite = flags.scaled_finite;
        scaled_nonnegative = flags.scaled_nonnegative;
        symmetric = flags.symmetric;
        repair_finite = flags.repair_finite;
        repair_nonnegative = flags.repair_nonnegative;
    } else {
        const BLOCK: usize = 32;
        for ii in (0..n).step_by(BLOCK) {
            for jj in (ii..n).step_by(BLOCK) {
                for i in ii..(ii + BLOCK).min(n) {
                    for j in jj.max(i)..(jj + BLOCK).min(n) {
                        let (x, y) = match values {
                            Some(v) if row_major => (v[i * n + j], v[j * n + i]),
                            Some(v) if column_major => (v[j * n + i], v[i * n + j]),
                            _ => (a[(i, j)], a[(j, i)]),
                        };
                        raw_finite &= x.is_finite() && y.is_finite();
                        let sx = x * scale;
                        let sy = y * scale;
                        scaled_finite &= sx.is_finite() && sy.is_finite();
                        symmetric &= sx == sy;
                        let value = (sx + sy) / 2.;
                        repair_finite &= value.is_finite();
                        if i == j {
                            raw_nonnegative &= x >= 0.;
                            scaled_nonnegative &= sx >= 0.;
                            repair_nonnegative &= value >= 0.;
                        }
                        // SAFETY: both indices are in range; every C-order output
                        // element is written once, under the input borrow and GIL.
                        unsafe {
                            repaired.data().add(i * n + j).write(value);
                            if i != j {
                                repaired.data().add(j * n + i).write(value);
                            }
                        }
                    }
                }
            }
        }
    }
    // Preserve raw validation before scaled validation, then the warning and
    // repair validation. Do not report repair overflow on a symmetric input.
    if !raw_finite {
        return Err(PyValueError::new_err("R must contain only finite values"));
    }
    if !raw_nonnegative {
        return Err(PyValueError::new_err("R diagonal must be nonnegative"));
    }
    if !scaled_finite {
        return Err(PyValueError::new_err("XtX must contain only finite values"));
    }
    if !scaled_nonnegative {
        return Err(PyValueError::new_err("XtX diagonal must be nonnegative"));
    }
    if symmetric {
        return Ok((true, None, None));
    }
    let error = if !repair_finite {
        Some("XtX must contain only finite values".into())
    } else if !repair_nonnegative {
        Some("XtX diagonal must be nonnegative".into())
    } else {
        None
    };
    Ok((false, Some(repaired), error))
}

/// Preparation-only executor entry. The original helper above remains the
/// authoritative serial fallback, including its independent validation flags.
#[pyfunction]
fn _validate_and_repair_rss_with_executor<'py>(
    py: Python<'py>,
    array: Bound<'py, PyArray2<f64>>,
    scale: f64,
    executor: &Bound<'py, crate::executor::FitExecutor>,
) -> PyResult<(bool, Option<Bound<'py, PyArray2<f64>>>, Option<String>)> {
    let result = parallel_repair_impl(py, array, scale, executor);
    if result.is_err() {
        crate::executor::close_after_error(executor);
    }
    result
}

fn parallel_repair_impl<'py>(
    py: Python<'py>,
    array: Bound<'py, PyArray2<f64>>,
    scale: f64,
    executor: &Bound<'py, crate::executor::FitExecutor>,
) -> PyResult<(bool, Option<Bound<'py, PyArray2<f64>>>, Option<String>)> {
    use prusie_core::input_validation::RssPairLayout;
    use prusie_core::PreparationAttempt;

    // This inner scope ends every source/output borrow and executor lease
    // before serial fallback. Aborted work has already joined in the core;
    // its possibly partial private output is dropped here, never exposed.
    let attempt = (|| {
        ensure_layout(&array)?;
        let input = array
            .try_readonly()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let a = input.as_array();
        let n = a.nrows();
        if a.ncols() != n {
            return Err(PyValueError::new_err("R must be square"));
        }
        if n.checked_mul(n)
            .and_then(|values| values.checked_mul(8))
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .is_none()
        {
            return Err(PyValueError::new_err(
                "Matrix must be square with addressable storage",
            ));
        }
        if !a.is_standard_layout() || a.strides() != [n as isize, 1] {
            return Ok(None);
        }
        let Some(values) = a.as_slice_memory_order() else {
            return Ok(None);
        };
        // No option extraction or conversion runs under this lease. The output
        // allocation stays on this caller; workers are strictly Python-free.
        // Architecture/control admission and every worker check are read-only.
        let mut owner = executor.try_borrow_mut()?;
        owner.preparation_lease()?;
        let threads = owner.threads;
        if !owner.inner.begin_preparation(threads, n) {
            owner.healthy()?;
            return Ok(None);
        }
        if scale.abs() <= f64::MAX * 0.5 {
            match owner
                .inner
                .exact_rss_proof(values, n, scale, RssPairLayout::C)
            {
                PreparationAttempt::Complete(true) => {
                    owner.healthy()?;
                    return Ok(Some((true, None, None)));
                }
                PreparationAttempt::Complete(false) => {}
                PreparationAttempt::NotApplicable | PreparationAttempt::Aborted => {
                    owner.healthy()?;
                    return Ok(None);
                }
            }
        }
        // SAFETY: checked square/address bounds, private NumPy allocation, no
        // caller alias. Only MaybeUninit references cover unwritten cells. The
        // core's rank writer initializes each cell once and joins before return.
        let repaired = unsafe { PyArray2::<f64>::new(py, [n, n], false) };
        let output = unsafe {
            std::slice::from_raw_parts_mut(
                repaired.data().cast::<std::mem::MaybeUninit<f64>>(),
                n * n,
            )
        };
        let flags = match owner
            .inner
            .validate_rss_owned(values, n, scale, RssPairLayout::C, output)
        {
            PreparationAttempt::Complete(flags) => flags,
            PreparationAttempt::NotApplicable | PreparationAttempt::Aborted => {
                owner.healthy()?;
                return Ok(None);
            }
        };
        owner.healthy()?;
        // Same seven independent predicates and priority as the untouched
        // serial helper: raw, scaled, symmetric, then deferred repair errors.
        if !flags.raw_finite {
            return Err(PyValueError::new_err("R must contain only finite values"));
        }
        if !flags.raw_nonnegative {
            return Err(PyValueError::new_err("R diagonal must be nonnegative"));
        }
        if !flags.scaled_finite {
            return Err(PyValueError::new_err("XtX must contain only finite values"));
        }
        if !flags.scaled_nonnegative {
            return Err(PyValueError::new_err("XtX diagonal must be nonnegative"));
        }
        if flags.symmetric {
            return Ok(Some((true, None, None)));
        }
        let error = if !flags.repair_finite {
            Some("XtX must contain only finite values".into())
        } else if !flags.repair_nonnegative {
            Some("XtX diagonal must be nonnegative".into())
        } else {
            None
        };
        Ok(Some((false, Some(repaired), error)))
    })();
    match attempt? {
        Some(output) => Ok(output),
        // Construction/FP admission or a recoverable joined task panic uses
        // the complete original validator. Never route to legacy helper workers.
        None => validate_and_repair_rss(py, array, scale, 1),
    }
}

/// Explicit destructive RSS storage option. Python rejects unsupported input
/// owners, aliases and hooks before this entry; native layout/borrow/dimension
/// checks remain independent. The GIL is held for every proof and write.
#[pyfunction]
#[pyo3(signature = (array, scale, proof_threads=1))]
fn validate_and_overwrite_rss<'py>(
    py: Python<'py>,
    array: Bound<'py, PyArray2<f64>>,
    scale: f64,
    proof_threads: usize,
) -> PyResult<(
    bool,
    Option<Bound<'py, PyArray2<f64>>>,
    Option<String>,
    &'static str,
)> {
    ensure_layout(&array)?;
    // Read the C-API writeability flag directly, without invoking Python hooks.
    // SAFETY: Bound<PyArray2<f64>> retains the live PyArrayObject under the GIL.
    if unsafe { (*array.as_array_ptr()).flags & numpy::npyffi::NPY_ARRAY_WRITEABLE == 0 } {
        return Err(PyValueError::new_err("overwrite_ld requires writable R"));
    }
    if !array.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "overwrite_ld requires C-contiguous R",
        ));
    }
    let witness = {
        // This read borrow and every view/slice derived from it END before a
        // mutable borrow is requested. No Python callback runs between phases.
        let input = array
            .try_readonly()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let a = input.as_array();
        let n = a.nrows();
        if n == 0
            || a.ncols() != n
            || n.checked_mul(n)
                .and_then(|v| v.checked_mul(8))
                .filter(|&v| v <= isize::MAX as usize)
                .is_none()
        {
            return Err(PyValueError::new_err(
                "R must be square with addressable storage",
            ));
        }
        #[cfg(target_arch = "x86_64")]
        if scale.abs() <= f64::MAX * 0.5 && exact_contiguous_rss_proof(a, scale, proof_threads) {
            return Ok((true, None, None, "symmetric_no_write"));
        }
        let values = input
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut witness = false;
        'rows: for i in 0..n {
            for j in (i + 1)..n {
                let x = values[i * n + j];
                let y = values[j * n + i];
                let sx = x * scale;
                let sy = y * scale;
                if x.is_finite() && y.is_finite() && sx.is_finite() && sy.is_finite() && sx != sy {
                    witness = true;
                    break 'rows;
                }
            }
        }
        witness
    };
    if !witness {
        // Speculative averages can overflow even for a valid scaled-symmetric
        // input. Keep the original owned helper and its untouched-caller rules.
        let (symmetric, repaired, error) =
            validate_and_repair_rss(py, array, scale, proof_threads)?;
        return Ok((symmetric, repaired, error, "no_witness_eager_no_write"));
    }
    let flags = {
        let mut input = array
            .try_readwrite()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let n = input.shape()[0];
        let values = input
            .as_slice_mut()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        prusie_core::input_validation::rss_pair_validation_in_place(values, n, scale)
            .ok_or_else(|| PyValueError::new_err("R must be square with addressable storage"))?
    };
    // Complete independent flags, in the parent's exact error priority. Caller
    // storage may already be consumed when an invalid input is discovered.
    if !flags.raw_finite {
        return Err(PyValueError::new_err("R must contain only finite values"));
    }
    if !flags.raw_nonnegative {
        return Err(PyValueError::new_err("R diagonal must be nonnegative"));
    }
    if !flags.scaled_finite {
        return Err(PyValueError::new_err("XtX must contain only finite values"));
    }
    if !flags.scaled_nonnegative {
        return Err(PyValueError::new_err("XtX diagonal must be nonnegative"));
    }
    debug_assert!(
        !flags.symmetric,
        "An independently finite scaled witness cannot be symmetric"
    );
    let error = if !flags.repair_finite {
        Some("XtX must contain only finite values".into())
    } else if !flags.repair_nonnegative {
        Some("XtX diagonal must be nonnegative".into())
    } else {
        None
    };
    Ok((false, Some(array), error, "ordered_scaled_xtx"))
}
