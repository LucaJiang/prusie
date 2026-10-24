//! NumPy/PyO3 interface of the native extension; the IBSS solver is in prusie-core.
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

mod array_layout;
use prusie_core as solver;
mod design;
mod diagnostic_helpers;
mod executor;
mod options;
mod output;
mod prepare_helpers;
mod ser;
mod summary_helpers;

#[pyfunction]
fn backend_version() -> &'static str {
    concat!(
        "prusie-rust/",
        env!("CARGO_PKG_VERSION"),
        "; susieR-reference/0.16.6@8e56a8e038e989856d106d9ca5175cc664fea9d2"
    )
}

#[pyfunction]
fn fit<'py>(
    py: Python<'py>,
    xtx: &Bound<'py, PyArray2<f64>>,
    xty: &Bound<'py, PyArray1<f64>>,
    yty: f64,
    n: f64,
    options: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    fit_impl(py, xtx, xty, yty, n, options, None)
}

#[pyfunction]
fn _fit_with_executor<'py>(
    py: Python<'py>,
    xtx: &Bound<'py, PyArray2<f64>>,
    xty: &Bound<'py, PyArray1<f64>>,
    yty: f64,
    n: f64,
    options: &Bound<'py, PyDict>,
    executor: &Bound<'py, executor::FitExecutor>,
) -> PyResult<Bound<'py, PyDict>> {
    let result = fit_impl(py, xtx, xty, yty, n, options, Some(executor));
    if result.is_err() {
        executor::close_after_error(executor);
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn fit_impl<'py>(
    py: Python<'py>,
    xtx: &Bound<'py, PyArray2<f64>>,
    xty: &Bound<'py, PyArray1<f64>>,
    yty: f64,
    n: f64,
    options: &Bound<'py, PyDict>,
    executor: Option<&Bound<'py, executor::FitExecutor>>,
) -> PyResult<Bound<'py, PyDict>> {
    let (settings, control) = options::parse(options)?;
    // Python numeric/sequence conversion can execute callbacks. Finish every
    // option extraction before checking raw array metadata or borrowing memory.
    let global = options
        .get_item("matrix_global_scale")?
        .map(|v| v.extract::<f64>())
        .transpose()?;
    let (inverse, scales): (Vec<f64>, Vec<f64>) = if global.is_some() {
        (
            options::vector_option(options, "matrix_inverse_scales")?,
            options::vector_option(options, "matrix_scales")?,
        )
    } else {
        (Vec::new(), Vec::new())
    };
    let scaling = global.map(|global| solver::MatrixScaling {
        global,
        inverse: &inverse,
        scales: &scales,
    });

    // A conversion callback can also change dtype or rank on the same NumPy
    // object. Recheck the complete typed-array contract after all extraction.
    let xtx = xtx.as_any().downcast::<PyArray2<f64>>().map_err(|_| {
        PyValueError::new_err(
            "Native XtX must remain a two-dimensional float64 array after option conversion",
        )
    })?;
    let xty = xty.as_any().downcast::<PyArray1<f64>>().map_err(|_| {
        PyValueError::new_err(
            "Native Xty must remain a one-dimensional float64 array after option conversion",
        )
    })?;

    // rust-numpy 0.23 forms Rust references without validating byte alignment,
    // and converts byte strides to element strides by integer division. Check
    // those preconditions before constructing any Rust array view or slice.
    let align = std::mem::align_of::<f64>();
    let item = std::mem::size_of::<f64>() as isize;
    let compatible = !xtx.data().is_null()
        && !xty.data().is_null()
        && (xtx.data() as usize).is_multiple_of(align)
        && (xty.data() as usize).is_multiple_of(align)
        && xtx
            .shape()
            .iter()
            .zip(xtx.strides())
            .all(|(&n, &step)| step.checked_abs().is_some() && (n <= 1 || step % item == 0))
        && xty
            .shape()
            .iter()
            .zip(xty.strides())
            .all(|(&n, &step)| step.checked_abs().is_some() && (n <= 1 || step % item == 0));
    if !compatible {
        return Err(PyValueError::new_err(
            "Native arrays must be aligned with element-compatible byte strides",
        ));
    }
    // Check metadata before acquiring a PyReadonlyArray borrow: the borrow
    // key of rust-numpy computes stride GCDs and address ranges, including for
    // empty and singleton arrays. The typed Bound arguments hold no borrow yet.
    let raw_p = xty.shape()[0];
    if raw_p == 0 || xtx.shape() != [raw_p, raw_p] {
        return Err(PyValueError::new_err(
            "Native vector must be nonempty and XtX must be square with matching dimensions",
        ));
    }
    if raw_p
        .checked_mul(raw_p)
        .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .is_none()
    {
        return Err(PyValueError::new_err(
            "Native matrix dimensions exceed addressable storage",
        ));
    }
    if !xtx.is_contiguous() || !xty.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "Native matrix must be C- or F-contiguous float64 and vector C-contiguous",
        ));
    }
    let xtx = xtx
        .try_readonly()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let xty = xty
        .try_readonly()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let matrix = xtx.as_array();
    let vector = xty.as_array();
    if matrix.nrows() != vector.len() || matrix.ncols() != vector.len() {
        return Err(PyValueError::new_err(
            "XtX dimensions must match Xty length",
        ));
    }
    if !(matrix.is_standard_layout() || matrix.t().is_standard_layout())
        || !vector.is_standard_layout()
    {
        return Err(PyValueError::new_err(
            "Native matrix must be C- or F-contiguous float64 and vector C-contiguous",
        ));
    }
    let p = vector.len();

    // The GIL and read borrows remain held through the solver call. The
    // solver reads the matrix directly and mutates only its owned buffers.
    let values = matrix
        .as_slice_memory_order()
        .ok_or_else(|| PyValueError::new_err("Matrix must be contiguous"))?;
    let output = if let Some(executor) = executor {
        // No Python conversion runs while the owner is borrowed. The core
        // independently validates all inputs and selects/reuses the pool only
        // at its original post-validation point; it joins on every exit.
        let mut owner = executor.try_borrow_mut()?;
        // Context origin/count/phase errors are deferred to the original core
        // pool point, after its dimensions/n/yty/factors/diagonal validation.
        let result = solver::fit_controlled_with_executor(
            values,
            xty.as_slice()?,
            yty,
            n,
            &settings,
            matrix.t().is_standard_layout(),
            scaling.as_ref(),
            &control,
            &mut owner.inner,
        );
        owner.core_finished();
        match result {
            Err(error) => return Err(PyValueError::new_err(error)),
            Ok(output) => {
                owner.healthy()?;
                output
            }
        }
    } else {
        solver::fit_controlled(
            values,
            xty.as_slice()?,
            yty,
            n,
            &settings,
            matrix.t().is_standard_layout(),
            scaling.as_ref(),
            &control,
        )
        .map_err(PyValueError::new_err)?
    };
    output::fit_result(py, output, p, control.track)
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    executor::register(module)?;
    summary_helpers::register(module)?;
    prepare_helpers::register(module)?;
    design::register(module)?;
    ser::register(module)?;
    module.add("MATRIX_OPERATOR_VERSION", 1)?;
    module.add_function(wrap_pyfunction!(fit, module)?)?;
    module.add_function(wrap_pyfunction!(_fit_with_executor, module)?)?;
    module.add_function(wrap_pyfunction!(backend_version, module)?)?;
    module.add_function(wrap_pyfunction!(ser_math_backend, module)?)?;
    Ok(())
}

#[pyfunction]
fn ser_math_backend() -> &'static str {
    solver::ser_math_backend()
}
