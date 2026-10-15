//! Thin NumPy/PyO3 boundary. Iterative inference lives in rust/core.rs.
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::PyTypeInfo;

mod array_layout;
#[path = "../rust/core.rs"]
mod core;
mod diagnostic_helpers;
mod prepare_helpers;
mod summary_helpers;

fn option<'py, T: FromPyObject<'py>>(options: &Bound<'py, PyDict>, name: &str) -> PyResult<T> {
    options
        .get_item(name)?
        .ok_or_else(|| PyValueError::new_err(format!("Missing native option: {name}")))?
        .extract()
}

fn vector_option(options: &Bound<'_, PyDict>, name: &str) -> PyResult<Vec<f64>> {
    let value = options
        .get_item(name)?
        .ok_or_else(|| PyValueError::new_err(format!("Missing native option: {name}")))?;
    if let Ok(array) = value.downcast::<PyArray1<f64>>() {
        // This conversion immediately owns a snapshot, at the same extraction
        // point as the former Vec conversion. Its short borrow ends before any
        // later Python conversion callback or the long-lived matrix borrow.
        // Noncontiguous/private sequence arguments retain scalar extraction.
        if value
            .get_type()
            .is(&PyArray1::<f64>::type_object(value.py()))
            && array.is_c_contiguous()
            && array_layout::ensure_layout(array).is_ok()
        {
            let values = array
                .try_readonly()
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            return Ok(values.as_slice()?.to_vec());
        }
    }
    value.extract()
}

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
    let settings = core::FitOptions {
        l: option(options, "l")?,
        prior_variance: option(options, "prior_variance")?,
        residual_variance: option(options, "residual_variance")?,
        prior_weights: vector_option(options, "prior_weights")?,
        estimate_prior_variance: option(options, "estimate_prior_variance")?,
        estimate_prior_method: option(options, "estimate_prior_method")?,
        estimate_residual_variance: option(options, "estimate_residual_variance")?,
        check_null_threshold: option(options, "check_null_threshold")?,
        max_iter: option(options, "max_iter")?,
        tol: option(options, "tol")?,
        check_prior: option(options, "check_prior")?,
        prior_tol: option(options, "prior_tol")?,
        null_index: option(options, "null_index")?,
    };
    let mut control = core::diagnostics::FitControl::default();
    macro_rules! optional {
        ($key:literal, $field:ident) => {
            if let Some(value) = options.get_item($key)? {
                control.$field = value.extract()?;
            }
        };
    }
    optional!("track_level", track);
    optional!("criterion", criterion);
    optional!("alpha_tol", alpha_tol);
    optional!("pip_tol", pip_tol);
    optional!("min_iter", min_iter);
    optional!("cycle_lag", cycle_lag);
    optional!("hard_exclusion", hard_exclusion);
    if options.contains("init_alpha")? {
        control.initial = Some(core::diagnostics::InitialState {
            alpha: vector_option(options, "init_alpha")?,
            mu: vector_option(options, "init_mu")?,
            mu2: vector_option(options, "init_mu2")?,
            v: if options.contains("init_V")? {
                Some(vector_option(options, "init_V")?)
            } else {
                None
            },
            sigma2: options
                .get_item("init_sigma2")?
                .map(|v| v.extract())
                .transpose()?,
        });
    }
    // Python numeric/sequence conversion can execute callbacks. Finish every
    // option extraction before checking raw array metadata or borrowing memory.
    let global = options
        .get_item("matrix_global_scale")?
        .map(|v| v.extract::<f64>())
        .transpose()?;
    let (inverse, scales): (Vec<f64>, Vec<f64>) = if global.is_some() {
        (
            vector_option(options, "matrix_inverse_scales")?,
            vector_option(options, "matrix_scales")?,
        )
    } else {
        (Vec::new(), Vec::new())
    };
    let scaling = global.map(|global| core::MatrixScaling {
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

    // rust-numpy0.23 forms Rust references without validating byte alignment,
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
    // Check metadata before acquiring a PyReadonlyArray borrow: that crate's
    // borrow key computes stride GCDs and address ranges, including for empty
    // and singleton arrays. The typed Bound arguments have not acquired it.
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

    // Keep the GIL while borrowing NumPy memory. No concurrent Python mutation.
    // All iterations still run in native Rust; LD never crosses a list/JSON boundary.
    let values = matrix
        .as_slice_memory_order()
        .ok_or_else(|| PyValueError::new_err("Matrix must be contiguous"))?;
    let output = core::fit_controlled(
        values,
        xty.as_slice()?,
        yty,
        n,
        &settings,
        matrix.t().is_standard_layout(),
        scaling.as_ref(),
        &control,
    )
    .map_err(PyValueError::new_err)?;
    let result = PyDict::new(py);
    result.set_item(
        "diagnostics",
        diagnostic_helpers::export(py, &output.diagnostics, settings.l, p, control.track)?,
    )?;
    for (name, values) in [
        ("alpha", output.alpha),
        ("mu", output.mu),
        ("mu2", output.mu2),
        ("lbf_variable", output.lbf_variable),
    ] {
        let array = ndarray::Array2::from_shape_vec((settings.l, p), values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let array: Bound<'py, PyArray2<f64>> = array.into_pyarray(py);
        result.set_item(name, array)?;
    }
    for (name, values) in [
        ("lbf", output.lbf),
        ("V", output.v),
        ("KL", output.kl),
        ("elbo", output.elbo),
        ("XtXr", output.xtxr),
    ] {
        let array: Bound<'py, PyArray1<f64>> = values.into_pyarray(py);
        result.set_item(name, array)?;
    }
    result.set_item("sigma2", output.sigma2)?;
    result.set_item("niter", output.niter)?;
    result.set_item("converged", output.converged)?;
    result.set_item("warnings", output.warnings)?;
    Ok(result)
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    summary_helpers::register(module)?;
    prepare_helpers::register(module)?;
    module.add("MATRIX_OPERATOR_VERSION", 1)?;
    module.add_function(wrap_pyfunction!(fit, module)?)?;
    module.add_function(wrap_pyfunction!(backend_version, module)?)?;
    module.add_function(wrap_pyfunction!(ser_math_backend, module)?)?;
    Ok(())
}

#[pyfunction]
fn ser_math_backend() -> &'static str {
    core::ser_math_backend()
}
