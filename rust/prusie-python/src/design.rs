//! Thin dense/CSC bridges to the matrix-free individual-data solver.
use crate::{array_layout, options, output};
use numpy::{Element, PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

pub(crate) fn vector<T: Element + Copy>(array: &Bound<'_, PyArray1<T>>) -> PyResult<Vec<T>> {
    // Recheck dtype and rank after all Python option conversions. Empty sparse
    // data are valid; avoid constructing a rust-numpy read borrow for them.
    let array = array.as_any().downcast::<PyArray1<T>>().map_err(|_| {
        PyValueError::new_err("Native vector dtype or rank changed during option conversion")
    })?;
    if array.shape()[0] == 0 {
        return Ok(Vec::new());
    }
    array_layout::ensure_layout(array)?;
    let borrowed = array
        .try_readonly()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(borrowed.as_slice()?.to_vec())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn fit_design_dense<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyArray2<f64>>,
    means: &Bound<'py, PyArray1<f64>>,
    scales: &Bound<'py, PyArray1<f64>>,
    xty: &Bound<'py, PyArray1<f64>>,
    yty: f64,
    n: f64,
    settings: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let (settings, control) = options::parse(settings)?;
    let means = vector(means)?;
    let scales = vector(scales)?;
    let xty = vector(xty)?;
    array_layout::ensure_layout(x)?;
    let x = x
        .try_readonly()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let matrix = x.as_array();
    if !(matrix.is_standard_layout() || matrix.t().is_standard_layout()) {
        return Err(PyValueError::new_err(
            "Native dense design must be C- or F-contiguous",
        ));
    }
    let values = matrix
        .as_slice_memory_order()
        .ok_or_else(|| PyValueError::new_err("Design must be contiguous"))?;
    let result = prusie_core::fit_design_dense(
        values,
        matrix.nrows(),
        matrix.ncols(),
        matrix.t().is_standard_layout(),
        &means,
        &scales,
        &xty,
        yty,
        n,
        &settings,
        &control,
    )
    .map_err(PyValueError::new_err)?;
    output::fit_result(py, result, xty.len(), control.track)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn fit_design_csc<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyArray1<f64>>,
    indices: &Bound<'py, PyArray1<i64>>,
    indptr: &Bound<'py, PyArray1<i64>>,
    nrow: usize,
    means: &Bound<'py, PyArray1<f64>>,
    scales: &Bound<'py, PyArray1<f64>>,
    xty: &Bound<'py, PyArray1<f64>>,
    yty: f64,
    n: f64,
    settings: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let (settings, control) = options::parse(settings)?;
    // These are O(nnz + n + p) copies; no dense design or p × p Gram is formed.
    let means = vector(means)?;
    let scales = vector(scales)?;
    let xty = vector(xty)?;
    let data = vector(data)?;
    let indices = vector(indices)?;
    let indptr = vector(indptr)?;
    let result = prusie_core::fit_design_csc(
        &data,
        &indices,
        &indptr,
        nrow,
        means.len(),
        &means,
        &scales,
        &xty,
        yty,
        n,
        &settings,
        &control,
    )
    .map_err(PyValueError::new_err)?;
    output::fit_result(py, result, xty.len(), control.track)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn fit_design_trend<'py>(
    py: Python<'py>,
    nrow: usize,
    order: usize,
    means: &Bound<'py, PyArray1<f64>>,
    scales: &Bound<'py, PyArray1<f64>>,
    xty: &Bound<'py, PyArray1<f64>>,
    yty: f64,
    n: f64,
    settings: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let (settings, control) = options::parse(settings)?;
    let means = vector(means)?;
    let scales = vector(scales)?;
    let xty = vector(xty)?;
    let result = prusie_core::fit_design_trend(
        nrow, order, &means, &scales, &xty, yty, n, &settings, &control,
    )
    .map_err(PyValueError::new_err)?;
    output::fit_result(py, result, xty.len(), control.track)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(fit_design_dense, module)?)?;
    module.add_function(wrap_pyfunction!(fit_design_csc, module)?)?;
    module.add_function(wrap_pyfunction!(fit_design_trend, module)?)?;
    Ok(())
}
