//! One unfinalized SER update for model-specific Python outer loops.
use numpy::{IntoPyArray, PyArray1};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

use crate::{design::vector, options};

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (xtr, diagonal, yy, n, settings, inflation=None))]
fn ser_update<'py>(
    py: Python<'py>,
    xtr: &Bound<'py, PyArray1<f64>>,
    diagonal: &Bound<'py, PyArray1<f64>>,
    yy: f64,
    n: f64,
    settings: &Bound<'py, PyDict>,
    inflation: Option<&Bound<'py, PyArray1<f64>>>,
) -> PyResult<Bound<'py, PyDict>> {
    let (settings, control) = options::parse(settings)?;
    let xtr = vector(xtr)?;
    let diagonal = vector(diagonal)?;
    let inflation = inflation.map(vector).transpose()?;
    let output = prusie_core::ser_update(
        &xtr,
        &diagonal,
        inflation.as_deref(),
        yy,
        n,
        &settings,
        &control.model,
        control.hard_exclusion,
    )
    .map_err(PyValueError::new_err)?;
    let result = PyDict::new(py);
    for (name, values) in [
        ("alpha", output.alpha),
        ("mu", output.mu),
        ("mu2", output.mu2),
        ("lbf_variable", output.lbf_variable),
    ] {
        result.set_item(name, values.into_pyarray(py))?;
    }
    result.set_item("lbf", output.lbf)?;
    result.set_item("V", output.v)?;
    if let Some(values) = output.lbf_grid {
        let p = xtr.len();
        if p == 0 || values.len() % p != 0 {
            return Err(PyValueError::new_err("Invalid mixture result shape"));
        }
        let k = values.len() / p;
        let values = ndarray::Array2::from_shape_vec((p, k), values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        result.set_item("lbf_grid", values.into_pyarray(py))?;
    }
    for (name, value) in [
        ("rv", output.rv),
        ("marginal_loglik", output.marginal_loglik),
        ("KL", output.kl),
    ] {
        if let Some(value) = value {
            result.set_item(name, value)?;
        }
    }
    Ok(result)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(ser_update, module)?)?;
    Ok(())
}
