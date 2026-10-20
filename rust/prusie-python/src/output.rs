//! Convert owned solver outputs into NumPy arrays.
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::{prelude::*, exceptions::PyValueError, types::PyDict};
use prusie_core::FitOutput;
use crate::diagnostic_helpers;

pub(crate) fn fit_result<'py>(py: Python<'py>, output: FitOutput, p: usize, track: u8) -> PyResult<Bound<'py, PyDict>> {
    let l = output.v.len();
    let result = PyDict::new(py);
    result.set_item(
        "diagnostics",
        diagnostic_helpers::export(py, &output.diagnostics, l, p, track)?,
    )?;
    for (name, values) in [
        ("alpha", output.alpha),
        ("mu", output.mu),
        ("mu2", output.mu2),
        ("lbf_variable", output.lbf_variable),
    ] {
        let array = ndarray::Array2::from_shape_vec((l, p), values)
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
    if let Some(values) = output.lbf_grid {
        let entries = l.checked_mul(p).filter(|&entries| entries > 0)
            .ok_or_else(|| PyValueError::new_err("Invalid mixture result shape"))?;
        if values.len() % entries != 0 {
            return Err(PyValueError::new_err("Invalid mixture result shape"));
        }
        let k = values.len() / entries;
        let grid = ndarray::Array3::from_shape_vec((l, p, k), values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        result.set_item("lbf_grid", grid.into_pyarray(py))?;
    }
    for (name, values) in [("slot_weights", output.slot_weights),
                           ("rv", output.rv), ("marginal_loglik", output.marginal_loglik)] {
        if let Some(values) = values { result.set_item(name, values.into_pyarray(py))?; }
    }
    Ok(result)
}
