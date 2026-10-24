//! Private, non-serializable owner of one public RSS call's native executor.
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

#[pyclass(name = "_FitExecutor", module = "prusie._native")]
pub(crate) struct FitExecutor {
    pub(crate) inner: prusie_core::FitExecutor,
    pub(crate) threads: usize,
    origin: std::thread::ThreadId,
    helper_used: bool,
    closed: bool,
}

impl FitExecutor {
    pub(crate) fn healthy(&self) -> PyResult<()> {
        match self.inner.failure() {
            Some(error) => Err(PyRuntimeError::new_err(error.to_owned())),
            None => Ok(()),
        }
    }

    fn invalid_lease(&mut self) -> PyErr {
        self.inner.close();
        self.closed = true;
        PyRuntimeError::new_err("Invalid private fit executor origin or call phase")
    }

    pub(crate) fn preparation_lease(&mut self) -> PyResult<()> {
        if self.closed || self.helper_used || self.origin != std::thread::current().id() {
            return Err(self.invalid_lease());
        }
        self.helper_used = true;
        self.healthy()
    }

    pub(crate) fn core_finished(&mut self) {
        self.closed = true;
    }
}

#[pymethods]
impl FitExecutor {
    #[getter]
    fn has_started(&self) -> PyResult<bool> {
        if self.origin != std::thread::current().id() {
            return Err(PyRuntimeError::new_err(
                "Invalid private fit executor origin thread",
            ));
        }
        Ok(self.inner.has_started())
    }

    fn close(&mut self) -> PyResult<()> {
        self.inner.close();
        self.closed = true;
        self.healthy()
    }

    fn __reduce__(&self) -> PyResult<Py<PyAny>> {
        Err(PyTypeError::new_err(
            "A private fit executor cannot be serialized",
        ))
    }
}

#[pyfunction]
fn _new_fit_executor(threads: &Bound<'_, PyAny>) -> PyResult<FitExecutor> {
    if threads.is_instance_of::<pyo3::types::PyBool>() {
        return Err(PyValueError::new_err(
            "n_threads must be an integer between 1 and 32",
        ));
    }
    let threads: usize = threads.extract()?;
    if !(1..=32).contains(&threads) {
        return Err(PyValueError::new_err(
            "n_threads must be an integer between 1 and 32",
        ));
    }
    Ok(FitExecutor {
        inner: prusie_core::FitExecutor::new(),
        threads,
        origin: std::thread::current().id(),
        helper_used: false,
        closed: false,
    })
}

/// An error in argument parsing, array checks or result construction can occur
/// outside the core's cleanup boundary. Join here without replacing that error.
pub(crate) fn close_after_error(executor: &Bound<'_, FitExecutor>) {
    if let Ok(mut owner) = executor.try_borrow_mut() {
        owner.inner.close();
        owner.closed = true;
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<FitExecutor>()?;
    module.add_function(wrap_pyfunction!(_new_fit_executor, module)?)?;
    Ok(())
}
