//! Extract Python settings completely before borrowing any input arrays.
use numpy::{PyArray1, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::{prelude::*, exceptions::PyValueError, types::PyDict, PyTypeInfo};
use prusie_core as solver;
use crate::array_layout;

fn option<'py, T: FromPyObject<'py>>(options: &Bound<'py, PyDict>, name: &str) -> PyResult<T> {
    options
        .get_item(name)?
        .ok_or_else(|| PyValueError::new_err(format!("Missing native option: {name}")))?
        .extract()
}

pub(crate) fn vector_option(options: &Bound<'_, PyDict>, name: &str) -> PyResult<Vec<f64>> {
    let value = options
        .get_item(name)?
        .ok_or_else(|| PyValueError::new_err(format!("Missing native option: {name}")))?;
    if let Ok(array) = value.downcast::<PyArray1<f64>>() {
        // Copy a contiguous float64 array immediately. The short read borrow
        // ends before any later Python conversion callback and before the
        // matrix is borrowed. Other inputs use element-wise extraction.
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

pub(crate) fn parse(options: &Bound<'_, PyDict>) -> PyResult<(solver::FitOptions, solver::diagnostics::FitControl)> {
    let estimate_prior_variance: bool = option(options, "estimate_prior_variance")?;
    let method_name: String = option(options, "estimate_prior_method")?;
    let prior_method = if estimate_prior_variance {
        method_name.parse().map_err(PyValueError::new_err)?
    } else {
        solver::options::PriorMethod::None
    };
    let settings = solver::FitOptions {
        l: option(options, "l")?,
        prior_variance: option(options, "prior_variance")?,
        residual_variance: option(options, "residual_variance")?,
        prior_weights: vector_option(options, "prior_weights")?,
        estimate_prior_variance,
        estimate_prior_method: prior_method,
        estimate_residual_variance: option(options, "estimate_residual_variance")?,
        check_null_threshold: option(options, "check_null_threshold")?,
        max_iter: option(options, "max_iter")?,
        tol: option(options, "tol")?,
        check_prior: option(options, "check_prior")?,
        prior_tol: option(options, "prior_tol")?,
        null_index: option(options, "null_index")?,
    };
    let mut control = solver::diagnostics::FitControl::default();
    macro_rules! optional {
        ($key:literal, $field:ident) => {
            if let Some(value) = options.get_item($key)? {
                control.$field = value.extract()?;
            }
        };
    }
    if options.contains("component_prior_variance")? {
        control.component_prior_variance =
            Some(vector_option(options, "component_prior_variance")?);
    }
    optional!("track_level", track);
    if let Some(value) = options.get_item("criterion")? {
        control.criterion = value
            .extract::<String>()?
            .parse()
            .map_err(PyValueError::new_err)?;
    }
    optional!("alpha_tol", alpha_tol);
    optional!("pip_tol", pip_tol);
    optional!("min_iter", min_iter);
    optional!("cycle_lag", cycle_lag);
    optional!("hard_exclusion", hard_exclusion);
    if options.contains("init_alpha")? {
        control.initial = Some(solver::diagnostics::InitialState {
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
    if let Some(value) = options.get_item("estimate_residual_method")? {
        control.model.residual_method = value.extract::<String>()?.parse().map_err(PyValueError::new_err)?;
    }
    if let Some(value) = options.get_item("residual_variance_lowerbound")? {
        control.model.residual_lowerbound = value.extract()?;
    }
    if let Some(value) = options.get_item("residual_variance_upperbound")? {
        control.model.residual_upperbound = value.extract()?;
    }
    if let Some(value) = options.get_item("alpha0")? {
        control.model.alpha0 = value.extract()?;
    }
    if let Some(value) = options.get_item("beta0")? {
        control.model.beta0 = value.extract()?;
    }
    if options.contains("prior_variance_grid")? {
        control.model.mixture = Some(solver::model::MixturePrior {
            variances: vector_option(options, "prior_variance_grid")?,
            weights: vector_option(options, "mixture_weights")?,
        });
    }
    if let Some(prior) = options.get_item("slot_prior")? {
        if !prior.is_none() {
            let prior = prior.downcast::<PyDict>()?;
            let family_name: String = option(prior, "family")?;
            let family = match family_name.as_str() {
                "betabinom" => solver::model::SlotPriorFamily::BetaBinomial {
                    a: option(prior, "a_beta")?, b: option(prior, "b_beta")?,
                },
                "poisson" => solver::model::SlotPriorFamily::GammaPoisson {
                    expected: option(prior, "C")?, shape: option(prior, "nu")?,
                },
                _ => return Err(PyValueError::new_err("Unknown slot prior family")),
            };
            let schedule = match option::<String>(prior, "update_schedule")?.as_str() {
                "sequential" => solver::model::UpdateSchedule::Sequential,
                "batch" => solver::model::UpdateSchedule::Batch,
                _ => return Err(PyValueError::new_err("Unknown slot update schedule")),
            };
            let initial = if prior.get_item("c_hat_init")?.is_some_and(|value| !value.is_none()) {
                Some(vector_option(prior, "c_hat_init")?)
            } else { None };
            control.model.slot_prior = Some(solver::model::SlotPrior {
                family, schedule, initial,
                skip_threshold_multiplier: option(prior, "skip_threshold_multiplier")?,
            });
        }
    }
    Ok((settings, control))
}
