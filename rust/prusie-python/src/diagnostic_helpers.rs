use prusie_core::diagnostics::{Diagnostics, Snapshot};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

fn snapshot<'py>(
    py: Python<'py>,
    s: &Snapshot,
    l: usize,
    p: usize,
    track: u8,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("iteration", s.iteration)?;
    d.set_item("phase", "post_sweep_pre_sigma2_update")?;
    d.set_item("elbo", s.elbo)?;
    d.set_item("delta_elbo", s.delta_elbo)?;
    d.set_item("relative_delta_elbo", s.relative_delta_elbo)?;
    d.set_item("max_abs_delta_alpha", s.delta_alpha)?;
    d.set_item("max_abs_delta_pip", s.delta_pip)?;
    d.set_item("max_abs_delta_bbar", s.delta_mean)?;
    d.set_item("delta_V", &s.delta_v)?;
    d.set_item("delta_sigma2", s.delta_sigma2)?;
    d.set_item("V", &s.v)?;
    d.set_item("sigma2", s.sigma2)?;
    d.set_item("top_variant", &s.top_variant)?;
    d.set_item("top_alpha", &s.top_alpha)?;
    d.set_item("entropy", &s.entropy)?;
    d.set_item(
        "effective_candidates",
        s.entropy.iter().map(|v| v.exp()).collect::<Vec<_>>(),
    )?;
    d.set_item("active", &s.active)?;
    d.set_item("cycle_lag", s.cycle_lag)?;
    d.set_item("cycle_alpha_distance", s.cycle_alpha_distance)?;
    d.set_item("cycle_pip_distance", s.cycle_pip_distance)?;
    d.set_item("events", &s.events)?;
    if track > 0 {
        d.set_item("pip", &s.pip)?;
        if track == 2 && !s.alpha.is_empty() {
            d.set_item("alpha", s.alpha.chunks(p).collect::<Vec<_>>())?;
            d.set_item("mu", s.mu.chunks(p).collect::<Vec<_>>())?;
            d.set_item("mu2", s.mu2.chunks(p).collect::<Vec<_>>())?;
            d.set_item("lossy", false)?;
        } else {
            d.set_item("alpha_entries", &s.alpha_entries)?;
            d.set_item("omitted_mass", &s.omitted_mass)?;
            d.set_item("row_threshold", 2. / p as f64)?;
            d.set_item("entry_threshold", 1. / p as f64)?;
            d.set_item("lossy", true)?;
        }
    }
    let _ = l;
    Ok(d)
}
pub fn export<'py>(
    py: Python<'py>,
    d: &Diagnostics,
    l: usize,
    p: usize,
    track: u8,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("stop_reason", &d.stop_reason)?;
    out.set_item("effective_criterion", &d.effective_criterion)?;
    out.set_item("fallback_iterations", &d.fallback_iterations)?;
    out.set_item("elbo_decrease_count", d.decrease_count)?;
    out.set_item("nonfinite_elbo_count", d.nonfinite_count)?;
    if let Some(last) = &d.last {
        out.set_item("last", snapshot(py, last, l, p, 0)?)?;
    }
    let trace = PyList::empty(py);
    for row in &d.trace {
        trace.append(snapshot(py, row, l, p, track)?)?;
    }
    out.set_item("trace", trace)?;
    let trim = PyDict::new(py);
    trim.set_item("phase", "post_loop_post_sigma2_update_and_trim")?;
    trim.set_item("trimmed_components", &d.trimmed_components)?;
    trim.set_item("max_abs_delta_alpha", d.trim_alpha_distance)?;
    trim.set_item("max_abs_delta_pip", d.trim_pip_distance)?;
    trim.set_item("sigma2", d.sigma2_after_loop)?;
    out.set_item("finalization", trim)?;
    Ok(out)
}
