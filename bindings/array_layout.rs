//! Check byte geometry before NumPy constructs any typed ndarray view.
use ndarray::Dimension;
use numpy::{Element, PyArray, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn ensure_layout<T: Element, D: Dimension>(array: &Bound<'_, PyArray<T, D>>) -> PyResult<()> {
    // A later Python numeric argument conversion may mutate dtype/rank after
    // initial argument extraction. Recheck the typed wrapper before any view.
    let array = array.as_any().downcast::<PyArray<T, D>>()
        .map_err(|_| PyValueError::new_err("Native array dtype or rank changed during argument conversion"))?;
    let item = std::mem::size_of::<T>() as isize;
    let alignment = std::mem::align_of::<T>();
    let mut span = 0usize;
    let valid = !array.data().is_null() && (array.data() as usize) % alignment == 0
        && array.shape().iter().zip(array.strides()).all(|(&size, &stride)| {
            // numpy 0.23 adjusts/negates negative strides even on empty axes.
            if size == 0 || stride % item != 0 { return false; }
            let Some(abs_stride) = stride.checked_abs() else { return false; };
            let Some(extent) = (size - 1).checked_mul(abs_stride as usize) else { return false; };
            let Some(total) = span.checked_add(extent) else { return false; };
            span = total;
            total <= isize::MAX as usize - std::mem::size_of::<T>()
        });
    if !valid {
        return Err(PyValueError::new_err("Native arrays require nonempty, aligned, element-compatible byte strides and bounded geometry"));
    }
    Ok(())
}
