//! Vector exponentials with a portable fallback; summation stays in Rust.
#[cfg(has_vector_math)]
use std::sync::OnceLock;

#[cfg(has_vector_math)]
// SAFETY: these declarations match the linked C shim's int/double/size_t ABI.
// Initialization is serialized below and keeps resolved symbols alive; callers
// check CPU support and pass exclusively borrowed slices of the stated length.
unsafe extern "C" {
    fn prusie_vector_init() -> i32;
    fn prusie_scalar_environment_required() -> i32;
    fn prusie_exp_shift_sse2(values: *mut f64, count: usize, maximum: f64);
    fn prusie_exp_shift_avx2(values: *mut f64, count: usize, maximum: f64);
}

#[cfg(has_vector_math)]
fn backend() -> i32 {
    static BACKEND: OnceLock<i32> = OnceLock::new();
    *BACKEND.get_or_init(|| {
        // SAFETY: C initialization is serialized by OnceLock, and its
        // immutable symbol pointers remain alive for the process lifetime.
        let available = unsafe { prusie_vector_init() };
        if available & 2 != 0 && std::is_x86_feature_detected!("avx2") {
            2
        } else if available & 1 != 0 {
            1
        } else {
            0
        }
    })
}

pub(crate) fn backend_name() -> &'static str {
    #[cfg(has_vector_math)]
    {
        // SAFETY: this C function only inspects the current thread's FP controls.
        if unsafe { prusie_scalar_environment_required() } != 0 {
            return "portable_scalar_exp_caller_fenv";
        }
        match backend() {
            2 => "glibc_libmvec_avx2_exp",
            1 => "glibc_libmvec_sse2_exp",
            _ => "portable_scalar_exp",
        }
    }
    #[cfg(not(has_vector_math))]
    {
        "portable_scalar_exp"
    }
}

/// Sample caller FP controls once for a private synchronous SER search.
#[cfg(all(has_vector_math, target_arch = "x86_64"))]
pub(crate) fn group_fill_avx2_allowed() -> bool {
    // SAFETY: the C helper only inspects the calling thread's FP controls.
    backend() == 2 && unsafe { prusie_scalar_environment_required() } == 0
}

pub(crate) fn exp_shift(values: &mut [f64], maximum: f64) {
    #[cfg(has_vector_math)]
    {
        let backend = backend();
        // SAFETY: this C function only reads the calling thread's FP controls.
        let scalar_environment = unsafe { prusie_scalar_environment_required() } != 0;
        if !scalar_environment {
            if backend == 2 {
                // SAFETY: AVX2 was detected above; C accesses exactly this
                // exclusively borrowed contiguous slice, including its tail.
                unsafe {
                    prusie_exp_shift_avx2(values.as_mut_ptr(), values.len(), maximum);
                }
                return;
            }
            if backend == 1 {
                // SAFETY: x86_64 guarantees SSE2; C accesses exactly this
                // exclusively borrowed contiguous slice, including its tail.
                unsafe {
                    prusie_exp_shift_sse2(values.as_mut_ptr(), values.len(), maximum);
                }
                return;
            }
        }
    }
    for x in values {
        *x = (*x - maximum).exp();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shifted_exponentials_cover_underflow_and_vector_tail() {
        let original = [
            1.,
            0.,
            -1.,
            -10.,
            -100.,
            -745.,
            -1000.,
            f64::NEG_INFINITY,
            2.,
        ];
        let mut values = original;
        exp_shift(&mut values, 2.);
        for (actual, old) in values.iter().zip(original) {
            let expected = (old - 2.).exp();
            assert!(
                (actual - expected).abs() <= 4. * f64::EPSILON * expected.max(f64::MIN_POSITIVE)
            );
        }
    }
}
