//! Optional system float64 vector exponential; scalar portable fallback.
use std::sync::OnceLock;

#[derive(Clone, Copy)]
enum Backend {
    Scalar,
    #[cfg(has_vector_math)]
    Sse2,
    #[cfg(has_vector_math)]
    Avx2,
}

fn backend() -> Backend {
    static BACKEND: OnceLock<Backend> = OnceLock::new();
    *BACKEND.get_or_init(|| {
        if std::env::var("PRUSIE_SER_MATH").as_deref() == Ok("scalar") {
            return Backend::Scalar;
        }
        #[cfg(has_vector_math)]
        {
            // SAFETY: OnceLock serializes initialization, and pointers are
            // immutable afterward. Missing library/symbols select scalar.
            let available = unsafe { prusie_vector_init() };
            if available & 2 != 0 && std::is_x86_feature_detected!("avx2") {
                return Backend::Avx2;
            }
            if available & 1 != 0 {
                return Backend::Sse2;
            }
        }
        Backend::Scalar
    })
}

pub(super) fn backend_name() -> &'static str {
    #[cfg(has_vector_math)]
    if !matches!(backend(), Backend::Scalar) && unsafe { prusie_scalar_environment_required() } != 0
    {
        return "portable_scalar_exp_caller_fenv";
    }
    match backend() {
        Backend::Scalar => "portable_scalar_exp",
        #[cfg(has_vector_math)]
        Backend::Sse2 => "optional_glibc_libmvec_sse2_masked_exp",
        #[cfg(has_vector_math)]
        Backend::Avx2 => "optional_glibc_libmvec_avx2_masked_exp",
    }
}

#[cfg(has_vector_math)]
extern "C" {
    fn prusie_vector_init() -> i32;
    fn prusie_scalar_environment_required() -> i32;
    fn prusie_exp_shift_sse2(values: *mut f64, count: usize, maximum: f64);
    fn prusie_exp_shift_avx2(values: *mut f64, count: usize, maximum: f64);
}

pub(super) fn exp_shift_inplace(values: &mut [f64], maximum: f64) {
    match backend() {
        Backend::Scalar => {
            for value in values {
                *value = (*value - maximum).exp();
            }
        }
        #[cfg(has_vector_math)]
        Backend::Sse2 => {
            // SAFETY: x86_64 guarantees SSE2. The C loop receives exclusive
            // Rust-owned storage and accesses exactly values.len() doubles.
            unsafe {
                prusie_exp_shift_sse2(values.as_mut_ptr(), values.len(), maximum);
            }
        }
        #[cfg(has_vector_math)]
        Backend::Avx2 => {
            // SAFETY: runtime CPU detection admits this function only on AVX2.
            // The C loop handles its scalar tail without unaligned vector casts.
            unsafe {
                prusie_exp_shift_avx2(values.as_mut_ptr(), values.len(), maximum);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::exp_shift_inplace;

    #[test]
    fn shifted_exponential_covers_extremes_and_vector_tails() {
        let inputs = [
            0.0,
            -1.0,
            -std::f64::consts::LN_2,
            -10.0,
            -700.0,
            -744.0,
            f64::NEG_INFINITY,
            -0.001,
            -0.5,
        ];
        for n in 0..=inputs.len() {
            let mut values = inputs[..n].to_vec();
            exp_shift_inplace(&mut values, 0.0);
            for (&actual, &input) in values.iter().zip(&inputs) {
                let expected = input.exp();
                assert!((actual - expected).abs() <= 3e-15 * expected + f64::from_bits(4));
            }
        }
        let mut shifted = [3.0, 2.0, 1.0, 0.0, -1.0];
        exp_shift_inplace(&mut shifted, 3.0);
        assert_eq!(shifted[0], 1.0);
        for (j, &value) in shifted.iter().enumerate() {
            assert!((value - (-(j as f64)).exp()).abs() <= 3e-15 * value);
        }
    }

    #[test]
    fn zero_range_mask_keeps_representable_subnormals() {
        let bound = -746.0_f64;
        let inputs = [
            f64::from_bits(bound.to_bits() + 1),
            bound,
            f64::from_bits(bound.to_bits() - 1),
            -745.14,
            -745.13,
            -745.0,
            -744.0,
            -710.0,
            -700.0,
            f64::NEG_INFINITY,
            -10000.0,
            0.0,
        ];
        let mut values = inputs;
        exp_shift_inplace(&mut values, 0.0);
        for (&actual, &input) in values.iter().zip(&inputs) {
            let expected = std::hint::black_box(input).exp();
            if input <= -746.0 {
                assert_eq!(actual.to_bits(), 0);
            } else {
                assert!((actual - expected).abs() <= expected * 3e-15 + f64::from_bits(4));
            }
        }
        assert!(values[5] > 0.0);
    }

    #[cfg(has_vector_math)]
    #[test]
    fn other_rounding_modes_use_scalar_without_changing_mode() {
        extern "C" {
            fn fegetround() -> i32;
            fn fesetround(mode: i32) -> i32;
        }
        struct Restore(i32);
        impl Drop for Restore {
            fn drop(&mut self) {
                unsafe {
                    fesetround(self.0);
                }
            }
        }
        let _restore = Restore(unsafe { fegetround() });
        // GNU x86_64 fenv constants: downward, upward, toward zero.
        for mode in [0x400, 0x800, 0xc00] {
            assert_eq!(unsafe { fesetround(mode) }, 0);
            let inputs = std::hint::black_box([-10000.0_f64, -746.0, -745.0, -1.0, 0.0]);
            let reference: Vec<_> = inputs
                .iter()
                .map(|&x| std::hint::black_box(x).exp())
                .collect();
            let mut values = inputs;
            exp_shift_inplace(&mut values, 0.0);
            assert_eq!(unsafe { fegetround() }, mode);
            for (&actual, &expected) in values.iter().zip(&reference) {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
    }

    #[cfg(has_vector_math)]
    #[test]
    #[allow(deprecated)]
    fn direct_simd_control_changes_preserve_scalar_behavior() {
        use std::arch::x86_64::{_mm_getcsr, _mm_setcsr};
        struct Restore(u32);
        impl Drop for Restore {
            fn drop(&mut self) {
                unsafe {
                    _mm_setcsr(self.0);
                }
            }
        }
        let original = unsafe { _mm_getcsr() };
        let initial_backend = super::backend_name();
        let _restore = Restore(original);
        // Independent SIMD rounding, flush-to-zero and denormals-are-zero.
        for mode in [0x2000, 0x4000, 0x6000, 0x8000, 0x0040, 0x8040] {
            unsafe {
                _mm_setcsr((original & !0xe040) | mode);
            }
            let inputs = std::hint::black_box([-10000.0_f64, -746.0, -745.0, -710.0, -1.0, 0.0]);
            let reference: Vec<_> = inputs
                .iter()
                .map(|&x| std::hint::black_box(x).exp())
                .collect();
            let mut values = inputs;
            exp_shift_inplace(&mut values, 0.0);
            assert!(super::backend_name().starts_with("portable_scalar_exp"));
            assert_eq!(unsafe { _mm_getcsr() } & 0xe040, mode);
            for (&actual, &expected) in values.iter().zip(&reference) {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
        unsafe {
            _mm_setcsr(original);
        }
        assert_eq!(super::backend_name(), initial_backend);
    }
}
