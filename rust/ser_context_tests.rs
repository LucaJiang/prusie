//! Tests that each SER search samples the current floating-point controls.
#![allow(deprecated)] // Test-only access to the caller's complete MXCSR controls.
use super::{SerBuffers, SerLikelihood, SerSampling};
use std::arch::x86_64::{_mm_getcsr, _mm_setcsr};

#[test]
#[allow(deprecated)]
fn every_constructor_resamples_caller_controls() {
    extern "C" {
        fn fegetround() -> i32;
        fn fesetround(mode: i32) -> i32;
    }
    struct Restore(i32, u32);
    impl Drop for Restore {
        fn drop(&mut self) {
            unsafe {
                fesetround(self.0);
                _mm_setcsr(self.1);
            }
        }
    }
    let original = Restore(unsafe { fegetround() }, unsafe { _mm_getcsr() });
    let beta = [4., -1., 0., 0.5, -0.25, 2., 0.1, -0.1, 1.];
    let diagonal = [99.; 9];
    let prior = [-2.1972245773362196; 9];
    let sampling = SerSampling::new(&diagonal, 1., true);
    let mut buffers = SerBuffers::default();
    for (rounding, mask) in [
        (0, 0),
        (0x400, 0),
        (0, 0),
        (0x800, 0),
        (0xc00, 0),
        (0, 0x8000),
        (0, 0),
        (0, 0x0040),
        (0, 0x8040),
        (0, 0x2000),
        (0, 0x4000),
        (0, 0x6000),
        (0, 0),
    ] {
        unsafe {
            _mm_setcsr(original.1);
            assert_eq!(fesetround(rounding), 0);
        }
        if mask != 0 {
            unsafe {
                _mm_setcsr((_mm_getcsr() & !0xe040) | mask);
            }
        }
        let expected = rounding == 0
            && mask == 0
            && super::vector_math::backend_name() == "optional_glibc_libmvec_avx2_masked_exp";
        let context = SerLikelihood::from_sampling_reusing(&beta, &sampling, &prior, &mut buffers);
        assert_eq!(context.group_fill_avx2, expected);
        context.return_buffers(&mut buffers);
        let context = SerLikelihood::new_reusing(&beta, &sampling.shat2, &prior, &mut buffers);
        assert_eq!(context.group_fill_avx2, expected);
        context.return_buffers(&mut buffers);
        assert_eq!(unsafe { fegetround() }, rounding);
    }
}
