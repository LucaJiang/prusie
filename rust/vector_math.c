/* Optional GNU vector-function ABI shim. No fast-math compilation is used.
 * Memory belongs to a Rust Vec, never a borrowed Python input array. */
#include <math.h>
#include <stddef.h>
#include <string.h>
#include <immintrin.h>
#include <fenv.h>
#include <dlfcn.h>

typedef double pair_double __attribute__((vector_size(16)));
typedef double quad_double __attribute__((vector_size(32)));
typedef pair_double (*pair_exp_function)(pair_double);
typedef quad_double (*quad_exp_function)(quad_double);
static pair_exp_function pair_exp = NULL;
static quad_exp_function quad_exp = NULL;
static void *math_library = NULL;

static int scalar_environment_required(void) {
    /* fegetround covers the C environment; callers can also alter MXCSR
     * independently. Preserve scalar behavior for SIMD rounding changes and
     * caller-selected flush-to-zero/denormals-are-zero modes. Never modify it. */
    return fegetround() != FE_TONEAREST || (_mm_getcsr() & 0xe040u) != 0;
}

/* Read-only diagnostic; the hot kernels keep their inline static check. */
int prusie_scalar_environment_required(void) {
    return scalar_environment_required();
}

/* Called exactly once by Rust OnceLock. The optional system library and
 * resolved function pointers remain live for the process lifetime. No Python
 * memory or floating-point environment state is mutated during selection. */
int prusie_vector_init(void) {
    math_library = dlopen("libmvec.so.1", RTLD_NOW | RTLD_LOCAL);
    if (math_library == NULL) return 0;
    void *pair_symbol = dlsym(math_library, "_ZGVbN2v_exp");
    void *quad_symbol = dlsym(math_library, "_ZGVdN4v_exp");
    /* POSIX dlsym function-pointer conversion, expressed without aliasing. */
    _Static_assert(sizeof(pair_exp) == sizeof(pair_symbol), "POSIX function-pointer size");
    _Static_assert(sizeof(quad_exp) == sizeof(quad_symbol), "POSIX function-pointer size");
    memcpy(&pair_exp, &pair_symbol, sizeof(pair_exp));
    memcpy(&quad_exp, &quad_symbol, sizeof(quad_exp));
    if (pair_exp == NULL && quad_exp == NULL) {
        dlclose(math_library);
        math_library = NULL;
    }
    return (pair_exp != NULL) | ((quad_exp != NULL) << 1);
}

void prusie_exp_shift_sse2(double *values, size_t count, double maximum) {
    if (scalar_environment_required()) {
        for (size_t j = 0; j < count; ++j) values[j] = exp(values[j] - maximum);
        return;
    }
    size_t i = 0;
    const __m128d shift = _mm_set1_pd(maximum);
    const __m128d threshold = _mm_set1_pd(-746.0);
    for (; i + 2 <= count; i += 2) {
        __m128d x = _mm_sub_pd(_mm_loadu_pd(values + i), shift);
        const __m128d zero_result = _mm_cmple_pd(x, threshold);
        // exp(x) necessarily rounds to +0 for x≤−746. Mask those lanes
        // before the library call to avoid its exceptional-range fallback.
        // All representable subnormal exponentials still use the library.
        x = pair_exp(_mm_andnot_pd(zero_result, x));
        _mm_storeu_pd(values + i, _mm_andnot_pd(zero_result, x));
    }
    for (; i < count; ++i) values[i] = exp(values[i] - maximum);
}

__attribute__((target("avx2")))
void prusie_exp_shift_avx2(double *values, size_t count, double maximum) {
    if (scalar_environment_required()) {
        for (size_t j = 0; j < count; ++j) values[j] = exp(values[j] - maximum);
        return;
    }
    size_t i = 0;
    const __m256d shift = _mm256_set1_pd(maximum);
    const __m256d threshold = _mm256_set1_pd(-746.0);
    for (; i + 4 <= count; i += 4) {
        __m256d x = _mm256_sub_pd(_mm256_loadu_pd(values + i), shift);
        const __m256d zero_result = _mm256_cmp_pd(x, threshold, _CMP_LE_OQ);
        x = quad_exp(_mm256_andnot_pd(zero_result, x));
        _mm256_storeu_pd(values + i, _mm256_andnot_pd(zero_result, x));
    }
    for (; i < count; ++i) values[i] = exp(values[i] - maximum);
}
