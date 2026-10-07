/* Optional Linux R BLAS identity probe; build with R CMD SHLIB -ldl. */
#define _GNU_SOURCE
#include <R.h>
#include <Rinternals.h>
#include <dlfcn.h>
#include <time.h>

SEXP prusie_monotonic(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC,&value)) error("clock_gettime failed");
    return ScalarReal(value.tv_sec + value.tv_nsec * 1e-9);
}

SEXP prusie_blas_info(void) {
    SEXP out = PROTECT(allocVector(VECSXP, 4));
    SEXP names = PROTECT(allocVector(STRSXP, 4));
    const char *keys[] = {"threads", "config", "dgemm_library", "dpotrf_library"};
    for (int i=0; i<4; i++) SET_STRING_ELT(names,i,mkChar(keys[i]));
    int (*threads)(void) = dlsym(RTLD_DEFAULT,"openblas_get_num_threads");
    char *(*config)(void) = dlsym(RTLD_DEFAULT,"openblas_get_config");
    SET_VECTOR_ELT(out,0,ScalarInteger(threads ? threads() : NA_INTEGER));
    SET_VECTOR_ELT(out,1,mkString(config ? config() : "not detected"));
    const char *symbols[] = {"dgemm_", "dpotrf_"};
    for (int i=0; i<2; i++) {
        void *symbol = dlsym(RTLD_DEFAULT,symbols[i]);
        Dl_info info;
        SET_VECTOR_ELT(out,i+2,mkString(symbol && dladdr(symbol,&info) ? info.dli_fname : "unknown"));
    }
    setAttrib(out,R_NamesSymbol,names);
    UNPROTECT(2);
    return out;
}
