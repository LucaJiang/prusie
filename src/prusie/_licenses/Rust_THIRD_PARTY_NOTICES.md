# Current implementation and historical provenance

The current target is official susieR0.16.6, commit
8e56a8e038e989856d106d9ca5175cc664fea9d2. See docs/reference_mapping.md in
the source distribution for the current constructor, SER, IBSS and CS mapping,
and the root THIRD_PARTY_NOTICES.md for retained upstream licenses.

The current implementation contains runtime-guarded SIMD intrinsics and an
optional vector-math FFI, with scalar fallbacks. It does not promise R's exact
accumulation order. Matrix operators and exact signed-column redundancy can
avoid the historical dense copy; their complete fallback preserves the input
values. The historical architectural statements below (no unsafe native code,
one dense copy and matched reduction order) describe the original0.14.2
implementation and are not current architecture or safety claims.

The following original attribution and audit record is retained for provenance.

# Historical Rust numerical source provenance

`core.rs` independently implements the mathematical operations and update
semantics in CRAN susieR 0.14.2, whose locked source is retained at
`vendor/upstream/susieR`. The package license is BSD-3-Clause; the complete
upstream copyright is preserved in `LICENSE-susieR`, with the BSD-3-Clause
license text in `LICENSE-susieR-BSD-3-clause`.

`optimizer.rs` adapts the scalar `Brent_fmin` routine from R's
`src/library/stats/src/optimize.c`, R 4.4 branch, retrieved on 2026-10-01 from
https://svn.r-project.org/R/branches/R-4-4-branch/src/library/stats/src/optimize.c.
This routine implements Richard Brent's *Algorithms for Minimization without
Derivatives* (1973), with R's step selection and stopping rules. Copyright
1995–1996 Robert Gentleman and Ross Ihaka; 2003–2004 The R Foundation;
1998–2023 The R Core Team. The adapted optimizer is GPL-2.0-or-later;
the full license is provided in `LICENSE-R`. The enclosing package distribution
uses GPL-3.0-or-later, which is compatible with this license.

R's finite log-normal-density operation ordering was audited at
https://svn.r-project.org/R/branches/R-4-4-branch/src/nmath/dnorm.c.
No R library is required at build time or runtime.

# Mapping to locked susieR functions

| Native implementation | Locked susieR source/function |
| --- | --- |
| `fit` initialization | `R/initialize.R`: `init_setup`, `init_finalize` |
| `fit` component loop | `R/update_each_effect_ss.R`: `update_each_effect_ss` |
| `single_effect` | `R/single_effect_regression_ss.R`: `single_effect_regression_ss` |
| `single_effect` optimization | `R/single_effect_regression.R`: `optimize_prior_variance`, `loglik` |
| component KL | `R/elbo_ss.R`: `SER_posterior_e_loglik_ss` |
| `expected_residual_sum_squares` | `R/elbo_ss.R`: `get_ER2_ss` |
| `fit` objective | `R/elbo_ss.R`: `Eloglik_ss`, `get_objective_ss` |
| `fit` residual update | `R/estimate_residual_variance.R`: `estimate_residual_variance_ss` |
| `fit` stopping/prior check | `R/susie_ss.R`: `susie_suff_stat` |

The core follows the upstream additive √ε prior-weight floor, log posterior
odds 0 for zero-diagonal columns, signed ELBO improvement stopping rule,
post-objective residual update, log-V bounds [−30, 15], old-V safeguard,
and exact-zero comparison with `check_null_threshold`. EM updates V after
the posterior without recomputing it, as upstream does.

Dense products use ndarray's `general_mat_vec_mul`; there is no handwritten
matrix multiplication, BLAS ABI dependency, unsafe native code, or Python/R
callback in the iterative solver. The borrowed row-major input is copied once
into exact-valued column-major storage (8p² bytes). ndarray 0.16.1 then reduces
each strided row in increasing column order, matching the pinned R environment's
reference BLAS DGEMV. Its contiguous-row reduction instead groups eight partial
sums, which was measured to perturb later prior-variance optimization. This
layout choice preserves all input values and has a dedicated cancellation
regression test. R's matrix-vector BLAS dispatch was audited in R 4.4
`src/main/array.c`, `matprod`, at
https://svn.r-project.org/R/branches/R-4-4-branch/src/main/array.c.

Float64 compensated scalar sums approximate R's extended precision reductions;
the implementation does not claim bit-identical results across arbitrary R
BLAS libraries, compiler targets, or platforms.

Explicit native limitations: initial fits, refinement, and fit tracing belong
to the unsupported public option set. A nonpositive estimated residual variance
returns a descriptive error immediately; R checks only negativity at that
point and may fail later for an exact zero. Array-scale input validation is
also performed defensively at the Rust boundary.
