# Model support

prusie implements the Gaussian SuSiE paths of susieR 0.16.6. The Python API
accepts NumPy-compatible arrays and returns explicit zero-based result indices.

| Capability | Supported behavior |
| --- | --- |
| Summary statistics | Signed z + signed R + known n; or beta/SE + R, with optional known phenotype variance |
| Sufficient statistics | Centered XᵀX, Xᵀy, yᵀy and n; optional original means for the intercept |
| Missing sample size | Explicit noncentrality-parameter likelihood with a warning |
| Effect priors | Scalar Gaussian variance; fixed, Brent-optimized, EM or simple update; normalized variant weights and optional null weight |
| Residual variance | Fixed or estimated; RSS defaults to fixed, sufficient statistics to estimated |
| Summaries | PIP, posterior means/second moments, natural-log Bayes factors, ELBO, component variances, credible sets and full-pair purity |
| Computation | CPU float64; sequential IBSS; dense signed matrices, including singular LD |

## Settings and result semantics

RSS defaults are `L=10`, `max_iter=50` and `tol=0.0001`; the sufficient-statistic
iteration maximum is 100. Evaluation settings are explicitly recorded separately.
Known-n RSS uses the Wald finite-sample adjustment. Supplied `var_y` sets response
variance, including with z-only input.

Finite ELBO convergence requires a nonnegative improvement below tolerance.
Nonfinite increments use the alpha/PIP fallback with warnings. After iteration,
components with V&lt;prior_tol are reset to the normalized prior with zero variance,
moments, log BFs and KL. The stored trace and XtXr describe the iterations before
that trimming. Degenerate columns retain the reference's no-information behavior.

Matrix-input purity examines the complete selected submatrix even when
`n_purity` is finite. Returned `coverage` is actual component posterior mass;
`requested_coverage` is the target. Empty credible-set collections retain PIPs.
Setting `coverage=None` or `min_abs_corr=None` disables set construction.

## Boundaries

The package does not perform allele harmonization, imputation, variant filtering
or LD repair. Matrix finite/symmetry/diagonal/range checks remain active with
`check_input=False`. Invalid or nonfinite inputs raise errors; nonpositive
estimated residual variance raises an error.

Individual-level regression, NIG/mixture priors, infinitesimal effects, greedy
components, slot priors, LD-mismatch models, low-rank X input, trend filtering and
kriging diagnostics are outside this API. Nondefault initialization, refinement,
MAF filtering, verbose output and trace snapshots raise `NotImplementedError`.
No GPU backend is provided. The declared Python minimum is 3.10; executed local
package checks cover CPython 3.12 on GNU Linux x86-64. Other platforms require
build and numerical validation.

See [statistical model](model.md), [API](api.md),
[numerical accuracy](numerical_accuracy.md) and [source mapping](reference_mapping.md).
