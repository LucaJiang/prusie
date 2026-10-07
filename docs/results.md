# SusieResult schema

All indices are zero-based. m is the number of biological variants; q=m+1 when
an explicit null column is present, otherwise q=m. L is the returned effect
count, capped at the available columns. `as_dict()` returns a shallow mapping
without copying arrays. Results are mutable; callers own any copies they need.

| Field | Shape/type | Meaning |
|---|---|---|
| variant_ids | (m,) strings | Original caller SNP order, excluding null |
| alpha | (L,q) float64 | Single-effect posterior probabilities |
| mu | (L,q) float64 | Conditional effect means on fitted predictor scale |
| mu2 | (L,q) float64 | Conditional second moments, not variances |
| lbf_variable | (L,q) float64 | Natural-log component/SNP Bayes factors |
| lbf | (L,) float64 | Component log Bayes factors |
| V | (L,) float64 | Component prior variances after final trimming |
| sigma2 | float | Residual variance |
| pip | (m,) float64 | Marginal inclusion probability; null excluded |
| elbo | (niter,) float64 | IBSS objective trace before final low-V trim |
| niter, converged | int, bool | Iterations executed and convergence status |
| null_index | int or None | Last internal null-column index when enabled |
| sets | mapping | Credible-set data below |
| params | mapping | Actual model/options, including effective prior weights |
| backend_version | string | Installed native implementation version |
| warnings | list of strings | Numerical/model/convergence warnings |
| qc | mapping | Per-call input/transform diagnostics; not external alignment proof |
| variant_metadata | mapping of (m,) arrays | Caller metadata copied into result |
| X_column_scale_factors | (q,) float64 | Predictor scaling, including null when present |
| intercept | float | Original-scale intercept using supplied means; NaN if means unavailable |
| KL | (L,) float64 | Component divergence terms |
| XtXr | (q,) float64 | Crossproduct times posterior coefficients, not a residual vector |
| coef | (m,) property | Σ(alpha*mu) divided by original predictor scales |

`sets.cs[k]` contains members for original effect `sets.cs_index[k]`.
Indices need not be consecutive and return order is not a relabeling.
`coverage` and its equal alias `actual_coverage` are posterior mass of each
returned set. `requested_coverage` is the target. `purity` holds
min_abs_corr, mean_abs_corr and median_abs_corr for complete retained sets.
Complete purity includes every selected pair; n_purity never subsamples the
supplied matrix. No correlation matrix is returned inside sets.

No-CS returns empty member/index/coverage/purity collections while preserving
PIP and the requested threshold. `coverage=None` or `min_abs_corr=None` disables
CS construction. A CS is not a marginal region posterior and its posterior mass
is not empirical frequentist coverage. Nonconvergence is independent of no-CS.

After iteration, V&lt;prior_tol components have zero V/moments/BFs/KL and alpha
restored to the normalized prior. ELBO and XtXr preserve pre-trim iteration
values. Numerical differences in PIPs, intermediate arrays, credible sets and Bayes
factors are reported separately in [numerical accuracy](numerical_accuracy.md). Persist arrays with NPZ plus JSON metadata if
needed; pickle and R are not required by this result contract.

## Saving results

Prefer NPZ arrays plus JSON metadata for portable results. Python pickles retain
class module paths and require a compatible environment. For trusted results
saved by an older package, export portable arrays using that original environment;
old class paths are not automatically remapped.
