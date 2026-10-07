# Python API

The top-level exports are `susie_rss`, `susie_suff_stat`, `SusieResult` and
`load_example`. All numerical inputs are converted to float64 as needed.
Result indices are zero-based. See [inputs](inputs.md) for alignment and
[results](results.md) for the complete return schema.

## susie_rss

Fit Gaussian SuSiE from signed association statistics and LD:

```python
fit = prusie.susie_rss(
    z=z, R=R, n=n,
    L=10, max_iter=50, tol=0.0001,
    estimate_residual_variance=False,
    coverage=0.95, min_abs_corr=0.5,
    variant_ids=variant_ids,
)
```

Choose one of the following input routes:

| Arguments | Interpretation |
| --- | --- |
| `z`, `R`, `n` | Signed z, signed correlation matrix and known sample size >1; finite-sample Wald adjustment |
| `bhat`, `shat`, `R`, `n` | Effect estimates and positive standard errors, converted to signed z; shat may be scalar or a vector |
| `bhat`, `shat`, `R`, `n`, `var_y` | Known phenotype variance, with original predictor-scale reconstruction |
| `z`, `R`, `n=None` | Explicit large-sample noncentrality likelihood, with a warning; `prior_variance=50` initializes V |

`z` is mutually exclusive with `bhat`/`shat`. `R` must match the complete input
order and effect alleles. A supplied `var_y` also sets response variance with
z-only inputs. Nonzero `z_ld_weight` is unsupported.

## susie_suff_stat

Fit from centered sufficient statistics:

```python
fit = prusie.susie_suff_stat(
    XtX=Q, Xty=g, yty=yty, n=n,
    L=10, max_iter=100, tol=0.0001,
    estimate_residual_variance=True,
    coverage=0.95, min_abs_corr=0.5,
    variant_ids=variant_ids,
)
```

Q is a symmetric p×p predictor crossproduct, g is a length-p predictor-response
crossproduct, yty is positive and n>1. Optional `X_colmeans` and `y_mean` recover
an original-scale intercept; it is NaN when means are unavailable.

## Shared fitting options

| Option | Default | Meaning |
| --- | --- | --- |
| `L` | 10 | Maximum number of effects, capped at internal predictor count |
| `scaled_prior_variance` | 0.2 | Initial component variance divided by yty/(n−1); ≤1 when standardized |
| `residual_variance` | None | Initial/fixed residual variance; None uses yty/(n−1) |
| `estimate_prior_variance` | True | Whether to update component V |
| `estimate_prior_method` | `'optim'` | `'optim'`, `'EM'` or `'simple'` |
| `check_null_threshold` | 0 | Log-BF threshold for the applicable null-V comparison |
| `prior_weights` | None | Uniform prior by default; finite nonnegative weights with positive total otherwise |
| `null_weight` | 0 | Optional last internal null column, in [0,1) |
| `standardize` | True | Scale predictors to unit sample variance |
| `prior_tol` | 1e-9 | Final trim and PIP threshold; CS activity uses its fixed 1e-9 threshold |
| `coverage` | 0.95 | Requested credible-set posterior mass |
| `min_abs_corr` | 0.5 | Minimum complete-pair absolute correlation for retained sets |
| `n_purity` | None | Accepted option; matrix-input purity always uses all selected pairs |
| `check_input` | False | Additional eigendecomposition/projection diagnostics; basic validity checks stay active |
| `r_tol` | 1e-8 | Tolerance for optional sufficient-statistic eigenvalue checks |
| `check_prior` | True in RSS, False in sufficient statistics | Check for unreasonably large estimated prior variance |
| `variant_ids` | None | Distinct nonempty strings; omitted IDs become positional labels |
| `variant_metadata` | None | Aligned length-p columns copied into the result |

RSS defaults to `max_iter=50`; sufficient statistics to 100. Both default to
`tol=0.0001`. The tolerance concerns finite nonnegative ELBO improvement.
`estimate_residual_variance` defaults to False in RSS and True in sufficient
statistics. Use fixed variance with external LD unless estimation is justified.
`coverage=None` or `min_abs_corr=None` disables CS construction while retaining PIP.

Nondefault `s_init`, `refine`, `track_fit` and `verbose` are unsupported.
The sufficient-statistic `maf`/`maf_thresh` filtering options are unsupported.
These requests raise `NotImplementedError`; invalid inputs raise `ValueError`.
Full signatures and docstrings are available through Python `help()` and the
[API source](https://github.com/LucaJiang/prusie/blob/main/src/prusie/api.py).

## SusieResult

Both fitting functions return a `SusieResult`, including `pip`, `alpha`, `mu`,
`mu2`, `lbf_variable`, `lbf`, `V`, `sigma2`, `elbo`, `KL`, `sets`, `niter`,
`converged` and execution metadata. `fit.coef` gives original-scale posterior
coefficient means. `fit.as_dict()` returns a shallow mapping; it does not copy
arrays. See [results](results.md) for shapes and finalization details.

## Posterior helpers

`prusie.posterior.marginal_pip(alpha, V, *, null_index=None, prior_tol=1e-9)`
computes marginal inclusion probabilities from active components and excludes
the optional null position.

`prusie.posterior.credible_sets` accepts alpha, V and a signed correlation
matrix, with keyword options `coverage`, `min_abs_corr`, `null_index`,
`n_purity`, `covariance_scales` and `covariance_multiplier`. Returned component
IDs retain their original identity. `coverage` is actual posterior mass;
`actual_coverage` is an equal alias and `requested_coverage` records the target.
The scale options are used when the matrix contains covariance rather than
correlation. Normal fitting constructs these arguments internally.

## load_example

```python
example = prusie.load_example()
fit = prusie.susie_rss(**example["inputs"], **example["parameters"])
```

The synthetic toy example is loaded as a mapping with `inputs`, `parameters` and `metadata`. Inputs contain
z, R, n and variant IDs; metadata includes the synthetic method, seed, true
effects and sample size. Input bytes are checked against packaged hashes and
returned arrays are owned by the caller. The loader works from any directory
without network access. See the [toy example tutorial](example.md).
