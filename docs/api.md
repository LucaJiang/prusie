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

Input routes, shapes and alignment requirements are centralized in
[Inputs](inputs.md#supported-model-and-input-routes). The
[model page](model.md#from-summaries-to-working-sufficient-statistics) gives
separate working-scale formulas for each route.

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

`XtX`, `Xty`, `yty` and `n` describe the same centered data, as detailed in
[Inputs](inputs.md#shapes-and-numerical-checks). `X_colmeans` and `y_mean` are
optional original means for recovering the intercept.

## Shared fitting options

| Option | Default | Meaning |
| --- | --- | --- |
| `L` | 10 | Maximum number of effects, capped at internal predictor count |
| `scaled_prior_variance` | 0.2 | Initial component variance divided by yty/(n−1); ≤1 when standardized |
| `residual_variance` | None | Initial/fixed residual variance; None uses yty/(n−1) |
| `estimate_prior_variance` | True | Whether to update component V |
| `estimate_prior_method` | `'optim'` | Continuous marginal-likelihood search; alternatives `'EM'` and `'simple'` follow the [model's update schedule](model.md#sequential-ibss-and-variance-estimation) |
| `check_null_threshold` | 0 | Log-BF threshold for the applicable null-V comparison |
| `prior_weights` | None | Uniform prior by default; finite nonnegative weights with positive total otherwise |
| `null_weight` | 0 | Optional last internal null column, in [0,1) |
| `standardize` | True | Scale predictors to unit sample variance |
| `prior_tol` | 1e-9 | Final trim/PIP threshold; precise inequalities and separate CS activity are in [Results](results.md#iterations-and-finalization) |
| `coverage` | 0.95 | Requested credible-set posterior mass |
| `min_abs_corr` | 0.5 | Minimum complete-pair absolute correlation for retained sets |
| `n_purity` | None | Accepted option; matrix-input purity always uses all selected pairs |
| `check_input` | False | Additional eigendecomposition/projection diagnostics; basic validity checks stay active |
| `r_tol` | 1e-8 | Tolerance for optional sufficient-statistic eigenvalue checks |
| `check_prior` | True in RSS, False in sufficient statistics | Check for unreasonably large estimated prior variance |
| `variant_ids` | None | Distinct nonempty strings; omitted IDs become positional labels |
| `variant_metadata` | None | Aligned length-p columns copied into the result |

RSS defaults to `max_iter=50`; sufficient statistics to 100. Both default to
`tol=0.0001`. See [Results](results.md#iterations-and-finalization) for the stopping criterion.
`estimate_residual_variance` defaults to False in RSS and True in sufficient
statistics. Use fixed variance with external LD unless estimation is justified.
`coverage=None` or `min_abs_corr=None` disables CS construction while retaining PIP.

The following exposed options accept only their listed defaults:

| Parameter | Allowed value | Error for another value |
| --- | --- | --- |
| `s_init` | None | `NotImplementedError: s_init is not implemented` |
| `refine` | False | `NotImplementedError: refine is not implemented` |
| `track_fit` | False | `NotImplementedError: track_fit is not implemented` |
| `verbose` | False | `NotImplementedError: verbose is not implemented` |
| `maf`, `maf_thresh` (sufficient statistics) | None, 0 | `NotImplementedError: maf is not implemented` |
| `z_ld_weight` (RSS) | 0 | `NotImplementedError: Nonzero z_ld_weight is not implemented` |

Invalid shapes, nonfinite values and parameter ranges raise `ValueError`.
`L`, `max_iter` and supplied `n_purity` must be positive integers; `tol`,
`prior_tol`, `r_tol` and `check_null_threshold` must be finite and nonnegative.
`coverage` lies in (0, 1], `min_abs_corr` in [0, 1], or either may be None.
In missing-n mode, `prior_variance` (default 50) replaces
`scaled_prior_variance`; changing the latter from 0.2 raises an error.

Full signatures and docstrings are available through Python `help()` and the
[API source](https://github.com/LucaJiang/prusie/blob/main/src/prusie/api.py).

## SusieResult

Both fitting functions return a `SusieResult`, including `pip`, `alpha`, `mu`,
`mu2`, `lbf_variable`, `lbf`, `V`, `sigma2`, `elbo`, `KL`, `sets`, `niter`,
`converged` and execution metadata. `fit.coef` gives posterior coefficient means on the scale defined by the input
route. `fit.as_dict()` returns a shallow mapping; it does not copy
arrays. See [results](results.md) for shapes and finalization details.

## Posterior helpers

`prusie.posterior.marginal_pip(alpha, V, *, null_index=None, prior_tol=1e-9)`
computes marginal inclusion probabilities from active components and excludes
the optional null position.

`prusie.posterior.credible_sets` accepts alpha, V and a signed correlation
matrix, with keyword options `coverage`, `min_abs_corr`, `null_index`,
`n_purity`, `covariance_scales` and `covariance_multiplier`. See [Results](results.md#credible-set-fields) for original component IDs,
selection order, posterior mass and aliases.
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
