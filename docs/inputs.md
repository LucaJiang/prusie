# Inputs, alignment and priors

Choose the input route that matches the available study data and apply any
required covariate adjustment before fitting. For association summary
statistics, align the summaries and linkage disequilibrium (LD) matrix to the
same variant order and effect alleles.

prusie checks array shapes and numerical values, but these checks do not
establish allele alignment or agreement between reference LD and the study.
[Statistical model](model.md#from-summaries-to-working-sufficient-statistics)
explains how the supplied summaries determine the working likelihood and
effect scale.

## Supported model and input routes

prusie fits Gaussian SuSiE on CPU using float64 arithmetic. The input route
determines how regression quantities and effect scales are obtained:

| Entry point | Inputs | Interpretation |
| --- | --- | --- |
| `susie` | `X`, `y` | Individual-level regression with optional centering and predictor standardization |
| `susie_suff_stat` | `XtX`, `Xty`, `yty`, `n` | Sufficient statistics from the same centered data |
| `susie_rss` | `z` or both `bhat` and `shat`, plus `R`, `n` | Standardized working scale; Wald adjustment by default |
| `susie_rss` | `bhat`, `shat`, `R`, `n`, `var_y` | Reconstructs regression scale; `coef` returns effects per original predictor unit in phenotype units |
| `susie_rss` | `z`, `R`, `n`, `var_y` | Sets `yty` and default initial variances; leaves z-based `XtX` and `Xty` unchanged |
| `susie_rss` | `z` or both `bhat` and `shat`, plus `R`, without `n` | Noncentrality effects; warns and uses `prior_variance` |
| `susie_rss_lambda` | `z`, `R` or reference `X`, and `lambda_` | RSS with a fixed or estimated isotropic covariance component |
| `susie_ser` | `z` or both `bhat` and `shat` | Single-effect regression without LD |

For `susie`, rows of `X` are observations and columns are predictors.
`intercept=True` centers `X` and `y`; `standardize=True` enables predictor
standardization. SciPy sparse input stays sparse. `na_rm=True` removes rows
with missing outcomes; predictor entries must be finite.

Single-panel `susie_rss` accepts either `z` or both `bhat` and `shat`; effect
estimates give `z = bhat / shat`. With known `n`, `z_method="wald"` applies
the finite-sample Wald adjustment; `z_method="score"` skips it. Supply `var_y`
as the measured sample phenotype variance. Without `n`, it is checked and
recorded but does not change the likelihood. The
[model derivation](model.md#from-summaries-to-working-sufficient-statistics)
explains these transformations and their scale assumptions.

`susie_rss` also accepts reference genotypes `X` instead of `R`; the reference
row count may differ from the association sample size `n`. Ordinary
single-panel products do not require dense LD. Finite-reference or mismatch
corrections, multiple panels, RSS-lambda, and reconstruction using `bhat`,
`shat` and `var_y` form dense matrices from reference `X`.

Multiple reference panels require `n` and a shared `z` vector. They and the
`R_finite` and `R_mismatch` options change the RSS model and require aligned
inputs. See [additional model options](model.md#additional-model-options)
and [API controls](api.md#shared-fitting-options).

## Shapes and numerical checks

For `susie_rss` with `R` and for `susie_suff_stat`, let $p$ be the number of
input variants. The vectors `z`, `bhat` and `Xty` have length $p$, and `R`
and `XtX` have shape $(p,p)$. Supplied `variant_ids` are converted to strings
and must contain $p$ distinct nonempty IDs. Each `variant_metadata` column
must have length $p$ and is copied into the result.

For RSS effect-estimate inputs, `bhat` must be finite. `shat` is a strictly
positive scalar or a length-$p$ vector of positive standard errors, without
missing values. Positive infinity is accepted: a finite `bhat` then gives
`z = 0`. For z-score input, missing values (`NaN`) are replaced with zero
with a `RuntimeWarning`; infinite z-scores are rejected.

`susie_suff_stat` requires crossproducts from the same centered data, a finite
nonnegative `yty`, and finite $n>1$. With the default `check_input=False`,
missing `Xty` entries are replaced with zero with a `RuntimeWarning`;
`check_input=True` rejects them. Infinite `Xty` entries are always rejected.
For `yty=0`, `Xty` must be zero, `residual_variance` must be explicitly
positive, and `estimate_residual_variance=False` is required. `X_colmeans`
and `y_mean` recover the original-scale intercept; omitting either leaves it
`NaN`. See [data, notation and scale](model.md#data-notation-and-scale) for
centering, standardization and coefficient units.

`R` must represent signed LD correlations. Singular LD is allowed. Matrix
entries must be finite and diagonals nonnegative; ordinary RSS does not
separately check for a unit diagonal or entries in $[-1,1]$. Supplying
correctly scaled LD remains the caller's responsibility.

For ordinary z-score RSS, symmetry is checked exactly on the working
`XtX = (n - 1) * R`, or `XtX = R` without `n`. Unequal off-diagonal pairs
trigger a `RuntimeWarning` and averaging as `(XtX + XtX.T) / 2`, with scaling
performed before averaging. Direct `XtX` uses the same exact check and
averaging rule. With `bhat`, `shat`, `n` and `var_y`, the reconstructed
`XtX` is always averaged. Finite-value and diagonal checks also apply to the
averaged matrix; overflow raises `ValueError`.

Caller arrays are preserved by default. The opt-in `overwrite_ld=True` can
consume `R` even if the call later fails. Reload the original LD before reuse;
[Optional LD workspace](api.md#optional-ld-workspace) gives the supported
inputs and preservation contract.

Shape, finite-value and diagonal checks always run. `check_input=True` adds
an eigendecomposition before predictor standardization. It rejects negative
eigenvalues at or below `-r_tol`; zero eigenvalues remain legal, including
with `r_tol=0`. Let $\epsilon$ be float64 machine epsilon. A `RuntimeWarning`
is issued when the fraction of squared `Xty` norm projected onto eigenvectors
with eigenvalues at most $\epsilon$ exceeds $\epsilon$. This relative check
has no absolute cutoff on `Xty` magnitude. No ridge adjustment, eigenvalue
clipping or positive-semidefinite repair is applied.

## Variant order and alleles

`variant_ids` label the input order; the fitter does not match association
summaries to LD by ID. Apply each permutation or subset to both axes of `R`
or `XtX`, all variant-level vectors (including `prior_weights` and `maf`),
IDs and metadata columns. For `X`, apply the selection to its columns.

Record genome build, counted allele, association effect allele, population,
reference sample selection and variant order. Resolve ambiguous strand or
palindromic alleles using appropriate evidence before fitting; the fitter
does not infer allele orientation.

For aligned NumPy arrays `z` and `R`, let `signs` be a length-$p$ vector
containing `-1` for variants whose effect allele is reversed and `+1`
otherwise. Apply the signs to z-scores and both LD axes:

```python
z_new = signs * z
R_new = signs[:, None] * R * signs[None, :]
```

For effect-estimate inputs, multiply `bhat` by the same `signs` and leave
`shat` unchanged. Update the allele metadata consistently.

Filtering is explicit. When `maf` is supplied, `susie_rss` and
`susie_suff_stat` retain variants with `maf > maf_thresh` (default
`maf_thresh=0.0`), applying the same selection to the working matrix,
summary vector, prior weights, IDs and metadata.

## Priors and zero-information columns

Let $L$ be the requested maximum number of single-effect components.
In `susie`, `susie_suff_stat` and `susie_rss`, `scaled_prior_variance`
accepts a finite nonnegative scalar or length-$L$ vector, including values
above one. The
[model definition](model.md#a-sum-of-single-effects) explains its scale.
Without `n`, `susie_rss` uses `prior_variance` and ignores
`scaled_prior_variance`.

`prior_weights` supply one weight per input variant, in the supplied order.
They must be finite and nonnegative with a positive finite total; default
weights are uniform. Normalization gives prior location probabilities
$\pi_j$, shared across components.

A positive `null_weight` appends a column with zero crossproducts. Write
$w_0=\mathtt{null\_weight}$, with $0<w_0<1$. For $p$ retained variants with
weights $w_1,\ldots,w_p$ before normalization, the augmented vector is

$$
\bigl((1-w_0)w_1,\ldots,(1-w_0)w_p,w_0\bigr).
$$

The whole vector is normalized. The normalized null prior equals $w_0$ when
the variant weights sum to one; rescaling unnormalized weights can change
it. For these three fitters, `null_weight=0` (the default) or `None` adds no
column. Supply only biological variant IDs and
metadata. For `susie_rss_lambda`, the cap on $L$ remains the variant count
after MAF filtering, excluding the null column.

The ordinary Gaussian SER uses $\log(\pi_j+\sqrt{\epsilon})$ for prior
weighting, where $\epsilon$ is float64 machine epsilon. A supplied zero
weight can therefore receive positive posterior probability. To exclude a
variant, remove it consistently from the inputs as described under
[Variant order and alleles](#variant-order-and-alleles).

A zero working crossproduct diagonal produces nonfinite internal marginal
quantities. The ordinary Gaussian SER assigns that column log BF zero and
zero conditional moments; its posterior location probability can remain
positive. For finite sampling variances, log Bayes-factor calculations use
a floor of $\epsilon$.

`lbf_variable` stores log Bayes factors before variant weighting; posterior
probabilities incorporate the weights. [Results](results.md#iterations-and-finalization)
describes component activity thresholds and final trimming.

`model_init` (alias `s_init`) accepts a previous fit or a working-coordinate
mapping containing `alpha`, `mu` and `mu2`. New fits may use different
summaries; supplied IDs, allele labels, scales and null layout are checked
for agreement. Component rows keep their input order. If the state has more
components than requested $L$, its count is retained with a warning. Source
data retention is unnecessary for a new fit;
[continuation](post_analysis.md#continue-first-then-consider-refinement)
requires the original retained context and integrity checks.

## Choosing variance and iteration settings

For the ordinary Gaussian model, `susie_rss` fixes residual variance by
default. Estimating it requires in-sample LD consistent with the association
summaries; requesting estimation emits a warning. `susie` and
`susie_suff_stat` enable estimation by default. A nonpositive or nonfinite
residual-variance estimate raises `ValueError` before
`residual_variance_lowerbound` and `residual_variance_upperbound` are applied.
See [residual-variance updates](model.md#elbo-and-residual-variance)
for the update and its order relative to stopping.

In the ordinary solver, `check_prior=True` (the `susie_rss` default) checks
component prior variances after each sweep, before updating residual variance.
Let $\mathbf Q$ and
$\mathbf g$ denote the working predictor and predictor–response crossproducts,
and $\sigma^2$ the current residual variance. When any $Q_{jj}>0$, the check
rejects a component prior variance exceeding

$$
100\max_{j:Q_{jj}>0}\frac{g_j^2}{Q_{jj}\sigma^2}.
$$

Passing this diagnostic does not establish allele alignment or agreement
between LD and the association study.

`susie_rss` defaults to `max_iter=50` and `susie_suff_stat` to
`max_iter=100`. These cap IBSS sweeps per fit; stopping can occur earlier.
Both default to `tol=1e-4`. Passing `tol=None` selects `1e-6` with
`estimate_residual_method="NIG"` and `1e-4` otherwise. See
[stopping rules](model.md#stopping-rules) and
[entry-point defaults](model.md#defaults-by-entry-point) for model-specific
settings, and [Results](results.md#iterations-and-finalization) for returned
status and finalization.

A dense float64 matrix with $p$ variants requires $8p^2$ bytes before working
arrays and results. See [memory use](optimization.md#memory) for copies and
additional storage.
