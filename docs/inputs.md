# Inputs, alignment and priors

Align association statistics and LD to the same variant order and effect
alleles before fitting. prusie checks numerical arrays; the study design,
allele harmonization and covariate adjustment belong to the input preparation.

## Supported model and input routes

prusie fits Gaussian SuSiE on CPU in float64, using individual-level data,
summary statistics or centered sufficient statistics. Choose the route that preserves the scale
information available for the study:

| Inputs | Working representation and coefficient interpretation |
| --- | --- |
| `X`, `y` via `susie` | Dense or sparse individual data, with optional centering and standardization |
| `z`, `R`, `n` | Finite-sample Wald-adjusted summaries on a standardized working scale |
| `bhat`, `shat`, `R`, `n` | Forms signed z as bhat/shat, then uses the same standardized route |
| `bhat`, `shat`, `R`, `n`, `var_y` | Reconstructs predictor crossproducts using the known phenotype sample variance; `coef` returns original-scale coefficients |
| `z`, `R`, `n`, `var_y` | Sets response sum of squares and initial variance scales; leaves z-based crossproducts unchanged |
| `z` or `bhat/shat`, `R`, with n omitted | Noncentrality-parameter likelihood; emits a warning and uses `prior_variance` |
| `XtX`, `Xty`, `yty`, `n` | Centered sufficient statistics via `susie_suff_stat` |
| Summaries and reference `X` via `susie_rss` | Reference rows may differ from association n; ordinary single-panel products are matrix-free |
| `z`, `R` via `susie_rss_lambda` | RSS with fixed or estimated regularization |
| Summaries via `susie_ser` | One-effect regression without LD |

For individual-level data, rows are observations and columns are predictors.
SciPy sparse input remains sparse. `na_rm=True` removes missing-outcome rows;
predictor entries must be finite. `intercept=True` centers predictors and outcome.
Do not pass unadjusted genotypes and outcomes when covariate adjustment is required
by the scientific analysis: perform that adjustment before fitting.

Advanced RSS options include score-statistic preparation, multiple reference
panels, mismatch variance and finite-reference corrections. These are explicit
model choices; they are not automatic repairs of the ordinary RSS input route.
Reference `X` is converted to dense correlation matrices for finite-reference or
mismatch corrections, panel lists, RSS-lambda, and effect reconstruction using
`bhat`, `shat` and `var_y`.
See [the API](api.md#shared-fitting-options) for their controls.

The [model page](model.md#from-summaries-to-working-sufficient-statistics)
derives each transformation. Provide either z or both bhat and shat. `var_y`
is a measured sample phenotype variance, not a value inferred from a trait name
or normalized-looking statistics. In missing-n mode it is checked and recorded
but does not change the working likelihood.

## Shapes and numerical checks

For $p$ biological variants, z or bhat and `variant_ids` have length $p$;
R has shape $(p,p)$. `shat` is either a strictly positive scalar shared by all
variants or a length-$p$ vector of positive standard errors. Positive infinite
standard errors are accepted: a finite bhat then produces z = 0. Missing
effect estimates or standard errors remain invalid. IDs are distinct
nonempty strings. `variant_metadata` contains aligned length-$p$ columns copied
into the result; metadata does not itself establish allele correctness.

`R` should contain **signed correlation r**; supplying correctly scaled LD
remains the caller's responsibility. Singular LD is allowed. Every entry must
be finite and diagonal entries nonnegative. As in ordinary dense susieR RSS,
the fitter does not impose a separate unit-diagonal or correlation-range check.
For ordinary z/R input, symmetry follows susieR 0.16.6's dense
numeric path: off-diagonal entries of the working `XtX = (n - 1) * R` must
compare exactly equal. There is no numerical symmetry tolerance. With missing
n, this check applies to `XtX = R`. If the check fails, the API issues a
`RuntimeWarning` and uses `(XtX + XtX.T) / 2` internally before fitting.
Scaling precedes averaging. With bhat, shat, n and var_y, the reconstructed
original-scale XtX is always averaged, as in susieR. Caller arrays are never
modified. No ridge adjustment or eigenvalue clipping is applied.

`susie_suff_stat` requires a finite predictor crossproduct `XtX`,
nonnegative finite `yty`, and $n>1$. Missing z-scores are replaced with zero;
missing `Xty` entries are replaced with zero with a warning. Infinite summaries
are rejected. With `check_input=True`, `Xty` must be finite before the optional
projection check. Zero `yty` requires zero `Xty` and an explicitly fixed positive
residual variance. These quantities describe the same centered data.
`X_colmeans` and `y_mean` recover an original-scale intercept; missing means
leave it NaN. Predictor standardization is enabled by default, and `coef`
reverses its stored scales. Direct XtX uses the same exact check, warning and
internal averaging. Basic finite/diagonal checks always run, including after
averaging; overflow is rejected even where R's constructor permits it.
`check_input=True` adds an eigendecomposition, rejects negative eigenvalues at
or below `-r_tol`, and warns when the squared projection fraction onto eigenvectors
with eigenvalues at most machine epsilon exceeds machine epsilon. This check
has no absolute response-magnitude floor; zero eigenvalues remain legal when
`r_tol=0`. It does not
apply a positive-semidefinite repair or certify its external provenance.

## Variant order and alleles

Preserve genome build, counted allele, association effect allele, population,
reference sample selection and SNP order in the analysis record. Resolve
ambiguous strand or palindromic alleles from appropriate evidence before
fitting. A permutation must be applied to both R axes, every association vector
and every metadata column.

If allele recoding multiplies each effect by $d_j\in\{-1,+1\}$, apply the same
signs to z and both LD axes:

This illustrative fragment assumes aligned NumPy arrays `z`, `R` and a vector
`signs` containing the required +1 or −1 recoding for each variant:

```python
z_new = signs * z
R_new = signs[:, None] * R * signs[None, :]
```

For effect-estimate inputs, recode bhat in the same way and leave shat unchanged.
Update the allele metadata consistently. The API does not infer these signs or
automatically filter variants. Explicit `maf`/`maf_thresh` filtering applies before fitting.

## Priors and zero-information columns

`scaled_prior_variance` accepts a nonnegative scalar or length-L vector,
including values above one. With missing n, the likelihood instead uses
`prior_variance`, and the unused `scaled_prior_variance` is ignored. An omitted
`null_weight` or `null_weight=None` means no added null column. `tol=None`
resolves to the ordinary Gaussian default `1e-4`.

Warm starts can initialize a new fit on different summaries. A minimal
`s_init` mapping contains working-coordinate `alpha`, `mu` and `mu2`; supplied
variant/allele/scale metadata is checked rather than silently reordered or
transformed. New fits do not require a retained-data hash. Continuation still
requires the original retained data and its integrity checks. If the initial
state has more effects than requested L, its effect count is retained, as in
pinned susieR. Initialization retains original component row order; credible-set
return order does not reorder the supplied alpha matrix.

`prior_weights` are finite nonnegative variant weights with a positive finite
total. Default weights are uniform. With a null weight $w_0$, the raw vector
is augmented as $((1-w_0)w_1,\ldots,(1-w_0)w_p,w_0)$, then the **whole** vector
is normalized. Consequently the final null probability equals `null_weight`
when the biological weights already sum to one. Arbitrarily rescaling raw
weights in the presence of a null column can change that probability. The
null column is appended internally with zero crossproducts; provide only
biological IDs and metadata.

SER computes log weights using $\log(\pi_j+\sqrt{\epsilon})$, where
$\epsilon$ is float64 machine epsilon. Thus zero supplied weights are smoothed
rather than enforcing absolute exclusion. The likelihood uses sampling
variances floored at machine epsilon. A nonfinite marginal estimate or sampling
variance, as produced by a zero-information column, receives log BF zero and
zero conditional moments. Ordinary positive-variance updates follow the
[Gaussian formulas](model.md#one-single-effect-regression). A zero V also has
zero conditional moments for an informative column.

These priors enter fitting. `lbf_variable` contains Bayes factors before variant
weighting; posterior probabilities already incorporate the weights. Active
component thresholds and final trimming are specified once in
[Results](results.md#iterations-and-finalization).

## Choosing variance and iteration settings

RSS fixes residual variance by default, appropriate when using external LD.
Estimating it requires in-sample LD consistent with the association summaries;
requesting it through RSS emits a warning. Sufficient-statistic fitting enables
estimation by default. A nonpositive or nonfinite estimate raises an error.

With `check_prior=True`, the prior-variance sanity check uses the **current**
residual variance at the post-sweep check, before that sweep's optional residual
variance update. In working sufficient-statistic coordinates $Q$ and $u$,
informative columns have $z_j^2=u_j^2/(Q_{jj}\sigma^2)$. The check rejects a
component variance above $100\max_j z_j^2$. Zero-information columns do not
provide a finite marginal z. This check is a diagnostic for extreme fitted
variance or summary/LD inconsistency, not proof that the input LD is appropriate.

The public RSS iteration limit is 50 and the sufficient-statistic limit is 100;
these are caps, with early stopping controlled by `tol`. Both default to
`tol=0.0001`. [API](api.md#shared-fitting-options) lists parameter defaults and
errors; [Results](results.md#iterations-and-finalization) explains the returned
status. The settings of numerical comparisons are recorded separately on
[Numerical accuracy](numerical_accuracy.md).

A dense float64 matrix requires $8p^2$ bytes before working arrays and results.
See [Implementation](implementation.md#dense-storage-and-matrix-products) for copies,
layouts and additional storage.
