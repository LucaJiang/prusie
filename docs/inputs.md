# Inputs, alignment and priors

Align association statistics and LD to the same variant order and effect
alleles before fitting. prusie checks numerical arrays; the study design,
allele harmonization and covariate adjustment belong to the input preparation.

## Supported model and input routes

prusie fits Gaussian SuSiE on CPU in float64, using summary statistics or
centered sufficient statistics. Choose the route that preserves the scale
information available for the study:

| Inputs | Working representation and coefficient interpretation |
| --- | --- |
| `z`, `R`, `n` | Finite-sample Wald-adjusted summaries on a standardized working scale |
| `bhat`, `shat`, `R`, `n` | Forms signed z as bhat/shat, then uses the same standardized route |
| `bhat`, `shat`, `R`, `n`, `var_y` | Reconstructs predictor crossproducts using the known phenotype sample variance; `coef` returns original-scale coefficients |
| `z`, `R`, `n`, `var_y` | Sets response sum of squares and initial variance scales; leaves z-based crossproducts unchanged |
| `z` or `bhat/shat`, `R`, with n omitted | Noncentrality-parameter likelihood; emits a warning and uses `prior_variance` |
| `XtX`, `Xty`, `yty`, `n` | Centered sufficient statistics via `susie_suff_stat` |

The [model page](model.md#from-summaries-to-working-sufficient-statistics)
derives each transformation. Provide either z or both bhat and shat. `var_y`
is a measured sample phenotype variance, not a value inferred from a trait name
or normalized-looking statistics. In missing-n mode it is checked and recorded
but does not change the working likelihood.

## Shapes and numerical checks

For $p$ biological variants, z or bhat and `variant_ids` have length $p$;
R has shape $(p,p)$. `shat` is either a strictly positive scalar shared by all
variants or a length-$p$ vector of positive standard errors. IDs are distinct
nonempty strings. `variant_metadata` contains aligned length-$p$ columns copied
into the result; metadata does not itself establish allele correctness.

`R` contains **signed correlation r**. Singular LD is allowed. Every entry must
be finite; the diagonal must be within $10^{-8}$ of one and magnitudes at most
$1+10^{-8}$. Symmetry uses both directions of the bound
$|R_{ij}-R_{ji}|\leq10^{-12}+10^{-12}|R_{ji}|$. The API does not repair LD by
ridge adjustment, eigenvalue clipping or silently symmetrizing invalid input.

`susie_suff_stat` requires a finite symmetric predictor crossproduct `XtX`,
finite `Xty`, positive `yty`, and $n>1$. These describe the same centered data.
`X_colmeans` and `y_mean` recover an original-scale intercept; missing means
leave it NaN. Predictor standardization is enabled by default, and `coef`
reverses its stored scales. Basic finite/symmetry/diagonal checks always run.
`check_input=True` adds an eigendecomposition, rejects eigenvalues below
`-r_tol`, and warns if Xty is outside the nonzero eigenspace. It does not
repair the matrix or certify its external provenance.

## Variant order and alleles

Preserve genome build, counted allele, association effect allele, population,
reference sample selection and SNP order in the analysis record. Resolve
ambiguous strand or palindromic alleles from appropriate evidence before
fitting. A permutation must be applied to both R axes, every association vector
and every metadata column.

If allele recoding multiplies each effect by $d_j\in\{-1,+1\}$, apply the same
signs to z and both LD axes:

```python
z_new = signs * z
R_new = signs[:, None] * R * signs[None, :]
```

For effect-estimate inputs, recode bhat in the same way and leave shat unchanged.
Update the allele metadata consistently. The API does not infer these signs or
filter variants on the caller's behalf.

## Priors and zero-information columns

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

The public RSS iteration limit is 50 and the sufficient-statistic limit is 100;
these are caps, with early stopping controlled by `tol`. Both default to
`tol=0.0001`. [API](api.md#shared-fitting-options) lists parameter defaults and
errors; [Results](results.md#iterations-and-finalization) explains the returned
status. The settings of numerical comparisons are recorded separately on
[Numerical accuracy](numerical_accuracy.md).

A dense float64 matrix requires $8p^2$ bytes before working arrays and results.
See [Implementation](implementation.md#layout-ownership-and-memory) for copies,
layouts and additional storage.
