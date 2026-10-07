# Inputs, alignment and priors

`susie_rss` accepts signed z and a signed correlation matrix R. Alternatively
provide bhat and positive shat; known n is needed for finite-sample adjustment.
The bhat/shat+var_y branch uses the declared phenotype variance. Never infer
var_y=1 from a trait name or normalized-looking statistics. With n omitted,
the explicit noncentrality likelihood is a distinct model path. Supplying z
together with bhat or shat raises an error; choose one input route. See [API](api.md).

For m biological variants, z or bhat and variant_ids have length m; R has
shape (m,m). shat can be a positive scalar shared by all variants or a
length-m vector of positive standard errors. IDs must be distinct and nonempty. variant_metadata contains
length-m columns copied into the result. It records caller metadata and does
not establish allele correctness. Preserve genome build, counted allele,
summary effect allele, population, reference sample selection and SNP order.
Unknown strand/palindromic orientation needs evidence before fitting.

A SNP permutation must be applied to both R axes, every vector and metadata.
For allele recoding with signs d[j]∈{−1,1}, use z_new=d*z and
R_new=d[:,None]*R*d[None,:], and update allele metadata consistently. Neither
transformation means deleting variants. Singular LD is allowed; approximate LD,
automatic ridge, eigenvalue clipping and silent symmetrization are not performed.
Finite/symmetry/diagonal/range checks remain active even with check_input=False;
that option controls additional sufficient-statistic consistency diagnostics.

`susie_suff_stat` takes centered XtX, Xty, positive yty and n>1. Supply original
X_colmeans and y_mean if an original-scale intercept is needed. Standardization
changes the fitted predictor scale; use result.coef to recover original-scale
posterior coefficients. All numerical work uses float64 and sequential IBSS
component updates.

`prior_weights` are finite nonnegative SNP weights with a positive total.
They are normalized; partial-zero priors are supported. The SER reference
semantics add sqrt(machine epsilon) before taking logarithms, so a supplied
zero is not a hard exclusion from every component posterior. Final low-V trim
restores the exact normalized prior. An optional null_weight∈[0,1) adds a last
null column; do not append null IDs or allele records yourself.

Prior weights enter fitting; do not multiply them into the returned component
log Bayes factors again. A PIP is marginal across effects; component BFs
describe one effect's evidence, and credible sets select within a component.
None is interchangeable with another.

Keep residual variance fixed for external reference LD unless the model and
data justify estimation. The RSS default does this. Sufficient-statistic
residual-variance estimation defaults to enabled. Public RSS defaults are max_iter=50 and tol=.0001; validation settings
are recorded separately in the [numerical accuracy](numerical_accuracy.md). A nonconverged result remains
nonconverged and emits a warning; the library does not silently refit it.

A dense float64 LD matrix uses 8m² bytes for m variants, before fitting workspace
and results. See [implementation](implementation.md) for memory layouts and working arrays.
