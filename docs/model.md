# Statistical model

prusie implements Gaussian SuSiE fine-mapping from association summaries and
signed LD, or from centered sufficient statistics. The model represents a
multiple-regression coefficient vector as a sum of a small number of effects.

## Sum of single effects

For centered phenotype y and predictors X, the working model is

**y = Xb + ε**, ε ∼ N(0, σ²I), **b = b₁ + ⋯ + bL**.

Each bℓ has one potentially nonzero coefficient. Its location has prior
probabilities π₁,…,πp, and its nonzero value has a zero-mean Gaussian prior with
variance Vℓ. L is an upper bound on the number of components, not a number of
signals guaranteed to be present. Estimated Vℓ can be zero. The approximation
factorizes across components and is fitted by iterative Bayesian stepwise
selection (IBSS).

IBSS updates one component conditional on the current posterior means of all
other components. Later components in a sweep use earlier components' updated
means. Each single-effect regression computes variant probabilities αℓj,
conditional means μℓj and second moments μ²ℓj. A finite, nonnegative ELBO increase
below `tol` indicates convergence after at least two sweeps. Warnings and the
iteration count are retained; reaching `max_iter` returns a diagnostic fit.

## Input representations

`susie_suff_stat` takes Q = XᵀX, g = Xᵀy, yᵀy and sample size n. These must refer
to the same centered data. Predictor standardization is explicit; `fit.coef`
transforms posterior means back to the original predictor scale. Means supplied
through `X_colmeans` and `y_mean` permit recovery of an intercept.

`susie_rss` accepts signed z statistics or effect estimates and standard errors,
together with an LD correlation matrix R. In the known-n z path, the Wald
adjustment is z̃j = zj √[(n − 1)/(zj² + n − 2)]. The working quantities are
Q = (n − 1)R, g = √(n − 1) z̃ and yᵀy = n − 1 when phenotype variance is not
supplied. Providing `bhat`, `shat`, `n` and `var_y` reconstructs predictor and
phenotype scales. See [inputs](inputs.md) for the separate missing-n
noncentrality-parameter path.

The LD input is **signed correlation r**, not r² or |r|. Every vector and both
matrix axes must use the same variant order and counted allele. Genome build,
alleles, population and sample provenance are supplied by the researcher;
prusie checks array validity but does not harmonize association files.

With external LD, keep residual variance fixed unless a justified model supports
estimation. In-sample LD consistent with the summaries is needed for the
residual-variance estimator. A small or mismatched reference panel can change
inference even when a software comparison is numerically accurate. Case-control
summary statistics use the Gaussian summary approximation; this interface does
not fit individual-level logistic regression.

## Priors and posterior summaries

Variant prior weights are normalized nonnegative weights. The implementation
follows the reference's √machine-epsilon offset in single-effect log weights;
a zero supplied weight is therefore not an absolute exclusion. An optional
null component position can carry prior mass. `scaled_prior_variance` initializes
V relative to phenotype variance; V may be fixed or updated by `optim`, `EM`
or `simple`. Residual variance is fixed by default in RSS and estimated by
default for sufficient statistics.

For active components, variant inclusion probability is

**PIPj = 1 − ∏ℓ (1 − αℓj)**.

A credible set is formed by adding variants in decreasing α within one original
component until their accumulated posterior mass reaches the requested
`coverage`. Sets are deduplicated, filtered by active V and minimum absolute LD,
and returned with their original component IDs. `sets['coverage']` is the
**actual returned posterior mass**; `requested_coverage` stores the target.
Purity reports the minimum, mean and median absolute correlation over all
selected pairs. A high PIP, component Bayes factor and credible-set mass answer
different questions; set mass is not repeated-sampling coverage.

These are the SuSiE and SuSiE-RSS methods of
[Wang et al. (2020)](https://doi.org/10.1111/rssb.12388) and
[Zou et al. (2022)](https://doi.org/10.1371/journal.pgen.1010299).
See [model support](compatibility.md), [results](results.md) and
[implementation](implementation.md) for exact options and returned quantities.
