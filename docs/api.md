# API reference

`susie_rss`, `susie_suff_stat` and `SusieResult` are top-level exports. Posterior helpers are in `prusie.posterior`.

The following signatures and parameter descriptions describe the public API. See the input and result guides for alignment and interpretation.

## susie_rss

```python
susie_rss(z: ArrayLike | None=None, R: ArrayLike | None=None, n: float | None=None, *, bhat: ArrayLike | None=None, shat: ArrayLike | None=None, var_y: float | None=None, z_ld_weight=0.0, estimate_residual_variance=False, prior_variance=50.0, check_prior=True, L=10, scaled_prior_variance=0.2, residual_variance=None, estimate_prior_variance=True, estimate_prior_method='optim', check_null_threshold=0.0, prior_tol=1e-09, r_tol=1e-08, prior_weights=None, null_weight=0.0, standardize=True, max_iter=50, s_init=None, coverage=0.95, min_abs_corr=0.5, tol=0.0001, verbose=False, track_fit=False, check_input=False, refine=False, n_purity: int | None=None, variant_ids: Sequence[str] | None=None, variant_metadata: Mapping[str, ArrayLike] | None=None) -> SusieResult
```

Fit SuSiE-RSS using z + signed R + sample size, or bhat/shat + R.

Parameters
----------
z : array_like, shape (m,), optional
    Signed association statistics; mutually exclusive with bhat/shat.
R : array_like, shape (m, m)
    Signed LD correlation r in the exact variant/effect-allele order.
    Singular matrices are allowed; no ridge or PSD clipping is applied.
n : float, optional
    Known sample size > 1. If absent, use the upstream large-n likelihood
    (XtX=R, Xty=z, n_internal=2, yty=1), with a recorded warning.
bhat, shat : array_like, optional
    Effect estimates and positive standard errors, instead of z. shat may
    be scalar. With var_y and n, reconstruct the original predictor scale.
var_y : float, optional
    Sample phenotype variance for known n; sets the response variance.
    With bhat/shat also reconstructs the original predictor scale.
estimate_residual_variance : bool, default False
    Fixed residual variance is recommended for external reference LD.
prior_variance : float, default 50
    Initial prior variance for missing-n mode only.
z_ld_weight : float, default 0
    Nonzero legacy LD modification is explicitly unsupported.
L, scaled_prior_variance, residual_variance, estimate_prior_variance,
estimate_prior_method, check_null_threshold, prior_tol, r_tol,
prior_weights, null_weight, standardize, max_iter, s_init, coverage,
min_abs_corr, tol, verbose, track_fit, check_input, refine, check_prior,
n_purity, variant_ids, variant_metadata
    See ``susie_suff_stat``. check_prior defaults True in this interface.
    RSS defaults max_iter=50 and tol=1e-4; matrix purity is complete.

Returns
-------
SusieResult
    Native fit with exact RSS transformation parameters recorded.

Raises
------
ValueError
    Ambiguous input, invalid correlation or nonfinite values.
NotImplementedError
    Unsupported nondefault option (including initialization/refinement).

Notes
-----
Known n applies z_tilde = z sqrt((n-1)/(z²+n-2)); this is not the
infinite-sample approximation. With z alone the effects have standardized
phenotype units. External-LD misspecification remains the user's modeling
assumption; large prior and convergence warnings are retained.

References
----------
Zou et al. (2022), PLoS Genetics 18, e1010299; susieR 0.16.6 R/susie_constructors.R.

Examples
--------
>>> fit = susie_rss([4., 0.], [[1., 0.], [0., 1.]], 100, L=1)
>>> fit.pip.shape
(2,)

## susie_suff_stat

```python
susie_suff_stat(XtX: ArrayLike, Xty: ArrayLike, yty: float, n: float, *, X_colmeans: ArrayLike | None=None, y_mean: float | None=None, L: int=10, scaled_prior_variance=0.2, residual_variance=None, estimate_residual_variance=True, estimate_prior_variance=True, estimate_prior_method='optim', check_null_threshold=0.0, prior_tol=1e-09, r_tol=1e-08, prior_weights=None, null_weight=0.0, standardize=True, max_iter=100, s_init=None, coverage=0.95, min_abs_corr=0.5, tol=0.0001, verbose=False, track_fit=False, check_input=False, refine=False, check_prior=False, n_purity: int | None=None, variant_ids: Sequence[str] | None=None, variant_metadata: Mapping[str, ArrayLike] | None=None, maf=None, maf_thresh=0.0) -> SusieResult
```

Fit SuSiE to centered sufficient statistics using sequential Rust IBSS.

Parameters
----------
XtX : array_like, shape (m, m)
    Centered predictor crossproduct, float64, symmetric and finite.
Xty : array_like, shape (m,)
    Centered predictor-response crossproduct in exactly the same order.
yty : float
    Centered response sum of squares, strictly positive.
n : float
    Sample size greater than one, never inferred from LD dimension.
L : int, default 10
    Maximum single effects, capped at the number of columns including null.
scaled_prior_variance : float, default 0.2
    Initial V divided by yty/(n-1); at most one when standardized.
residual_variance : float or None
    Initial or fixed sigma2; None uses yty/(n-1).
estimate_residual_variance, estimate_prior_variance : bool
    Update sigma2 and V respectively. Both default True for this interface.
estimate_prior_method : {'optim', 'EM', 'simple'}
    Pinned R method, default 'optim'. The native solver carries optimization.
prior_weights : array_like, optional
    Nonnegative finite variant weights; normalized after optional null.
null_weight : float, default 0
    Optional last, zero-crossproduct null column. Must be in [0, 1).
standardize : bool, default True
    Scale predictors to unit sample variance before inference.
max_iter, tol : int, float
    IBSS iteration limit (100 here; 50 for RSS) and finite nonnegative
    ELBO improvement threshold (1e-4).
coverage, min_abs_corr : float or None
    Requested CS posterior mass (0.95) and minimum absolute correlation
    (0.5). Returned ``sets.coverage`` is each retained set's actual mass;
    ``sets.requested_coverage`` records the target.
    None disables CS construction; marginal PIP is still returned.
n_purity : int or None
    Matrix-input purity is always complete, matching the pinned reference;
    a finite n_purity does not subsample the matrix.
prior_tol : float, default 1e-9
    V threshold for final null-component trim and PIP. The CS threshold
    stays fixed at 1e-9. Trim also runs at the iteration cap.
check_input, r_tol : bool, float
    Optional PSD/projection checks, eigenvalue tolerance 1e-8.
check_prior, check_null_threshold : bool, float
    Large-V guard and log-BF threshold for setting V to zero.
X_colmeans, y_mean : array_like or float, optional
    Means needed to recover intercept; without them intercept is NaN.
variant_ids, variant_metadata : sequence, mapping, optional
    Ordered biological IDs and aligned chromosome/allele metadata.
    Omitted IDs are positional labels, not inferred biological identifiers.
s_init, refine, track_fit, maf, maf_thresh, verbose
    Nondefault initialization, refinement, trace, MAF filtering and verbose
    mode raise NotImplementedError. These options are never silently ignored.

Returns
-------
SusieResult
    Posterior arrays, BF, variance, ELBO, CS, PIP, and execution metadata.

Raises
------
ValueError
    Invalid shape, nonfinite input, invalid parameters or native failure.
NotImplementedError
    An explicitly unsupported option is requested.

Notes
-----
Supported Gaussian updates follow pinned susieR 0.16.6. Matrix-input CS
purity uses the complete submatrix.
Asymmetric or missing inputs error rather than being silently repaired.
Component mu/mu2 remain on the standardized predictor scale; ``coef``
applies the stored inverse scale transformation.

References
----------
Wang et al. (2020), JRSS B 82, 1273-1300; susieR 0.16.6 R/sufficient_stats_methods.R.

Examples
--------
>>> fit = susie_suff_stat([[99., 0.], [0., 99.]], [30., 0.], 99., 100,
...                       L=1, estimate_residual_variance=False)
>>> fit.alpha.shape
(1, 2)

## credible_sets

```python
credible_sets(alpha, V, correlation, *, coverage=0.95, min_abs_corr=0.5, null_index=None, n_purity=None, covariance_scales=None, covariance_multiplier=1.0)
```

Construct deduplicated CSs with complete pairwise absolute-r purity.

The current matrix-input reference uses complete purity regardless of
``n_purity``. None for coverage or min_abs_corr disables set construction,
following the high-level sufficient-statistics route.
Returned ``coverage[k]`` is the posterior mass of ``cs[k]`` in its original
component ``cs_index[k]``. ``actual_coverage`` is an equal alias; the target
threshold is recorded separately as ``requested_coverage``.

## marginal_pip

```python
marginal_pip(alpha, V, *, null_index=None, prior_tol=1e-09)
```

Marginal inclusion probabilities, excluding small-V effects and null.
