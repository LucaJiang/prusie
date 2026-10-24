# Python API

The Gaussian SuSiE fitters return a typed `SusieResult`. Inputs are converted
to float64 as needed; variant and component indices are zero-based. See
[inputs](inputs.md) for alignment and [results](results.md) for returned fields.
Python `help()` provides complete signatures and NumPy-style docstrings.

## Fitting entry points

| Function | Input and purpose |
| --- | --- |
| `susie(X, y, *, ...)` | Individual-level dense NumPy or SciPy sparse data |
| `susie_suff_stat(XtX, Xty, yty, n, *, ...)` | Centered sufficient statistics; `susie_ss` is an alias |
| `susie_rss(z=..., R=..., n=..., ...)` | Signed summary statistics with LD; accepts `bhat`/`shat` and reference `X` routes |
| `susie_rss_lambda(z, R, *, lambda_=..., ...)` | RSS with fixed or estimated LD regularization |
| `susie_ser(z=..., ...)` | One-effect summary-statistic model without an LD matrix |
| `susie_auto(X, y, *, ...)` | Increase the effect count using the susieR automatic fitting rule |
| `susie_trendfilter(y, *, order=0, ...)` | Trend filtering with an implicit cumulative design |

```python
import prusie

fit = prusie.susie_rss(
    z=z, R=R, n=n,
    L=10, max_iter=100,
    estimate_residual_variance=False,
    variant_ids=variant_ids,
)
```

Here `z`, `R`, `n` and `variant_ids` must describe the same ordered variants.
For a runnable offline script, start with [the toy example](example.md).
Individual-level fitting centers the outcome and, by default, the design;
`intercept=False` disables centering. `standardize` controls predictor scaling,
`na_rm=True` drops rows with missing outcomes, and `compute_univariate_zscore`
requests marginal z-scores. Dense and CSC sparse designs use matrix-vector
operators without constructing a dense Gram matrix; sparse designs are not
densified. Trend filtering also avoids storing its dense design matrix.

The reference-factor `X` route in `susie_rss` uses the number of reference
rows to estimate LD, independently of the association sample size `n`.
Lists of reference panels are supported where susieR accepts them. SciPy is
required for sparse inputs and advanced RSS linear algebra.

`susie`, `susie_suff_stat`, `susie_rss` and `susie_rss_lambda` accept
`init_only=True` to return a `SusieInputs` specification;
`susie_workhorse(inputs)` fits it later. Panel-list RSS initialization performs
per-panel selection fits before returning that specification. `model_init` accepts a previous fit
or posterior state; `s_init` remains its supported alias. Supplying both raises
an error. The existing retained-data continuation interface remains available.
With the default `use_mad=True`, trend-filter initialization follows susieR's
nested preparation; use `use_mad=False` for a directly reusable `SusieInputs`.
`susie_auto` requires fitting and does not accept `init_only=True`.

Warning filters, including `error`, and warning recording are supported. Callbacks that mutate input arrays during a fit, and replacing `warnings.warn` before importing prusie, are unsupported.

## Shared fitting options

| Option | Meaning |
| --- | --- |
| `L=10` | Maximum number of effects, capped at the internal predictor count; RSS-lambda uses retained variants after MAF filtering, excluding any null column (`L=None` is an alias for 10) |
| `n_threads=1` | Per-fit native thread limit, an integer from 1 to 32; small kernels stay serial |
| `scaled_prior_variance=0.2` | Gaussian initial variance relative to outcome variance; NIG slab variance relative to residual variance; scalar or length-L vector |
| `residual_variance=None` | Initial or fixed residual variance; defaults to the working outcome variance |
| `estimate_prior_variance=True` | Update component prior variances |
| `estimate_prior_method="optim"` | Prior update: `optim`, `EM` or `simple` (`susie_ser` accepts `optim` or `simple`); disable estimation with `estimate_prior_variance=False` |
| `prior_variance_grid`, `mixture_weights` | Fixed mixture of positive prior variances; weights default to uniform |
| `estimate_residual_method="MoM"` | Residual update: `MoM`, `MLE` or conjugate `NIG` |
| `alpha0`, `beta0` | Positive NIG hyperparameters; defaults are 1/√n |
| `residual_variance_lowerbound`, `residual_variance_upperbound` | Bounds for residual variance updates |
| `prior_weights=None`, `null_weight=0` | Variant prior probabilities and optional null-effect probability |
| `slot_prior=None` | Optional effect-activity model from `slot_prior_betabinom` or `slot_prior_poisson` |
| `L_greedy`, `greedy_lbf_cutoff` | Greedy effect-count extension controls |
| `convergence_method` | Effective ordinary default `elbo`; SS/RSS default `None` preserves configured controls. `pip` checks complete alpha and aggregate PIP |
| `coverage=0.95`, `min_abs_corr=0.5` | Credible-set mass and purity thresholds |
| `n_purity` | Defaults to `None` for sufficient-statistic/ordinary RSS and `"auto"` for individual data, RSS-lambda and CS getters; correlation-matrix purity uses all pairs |
| `median_abs_corr`, `cs_extension_corr` | Alternative purity criterion and correlated-variant CS extension |
| `maf`, `maf_thresh` | Filter predictors before fitting using aligned minor-allele frequencies |
| `verbose=False` | Emit progress through the `prusie` Python logger |
| `variant_ids`, `variant_metadata` | Ordered identifiers and aligned annotations |

RSS defaults to `max_iter=50` and fixed residual variance; sufficient-statistic
and individual-level fitting default to `max_iter=100` and estimated residual
variance. Ordinary Gaussian ELBO tolerance is `1e-4`. Passing `tol=None`
selects method-specific defaults (`1e-6` for NIG); omitted `tol` retains the
existing `1e-4` default in `susie_rss` and `susie_suff_stat`.
Individual-level NIG uses `1e-6` when tolerance is omitted.
NIG integrates out residual variance unless a fixed mixture selects Gaussian
SER updates; it enables residual estimation and uses posterior stopping when
L > 1. Slot priors also select
posterior stopping. Use fixed residual variance with external LD unless
estimation is justified.

`n_threads` is accepted by `susie`, `susie_suff_stat`, `susie_rss` and
`susie_rss_lambda`, and the requested value is recorded in `fit.params`.
The current parallel kernels are dense Gram-matrix products, including proved
exact-class compact products. They use a private pool for the fit and fixed
output partitions. Component updates remain sequential. Validation, design-matrix
operators and dynamic RSS likelihoods remain serial in this implementation;
requesting more threads does not imply a speedup for those stages or models.
Small matrices and compact products with fewer than 1024 rows remain serial.
The process-wide BLAS, OpenMP and Rayon configuration is unchanged.

The default is explicitly one native thread. Environment-only requests through
`PRUSIE_NUM_THREADS` no longer select public fitting parallelism; pass
`n_threads=4`, for example. Booleans and floating-point values are not thread
counts. Determinism is defined for the same input and requested thread count;
numerical equality between different thread counts is reported, not guaranteed.

## Optional LD workspace

`susie_rss(..., overwrite_ld=False)` preserves caller arrays by default. The
explicit `True` option can use R and its external aliases as destructive
workspace, avoiding an owned full repair matrix when scaled asymmetry needs
repair. It accepts only Python `bool` or NumPy `bool_` values.

The option requires ordinary single-panel `z`, `R`, and provided `n`, with
`var_y=None`. R must be a plain, aligned, writable native float64 NumPy array
with ordinary positive C-order strides and NumPy-owned heap storage. Reshape
views of such an owner are supported. Array subclasses, memory maps and views
of them, external buffers, readonly, F-order or strided arrays are rejected.
Other arguments must use plain scalars or flat nonobject numeric/string vectors
and plain metadata dictionaries; custom conversion/warning hooks and any
overlap with R, including nested or zero-dimensional views, are rejected before
writes. External aliases that are not arguments are consumed with R and must
not be accessed concurrently.
The option requires standard warning display or plain-list recording through
`warnings.catch_warnings(record=True)`; custom warning dispatch/display and
`logging.captureWarnings` are unsupported. Ordinary warning-as-error filters
remain supported. `R_finite` must be `None` or the plain Python value `False`;
the NumPy boolean scalar is outside this narrow support boundary.

This option excludes reference `X`, `bhat`/`shat`, finite-reference/mismatch or
control objects, MAF filtering, null-weight augmentation, mixture/slot/NIG
models, residual estimation methods other than MoM, greedy fitting, proxy extension, warm starts, explicit convergence
objects, `check_input=True`, tracking, retained data, refinement and
`init_only=True`. Ordinary Gaussian controls, prior weights, standardization,
basic CS/purity options and `n_threads` remain available. Unsupported opt-in
requests raise before writes; use `overwrite_ld=False` for the other routes.

The native helper records its actual action in `fit.qc['ld_overwrite']`, with
the original RSS scale n−1:

| Action | State of R after a successful call |
| --- | --- |
| `ordered_scaled_xtx` | Both orientations contain `(Rij*(n−1) + Rji*(n−1))/2` in float64, in that operation order |
| `symmetric_no_write` | Original bytes retained after the complete exact-symmetry proof |
| `no_witness_eager_no_write` | Original bytes retained; original owned preparation was used because no finite scaled-asymmetry witness exists |

`fit.params['overwrite_ld']` records the requested flag. A consumed matrix is
scaled XtX, so reload the original LD before a later fit. **After writes begin,
an error or a warning promoted to an exception can leave R and its aliases
changed.** There is no rollback guarantee. This includes later matrix/fit/option
validation failures; no result metadata is available on an aborted call.
Default calls and every other API argument retain their original preservation
contract.

For RSS, `z_method="wald"` or `"score"` selects the working statistic;
`R_mismatch` and `R_finite` control reference mismatch and finite-reference
adjustments. `susie_rss_control(...)` creates a validated control object for
these options. `susie_rss_lambda` uses `lambda_="estimate"` for automatic
regularization and exposes `intercept_value`, `check_R` and `check_z`.
Its residual method is `"MLE"` only, with default tolerance `1e-3`.

The legacy `z_ld_weight=0` argument is retained. Nonzero values are rejected:
this parameter has no defined semantics in the pinned susieR 0.16.6 source.
SuSiE-inf, SuSiE-ash and Mr.ASH are outside the supported model set.

## SusieResult

Results include documented fields `pip`, `alpha`, `mu`, `mu2`, `lbf_variable`,
`lbf`, `V`, `sigma2`, `elbo`, `KL`, `sets`, `niter` and `converged`. Mixture and
activity models additionally return grid Bayes factors and slot probabilities.
`input_quality` gives typed input diagnostics; `fit.qc` retains the existing
mapping interface. `fit.as_dict()` returns a shallow mapping and does not copy arrays.

`params`, `qc` and entries in `cs_audit` use the exported `FitParameters`,
`InputChecks` and `CredibleSetAudit` schemas. These are `TypedDict` definitions:
dictionary access and saved-result formats are unchanged. `input_missing_policy`
is `"reject"` or `"zero"`; the latter records zero-filled missing summary
statistics. Retained working inputs are private, read-only snapshots used by
continuation and refinement, not additional caller-owned data.

Analysis, comparison, LD diagnostics, reports and visualization modules load
only when requested. Ordinary `import prusie` does not import SciPy or Matplotlib.

## Posterior and model helpers

| Functions | Output |
| --- | --- |
| `susie_get_pip`, `susie_get_cs`, `susie_get_cs_attainable` | Inclusion probabilities and credible sets |
| `susie_get_posterior_mean`, `susie_get_posterior_sd`, `susie_get_lfsr` | Marginal moments and local false-sign rates |
| `susie_get_posterior_samples(..., seed=...)` | Reproducible posterior draws using an explicit NumPy RNG seed |
| `susie_get_objective`, `susie_get_niter` | Objective history and completed iterations |
| `susie_get_prior_variance`, `susie_get_residual_variance` | Fitted variance parameters |
| `get_cs_correlation` | Correlations between credible-set representative variants |
| `coef`, `predict`, `summary` | Coefficients, predictions and tabular model summaries |
| `compute_suff_stat`, `univariate_regression`, `calc_z` | Individual-level data preparation |
| `susie_init_coef` | Construct a sparse coefficient initialization |
| `susie_post_outcome_configuration` | Shared-variant posterior outcome configurations from fitted models |

`susie_get_lfsr` computes Gaussian sign probabilities from each matched posterior
mean and standard deviation. It intentionally avoids susieR 0.16.6's recycled
matrix indexing in this auxiliary calculation. NumPy posterior samples are
reproducible for a fixed seed; they do not reproduce R's random-number stream.
X-based purity subsampling also uses a local NumPy RNG. Sparse-X credible-set
extension evaluates the requested correlation blocks without densifying X;
susieR skips this extension for sparse designs.

Lower-level `prusie.posterior.marginal_pip` and `credible_sets` also accept
posterior arrays directly. Covariance scale options allow purity calculations
without materializing a correlation matrix. Original component IDs are retained
in credible-set results; see [results](results.md#credible-set-fields).

## load_example

```python
import prusie

example = prusie.load_example()
fit = prusie.susie_rss(**example["inputs"], **example["parameters"])
```

The synthetic toy example is loaded as a mapping with `inputs`, `parameters` and `metadata`. Inputs contain
z, R, n and variant IDs; metadata includes the synthetic method, seed, true
effects and sample size. Input bytes are checked against packaged hashes and
returned arrays are owned by the caller. The loader works from any directory
without network access. See the [toy example tutorial](example.md).

## Convergence and post-analysis

The core fitters support `s_init=None`, `refine=False`,
`track_fit=False`, `convergence=None` and `retain_data=False`.
The retained continuation, refinement and objective helpers support static,
materialized Gaussian sufficient-statistic and R-based RSS models using the
default or `ConvergenceConfig` stopping controls. Matrix-free designs, dynamic
RSS, mixture/NIG/slot models, susieR `pip` stopping, greedy component growth,
`median_abs_corr` and `cs_extension_corr` are outside that scope.
For these combinations, `refine=True` refits the model with susieR's
credible-set block coordinate ascent (except multi-panel RSS) and records the
search in `refinement`. Explicit `retain_data=True` or a `RefinementConfig`
raises before fitting; ordinary fitting and tracking remain available without
a retained snapshot. No dense Gram matrix is constructed to enable retention of
a matrix-free design.

For supported models, `retain_data=True` enables later continuation,
refinement or objective evaluation. Tracking, initialization and refinement
also retain inputs within this scope. A new fit with `model_init`/`s_init`
does not require source data retention. `data_signature` is a content hash of
the owned input snapshot, computed once and reused; unretained fits return `None`.
`ConvergenceConfig` selects exact ELBO, alpha/PIP or joint stopping.
The separate `convergence_method="pip"` uses the susieR posterior stopping rule.
Finite-ELBO stopping uses signed improvement; lagged posterior cycles remain
diagnostics in that mode. Successful and statistical cycle exits respect
`min_iter`; numerical failure is reported separately.
`track_fit=True` records compact exact metrics and a labelled lossy alpha display;
`track_fit='dense'` retains full states for small debugging inputs.

`continue_fit`, `refine_fit`, `evaluate_objective`, `summarize`, `diagnose_ld`,
`compare_fits` and `run_sensitivity` are independent explicit operations.
`RefinementConfig` separates search thresholds/budgets from inner IBSS tolerances.
See [convergence and refinement](post_analysis.md) for costs, initialization
checks, phases and runnable examples. `continue_fit(fit, max_iter=100, tol=None)`
inherits the original ELBO tolerance when `tol` is omitted. Requested convergence
controls persist across continuation, refinement and sensitivity unless explicitly
overridden; effective runtime fallback is recorded separately. Refinement's
`inner_tol` sets its inner ELBO tolerance independently of those inherited controls.
Continuation starts from the supplied posterior; refinement stage A is a cold
excluded-prior fit and stage B warm-starts its posterior moments under the restored
original model. Reuse the returned `refined` in plots and comparisons to avoid
starting another search.

## Summaries, comparisons and reports

| Function | Purpose |
| --- | --- |
| `summarize(fit, R=None, *, z=None, ...)` | Component, variant, CS, proxy and prior-influence tables; no new fit |
| `summary_text(summary, *, ld=None, comparison=None)` | Deterministic short string for `print`; no fitting, eigendecomposition or plotting |
| `compare_fits(fits, *, baseline=None, mode="union", R=None, ...)` | Compare a named mapping of existing fits using explicit ID/support rules |
| `run_sensitivity(inputs, plan, *, baseline=None, baseline_settings=None, ...)` | Execute a finite named one-factor plan; report-only choices reuse a posterior |
| `diagnose_ld(z, R, n=None, ...)` | Explicit conditional z/LD and allele-flip diagnostics |
| `write_report(summary, directory, *, figures=None, ld=None, comparison=None, ...)` | Offline HTML, text and full evidence exports; include only supplied figures |

The first five functions are exported at the package top level. `write_report`,
`save_analysis`, `load_analysis` and `analysis_summary` live in `prusie.report`.
The [quick start](example.md#write-an-offline-report) gives a minimal report call;
[sensitivity and LD](sensitivity.md) specifies prior remapping, alternative inputs,
common/union support and missing-assessment states.

## Figures and local style

Use `from prusie.viz import plot_locus` for plotting. `prusie.plots` remains a
compatibility import for existing scripts. Install the `analysis` extra for
Matplotlib and SciPy; neither is required for ordinary dense Gaussian fits.

Every function in `prusie.viz` returns `(figure, axes, plotted_data)`. The third
item records selection rules and plotted values alongside the complete evidence
needed to interpret the view. Existing Matplotlib axes can be passed with
`axes=...`; their count must match the selected layout.

| Function | View and layout choices |
| --- | --- |
| `plot_locus(summary, *, R=None, layout="full", ...)` | Association, PIP, CS and reference r² on the same coordinates; optional annotated gene track |
| `plot_convergence(fit_or_summary, *, after=None, layout="full", ...)` | `compact`: ELBO, signed ΔELBO, Δalpha, ΔPIP; `variance`: V and sigma²; `full`: all six metrics |
| `plot_refinement(before, after, *, layout="full", ...)` | Candidate PIPs, refit ELBO gains and accepted ELBO path; `compact` moves long evidence out of the figure |
| `plot_sensitivity(comparison, *, layout="full", candidates_only=True, layer="signals", ...)` | Baseline-first candidate heatmap and main signal relations; `components` retains all active-component relations |
| `plot_signal(summary, component_id=None, *, variant_id=None, layout="full", ld_measure="r", ...)` | Component posterior, cumulative mass, CS, resolution and LD; `ld_measure="r2"` selects an explicitly labelled r² view |
| `plot_prior_influence(summary, component_id=None, *, variant_id=None, variant_ids=None, layout="full", ...)` | Supplied/effective prior, conditional log BF and alpha in aligned variant order |
| `plot_ld_diagnostics(diagnostics, *, layout="full", ...)` | Assessed states, conditional residuals, flip evidence and the actual joint rule |

`compact` is an explicit screen layout; `full` preserves access to larger views.
Formal CS members remain included in signal candidate selections. Sensitivity
`display_names={stored_name: readable_label}` changes labels, not settings or
fit identities. `table_page` and `table_page_size`, where supported, paginate
relation/details tables without discarding returned records.

Use the shared white-background style or override Matplotlib settings locally:

```python
from prusie.viz import plot_convergence

figure, axes, data = plot_convergence(
    fit, layout="compact", style={"font.size": 11, "axes.labelsize": 11},
)
figure.savefig("convergence.svg", bbox_inches="tight")
```

`style=False` uses the caller's current Matplotlib style. The function restores
global rcParams on exit and does not choose a backend. Shared presentation tokens
in `prusie._style` define fonts, spacing, colors and readable number formatting;
use the public `style` argument for individual figures. Before/after colors are
consistent, missing values are separate, and probability/r² scales stay in [0,1].
Displayed rounding never replaces full-precision returned or exported values.

### Heatmap scales and labels

PIP heatmaps default to **Blues**, with a fixed linear scale from 0 to 1. CS LD
uses **Reds** for the explicit `ld_measure="r2"` display and **RdBu_r** for the
default signed-r display, with fixed ranges [0,1] and [−1,1] respectively.
The separate colormap roles prevent a PIP style change from altering LD views.
Pass `cmap=` to `plot_sensitivity` or `plot_signal` to override the displayed
colormap; the fixed numeric range and original matrix are retained.

Compact sensitivity cells use `0`, `<0.01`, two decimal places, `>0.99`, `1` and
`NA` as appropriate. This formatting affects labels only: returned display text
and policy are separate from full-precision PIPs. Small selected matrices label
every candidate row; large matrices retain the candidate records even when
per-cell annotations are omitted. CS members on the declared comparison support
are not dropped to shrink the figure; common-mode filtering is described in the
[support rules](sensitivity.md#common-and-union-comparison-support). Missing observations have a distinct grey color, separate from the
color for a true zero.

Within-CS LD cells are square. Small matrices label non-endpoint correlations
with sufficient precision to distinguish high LD from exact identity. Compact
singleton CS views keep the underlying matrix and purity while using a short
singleton display. See the [worked 86/90 example](worked_examples.md#what-remains-unresolved).

### Other display controls

For large-magnitude objectives, `plot_convergence(..., relative_elbo=True)` plots
each trace relative to its first finite ELBO, with that origin labelled. It
preserves the original trace and signed increments in the returned data.

Locus plots use physical position only when provided, otherwise a labelled
variant index. `position_unit="auto"` chooses a displayed bp/kb/Mb scale;
`position_unit="bp"`, `"kb"` or `"Mb"` selects it explicitly without changing
source coordinates. `label_variants` chooses annotation IDs, while
`color_by_ld=True` colors association points by r² to the named reference. LD coloring names its reference variant. `plot_signal` defaults
to signed r; the worked reports explicitly request r². Local gene tables/GTF
require source, build, transcript selection and a coordinate convention; known
chromosome/build conflicts are rejected. No gene identity is inferred from an
rsID or arbitrary example coordinate. LD-diagnostic `auxiliary="residual"` or
`"log_lr"` selects the supporting panel; labelled extreme residuals remain
distinct from variants satisfying the joint flag rule.
