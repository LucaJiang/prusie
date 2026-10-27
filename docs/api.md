# Python API

prusie fits SuSiE fine-mapping models from individual-level data, centered
sufficient statistics or association summary statistics with LD. Fitted models
are returned as `SusieResult` objects.

Numerical arrays are converted to float64 as needed; variant and component
indices are zero-based. See [Inputs](inputs.md) for preparation and alignment,
[Results](results.md) for posterior fields and [the toy example](example.md)
for a runnable workflow. Use Python `help()`, for example
`help(prusie.susie_rss)`, for signatures and parameter documentation.

## Fitting entry points

Choose the entry point that matches the available data.

| Function | Inputs and purpose |
| --- | --- |
| `susie` | Individual-level `X`, `y`; dense or SciPy sparse predictors |
| `susie_suff_stat` | Centered `XtX`, `Xty`, `yty` and sample size `n`; `susie_ss` is an alias |
| `susie_rss` | `z` or both `bhat` and `shat`, with LD `R` or reference genotypes `X` |
| `susie_rss_lambda` | `z`, either `R` or reference `X`, and required `lambda_`; fixed or estimated isotropic covariance component |
| `susie_ser` | `z` or both `bhat` and `shat`; single-effect regression without LD |
| `susie_auto` | `X`, `y`; staged `susie` fits that double the component count |
| `susie_trendfilter` | Ordered `y`; trend filtering with an implicit cumulative design (`order=0` by default) |

For an RSS fit, let `z` contain signed marginal z-scores, `R` the matching
LD correlation matrix, `n` the association sample size and `variant_ids`
the ordered identifiers:

```python
import prusie

fit = prusie.susie_rss(
    z=z, R=R, n=n,
    L=10, max_iter=100,
    estimate_residual_variance=False,
    variant_ids=variant_ids,
)
```

Align the z-scores and LD to the same variant order and effect alleles, and
keep `variant_ids` in that order; see [Inputs](inputs.md#variant-order-and-alleles).

`susie` centers `X` and `y` with `intercept=True` and scales predictors with
`standardize=True`, both defaults. `na_rm=True` omits rows with missing
outcomes; `compute_univariate_zscore=True` adds marginal z-scores. Dense
and sparse designs use matrix-vector operators without a dense Gram matrix;
sparse inputs remain sparse. Trend filtering also uses an implicit design.

For `susie_rss(..., X=X)`, rows are reference samples and may differ in
number from `n`. Multiple panels require a shared `z` and `n`; supply a list
of LD matrices as `R` or reference genotype matrices as `X`. SciPy is required
for sparse inputs, RSS-lambda, finite-reference and mismatch models, and
multi-panel RSS.

`susie`, `susie_suff_stat`, `susie_rss` and `susie_rss_lambda` accept
`init_only=True` to return `SusieInputs`; `prusie.susie_workhorse(inputs)`
fits the prepared request. Multi-panel RSS still performs per-panel selection
fits during preparation.

`susie_trendfilter(..., init_only=True)` with default `use_mad=True` and no
warm start returns nested preparation that the workhorse rejects. Set
`use_mad=False` to obtain directly reusable inputs. `susie_auto` requires
fitting and does not support `init_only=True`.

`model_init` accepts a previous fit or a posterior-state mapping; `s_init`
is its alias. Supplying both raises `ValueError`. See [convergence and
refinement](post_analysis.md) for warm-start validation and continuation.

Python warning filters, including `error`, and warning recording are supported.
Callbacks that mutate input arrays during a fit, and replacing `warnings.warn`
before importing prusie, are unsupported.

## Shared fitting options

Several entry points share these controls. Defaults and supported combinations
differ; consult each function's signature and the
[entry-point defaults](model.md#defaults-by-entry-point).

| Option | Meaning |
| --- | --- |
| `L=10` | Requested maximum number of single-effect components; count limits are described below |
| `n_threads=1` | Per-fit native thread limit, from 1 to 32 |
| `scaled_prior_variance=0.2` | Gaussian initial variance relative to working outcome variance; NIG slab variance relative to residual variance; scalar or length-`L` vector |
| `prior_variance=50` | Initial effect variance for RSS without `n`, and for RSS-lambda |
| `residual_variance=None` | Initial or fixed residual variance, defaulting to working outcome variance; in RSS-lambda it denotes total variance before subtracting `lambda_` |
| `estimate_prior_variance=True` | Update component prior variances |
| `estimate_prior_method="optim"` | Prior update: `"optim"`, `"EM"` or `"simple"`; `susie_ser` accepts only `"optim"` or `"simple"` |
| `prior_variance_grid`, `mixture_weights` | Fixed mixture of positive prior variances; equal weights by default; disables scalar prior-variance estimation |
| `estimate_residual_method="MoM"` | `"MoM"`, `"MLE"` or conjugate `"NIG"`; RSS-lambda accepts `"MLE"` only |
| `alpha0`, `beta0` | Positive NIG hyperparameters, each defaulting to $1/\sqrt n$ |
| `residual_variance_lowerbound`, `residual_variance_upperbound` | Bounds for residual variance updates |
| `prior_weights=None`, `null_weight=0` | Variant prior weights, uniform by default, and optional null-column weight |
| `slot_prior=None` | Optional effect-activity model from `slot_prior_betabinom` or `slot_prior_poisson` |
| `L_greedy`, `greedy_lbf_cutoff=0.1` | Grow the component count in steps, stopping at weak component evidence or the requested count |
| `convergence_method`, `convergence` | Ordinary default ELBO stopping; `"pip"` checks alpha and PIP. The sufficient-statistic/RSS default `None` preserves configured controls; `ConvergenceConfig` adds joint stopping where supported |
| `coverage=0.95`, `min_abs_corr=0.5` | Credible-set posterior mass and minimum absolute LD correlation thresholds |
| `n_purity` | X-based purity cap: default `None` in sufficient-statistic/RSS fits, `"auto"` in individual fits and RSS-lambda; LD matrices use all members |
| `median_abs_corr`, `cs_extension_corr` | Alternative median-purity threshold and correlated-variant CS extension |
| `maf`, `maf_thresh=0` | Keep variants with supplied `maf > maf_thresh`; available in sufficient-statistic and RSS entry points |
| `verbose=False` | Log sweep count, convergence and credible-set count through the `prusie` logger |
| `variant_ids`, `variant_metadata` | Ordered identifiers and aligned annotations |

For a cold start, `L` is capped at the predictor count including any null
column. RSS-lambda caps at retained variants after MAF filtering, excluding
null; `L=None` aliases 10 there. A larger supplied warm start can raise the
effective count.

`susie` and `susie_suff_stat` default to `max_iter=100` and estimated
residual variance. `susie_rss` and `susie_rss_lambda` default
to `max_iter=50` and fixed residual variance. Use fixed residual variance
with external LD unless estimation is justified; RSS warns when estimation
is requested.

Ordinary Gaussian tolerance is `1e-4`; RSS-lambda uses `1e-3`. In `susie`,
`susie_suff_stat` and `susie_rss`, `tol=None` selects `1e-6` for NIG and
`1e-4` otherwise. `susie` defaults to `None`; the other two default to
`1e-4`, including with NIG.

NIG forces residual-variance estimation and uses `EM` when estimating the
prior variance. A fixed mixture selects Gaussian SER updates instead of
conjugate NIG updates. Slot priors and NIG with multiple fitted components
use the PIP stopping rule. See [additional model options](model.md#additional-model-options)
and [convergence diagnostics](post_analysis.md#convergence-diagnostics).

`susie`, `susie_suff_stat`, `susie_rss` and `susie_rss_lambda` record
`n_threads` in `fit.params['n_threads']`. NumPy integers are accepted;
booleans, floats and counts outside 1–32 raise `ValueError`.
`PRUSIE_NUM_THREADS` does not override this public option. Large dense Gram
products can use a per-fit pool; small kernels stay serial, and IBSS
components remain sequential. This option does not parallelize
individual-level operators or dynamic RSS fitting. Eligible R-based RSS
calls can also parallelize LD preparation; `overwrite_ld=True` preparation
is serial. See [native threading](implementation.md#native-threading) for
build, storage and floating-point requirements. Process-wide BLAS, OpenMP
and Rayon settings are unchanged.

## Optional LD workspace

`susie_rss(..., overwrite_ld=False)` preserves caller arrays. Opting into
`overwrite_ld=True` allows `R` and all its aliases to become destructive
workspace for symmetry repair. This can avoid allocating a separate full
repair matrix. The flag accepts only Python `bool` or NumPy `bool_` values.

**After writes begin, later validation errors, fitting failures or warnings
promoted to exceptions can leave `R` and its aliases changed. There is no
rollback. Reload the original LD before reuse, including after a failed call.**
Do not access aliases concurrently with the fit. Other API arguments retain
their preservation contract.

The option requires a single `z`, `R` and supplied `n`, with `var_y=None`.
`R` must be a plain, square, aligned, writable native float64 NumPy array
with ordinary positive C-order strides and NumPy-owned heap storage. Reshape
views of such owners are accepted. Subclasses, memory maps and their views,
external buffers, read-only arrays, F-order arrays and strided arrays are
rejected. Other arguments must use supported plain scalars or flat nonobject
numeric/string vectors, with a plain `dict` for `variant_metadata`. Custom
conversion objects and any other argument overlapping `R`, including nested
or zero-dimensional views, are rejected before writes.

Keep `estimate_residual_method="MoM"`, `convergence_method=None` or `"elbo"`,
`R_mismatch="none"` and `control=None`. `R_finite` must be `None` or Python
`False`; NumPy `bool_(False)` is rejected. The option excludes reference `X`,
`bhat`/`shat`, MAF filtering, null columns, mixture/slot/NIG models,
`alpha0`/`beta0`, greedy growth, warm starts, `median_abs_corr`,
`cs_extension_corr`, explicit `convergence` objects, `check_input=True`,
tracking, retained data, refinement and `init_only=True`. Standardization,
prior weights, Gaussian fitting controls, basic CS options and `n_threads`
remain available. Unsupported storage and option combinations raise before
writes; use `overwrite_ld=False` for other routes.

Standard warning filters, recording and `logging.captureWarnings` are
supported. Warning display and tracing hooks do not affect eligibility;
callbacks must not modify input arrays during fitting.

On success, `fit.params["overwrite_ld"]` records the flag and
`fit.qc["ld_overwrite"]["action"]` records what happened. The accompanying
`original_rss_scale` is `n - 1`. Let `s = n - 1`, with `Rij` and `Rji`
denoting the original pair of matrix entries:

| Action | State of `R` after a successful call |
| --- | --- |
| `ordered_scaled_xtx` | Both orientations contain `(Rij*s + Rji*s)/2` in float64, in that operation order; `R` is now scaled `XtX` |
| `symmetric_no_write` | Exact-symmetry proof succeeded; original bytes retained |
| `no_witness_eager_no_write` | No finite pair had unequal scaled entries; ordinary owned preparation was used and original bytes retained |

See [dense storage and matrix products](implementation.md#dense-storage-and-matrix-products)
for the repair implementation. Other RSS likelihood controls, including
`z_method`, `R_finite`, `R_mismatch`, `susie_rss_control` and RSS-lambda's
`lambda_`, are described under [additional model options](model.md#additional-model-options).
Keep `z_ld_weight=0`; nonzero values raise `ValueError`. SuSiE-inf,
SuSiE-ash and Mr.ASH are not implemented.

## SusieResult

`SusieResult` is a mutable dataclass containing posterior arrays, credible
sets and the fitting record. Use attributes such as `fit.pip`, `fit.alpha`,
`fit.sets` and `fit.convergence`. See [Results](results.md#posterior-fields)
for field shapes, scales and model-specific outputs, including mixture-grid
Bayes factors, slot-activity probabilities and NIG quantities.

`params`, `qc` and entries in `cs_audit` are dictionaries with the exported
`FitParameters`, `InputChecks` and `CredibleSetAudit` `TypedDict` schemas.
`input_quality` is a frozen `InputQuality` dataclass describing recorded input
checks. Its `missing_summary_values` field records the policy (`"reject"` or
`"zero"`), even when no values were missing.

`fit.as_dict()` returns stored public fields without copying arrays or nested
containers, except that `input_quality` is converted to a dictionary. It omits
computed properties such as `coef` and private retained inputs. Copy values
before modifying them independently.

Retained working inputs are private, read-only snapshots. See
[convergence and post-analysis](#convergence-and-post-analysis) for supported
retention routes and later continuation or refinement.

## Posterior and model helpers

Use these top-level `prusie` functions to inspect a fit or prepare inputs.

| Functions | Output |
| --- | --- |
| `susie_get_pip` | Marginal PIPs from a fit or raw alpha matrix |
| `susie_get_cs` | Credible sets from a fit or raw alpha matrix; supply `X` or `Xcorr` for purity and extension |
| `susie_get_cs_attainable` | Credible sets screened by attainable mass and entropy, without LD filtering; `X` and `Xcorr` are ignored |
| `susie_get_posterior_mean`, `susie_get_posterior_sd` | Means and standard deviations over active components, on the input predictor scale; optional null included |
| `susie_get_lfsr` | Alpha-weighted Gaussian local false-sign rate for each component |
| `susie_get_posterior_samples` | Coefficient draws (`b`) and nonzero indicators (`gamma`); requires `num_samples` and keyword `seed` |
| `susie_get_objective`, `susie_get_niter` | Last recorded objective (`last_only=False` returns the trace), and completed IBSS sweeps |
| `susie_get_prior_variance`, `susie_get_residual_variance` | Fitted variance parameters |
| `get_cs_correlation` | Correlations between maximum-PIP CS members |
| `coef`, `predict`, `summary` | Coefficients including intercept, response predictions, and `vars`/`cs` lists of summary records |
| `compute_suff_stat` | Centered sufficient statistics, including dense `XtX` even for sparse `X` |
| `univariate_regression`, `calc_z` | Marginal coefficient estimates and standard errors, or their z-scores |
| `susie_init_coef` | Dense `alpha`, `mu` and `mu2` initialization arrays from nonzero working-scale coefficients |
| `susie_post_outcome_configuration` | Existing single-outcome fits: `method="susiex"` activation configurations or `"coloc_pairwise"` H0–H4 probabilities; requires SciPy |

`coef(fit)` prepends the intercept and includes any null column; `fit.coef`
contains only non-null predictor effects. Supply `predict` with columns in
`fit.variant_ids` order. Both `coef` and `predict` include `slot_weights`
for slot-prior fits. The moment, sign-rate and sampling helpers do not apply
these activity weights.

Sign rates and draws use a Gaussian distribution constructed from `mu` and
`mu2`; for mixture or NIG fits this is an approximation. `susie_get_lfsr`
pairs each mean with its own standard deviation; its output can differ from
susieR 0.16.6. Posterior sampling and X-based purity subsampling use local
NumPy RNGs; both accept `seed`. NumPy samples do not reproduce R's draws.
Sparse-X extension computes correlation blocks without densifying `X`.

`prusie.posterior.marginal_pip` and `credible_sets` accept alpha and
prior-variance arrays. For `credible_sets`, `covariance_scales` and
`covariance_multiplier` enable purity calculation directly from covariance.
Credible sets retain original component IDs; see
[credible-set fields](results.md#credible-set-fields).

## load_example

Load the packaged synthetic toy dataset and fit it with its supplied options:

```python
import prusie

example = prusie.load_example()
fit = prusie.susie_rss(**example["inputs"], **example["parameters"])
```

The returned dictionary contains `inputs` (z-scores `z`, signed LD `R`, sample
size `n` and ordered `variant_ids`), `parameters` (fitting settings) and
`metadata` (generation method, seed, true effects and sample size). The data
are simulated; no human study or genome build is implied.

Each call returns new, caller-owned input arrays. The loader reads installed
package resources from any working directory, without R or network access.
It checks SHA-256 hashes of the inputs and metadata, raising `ValueError` on
a mismatch. See the [toy example tutorial](example.md) for the workflow and
provenance.

## Convergence and post-analysis

Inspect `fit.convergence` before running more sweeps or searching for another
solution. `diagnose_convergence(fit)` copies the recorded diagnostics or reports
`stop_reason="not_available"` if none were stored. Native IBSS tracking with
`track_fit=True` or `"compact"` adds exact per-sweep metrics and a labelled
lossy alpha display to `fit.trace`; `"dense"` also stores full `alpha`, `mu`
and `mu2` arrays for debugging. Dynamic RSS traces contain metrics only.

`convergence=ConvergenceConfig(...)` selects `"elbo"`, `"alpha_pip"` or
`"joint"` where supported. The separate `convergence_method="pip"` uses the
susieR posterior rule. See [convergence diagnostics](post_analysis.md#convergence-diagnostics)
for stopping conditions, cycles and nonfinite objectives.

The following helpers use retained working inputs:

| Function | Behavior |
| --- | --- |
| `continue_fit` | Return a new fit from the saved posterior; `max_iter=100` gives an additional sweep budget and returned `niter` counts only those sweeps |
| `refine_fit` | Search alternative starting solutions for a higher logged ELBO; return a result with the search record in `refinement` |
| `evaluate_objective` | Recompute ELBO, KL and fitted products from the post-trimming posterior without running IBSS |

`refine_fit(..., track_fit=False)` leaves candidate sweeps unrecorded by default.
Use `True`, `"compact"` or `"dense"` to save each completed stage's trace in
`refinement["history"][i]["stages"][phase]["trace"]`. An accepted result keeps
its original stage-B trace; if no candidate is accepted, it keeps the input
fit's trace. The fitting-entry shortcut `refine=True` does not forward
`track_fit` to its search: entry-level tracking applies to the initial fit.
Call `refine_fit` explicitly to track the candidates.

Retention supports static, materialized Gaussian sufficient-statistic and RSS
fits with default or `ConvergenceConfig` stopping. Matrix-free designs, dynamic
RSS (including RSS-lambda and multiple panels), mixture/NIG/slot models,
`convergence_method="pip"`, `L_greedy`, `median_abs_corr` and `cs_extension_corr`
are outside this scope. `retain_data=True` or `refine=RefinementConfig(...)` is
rejected on unsupported fitting routes. On these routes, `refine=True` uses
a separate refitting search, except that multi-panel RSS rejects refinement.
Ordinary fitting and tracking remain
available without a snapshot; no dense Gram is built to enable retention of
a matrix-free design.

Set `retain_data=True` during a supported fit. Tracking, warm starts and
refinement also retain inputs within this scope. `data_signature` hashes the
owned snapshot once; unretained fits return `None`. A new fit using
`model_init`/`s_init` can use an unretained source posterior. Without retained
inputs, `continue_fit` and `evaluate_objective` raise `ValueError`, while
`refine_fit` records `status="unsupported"`.

Continuation and refinement inherit requested convergence controls unless
overridden, independently of runtime fallback. `continue_fit` with `tol=None`
inherits the fitting tolerance. `RefinementConfig` separates search thresholds
and budgets from candidate IBSS controls (`inner_tol`, `max_iter`). On the
returned result, refinement can overwrite `converged`; read
`convergence["ibss_converged"]` for IBSS status and `refinement["status"]` for
the search outcome. Reuse that result for plots and comparisons. See the
[convergence and refinement guide](post_analysis.md) for algorithms, costs,
warm-start validation and examples.

## Summaries, comparisons and reports

To inspect and share a fine-mapping analysis, summarize an existing fit and
pass the resulting mapping to the text or report helpers.

| Function | Purpose |
| --- | --- |
| `summarize` | Component, variant, credible-set, LD-proxy and prior-influence tables from one posterior; no refit or eigendecomposition |
| `summary_text` | Short string for `print`; no fitting, eigendecomposition or plotting |
| `compare_fits` | Compare named existing fits by variant ID, using `mode="union"` (default) or `"common"` |
| `run_sensitivity` | Execute a finite named plan: one-factor model changes, alternative inputs, posterior-only reports or continuation |
| `diagnose_ld` | Conditional z/LD diagnostics and possible allele-mismatch flags; no automatic flips or filtering |
| `write_report` | Offline HTML, text, CSV tables and full-precision evidence exports; include only supplied figures |

Supply optional `ld=` and `comparison=` records to `summary_text` or
`write_report`; neither function runs LD diagnostics or refits a model.
`write_report` returns artifact paths and checksums. Keep the report directory
together so its local links remain valid.

The first five functions are available from `prusie`. Import `write_report`
and the compact `analysis_summary` helper from `prusie.report`, which also
provides [`save_analysis` and `load_analysis`](results.md#metadata-and-persistence)
for JSON/NPZ persistence.
The [quick start](example.md#write-an-offline-report) gives a minimal report call;
[sensitivity and LD](sensitivity.md) specifies prior remapping, alternative inputs,
common/union support and missing-assessment states.

## Figures and local style

Use `prusie.viz` to inspect association signals, posterior uncertainty and
fitting diagnostics. Install the `analysis` extra for Matplotlib and SciPy.
`prusie.plots` exports the same functions for compatibility.

Each plotting function below returns `(figure, axes, plotted_data)`, with
numeric values and selection or display rules in `plotted_data`. All default
to `layout="full"`; `"compact"` reduces panels or spacing. Existing axes passed
with `axes=...` must belong to one figure and match the selected panel count.

| Function | View |
| --- | --- |
| `plot_locus` | Association, PIP, credible sets and reference $r^2$ from a summary; optional gene track |
| `plot_convergence` | Tracked fit or summary: `"compact"` shows ELBO, signed ΔELBO and maximum absolute changes in alpha and PIP; `"variance"` shows `V` and `sigma2`; `"full"` combines both |
| `plot_refinement` | Before/after PIPs, refit ELBO gains and accepted incumbent path |
| `plot_sensitivity` | Baseline-first PIP heatmap from a comparison; `candidates_only=True` and `layer="signals"` by default; `layer="components"` includes all active or CS-retained effects |
| `plot_signal` | One component's alpha, cumulative probability, within-CS LD and proxies from a summary; select with `component_id` or `variant_id` |
| `plot_prior_influence` | Supplied/effective prior, conditional log Bayes factors and alpha for one component |
| `plot_ld_diagnostics` | Observed/expected z-scores, assessment states, flip evidence and the joint flag rule from supplied diagnostics |

For LD views, pass `R=` in the summary's variant order to `plot_locus` or
`plot_signal`. Compact signal and prior-influence selections retain formal CS
members. Signal cumulative probabilities use the full ranking without
renormalizing. Sensitivity `display_names={stored_name: readable_label}` changes
labels while preserving fit identities and settings. `table_page` and
`table_page_size`, where supported, paginate visible tables while retaining
complete returned records.

The shared style uses a white background and consistent before/after colors.
Override Matplotlib settings locally with `style=`:

```python
from prusie.viz import plot_convergence

figure, axes, data = plot_convergence(
    fit, layout="compact", style={"font.size": 11, "axes.labelsize": 11},
)
figure.savefig("convergence.svg", bbox_inches="tight")
```

`style=False` uses the caller's current settings. Plotting restores global
rcParams on exit. Select a backend before plotting; `backend` is rejected as
a local style option.

### Heatmap scales and labels

PIP heatmaps use `Blues` on a fixed linear [0,1] scale. Within-CS LD defaults to
signed correlation $r$ with `RdBu_r` on [−1,1]; `ld_measure="r2"` selects $r^2$
with `Reds` on [0,1]. `cmap=` in `plot_sensitivity` or `plot_signal` changes that view's
colormap while preserving its numeric range and matrix. Missing PIPs are grey,
distinct from observed zero.

Sensitivity cells use exact `0`/`1`, `<0.01`/`>0.99` for interior tails, two
decimals otherwise, and `NA` for missing values. Label formatting leaves numeric
matrices unchanged; display text and policies are returned separately. All
selected candidate rows are labelled, including CS members on the declared
comparison support; large views
may omit cell annotations. See the
[support rules](sensitivity.md#common-and-union-comparison-support).

Within-CS LD cells are square. Small matrices distinguish near-unit correlations
from exact identity in their labels. Compact singleton views retain the matrix
and CS purity. See the [Toy example](worked_examples.md#what-remains-unresolved).

### Other display controls

`plot_convergence(..., relative_elbo=True)` displays each trace relative to its
first finite ELBO, with the origin labelled. Returned traces and signed
increments are unchanged.

Locus plots use physical positions when every variant has a finite position;
otherwise they use labelled variant indices. `position_unit="auto"` chooses
bp/kb/Mb; `"bp"`, `"kb"` or `"Mb"` selects the display unit explicitly without
changing source coordinates. `label_variants` chooses annotation IDs.
`plot_locus` shows one LD $r^2$ track per retained CS, using that component's
highest-alpha CS member; exact ties use lexical variant ID order. An explicit
`reference_variant` selects a single track. Without a retained CS, the reference
is the highest marginal PIP variant. `color_by_ld=True` colors association points
by the first displayed reference, which is named on the colorbar.

`read_gene_track` returns a mapping of intervals and provenance from a local GTF.
Gene tracks require physical variant positions, source, genome build, transcript
selection and `1-based closed` coordinates. Known chromosome/build conflicts
are rejected. No gene identity is inferred from rsIDs or arbitrary example
coordinates.

For LD diagnostics, `auxiliary="residual"` (default) or `"log_lr"` selects the
supporting panel. Labelled extreme residuals remain distinct from variants
satisfying the joint flag rule.
