# Python API

The fitting entry points are `susie_rss` and `susie_suff_stat`, both returning
`SusieResult`. `load_example` supplies the packaged toy data; the analysis
helpers below operate on a completed fit. Numerical fitting inputs are converted
to float64 as needed.
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
[API source](https://github.com/LucaJiang/prusie/blob/post_analysis/src/prusie/api.py).

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

## Convergence and post-analysis

Both fitting interfaces support `s_init=None`, `refine=False`,
`track_fit=False`, `convergence=None` and `retain_data=False`.
`ConvergenceConfig` selects exact ELBO, alpha/PIP or joint stopping.
`track_fit=True` records compact exact metrics and a labelled lossy alpha display;
`track_fit='dense'` retains full states for small debugging inputs.

`continue_fit`, `refine_fit`, `evaluate_objective`, `summarize`, `diagnose_ld`,
`compare_fits` and `run_sensitivity` are independent explicit operations.
`RefinementConfig` separates search thresholds/budgets from inner IBSS tolerances.
See [convergence and refinement](post_analysis.md) for costs, initialization
checks, phases and runnable examples. `continue_fit(fit, max_iter=100, tol=None)`
inherits the original ELBO tolerance when `tol` is omitted. Requested convergence
controls persist across continuation, refinement and sensitivity; effective
fallback is recorded separately.

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
common/union support and missing-assessment states. Optional AI adapters live in
`prusie.ai`; ordinary summaries and reports are independent of them.

## Figures and local style

Every function in `prusie.plots` returns `(figure, axes, plotted_data)`. The third
item records selection rules and plotted values alongside the complete evidence
needed to interpret the view. Existing Matplotlib axes can be passed with
`axes=...`; their count must match the selected layout.

| Function | View and layout choices |
| --- | --- |
| `plot_locus(summary, *, R=None, layout="full", ...)` | Association, PIP, CS and reference r² on the same coordinates; optional annotated gene track |
| `plot_convergence(fit_or_summary, *, after=None, layout="full", ...)` | `compact`: ELBO, signed ΔELBO, Δalpha, ΔPIP; `variance`: V and sigma²; `full`: all six metrics |
| `plot_refinement(before, after, *, layout="full", ...)` | Candidate PIPs, stage-B gains and accepted objective path; `compact` moves long evidence out of the figure |
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
from prusie.plots import plot_convergence

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
