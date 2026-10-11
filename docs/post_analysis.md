# Convergence and refinement

A stopping rule tells you whether one IBSS trajectory has settled under its
selected criterion. Continuation gives an unfinished trajectory more iterations;
refinement searches for a better solution from different initial states. Read
these alongside [PIPs and credible sets](results.md), then compare justified
[alternative settings](sensitivity.md).

## Start with a tracked fit

```python
import prusie
from prusie.plots import plot_convergence, plot_refinement

example = prusie.load_example()
fit = prusie.susie_rss(**example["inputs"], **example["parameters"],
                       track_fit="compact", retain_data=True)
print(fit.convergence["stop_reason"])
figure, axes, plotted = plot_convergence(fit, layout="compact")
```

For this toy input the selected ELBO rule is satisfied after three sweeps.
The [worked examples](worked_examples.md) also show a ten-sweep fit that satisfies
its rule yet improves substantially under refinement.

The compact figure shows ELBO, signed ΔELBO, Δalpha and ΔPIP. Use
`layout="variance"` for V and residual variance, or `layout="full"` for all six
metrics. Selected stopping thresholds and diagnostic-only thresholds have
different line styles. A missing trace stays unavailable; it is not reconstructed
from the final fit.

Tracking is opt-in and does not select a different stopping rule. `retain_data=True`
owns a read-only copy of the validated matrix and scale metadata for explicit
continuation and refinement; tracking or initialization also retains those inputs.
Ordinary calls with these options disabled do not retain the matrix.

## Read the stopping evidence

For biological variant $j$, marginal PIP combines only active components:

$$
\mathrm{PIP}_j = 1-\prod_{l:V_l>\mathrm{prior\_tol}}(1-\alpha_{lj}).
$$

The internal null column is excluded. No active components gives zero PIP.
Component alpha and marginal PIP are different: components can exchange their
probability allocations while PIP stays unchanged.

At the end of each complete sequential IBSS sweep, before any residual variance
update, prusie records ELBO and exact distances from the preceding sweep:

$$
\Delta_\alpha=\max_{l,j}|\alpha^{(t)}_{lj}-\alpha^{(t-1)}_{lj}|,\qquad
\Delta_{\mathrm{PIP}}=\max_j|\mathrm{PIP}^{(t)}_j-\mathrm{PIP}^{(t-1)}_j|.
$$

Each PIP uses its own iteration's activity state. Alpha distance includes null;
PIP distance does not. The posterior-mean distance uses
$\bar b_j=\sum_l\alpha_{lj}\mu_{lj}$ on the working predictor scale. Signed
variance changes, top variants, entropy, activity and events accompany these
metrics. Online entropy includes the null column; the posterior resolution table
instead normalizes biological alpha and reports null mass separately.

The first sweep has no preceding recorded sweep: its distances are unavailable.
`fit.convergence` includes requested/effective criterion, thresholds, iteration
count, stop reason, last distances, nonfinite/decrease events and finalization.
`fit.converged` and `fit.ibss_converged` describe the returned **inner IBSS** fit.

The default finite-ELBO rule, starting at iteration two, is
$0\leq\Delta\mathrm{ELBO}<\mathrm{tol}$. Even a small negative increment does not
satisfy it. Relative change is separately reported as
$\Delta\mathrm{ELBO}/\max(1,|\mathrm{ELBO}_{t-1}|)$ and is not a default stopping
condition. A nonfinite increment uses a distinctly labelled alpha/PIP fallback;
a nonfinite objective is never eligible for refinement selection.

```python
rule = prusie.ConvergenceConfig(criterion='joint', alpha_tol=1e-6,
                                 pip_tol=1e-6, min_iter=2, cycle_lag=5)
fit = prusie.susie_rss(**example['inputs'], **example['parameters'],
                       convergence=rule, track_fit=True)
```

`alpha_pip` requires both probability distances to be strictly below their
separate tolerances. `joint` also requires the finite-ELBO condition. Tolerances
for alpha and PIP are absolute probability units; ELBO tolerance is an objective
increment. A lag-one fixed point is distinct from a short cycle. A detected cycle
of lag two through five retains the complete current posterior and reports
`cycle_detected`, with IBSS convergence false. Alpha is never averaged while
leaving its moments or variance parameters from another state.

After a non-stopping sweep, residual variance may be updated. Final trimming
then zeros components with $V_l<\mathrm{prior\_tol}$, including at the iteration
limit. Historical `elbo` and `XtXr` fields remain at their established phases;
`evaluate_objective(fit)` explicitly recomputes likelihood, KL and fitted
products from the complete saved state. `convergence['finalization']` separates
trim changes from IBSS iterations. The CS activity threshold remains $10^{-9}$;
a custom PIP `prior_tol` can therefore produce different boundary activity.

Compact tracking (`True` or `'compact'`) records exact online distances and
component summaries. Its alpha display retains rows with maximum alpha at least
$2/p$ and entries at least $1/p$, where $p$ includes any internal null column.
Thresholds, omitted mass and `lossy=True` are explicit. This display cannot
reconstruct the posterior. `'dense'` also records full alpha and moments and is
intended for small debugging examples. No history is fabricated when tracking
was disabled.

## Continue first, then consider refinement

An iteration-limit fit is unfinished under its stopping rule. Explicit
continuation can finish its inner optimization:

```python
continued = prusie.continue_fit(fit, max_iter=100, tol=1e-5, track_fit=True)
```

Continuation inherits the original requested convergence configuration unless
you supply `convergence=ConvergenceConfig(...)`. An omitted continuation `tol`
also inherits the fit's ELBO tolerance; an explicit value overrides it.
Refinement inherits the same
alpha/PIP thresholds, minimum iteration count and cycle controls; `inner_tol`
sets its separate inner ELBO tolerance. Runtime fallback from a nonfinite ELBO
is recorded as an effective rule and is not reused as the next requested rule.
Older saved fits without the requested configuration use a labelled compatibility
resolution; the result records resolved controls rather than claiming exact
inheritance. Inspect per-stage `solver_settings` and `convergence` for each
refinement A/B fit.

Alternatively, pass `s_init=fit` to the same-data fitting interface. Variant
IDs/order, supplied allele labels, null layout, working scales and the data
signature must agree. Unknown cross-input transformations are rejected. A minimal
state mapping must explicitly supply `alpha`, `mu`, `mu2`, `variant_ids`,
`scales`, `null_index` and `data_signature`, plus allele metadata when present.
Optional V and sigma2 are checked; estimated parameters can resume their values,
while fixed parameters follow the requested model. Growing L retains the first
components and adds zero-mean prior rows; shrinking L keeps the first L. Fitted
crossproducts are rebuilt from the current posterior and inputs. Old KL, BF and
fitted caches are never trusted.

Refinement searches other solutions after a valid converged baseline:

```python
refined = prusie.refine_fit(fit, prusie.RefinementConfig(
    max_rounds=10, absolute_tol=1e-3, relative_tol=None,
    max_iter=100, inner_tol=1e-4, candidate_budget=20))
```

For every retained CS in the round's fixed incumbent:

1. Temporarily zero its variant weights, normalize the remaining **full** prior
   including null, and fit from a cold start. Zero support is a hard exclusion.
2. Restore the exact original prior and model, transfer only alpha/mu/mu2, and
   refit. V and sigma2 restart at the original model initialization.

The best finite, valid, converged stage-B candidate is accepted only if its
common recomputed original-prior post-trim objective improves by **more than** the
absolute tolerance. Stage A has a different prior and cannot win this
comparison. Candidate states share immutable input matrices but own their mutable
posterior arrays. A failed final attempt never replaces an earlier accepted best.

The default absolute threshold avoids accepting numerical noise. The optional
relative threshold is disabled because its scale depends on sample size and
objective constants. A candidate budget counts exclusion attempts; an attempt
uses at most two fits. Round or candidate limits report `budget_exhausted`, not
a completed search. `refinement_status`, `refinement_stop_reason`, per-stage
stops/times, original CS identities, candidate improvements and before/after
summaries are separate from IBSS convergence. No CS, no improvement, unsupported
baseline, all candidates failed, and exhausted budgets have distinct outcomes.
No improvement describes this search only, not all possible local optima.

`evaluate_objective` follows the fitting convention `log(pi + sqrt(eps))` for
ordinary priors and zero KL for trimmed V=0 components. This explains small
logged-versus-final differences. Refinement deliberately corrects upstream's
approximate CS exclusions: the temporary stage-A zero weights remain exactly
zero. Returned default standard fits retain the established epsilon convention.

## Compare before and after

```python
refined = prusie.refine_fit(fit)
convergence_figure = plot_convergence(fit, after=refined, layout="compact")
search_figure = plot_refinement(fit, refined, layout="compact")
```

Before and returned-after trajectories use separate iteration axes. If the search
accepted no improvement, the label is **After search (unchanged)**. The refinement
view shows candidate PIPs, stage-B improvements relative to each round's incumbent,
and the accepted objective path. Full attempt logs distinguish failures,
unconverged candidates, gains below the acceptance threshold, valid candidates
that did not win, and accepted candidates.

Use the recomputed **post-trim original-model objective** to assess the gain.
The final logged sweep ELBO has a different phase and can differ slightly.
Candidate attempts are separate solutions; connecting their PIPs across variant
positions would not describe a fitted curve. `layout="full"` and the report's
complete tables expose additional variants and component relations.

## Compare explicit settings

Continue with [sensitivity and LD diagnostics](sensitivity.md#compare-explicit-settings)
for L, prior variance, variant priors, windows and report-only CS thresholds.
Requested convergence controls are inherited by sensitivity fits; an explicit
override is recorded. Runtime fallback never silently becomes the next requested
stopping rule.

## Figures, persistence and optional interpretation

See [the quick start](example.md#write-an-offline-report) for HTML, figure and
lossless data exports, and [the plotting API](api.md#figures-and-local-style) for
layouts and local styles. Reports summarize deterministic results without fitting
again. Optional AI interpretation remains separate from those results and requires
an explicit request and consent; offline analysis needs no model service.
