# Convergence and refinement

A fit can satisfy its stopping rule while a better solution remains available
under the same model. Refinement searches alternative starting solutions for a
higher evidence lower bound (ELBO).

Start by inspecting [why the fit stopped](#convergence-diagnostics).
`continue_fit` runs more sweeps from the saved posterior; `refine_fit` searches
for an improvement. Both require retained inputs. Compare PIPs and credible sets
before and after the search to assess its effect on the fine-mapping results.

## Start with a tracked fit

To see how a fit approached its stopping rule, set `track_fit="compact"` when
fitting. This records diagnostics after each complete pass through the
single-effect components. In the official susieR `FinemappingConvergence` toy example below, tracking retains
validated inputs for continuation and refinement.

```python
import prusie
from prusie.viz import plot_convergence, plot_refinement

example = prusie.load_convergence_example()
fit = prusie.susie_suff_stat(
    **example["inputs"],
    **example["parameters"],
    track_fit="compact",
)
print("Stop:", fit.convergence["stop_reason"], "Iterations:", fit.niter)
convergence_view = plot_convergence(fit, layout="compact")
```

Expected output:

```text
Stop: criterion_satisfied Iterations: 10
```

The compact plot shows the logged ELBO, its signed change, and the maximum
absolute changes in component inclusion probabilities (`alpha`) and marginal
PIPs between successive sweeps. `layout="variance"` shows the component prior
variances and residual variance; `layout="full"` combines both views.

## Continue first, then consider refinement

If a fit reaches its iteration limit, more IBSS sweeps may be enough to satisfy
its stopping rule. `continue_fit` starts from the saved posterior:

```python
fit_for_search = fit
if fit.convergence["stop_reason"] == "iteration_limit":
    fit_for_search = prusie.continue_fit(fit, max_iter=100, track_fit="compact")
print("IBSS converged:", fit_for_search.ibss_converged)
```

For the toy fit above, this branch is skipped because the stopping rule was
already satisfied. `max_iter` gives the continuation call an additional sweep
budget; its returned `niter` counts only those new sweeps. Unless overridden,
the call inherits the requested convergence controls and fitting tolerance.
Supply `tol` or `convergence=prusie.ConvergenceConfig(...)` to change the
corresponding controls. Inspect the [convergence diagnostics](#convergence-diagnostics)
before extending a fit with an objective anomaly or recorded cycle.

Refinement searches alternative starting solutions under the same model. It
requires retained inputs and a valid baseline with a finite ELBO; neither the
baseline nor the candidate IBSS fits need to have converged. Run a search with
explicit settings:

```python
search_config = prusie.RefinementConfig(
    max_rounds=10,
    absolute_tol=1e-3,
    relative_tol=None,
    max_iter=100,
    inner_tol=1e-3,
    candidate_budget=20,
)
refined = prusie.refine_fit(fit_for_search, search_config, track_fit="compact")
search = refined.refinement
print("Search:", search["status"])
print(
    "Accepted attempts:", sum(row["accepted"] for row in search["history"]),
    "of", len(search["history"]),
)
```

Refinement inherits the baseline's requested convergence controls, even after a
runtime fallback; `convergence=...` overrides them. `inner_tol` sets `tol` for
each inner fit, and `max_iter` limits its sweeps. `max_rounds` limits search
rounds and `candidate_budget` limits exclusion attempts, each using at most two
fits. `relative_tol=None` disables the relative-gain stopping rule.

Stage A sets the prior weights of one retained credible set to zero and fits
from a cold start. The zeroed weights are softened by ordinary prior
stabilization. Stage B restores the original prior and starts from stage A's
`alpha`, `mu` and `mu2`, resetting the prior and residual variances to the
original model's initial values.

The search ranks candidates by their last logged ELBO under the original prior,
before final component trimming. The best valid, finite candidate in a round
replaces the current fit only if its gain exceeds `absolute_tol`. The recomputed
post-trimming ELBO is a validity check and diagnostic. An unsuccessful search
does not establish a global optimum.

## Refinement compared with susieR

When comparing refinement results, match the search settings as well as the
model settings. For Gaussian RSS fits, `refine=True` calls `refine_fit`;
other fitting routes can use a
[separate refitting search](implementation.md#refinement-continuation-and-warm-starts).
The table compares `refine_fit` without an explicit
`RefinementConfig` with susieR **0.16.6**, commit
`8e56a8e038e989856d106d9ca5175cc664fea9d2`.

| Step or control | prusie | Pinned susieR |
| --- | --- | --- |
| Stage A | Cold start; zero CS weights retain the ordinary `log(pi + sqrt(eps))` stabilization | Same; zero weights do not impose strict exclusion |
| Stage B | Restore the original prior; transfer A's `alpha`, `mu` and `mu2`; reset `V` and `sigma2` to original initial values | Same posterior transfer, with ordinary variance initialization |
| Ranking objective | Stage B's last logged ELBO under the original prior | Same, through `susie_get_objective` |
| Required absolute gain | Strictly greater than the original fit's `tol` | Same |
| Relative stop after acceptance | Accepted gain divided by `max(1, abs(new ELBO))` is below the original fit's `tol` | Same |
| Inner IBSS convergence | Not required for the baseline or candidates; outer search status can replace `converged` | Not required; the outer search replaces `converged` |
| Search limits | 100 rounds; an explicit configuration can also set `candidate_budget` | 100 rounds; no separate attempt budget |
| Candidate validity and errors | Require finite logged and recomputed ELBOs; record and reject caught fitting or validation errors while retaining the current best fit | No per-candidate error recovery or nonfinite filtering in `run_refine`; errors propagate |

Without an explicit configuration, prusie forwards the original fit's `max_iter`
and `tol` to each inner fit, as susieR does. Constructing `RefinementConfig()`
instead selects `max_rounds=10`, `absolute_tol=1e-3`, `max_iter=100`,
`inner_tol=1e-4` and `relative_tol=None`. These settings can differ from the
baseline's; the preceding example also sets an attempt budget.

Use `refinement["status"]` to read the search outcome. When `refine_fit`
overwrites `converged`, it saves the preceding flag in
`convergence["inner_ibss_converged"]`. Agreement of fits without refinement does
not establish numerical equivalence of refinement. For the algorithm
definitions, see the
[prusie implementation](https://github.com/LucaJiang/prusie/blob/main/src/prusie/refinement.py),
[pinned R search](https://github.com/stephenslab/susieR/blob/8e56a8e038e989856d106d9ca5175cc664fea9d2/R/refinement.R)
and [pinned R objective accessor](https://github.com/stephenslab/susieR/blob/8e56a8e038e989856d106d9ca5175cc664fea9d2/R/susie_get_functions.R).
The [Toy example](worked_examples.md) documents this official fixture with its
software versions and settings. Read its outcome separately from the
[comparison of fits without refinement](numerical_accuracy.md).

## Compare before and after

Compare changes in variant support alongside the objective gains that led to
acceptance. Reuse `fit_for_search` and `refined` from above; these plotting calls
do not repeat the search:

```python
convergence_figure, _, _ = plot_convergence(
    fit_for_search, after=refined, layout="compact"
)
search_figure, _, _ = plot_refinement(fit_for_search, refined, layout="compact")
print("Same PIPs:", bool((fit_for_search.pip == refined.pip).all()))
```

This official example prints `Same PIPs: False`: refinement accepts a higher-ELBO
solution under the same model. Tracking captures the baseline's ten sweeps and
the accepted candidate's three original stage-B sweeps. Separate fits use
independent sweep axes.

The refinement plot compares the two fits' PIPs, each attempt's ELBO gain over
the best fit at the start of its round, and the sequence of best ELBOs. These
gains use the last logged ELBO under the original prior. The compact layout
displays selected variants; `candidates_only=False` shows all variants.
`layout="full"` adds credible-set memberships and attempt records.

Read `accepted`, `reason` and `stages` in `refined.refinement["history"]` to
distinguish selection from inner IBSS convergence. Refinement defaults to
`track_fit=False`; enable it explicitly to save candidate trajectories. It
does not reconstruct a trajectory for an already fitted result.

The [saved Toy example](worked_examples.md#a-stopped-fit-improved-by-refinement)
shows changed PIPs and credible sets after the initial fit satisfies its
stopping rule. Its own record defines the ELBO comparison used there.

## Convergence diagnostics

Read the stopping rule alongside changes in variant support. `fit.convergence`
records the requested and effective criteria, thresholds and stop reason;
`fit.convergence["last"]` holds the final sweep's metrics and events.
`fit.trace` holds the per-sweep records when tracking is enabled.

For the returned fit's inner IBSS status, read
`fit.convergence["ibss_converged"]`. The property `fit.ibss_converged` aliases
`fit.converged`, which refinement can overwrite with its search outcome.
Read `fit.refinement["status"]` separately.

Component probabilities can move while marginal PIPs stay similar. For sweep
$t$, the maximum absolute changes are

$$
\Delta_\alpha=\max_{\ell,j}|\alpha^{(t)}_{\ell j}-\alpha^{(t-1)}_{\ell j}|,\qquad
\Delta_{\mathrm{PIP}}=\max_j|\mathrm{PIP}^{(t)}_j-\mathrm{PIP}^{(t-1)}_j|.
$$

Here $\alpha_{\ell j}$ is the probability that component $\ell$ selects variant
$j$. Alpha distance includes any null column; PIP distance excludes it. PIPs
combine components with $V_\ell>\mathtt{prior\_tol}$, as described under
[PIP and credible sets](model.md#pip-and-credible-sets). The first sweep has no
preceding sweep, so its distances are unavailable.

The ordinary ELBO rule requires a finite objective and a finite signed increment
$0\leq\Delta\mathrm{ELBO}<\mathtt{tol}$. A negative increment does not satisfy
it or trigger a posterior fallback. Relative ELBO change is a diagnostic, rather
than this stopping condition.

`ConvergenceConfig` selects `elbo`, `alpha_pip` or `joint`. The `alpha_pip` rule
requires $\Delta_\alpha<\mathtt{alpha\_tol}$ and
$\Delta_{\mathrm{PIP}}<\mathtt{pip\_tol}$; both tolerances default to `1e-4`.
The `joint` rule also requires the ELBO condition. For example:

```python
rule = prusie.ConvergenceConfig(
    criterion="joint", alpha_tol=1e-6, pip_tol=1e-6
)
```

Pass `convergence=rule` when fitting, continuing or refining. `min_iter=2` is
the default; a larger value delays successful convergence and cycle stops.
Cycle detection checks for repeated alpha and PIP states up to `cycle_lag`
sweeps earlier (default 5). Repetition at lags greater than one is diagnostic
only under a finite ELBO in `elbo` mode. In explicit `alpha_pip` or `joint`
mode, an eligible cycle stops with `cycle_detected` and inner IBSS convergence
false if the requested success condition has not already held.

The [PIP stopping rule](model.md#stopping-rules), selected with
`convergence_method="pip"` or automatically for some models, uses `tol` for
both distances. An eligible cycle counts as convergence and averages `alpha`
over the cycle. That rule also seeds the preceding PIP at its first check using
the current component variances; later PIPs retain their own sweep's activity.

The following nonfinite-objective rules describe the native Gaussian loop.
The dynamic RSS path has its own posterior-validity checks; see
[convergence outcomes](implementation.md#convergence-outcomes).

| Objective state | Behavior |
| --- | --- |
| NaN ELBO in `elbo` mode | At or after `min_iter`, record `alpha_pip_nonfinite_fallback` and use posterior checks; the default ELBO fit without a `convergence` configuration uses the PIP rule above |
| NaN ELBO in `alpha_pip` mode | Apply the requested posterior rule and record the nonfinite objective |
| NaN ELBO in `joint` mode | Stop with `numerical_failure` and `converged=False`, even before `min_iter` |
| Infinite computed ELBO | Raise `ValueError` |

If a NaN ELBO accompanies nonfinite posterior arrays, fitting raises an error.
A nonfinite objective cannot win refinement selection. Read
`criterion_satisfied` with the effective rule and objective validity. If no
stopping condition holds at `max_iter`, the inner fit reports `iteration_limit`
and convergence false.

## Technical details

<details markdown="1">
<summary>Iteration phases and objective values</summary>

Online metrics describe each completed sweep before any residual-variance
update, cycle averaging or final trimming. For ordinary Gaussian fits,
posterior-mean distance uses
$\bar b_j=\sum_\ell\alpha_{\ell j}\mu_{\ell j}$ on the working predictor
scale. The relative ELBO increment is
$\Delta\mathrm{ELBO}/\max(1,|\mathrm{ELBO}_{t-1}|)$.
Online component entropy includes the null column; `summarize` instead
normalizes alpha over biological variants and reports null mass separately.

An iteration-limit exit can return a residual variance updated after the last
ELBO. Final trimming zeros `V`, `mu` and `mu2` for components with
$V_\ell<\mathtt{prior\_tol}$, retaining their rows. In the retained Gaussian
path, stored `elbo` and `XtXr` are not recomputed after trimming.
`fit.convergence["finalization"]` records trimming distances and the returned
residual variance. [Results](results.md#iterations-and-finalization) describes
these phases and the separate PIP and credible-set activity thresholds.

`evaluate_objective(fit)` recomputes the ELBO from retained inputs and the
returned posterior, without using cached KL, Bayes factors or fitted products.
It can materialize an additional dense working matrix. Ordinary prior
stabilization is `log(pi + sqrt(eps))`, where `pi` is a normalized prior weight
and `eps` is float64 machine epsilon. Components with `V=0` contribute zero KL.
Refinement stage A uses this same softened zero-weight convention.

</details>

<details markdown="1">
<summary>Retained inputs and warm starts</summary>

For [supported Gaussian models](api.md#convergence-and-post-analysis),
`retain_data=True` keeps read-only input snapshots and scale metadata.
Tracking, initialization and refinement also imply retention within this scope.
Continuation and refinement share these owned inputs and their cached content
signature, `data_signature`. Later changes to caller arrays do not alter the
snapshot. Ordinary unretained fits have `data_signature=None`; a missing
signature cannot establish input identity.

`save_analysis` preserves an available signature but excludes the private input
snapshot. A [reloaded fit](example.md#save-and-reload-a-result) can initialize a
new fit when inputs are supplied again with `model_init=restored` (`s_init` is
an alias). It cannot be passed directly to `continue_fit`.

A new fit can use a posterior without a retained-data signature.
Working-coordinate mappings require `alpha`, `mu` and `mu2`; supplied variant
IDs/order, allele labels, scales and null layout must agree with the new inputs.
Supplied `V` initializes even fixed-prior fits; `sigma2` is resumed only when
residual variance is estimated. Increasing $L$ preserves existing row order and
adds uniform alpha rows with zero moments. Requesting fewer effects preserves
the state's effect count. Fitted products are rebuilt from the posterior and
inputs; old KL, Bayes factors and fitted caches are not reused.

</details>

<details markdown="1">
<summary>Tracking detail and search records</summary>

Compact tracking (`True` or `"compact"`) records exact online distances and a
lossy alpha display. Let $q$ be `alpha.shape[1]`, including any null column.
The display retains rows whose maximum alpha is at least $2/q$ and entries
at least $1/q$. Thresholds, omitted mass and `lossy=True` are recorded; these
entries cannot reconstruct the posterior. `"dense"` also stores full `alpha`,
`mu` and `mu2`. Tracking level changes the stored history, rather than the fit
or stopping rule. [Cycle handling](#convergence-diagnostics) depends on that rule.

`refine_fit(..., track_fit=True)` records both stages of each candidate in
`refinement["history"][i]["stages"][phase]["trace"]`. An accepted result's
`trace` contains its original stage-B trajectory. If no candidate is accepted,
the returned copy keeps the input fit's trace. Candidate traces are not saved
by default, even if the baseline was tracked.

Requested controls are read from `params["convergence"]`, with labelled recovery
from diagnostics or retained options for older results. If those controls are
unavailable, the record identifies compatibility defaults. Completed inner fits
record their resolved controls in the stage's `solver_settings` and
`convergence` entries.

Each search round keeps its incumbent and credible-set list fixed. Candidates
share the retained inputs and own their posterior arrays; a later failed
attempt does not discard an earlier accepted best. `refinement["config"]`
records the resolved search settings; the
[default comparison](#refinement-compared-with-susier) explains how an explicit
configuration changes them. The candidate budget counts exclusion attempts,
each using at most two fits. Round or candidate limits report
`budget_exhausted`; read search status separately from IBSS stopping.

</details>

Continue with [sensitivity and LD diagnostics](sensitivity.md#compare-explicit-settings)
to compare modeling or input choices. The [quick start](example.md#write-an-offline-report)
and [plotting API](api.md#figures-and-local-style) cover reports, persistence and
local styles.
