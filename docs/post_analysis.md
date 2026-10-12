# Convergence and refinement

Read why a fit stopped, give an unfinished fit more iterations when appropriate,
then search for a better solution and compare the returned results. A stopping
rule describes one IBSS trajectory; refinement explores other starting solutions.

## Start with a tracked fit

```python
import prusie
from prusie.plots import plot_convergence, plot_refinement

example = prusie.load_example()
fit = prusie.susie_rss(**example["inputs"], **example["parameters"],
                       track_fit="compact", retain_data=True)
print("Stop:", fit.convergence["stop_reason"], "Iterations:", fit.niter)
convergence_view = plot_convergence(fit, layout="compact")
```

Expected output:

```text
Stop: criterion_satisfied Iterations: 3
```

Enable `track_fit` during fitting to retain iteration history. The compact plot
shows logged ELBO, signed ΔELBO, Δalpha and ΔPIP. `layout="variance"` shows prior
and residual variances separately; `layout="full"` includes all six metrics.
`retain_data=True` preserves validated inputs for continuation and refinement.

## Continue first, then consider refinement

Reuse a completed fit. For an iteration-limit exit, an explicit continuation can
give the same posterior more sweeps:

```python
fit_for_search = fit
if fit.convergence["stop_reason"] == "iteration_limit":
    fit_for_search = prusie.continue_fit(fit, max_iter=100, track_fit="compact")
print("Ready for search:", fit_for_search.ibss_converged)
```

This example prints `Ready for search: True` and reuses `fit`, because the
initial fit already satisfied its rule. A continuation call starts from the
supplied posterior, retains the original requested criterion and alpha/PIP,
minimum-iteration and cycle controls, and inherits the ELBO tolerance when `tol`
is omitted. Its `niter` counts the new call's sweeps. An explicit `tol` or
`convergence=ConvergenceConfig(...)` overrides the corresponding controls.
Investigate an objective anomaly or cycle before deciding to continue it.

Now run one refinement search with explicit search settings:

```python
search_config = prusie.RefinementConfig(
    max_rounds=10, absolute_tol=1e-3, relative_tol=None,
    max_iter=100, inner_tol=1e-4, candidate_budget=20,
)
refined = prusie.refine_fit(fit_for_search, search_config)
search = refined.refinement
print("Search:", search["status"])
print("Accepted attempts:", sum(row["accepted"] for row in search["history"]),
      "of", len(search["history"]))
```

For the packaged example:

```text
Search: no_improvement
Accepted attempts: 0 of 2
```

Refinement requires a converged baseline with retained inputs. It inherits the
baseline's **requested** convergence configuration unless explicitly overridden;
`search_config.inner_tol` separately sets the inner ELBO tolerance. A runtime
alpha/PIP fallback after a nonfinite ELBO is recorded as the effective rule and
does not replace the original request for later calls.

Each exclusion attempt has two stages: A temporarily excludes one retained CS
and fits from a cold start; B restores the original prior and model and starts
from A's posterior moments. Only valid, finite, converged B candidates compete
under the common original-prior **post-trim objective**. The best candidate must
improve it by more than `absolute_tol`. No accepted attempt means stability under
this executed search, not a global-optimum certificate.

## Compare before and after

Reuse the `refined` result and the configuration above; these plotting calls do
not repeat the search:

```python
convergence_figure = plot_convergence(fit_for_search, after=refined, layout="compact")
search_figure = plot_refinement(fit_for_search, refined, layout="compact")
print("Same PIPs:", bool((fit_for_search.pip == refined.pip).all()))
```

This example prints `Same PIPs: True`; the returned trajectory is labelled
**After search (unchanged)**. Before and after have independent iteration axes.
The refinement figure compares candidate PIPs, B-stage gains relative to each
round's incumbent, and the accepted objective path. Full records distinguish
failed, unconverged, below-threshold, valid but unselected, and accepted candidates.

The [problem example](worked_examples.md#a-stopped-fit-improved-by-refinement)
illustrates a different outcome: its initial fit satisfies the stopping rule,
yet refinement improves the comparable objective by 17.4583. The final logged
sweep ELBO precedes finalization and can differ from the post-trim objective;
use the latter for candidate comparisons.

## Read the stopping evidence

`fit.convergence` contains requested/effective criterion, thresholds, iteration
count, stop reason, last distances and recorded events. `fit.converged` and
`fit.ibss_converged` describe the returned **inner IBSS** fit, independently of
search status.

| Evidence | Reading question |
| --- | --- |
| Logged ELBO and signed ΔELBO | Is the recorded objective still improving, flat, decreasing or nonfinite? |
| Maximum absolute Δalpha | Are the component probabilities still moving? |
| Maximum absolute ΔPIP | Is marginal variant inclusion still moving? |
| Requested/effective criterion | Which condition was selected, and did a fallback occur? |
| Stop reason | Was the criterion satisfied, a cycle detected, or the iteration cap reached? |

For biological variant $j$, marginal PIP combines active components:

$$
\mathrm{PIP}_j = 1-\prod_{l:V_l>\mathrm{prior\_tol}}(1-\alpha_{lj}).
$$

The null column is excluded. Component allocations can move while marginal PIPs
stay similar. Exact distances compare consecutive complete IBSS sweeps:

$$
\Delta_\alpha=\max_{l,j}|\alpha^{(t)}_{lj}-\alpha^{(t-1)}_{lj}|,\qquad
\Delta_{\mathrm{PIP}}=\max_j|\mathrm{PIP}^{(t)}_j-\mathrm{PIP}^{(t-1)}_j|.
$$

The first recorded sweep has no preceding sweep, so its distances are unavailable.
Each PIP uses that iteration's component activity. Alpha distance includes any
null column; PIP distance does not.

The default finite-ELBO rule, starting at iteration two, is
$0\leq\Delta\mathrm{ELBO}<\mathrm{tol}$. A negative increment does not satisfy it.
The `alpha_pip` rule requires both probability distances below their separate
absolute tolerances; `joint` requires those conditions and the finite-ELBO rule.
Select a different requested rule explicitly when fitting or continuing:

```python
rule = prusie.ConvergenceConfig(criterion="joint", alpha_tol=1e-6,
                                 pip_tol=1e-6, min_iter=2, cycle_lag=5)
```

Pass `convergence=rule` to the desired operation; merely creating this object
leaves the completed example unchanged. Plot legends distinguish selected
stopping thresholds from diagnostic-only thresholds. Relative ELBO change is
reported separately and is not the default stopping condition. A nonfinite
increment uses the labelled alpha/PIP fallback; such a nonfinite objective
cannot win refinement selection.

## Technical details

<details markdown="1">
<summary>Iteration phases, finalization and objective conventions</summary>

Metrics are recorded after each complete sequential sweep, before any residual
variance update. Posterior-mean distance uses
$\bar b_j=\sum_l\alpha_{lj}\mu_{lj}$ on the working predictor scale. Signed
variance changes, top variants, entropy, activity and events accompany the
metrics. Online component entropy includes null; posterior resolution instead
normalizes biological alpha and reports null mass separately. Relative ELBO
change is $\Delta\mathrm{ELBO}/\max(1,|\mathrm{ELBO}_{t-1}|)$.

After a non-stopping sweep, residual variance may be updated. Final trimming
zeros components with $V_l<\mathrm{prior\_tol}$, including at the iteration
limit. Historical `elbo` and `XtXr` retain their established phases;
`evaluate_objective(fit)` explicitly recomputes likelihood, KL and fitted
products from the complete saved state. `convergence['finalization']` separates
trim changes from iterations. The CS activity threshold remains $10^{-9}$;
a custom PIP `prior_tol` can therefore produce different boundary activity.

Ordinary SER uses `log(pi + sqrt(eps))`; trimmed V=0 components contribute zero
KL. This convention and the recorded phase explain small logged-versus-final
objective differences. Stage A's hard-exclusion weights remain exactly zero;
B restores the original ordinary-prior convention.

</details>

<details markdown="1">
<summary>Warm-start validation and legacy configuration recovery</summary>

Tracking, initialization or `retain_data=True` retains validated input matrices
and scale metadata. Ordinary calls with these options disabled do not retain
the matrix. `continue_fit` reuses this context and starts from the saved posterior.
To restore a saved result without that context, pass the original inputs and
`s_init=restored` to the appropriate fitting interface.

Variant IDs/order, supplied allele labels, null layout, working scales and data
signature must agree. An explicit state mapping needs `alpha`, `mu`, `mu2`,
`variant_ids`, `scales`, `null_index`, `data_signature` and available allele
metadata. Optional V and sigma2 are checked; estimated parameters can resume
while fixed parameters follow the requested model. Growing L keeps existing
components and adds zero-mean prior rows; shrinking L keeps the first L.
Fitted crossproducts are rebuilt from the current posterior and inputs; old KL,
BF and fitted caches are not reused as current values.

Requested controls are read from `params['convergence']`, with labelled recovery
from diagnostics or retained options for older results. If no original controls
were recorded, compatibility defaults are identified rather than presented as
inheritance. Per-stage `solver_settings` and `convergence` record the resolved
controls for new refinement runs; historical files may not contain these fields.

</details>

<details markdown="1">
<summary>Tracking detail, cycle detection and refinement budgets</summary>

Compact tracking (`True` or `'compact'`) records exact online distances and
component summaries. Its alpha display retains rows with maximum alpha at least
$2/p$ and entries at least $1/p$, where $p$ includes any internal null column.
Thresholds, omitted mass and `lossy=True` are recorded. This display cannot
reconstruct the posterior. `'dense'` also records full alpha and moments and is
intended for small debugging inputs.

A lag-one fixed point is distinct from a short cycle. With the configuration
above, a detected cycle of lag two through five retains the complete current
posterior, reports `cycle_detected` and leaves IBSS convergence false. Components
are not averaged across incompatible states.

Every search round freezes an incumbent and its CS list. Stage A zeros excluded
variant weights and normalizes the remaining full prior, including null. Stage B
transfers alpha/mu/mu2, then resets V and sigma2 to the original model initialization.
Candidates share immutable inputs and own their posterior arrays. A failed final
attempt does not replace an earlier accepted best.

The default absolute improvement threshold avoids accepting numerical noise.
The optional relative threshold is disabled by default; after an accepted gain,
it can stop further rounds when the gain divided by
$\max(1,|\mathrm{incumbent\ objective}|)$ is below the requested value.
A candidate budget counts exclusion attempts, each using at most two fits.
Round or candidate limits report `budget_exhausted`. No CS, no improvement,
unsupported baseline, all candidates failed and exhausted budgets remain
separate outcomes from IBSS stopping.

</details>

Continue with [sensitivity and LD diagnostics](sensitivity.md#compare-explicit-settings)
to compare modeling or input choices. The [quick start](example.md#write-an-offline-report)
and [plotting API](api.md#figures-and-local-style) cover reports, persistence and
local styles. Optional AI interpretation is separate from deterministic analysis.
