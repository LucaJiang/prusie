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
alpha/PIP fallback after a NaN ELBO is recorded as the effective rule and
does not replace the original request for later calls.

Each refinement attempt first excludes one retained credible set by setting its
prior weights to zero and fits from a cold start. It then restores the original
prior and refits, starting from the excluded-variant solution. The saved search
records call these steps A and B, respectively.

Candidates are compared using the **ELBO of the full model with the original
prior**, evaluated after zeroing components with $V_\ell < \mathtt{prior\_tol}$.
Below, “ELBO” refers to this value; the iteration log stores the value before
that finalization step. A candidate must be valid, finite and converged, and
improve the ELBO by more than `absolute_tol`. An unsuccessful search does not
establish global optimality.

## Refinement compared with susieR

`refine=True` and `refine_fit` perform the same prusie search. The table compares
the Gaussian model with susieR **0.16.6**, commit
`8e56a8e038e989856d106d9ca5175cc664fea9d2`. It describes algorithm choices;
standard-fit R agreement alone does not validate refinement equivalence.

| Step or control | prusie | Pinned susieR |
| --- | --- | --- |
| Excluded-variant fit (A) | Cold fit; CS weights are zero with log weight −∞, enforcing hard exclusion | Cold fit; zero CS weights enter the ordinary `log(pi + sqrt(eps))` convention, retaining small prior support |
| Refit under the original prior (B) | Restore original prior and model; initialize alpha/mu/mu2 from A, reset V and sigma² to original initialization | Restore original prior; initialize alpha/mu/mu2 from A with ordinary variance initialization |
| Candidate ELBO | Full model after zeroing low-variance components; A's objective is diagnostic only | Last logged ELBO returned by `susie_get_objective`; A's objective is not compared with the baseline |
| Required absolute gain | Strictly greater than `absolute_tol=1e-3` | Strictly greater than fit `tol`, default `1e-4` |
| Relative round stop | Disabled (`relative_tol=None`); if set, accepted gain divided by `max(1, abs(incumbent objective))` is compared with it | Absolute logged gain divided by `max(1, abs(new objective))` is compared with fit `tol` |
| Convergence eligibility | Baseline and both A/B fits must be converged; returned `converged` describes IBSS separately from search status | No explicit baseline/A/B convergence eligibility test; the outer search overwrites `converged` with its own status |
| Search bound | `max_rounds=10`; optional candidate budget, unlimited by default | 100 outer rounds; no separate candidate budget |
| Invalid or failed candidate | Record failed stage, reject candidate, retain incumbent; require a finite recomputed objective | No per-candidate exception recovery or nonfinite filtering in `run_refine`; underlying fitting/objective errors propagate |

The prusie inner defaults are `max_iter=100` and `inner_tol=1e-4`, independent
of the outer improvement threshold. susieR forwards the original fit settings.
For exact definitions, see the
[prusie implementation](https://github.com/LucaJiang/prusie/blob/main/src/prusie/refinement.py),
[pinned R search](https://github.com/stephenslab/susieR/blob/8e56a8e038e989856d106d9ca5175cc664fea9d2/R/refinement.R)
and [pinned R objective accessor](https://github.com/stephenslab/susieR/blob/8e56a8e038e989856d106d9ca5175cc664fea9d2/R/susie_get_functions.R).
The saved [worked examples](worked_examples.md) report executed prusie searches
with their own source identities and settings; their search outcomes should be
read separately from the [standard-fit comparison](numerical_accuracy.md).

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
The refinement figure compares candidate PIPs, ELBO gains relative to the best
fit at the start of each round, and the ELBO of accepted solutions. Full records distinguish
failed, unconverged, below-threshold, valid but unselected, and accepted candidates.

The [problem example](worked_examples.md#a-stopped-fit-improved-by-refinement)
illustrates a different outcome: its initial fit satisfies the stopping rule,
yet refinement improves the ELBO by 17.4583. The final logged
sweep ELBO precedes finalization; use the finalized ELBO for candidate comparisons.

## Convergence diagnostics

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
| Stop reason | Was the criterion satisfied, an eligible cycle stop taken, the iteration cap reached, or a numerical failure detected? |

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
`min_iter=2` is the default; a larger value delays successful convergence and
any cycle stop until that sweep. Repetition of alpha/PIP at lags two through five
is **diagnostic only for a finite objective in `elbo` mode**. It cannot stop that
fit or override an already satisfied convergence criterion.
Select a different requested rule explicitly when fitting or continuing:

```python
rule = prusie.ConvergenceConfig(criterion="joint", alpha_tol=1e-6,
                                 pip_tol=1e-6, min_iter=2, cycle_lag=5)
```

Pass `convergence=rule` to the desired operation; merely creating this object
leaves the completed example unchanged. Plot legends distinguish selected
stopping thresholds from diagnostic-only thresholds. Relative ELBO change is
reported separately and is not the default stopping condition.

| Objective state | Behavior |
| --- | --- |
| First finite ELBO, following the initial −∞ sentinel | No convergence check or fallback; a preceding completed sweep is required |
| Finite ELBO with a negative or nonfinite increment | Does not satisfy the ELBO condition; no alpha/PIP fallback |
| NaN ELBO in `elbo` mode | At or after `min_iter`, use the labelled `alpha_pip_nonfinite_fallback` rule; success requires both posterior distances |
| NaN ELBO in `alpha_pip` mode | Use the explicitly requested posterior rule and record the nonfinite objective |
| NaN ELBO in `joint` mode, or an infinite ELBO in any mode | Stop with `numerical_failure` and `converged=False`, even before `min_iter` |

A nonfinite objective cannot win refinement selection. `criterion_satisfied`
means the effective rule succeeded; inspect that rule and objective validity
before interpreting the fit. If no stopping condition holds at `max_iter`, the
reason is `iteration_limit` and `converged=False`.

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
limit. Stored `elbo` and `XtXr` retain their pre-finalization values;
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
controls for refinement runs; older saved files may lack these fields.

</details>

<details markdown="1">
<summary>Tracking detail, cycle detection and refinement budgets</summary>

Compact tracking (`True` or `'compact'`) records exact online distances and
component summaries. Its alpha display retains rows with maximum alpha at least
$2/p$ and entries at least $1/p$, where $p$ includes any internal null column.
Thresholds, omitted mass and `lossy=True` are recorded. This display cannot
reconstruct the posterior. `'dense'` also records full alpha and moments and is
intended for small debugging inputs.

A lag-one fixed point is distinct from a short cycle. `cycle_lag=5` retains the
default diagnostic history in every mode; `cycle_lag=1` disables lag-greater-than-one
detection. Lagged repetition is recorded independently of the stopping decision.
In explicit `alpha_pip` or `joint` mode, or the NaN-ELBO fallback, an eligible
lag-two-through-five cycle stops only when the requested success criterion has
not already held and `min_iter` has been reached. Such an exit reports
`cycle_detected` and leaves IBSS convergence false. Finite-ELBO mode continues.
Every mode retains the complete current posterior; components are not averaged.
`track_fit=False`, `'compact'` and `'dense'` change recorded history, not fitted
values or stopping decisions.

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
local styles.
