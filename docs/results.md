# Results

Both fitting functions return `SusieResult`. All indices are zero-based and
biological variants remain in the caller's order. Let $p$ be their count and
$q=p+1$ when an internal null column is present, otherwise $q=p$. In the table,
$L$ is the returned component count, capped at the number of available columns.
`as_dict()` returns a shallow mapping without copying arrays. Results are
mutable; callers own any copies they need.

## Posterior fields

| Field | Shape/type | Meaning |
| --- | --- | --- |
| `variant_ids` | $(p,)$ strings | Caller variant order, excluding null |
| `alpha` | $(L,q)$ float64 | Component probabilities $q_\ell(\gamma_\ell=j)$ |
| `mu` | $(L,q)$ float64 | Conditional means on the fitted predictor scale |
| `mu2` | $(L,q)$ float64 | Conditional second moments, equal to squared conditional means plus conditional variances |
| `lbf_variable` | $(L,q)$ float64 | Natural-log component/variant Bayes factors before variant weights |
| `lbf` | $(L,)$ float64 | Component log weighted evidence |
| `V` | $(L,)$ float64 | Component prior variances after trimming |
| `sigma2` | float | Residual variance on the working response scale |
| `pip` | $(p,)$ float64 | Marginal inclusion probability over active components, excluding null |
| `coef` | $(p,)$ property | Sum of alpha times mu over components, divided by predictor scale factors |
| `KL` | $(L,)$ float64 | Component divergence terms from the fitting updates; trimmed rows are zero |
| `elbo` | `(niter,)` float64 | Objective trace before final low-V trimming |
| `XtXr` | $(q,)$ float64 | Retained fitted crossproduct before trimming; this is not the crossproduct residual |
| `X_column_scale_factors` | $(q,)$ float64 | Predictor scale factors including null, if present |
| `intercept` | float | Supplied response mean minus predictor means times posterior coefficients; NaN when means are unavailable |

`mu` is conditional on a variant being selected within one component, whereas
`alpha * mu` is the component's marginal coefficient mean. See the
[model explanation](model.md#component-posteriors-and-effect-means) for their
relationship and the separate [RSS coefficient scales](model.md#from-summaries-to-working-sufficient-statistics).

## Credible-set fields

`sets['cs'][k]` contains member indices for the original component
`sets['cs_index'][k]`. IDs need not be consecutive; sorting returned sets does
not relabel components. The mapping contains:

| Key | Meaning |
| --- | --- |
| `cs` | List of member-index arrays, ascending within each set |
| `cs_index` | Original zero-based component ID for each returned set |
| `coverage`, `actual_coverage` | Actual selected posterior mass in that component; equal aliases sharing the returned array for nonempty results |
| `requested_coverage` | Requested selection target |
| `purity` | Arrays `min_abs_corr`, `mean_abs_corr`, `median_abs_corr`, one entry per returned set |

Candidates accumulate alpha in descending order with stable ties, then sort
member indices. Deduplication retains the first occurrence of a membership set
**before** activity and purity filtering, so an earlier rejected candidate can
suppress a later duplicate. CS activity uses $V_\ell>10^{-9}$. Candidates
containing null receive the reference's negative purity sentinel and are not
retained at allowed thresholds. Singleton sets have purity one; larger retained
sets use every unordered pair's absolute correlation. `n_purity` does not
subsample matrix input. Returned sets sort by decreasing minimum purity with
stable ties.

Empty sets give empty member/index/mass/purity collections while retaining PIPs
and the target. `coverage=None` or `min_abs_corr=None` disables construction.
The [two-variant example](model.md#pip-and-credible-sets) distinguishes requested
mass, selected mass, LD purity and repeated-sampling coverage. The presence of
a credible set and convergence are separate properties.

## Iterations and finalization

`niter` counts executed sweeps; `max_iter` is the cap. With a finite objective
increment, convergence requires $0\leq\Delta\mathcal L<\mathtt{tol}$ after at
least two sweeps. An objective decrease beyond tolerance produces a warning
and does not satisfy this criterion. Reaching the cap returns
`converged=False` with a warning, unless the stopping criterion held on that
last sweep. The library does not silently refit.

A nonfinite objective increment invokes the recorded alpha/PIP fallback. It
checks a fixed point or a cycle of at most five states using the maximum
absolute alpha/PIP difference and `tol`; a detected multi-state cycle averages
alpha across that cycle. Other posterior arrays and retained products are not
recomputed by this exceptional fallback. Its warning is part of the fit and
should be retained when interpreting such outputs.

Residual variance is updated only after the objective and stopping check, so
it is unchanged on the convergence sweep. If estimation is enabled at an
iteration-limit exit, returned `sigma2` includes the final ER2/n update while
the last ELBO used the previous value. After iteration, V&lt;prior_tol components
have V, moments, log BFs and KL set to zero and alpha restored to the normalized
prior. ELBO and XtXr retain the completed iteration values before trimming.
PIP uses $V_\ell>\mathtt{prior\_tol}$; a component exactly at that threshold is
untrimmed but inactive for PIP. The separate CS threshold is fixed as described
above.

## Metadata and persistence

| Field | Meaning |
| --- | --- |
| `niter`, `converged` | Executed sweep count and stopping status |
| `null_index` | Last internal column index, or None |
| `params` | Actual options, normalized weights, effective dimensions and RSS input/scale information |
| `backend_version` | Installed Rust implementation and pinned reference identity |
| `warnings` | Numerical, model and convergence messages, also emitted as Python warnings |
| `qc` | Per-call matrix diagnostics; these do not certify external allele alignment |
| `variant_metadata` | Copies of caller-supplied length-$p$ metadata arrays |

Use NPZ for arrays and JSON for parameters/metadata when persisting portable
results; the [toy tutorial](example.md) has an example. Python pickles retain
class module paths and require a compatible environment. For trusted results
saved by an older package, export arrays with that original environment.
