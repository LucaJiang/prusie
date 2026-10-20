# Reading fine-mapping results

Start with the PIP ranking and retained credible sets, then ask how concentrated
each component is and whether diagnostics change the interpretation. A variant
can have a high marginal PIP while belonging to a set with several plausible
alternatives. Component IDs identify model rows, not ranks or known causal effects.

## PIP, alpha and credible sets

| Quantity | Question it answers | How to read it |
| --- | --- | --- |
| Marginal PIP | How much posterior support does this variant have across active components? | Rank candidates under the fitted model and assumptions |
| Component alpha | Where does this one single-effect component place its probability? | Read within a component; its rows can exchange roles across fits |
| Credible set (CS) | Which variants together reach the requested component posterior mass? | Read the actual mass, membership and original component ID |
| Purity | Are the CS members strongly correlated with one another? | Minimum, mean and median absolute LD correlation; a singleton has purity 1 by definition |
| Effective number of variants | How concentrated is the component posterior? | Entropy-based resolution, distinct from CS size and causal count |

For active components, marginal PIP is

$$
\mathrm{PIP}_j=1-\prod_{l:V_l>\mathrm{prior\_tol}}(1-\alpha_{lj}).
$$

The optional null position is excluded. A CS with target mass 0.95 can have
actual mass greater than 0.95 because variants are included whole. That target is
posterior mass under the model; it is not a claim of demonstrated repeated-sampling
coverage in a new study. PIP and CS support do not establish biological mechanism.

This fragment continues from the [quick start](example.md), with its `prusie`,
`fit` and `inputs` objects already defined:

```python
summary = prusie.summarize(fit, inputs["R"], z=inputs["z"])
print(prusie.summary_text(summary).splitlines()[0])
```

```text
500 variants; 2 retained credible sets; 2 active components.
```

For the [packaged example](example.md), `syn0100` has PIP about 0.9697 and
`syn0350` about 0.99994. Each has a singleton CS. Their effective numbers of variants
are approximately 1.146 and 1.001: most mass is concentrated, although neighboring
variants can still carry small probabilities. See the
[worked examples](worked_examples.md) for a retained two-variant CS whose members
remain difficult to resolve.

## Posterior summaries

`summarize` provides variant, component and CS tables, the recorded CS pipeline
audit, direct LD proxies and conditional prior/BF information. CS audit reasons
include inactive, duplicate, contains-null, low-purity, disabled and unavailable;
several reasons may apply. Changing summary coverage or purity re-extracts sets
from the same posterior without running IBSS.

The effective number of variants is $\exp(H_\ell)$, where
$H_\ell=-\sum_j a_j\log a_j$ is the entropy of component $\ell$ and
$a_j=\alpha_{\ell j}/\sum_{k\ne\mathrm{null}}\alpha_{\ell k}$ over non-null
variants. The result field is `effective_candidates`; null mass is reported
separately. This is a continuous resolution
summary, with no automatic cutoff declaring a component to be a causal signal.

Direct proxies require pairwise $|r|$ above the chosen threshold (default 0.99).
They do not add CS members or expand through chains of correlated variants.
Representatives use highest alpha, with lexical ID tie-breaking. Genotype
correlation supplements posterior resolution; it is not a test of whether
multiple conditional effects can be distinguished. CS count is not a known
causal count.

Prior-influence plots compare supplied and effective computational weights,
component-conditional log Bayes factors and alpha in the same variant order.
These Bayes factors depend on the other components and current settings. They
are not independent evidence for additional effects. Exact prior conventions
are described in the [model](model.md) and [sensitivity guide](sensitivity.md).

## Result object

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
[model explanation](model.md#single-effect-posteriors-and-effect-means) for their
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

<details markdown="1">
<summary>Exact CS extraction order and boundary cases</summary>

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

</details>

## Iterations and finalization

`niter` counts executed sweeps; `max_iter` is the cap. By default, convergence
requires a finite $0\leq\Delta\mathcal L<\mathtt{tol}$ after at least two sweeps.
An objective decrease does not satisfy this rule. Reaching the cap leaves IBSS
unconverged unless the criterion held on the last sweep. Optional alpha/PIP and
joint rules are selected by `ConvergenceConfig`; the library does not silently
refit when a rule fails.

`fit.convergence` records the requested and effective rules, actual stopping
reason, exact last distances and finalization. Finite-ELBO mode records lagged
alpha/PIP repetition as a diagnostic and continues until its own rule succeeds
or the iteration cap is reached. Explicit alpha/PIP and joint modes may stop
on a lagged cycle after `min_iter`, reporting `cycle_detected` with convergence
false. A successful criterion takes precedence over a cycle diagnostic.

Only an actual NaN objective in ELBO mode permits the labelled alpha/PIP
fallback. The initial −∞ sentinel or an overflowing difference between finite
objectives does not. A NaN objective in joint mode reports `numerical_failure`;
an infinite computed objective raises `ValueError`, as in the pinned R fitter.
Numerical failure may stop before
`min_iter`; successful and cycle exits respect it. Every stop retains the
complete current posterior; components are not averaged across states.
Read [convergence and refinement](post_analysis.md) for thresholds, continuation,
tracking costs and the distinction between stopping and finding a better solution.

<details markdown="1">
<summary>Residual-variance updates, trim thresholds and stored field phases</summary>

Residual variance is updated only after the objective and stopping check, so
it is unchanged on the convergence sweep. If estimation is enabled at an
iteration-limit exit, returned `sigma2` includes the final ER2/n update while
the last ELBO used the previous value. After iteration, V&lt;prior_tol components
have V, moments, log BFs and KL set to zero and alpha restored to the normalized
prior. ELBO and XtXr retain the completed iteration values before trimming.
PIP uses $V_\ell>\mathtt{prior\_tol}$; a component exactly at that threshold is
untrimmed but inactive for PIP. The separate CS threshold is fixed as described
above.

</details>

## Metadata and persistence

| Field | Meaning |
| --- | --- |
| `niter`, `converged`, `ibss_converged` | Executed sweep count and inner IBSS stopping status |
| `convergence`, `trace` | Requested/effective rule, stopping evidence and optional iteration history |
| `refinement` | Explicit search settings, attempts, accepted objective path and search status |
| `null_index` | Last internal column index, or None |
| `params` | Actual options, normalized weights, effective dimensions and RSS input/scale information |
| `backend_version` | Installed Rust implementation and pinned reference identity |
| `warnings` | Numerical, model and convergence messages, also emitted as Python warnings |
| `qc` | Per-call matrix diagnostics; these do not certify external allele alignment |
| `variant_metadata` | Copies of caller-supplied length-$p$ metadata arrays |

Use `save_analysis(fit, directory)` and `load_analysis(directory)` from
`prusie.report` to preserve the result's arrays, identifiers and metadata without
pickle. The [quick start](example.md#save-and-reload-a-result) shows the round trip;
manual NPZ/JSON export is available in its advanced section. Array shapes and
stored phases above remain part of the saved result. The retained optimization
context is omitted, so supply the original inputs when warm-starting a reloaded
posterior.

<details markdown="1">
<summary>Older saved results</summary>

Python pickles retain class module paths and require a compatible environment.
For trusted results saved by an older package, export arrays with that original
environment before importing them into a new workflow. Saved examples with
producer/render identities are described in
[worked-example provenance](worked_examples.md#sources-and-reproducibility).

</details>
