# Reading fine-mapping results

Fine-mapping results summarize uncertainty about which variants affect a trait
under the fitted model and input assumptions. Start with the marginal posterior
inclusion probabilities (PIPs) in `fit.pip` and retained credible sets in
`fit.sets`, then examine posterior concentration, LD purity and convergence
diagnostics.

A high marginal PIP can coexist with a credible set containing several plausible
variants. Component IDs refer to rows of `fit.alpha`; they do not rank signals
or identify known causal effects. The [statistical model](model.md#pip-and-credible-sets)
defines PIPs and credible sets.

## PIP, alpha and credible sets

Use PIPs to rank variants and alpha to examine uncertainty within a single-effect
component. A credible set (CS) summarizes that component's possible locations.

| Quantity | Where to read it | Interpretation |
| --- | --- | --- |
| Marginal PIP | `fit.pip` | Variant support across active components |
| Component alpha | `fit.alpha` | Probability allocation within one component |
| Credible set | `fit.sets` | Selected variants, attained posterior mass and original component ID |
| LD purity | `fit.sets["purity"]` | Minimum, mean and median absolute pairwise correlation within each set, when evaluated |
| Effective number of variants | `effective_candidates` in the [component summary](#posterior-summaries) | Entropy-based concentration of alpha, distinct from CS size or causal count |

For component $\ell$ and variant $j$, $\alpha_{\ell j}$ is the entry of
`fit.alpha` and $V_\ell$ is the component's prior variance in `fit.V`. For fits
without a slot-activity prior, let $\mathcal A$ contain the components with
$V_\ell>\mathtt{prior\_tol}$. The marginal PIP is

$$
\mathrm{PIP}_j=1-\prod_{\ell\in\mathcal A}(1-\alpha_{\ell j}).
$$

`fit.pip` excludes the optional null column. Active components can contribute
even without a retained CS. With a [slot-activity prior](model.md#additional-model-options),
each alpha is also weighted by its component's posterior activity probability
before computing the product.

CS members are selected by decreasing alpha until their cumulative probability
reaches `coverage` (default 0.95). The attained mass can exceed the target because
the last selected variant is included whole. Sets are then screened for activity,
duplicate membership and LD purity; see [credible-set fields](#credible-set-fields)
for the exact extraction order and options. This posterior mass does not establish
repeated-sampling coverage in a new study. PIP and CS support do not establish
biological mechanism.

High purity means that the set's members are strongly correlated; uncertainty
about which member carries the effect can remain. When evaluated, singleton
purity is 1 by convention and says nothing about LD with neighboring variants.

The [quick start](example.md#summarize-the-fitted-posterior) shows how to obtain
the summary tables. The [Toy example](worked_examples.md#resolution)
illustrates a retained CS whose members remain difficult to resolve.

## Posterior summaries

Use `summary = prusie.summarize(fit, R)` to tabulate variants, components and
credible sets, with CS audit records, LD proxies and prior-influence information.
Supply `R` as signed LD correlations in `fit.variant_ids` order. Passing
`coverage` or `min_abs_corr` re-extracts sets from the same posterior without
running IBSS or changing `fit`. Re-extraction requires an explicit `R` unless
CS construction is disabled.

Inspect `summary["cs_audit"]` for the extraction decisions. Reasons such as
`inactive` and `duplicate` can coexist; `unavailable` marks a missing audit
record.

Each entry in `summary["components"]` reports `effective_candidates`, the
entropy-based effective number of variants. Let $\mathcal J$ contain the
non-null variants. For component $\ell$ with positive non-null mass, define

$$
a_{\ell j}=\frac{\alpha_{\ell j}}{\sum_{k\in\mathcal J}\alpha_{\ell k}},
\qquad
H_\ell=-\sum_{j\in\mathcal J:\,a_{\ell j}>0}a_{\ell j}\log a_{\ell j}.
$$

`effective_candidates` is $\exp(H_\ell)$, and `null_mass` is reported separately.
If all probability is on the null position, the effective number is `None`.
This measures posterior concentration, with no cutoff declaring a component
to be a causal signal.

`summary["proxies"]` lists variants outside each retained CS with
$|r|\geq\mathtt{proxy\_r}$ to a CS member, where $r$ is their signed LD
correlation and `proxy_r=0.99` by default. Only direct pairs are listed; proxies
do not add CS members or expand through chains of correlated variants. Without
`R`, proxies are not assessed. Each CS representative is its highest-alpha
member, with ties broken by lexical variant ID. Genotype correlation does not
test whether multiple conditional effects can be distinguished, and CS count
does not identify the number of causal effects.

`summary["prior_influence"]` and `plot_prior_influence` compare stored and
effective prior weights, component-conditional log Bayes factors and alpha.
These Bayes factors depend on the other components and fit settings; they are
not independent evidence for additional effects. See the
[prior conventions](inputs.md#priors-and-zero-information-columns) and
[sensitivity guide](sensitivity.md#variant-alignment-and-prior-weights).

## Result object

After fitting, the fitting entry points return a mutable `SusieResult`
containing posterior arrays, credible sets and the fitting record. Variant and
component indices are zero-based; non-null variants retain input order after
any filtering.

For the [posterior field shapes](#posterior-fields) below, let $p$ be the number
of returned non-null variants (`len(fit.variant_ids)`). There are $q=p+1$ columns
when `fit.null_index` is present, otherwise $q=p$. The returned number of
single-effect components is $L$ (`fit.alpha.shape[0]`), including inactive rows.

`fit.as_dict()` returns the stored public fields without copying their arrays
or nested containers, except that `input_quality` is exported as a dictionary.
Computed properties such as `coef` are omitted. Copy any values you intend to
modify independently.

## Posterior fields

The dimensions $p$, $q$ and $L$ are defined [above](#result-object). Effect
moments use the working predictor scale; `coef` reverses the stored predictor
scaling.

| Field | Shape/type | Meaning |
| --- | --- | --- |
| `variant_ids` | $(p,)$ strings | Retained variant identifiers in input order, excluding null |
| `alpha` | $(L,q)$ float64 | Single-effect location probabilities; each row sums to one |
| `mu` | $(L,q)$ float64 | Conditional posterior means given the component's location |
| `mu2` | $(L,q)$ float64 | Conditional second moments: squared conditional means plus conditional variances |
| `lbf_variable` | $(L,q)$ float64 | Natural-log Bayes factors for each component and variant, before variant prior weighting |
| `lbf` | $(L,)$ float64 | Natural-log prior-weighted Bayes factor for each component |
| `V` | $(L,)$ float64 | Returned component prior variances after final trimming; interpretation depends on the prior |
| `sigma2` | float | Returned residual-variance parameter on the working response scale |
| `pip` | $(p,)$ float64 | Marginal inclusion probabilities over active components, including slot-activity weights when present; null excluded |
| `coef` | $(p,)$ property | Posterior mean effects on the input predictor scale, including slot-activity weights when present; null excluded |
| `KL` | $(L,)$ float64 | Stored component divergence terms; final-trimmed rows are zero |
| `elbo` | `(niter,)` float64 | Recorded objective after each sweep, before final trimming |
| `XtXr` | $(q,)$ float64 | Stored fitted crossproduct; its phase is described below |
| `X_column_scale_factors` | $(q,)$ float64 | Predictor scale divisors, including null when present |
| `intercept` | float | Reconstructed from response and predictor means; NaN when means are unavailable |

Without a slot prior, `alpha * mu` gives each component's marginal coefficient
mean. For slot-prior fits, `coef` also applies the fitted component activity
probabilities, `slot_weights`. The reconstructed `intercept` and individual-level
`fitted` values use the same activity-weighted posterior mean. Slot-prior
intercepts deliberately differ from susieR 0.16.6 exports, which omit these
weights. See the [posterior moments](model.md#single-effect-posteriors-and-effect-means)
and [RSS coefficient scales](model.md#from-summaries-to-working-sufficient-statistics)
for the interpretation of effect estimates.

`XtXr` stores a fitted crossproduct, rather than the crossproduct residual. The
ordinary Gaussian solver retains it from before final trimming; RSS fits using
finite-reference or mismatch models, multiple LD panels, or `susie_rss_lambda`
construct it from the trimmed posterior. See [iterations and finalization](#iterations-and-finalization)
for how stored quantities relate to the returned posterior.

Additional outputs depend on the fitting route and are `None` when absent.
Here $n$ is the number of observations used in the fit and $K$ is the number of
variances in `prior_variance_grid`.

| Field | Shape when present | Meaning |
| --- | --- | --- |
| `fitted` | $(n,)$ | Stored fitted responses for individual-level data |
| `univariate_zscore` | $(p,)$ | Marginal regression z-scores requested with `compute_univariate_zscore=True` |
| `slot_weights` | $(L,)$ | Posterior component activity probabilities under a slot prior |
| `lbf_grid` | $(L,q,K)$ | Natural-log Bayes factors for each component, variant and mixture variance |
| `rv` | $(L,)$ | Per-component residual-variance values for NIG |
| `marginal_loglik` | $(L,)$ | Integrated NIG single-effect log likelihoods, when evaluated |

The [additional model options](model.md#additional-model-options) explain mixture
moments and the returned `V` under mixture and NIG priors. When mixture SER takes
precedence over NIG SER, the NIG fields do not contain newly evaluated NIG
posteriors.

## Credible-set fields

Read CS membership together with its component ID, attained posterior mass and
purity. `fit.sets["cs"][k]` contains member indices for the component identified by
`fit.sets["cs_index"][k]`. These IDs index rows of `fit.alpha`; they need not be
consecutive, and sorting sets does not relabel them.

| Key | Meaning |
| --- | --- |
| `cs` | List of member-index arrays in ascending variant order within each set |
| `cs_index` | Original zero-based component ID for each returned set |
| `coverage`, `actual_coverage` | Attained component posterior mass; the two keys share one array when sets are retained |
| `requested_coverage` | Requested selection target, including when no sets are retained |
| `purity` | Arrays `min_abs_corr`, `mean_abs_corr`, `median_abs_corr`, one entry per returned set |

Empty or disabled results contain an empty `cs` list and empty index, mass and
purity arrays. PIPs remain available separately in `fit.pip`. In fitting calls,
`coverage=None` disables CS construction, as does setting both `min_abs_corr`
and `median_abs_corr` to `None`. A retained CS does not establish that the fit
converged; inspect the [stopping record](#iterations-and-finalization) separately.

<details markdown="1">
<summary>Exact CS extraction order and boundary cases</summary>

Variants are ranked by decreasing alpha, with ties retaining column order, and
added until their cumulative mass reaches `coverage`. Member indices are then
sorted. The CS definitions follow the
[pinned susieR reference](implementation.md#reference-source-mapping).

Deduplication records the first occurrence of each initial membership set
**before** activity and purity filtering. An earlier rejected candidate can
therefore suppress a later duplicate. CS activity requires $V_\ell>10^{-9}$;
this threshold is fixed independently of the configurable `prior_tol` used for
PIPs.

If `cs_extension_corr` is supplied, variants with absolute correlation strictly
above that threshold to any initially selected member are added before purity
filtering. Extension excludes the null column and does not follow correlation
chains. Extended memberships are not deduplicated again.

Candidates containing null are rejected when purity is evaluated. Singleton
purity is 1 by convention. Matrix-based purity uses every unordered pair within
a retained set; `n_purity` does not subsample matrix input. For purity computed
from individual-level `X`, `n_purity` can cap the calculation: sets exceeding
the cap use a random subset of members to estimate purity, while the full CS
membership is returned.

By default, the minimum absolute correlation must be at least `min_abs_corr`.
If `median_abs_corr` is also supplied, meeting either threshold is sufficient;
with `min_abs_corr=None`, only the median threshold is used. Returned sets sort
by decreasing minimum purity when its threshold is present, otherwise by
decreasing median purity, with ties retaining component order.

The [model explanation](model.md#pip-and-credible-sets) distinguishes the requested
mass, attained posterior mass, LD purity and repeated-sampling coverage.

</details>

## Iterations and finalization

`fit.niter` counts completed IBSS sweeps; `max_iter` caps the sweeps in one fit.
If no stopping condition holds on the last permitted sweep, `fit.converged` is
false. A convergence flag describes the selected stopping rule and does not
establish that the fit is a global optimum. Read `fit.convergence` for the
requested and effective rules, stop reason and fitting diagnostics.

For ordinary Gaussian fitting, the default ELBO rule requires a finite
$0\leq\Delta\mathcal L<\mathtt{tol}$, where $\Delta\mathcal L$ is the change
from the preceding sweep. The first check is after two sweeps; `min_iter` can
delay it. A decrease does not satisfy the rule. With a finite ELBO, repeated
alpha/PIP states are recorded only as diagnostics in `elbo` mode.

The susieR-compatible PIP rule, selected by `convergence_method="pip"` and some
[model-specific defaults](model.md#stopping-rules), can accept a short alpha/PIP
cycle as convergence. It averages `alpha` over the cycle; the other posterior
arrays are not averaged or refitted. For ordinary fits using an explicit
`ConvergenceConfig` with `criterion="alpha_pip"` or `"joint"`, an eligible cycle
instead returns the current state with `stop_reason="cycle_detected"` and
`converged=False`, unless the requested criterion already holds. Successful and
cycle exits respect `min_iter`.

In the ordinary solver's `elbo` mode, a NaN ELBO with finite posterior arrays
invokes the labelled alpha/PIP fallback at or after `min_iter`. Default controls use
the compatible PIP rule, including cycle averaging; an explicit ELBO
configuration uses its separate alpha/PIP tolerances without averaging. A finite
ELBO with a nonfinite increment does not trigger fallback. NaN in `joint` mode
reports `numerical_failure`, even before `min_iter`; an infinite computed ELBO
raises `ValueError`. See [convergence diagnostics](post_analysis.md#convergence-diagnostics)
and [continuation and refinement](post_analysis.md#continue-first-then-consider-refinement)
for how to investigate a stopped fit.

<details markdown="1">
<summary>Residual-variance updates, trim thresholds and stored field phases</summary>

Residual variance is updated after the objective and stopping check, and the
update is skipped when that check terminates the fit. At an iteration-limit
exit with variance estimation enabled, the final update can therefore return a
`sigma2` different from the value used for the last ELBO. For ordinary Gaussian
fits the update uses the expected residual sum of squares divided by the sample
size, subject to the variance bounds. Other likelihoods have
[different updates](model.md#elbo-and-residual-variance).

Final trimming acts on components with $V_\ell<\mathtt{prior\_tol}$. Their `V`,
`mu`, `mu2`, `lbf_variable`, `lbf` and `KL` are set to zero, and their alpha rows
are restored to the normalized prior weights. Rows are retained. NIG converts
its component prior variances using `rv` before this comparison; see the
[model options](model.md#additional-model-options).

The returned PIPs use the finalized arrays and require
$V_\ell>\mathtt{prior\_tol}$. A component exactly at that threshold is untrimmed
but inactive for PIP. CS extraction uses its separate fixed
[activity cutoff](#credible-set-fields).

The ELBO trace is not recomputed after trimming. Ordinary fits also retain the
pre-trimming `XtXr`; the [posterior fields](#posterior-fields) describe RSS routes
that reconstruct it from the trimmed posterior. Online distances describe the
stopping checks before cycle averaging and final trimming. Ordinary fits provide
`fit.convergence["last"]` and `fit.convergence["finalization"]` to distinguish the
last checked state from the returned variance and trimming changes; these
detailed records are not supplied by every fitting route.

</details>

## Metadata and persistence

Keep the fitting record with the posterior to track settings, input checks and
stopping behavior when comparing analyses.

| Field | Meaning |
| --- | --- |
| `niter` | Sweep count of the returned IBSS fit, excluding other refinement attempts |
| `converged`, `ibss_converged` | Returned status and its property alias; refinement can change their interpretation |
| `convergence`, `trace` | Requested/effective rule, stopping evidence and optional iteration history |
| `refinement` | Explicit search record; defaults to `status="not_executed"` when no search was requested |
| `null_index` | Index of the appended null column, or `None` |
| `params` | Resolved fitting options, normalized prior weights, effective dimensions and RSS input/scale information |
| `backend_version` | Native package version and pinned susieR reference identity |
| `warnings` | Collected numerical, model and convergence messages, also emitted as `RuntimeWarning` |
| `qc`, `input_quality` | Per-call input checks as a mapping and a typed `InputQuality` record |
| `variant_metadata` | Copied caller-supplied columns, filtered into `variant_ids` order |
| `data_signature` | SHA-256 digest of an owned input snapshot, when available; otherwise `None` |
| `R_finite_diagnostics` | Finite-reference LD and mismatch reliability summaries, when evaluated; otherwise an empty mapping |

After `refine_fit`, `converged` can describe the search outcome. Read
`fit.convergence["ibss_converged"]` for the returned fit's inner IBSS status and
`fit.refinement["status"]` for the search status; see
[convergence diagnostics](post_analysis.md#convergence-diagnostics).
Input checks do not certify external variant or allele alignment, and a missing
`data_signature` does not establish that inputs match.

Use `save_analysis(fit, directory)` from `prusie.report` to write the public
result fields to `analysis.json` and `arrays.npz`. Keep both files together.
`load_analysis(directory)` reconstructs a `SusieResult`, preserving array values,
dtypes, shapes and stored diagnostics without refitting or recomputing summaries.
The loader checks the array archive against the SHA-256 digest stored in JSON
and reads it with `allow_pickle=False`. The
[quick start](example.md#save-and-reload-a-result) shows the round trip;
[manual export](example.md#advanced-export-and-example-provenance) covers exchange
of selected arrays and metadata.

An available `data_signature` is saved, but the private input snapshot is omitted.
Consequently, `continue_fit(restored)` cannot run directly. To warm-start a new
fit, supply the original inputs and fitting options with `s_init=restored`; see
[warm-start validation](post_analysis.md#technical-details).

Keep input provenance and a full software environment record alongside saved fits.
The [Toy example provenance](worked_examples.md#sources-and-reproducibility)
distinguishes the software that produced saved results from the current renderer.
