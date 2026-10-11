# Diagnose and explain a fine-mapping result

Your fit has stopped, but does it localize one candidate per signal, or spread
probability across correlated variants? Start with credible sets and PIPs,
then inspect stopping evidence and test whether refinement changes the result.
These steps answer different questions: a stopping condition does not establish
a global optimum, and a small CS does not establish causal identity.

From a source checkout, install the optional analysis dependencies:

```sh
python -m pip install '.[analysis]'
```

This minimal workflow fits the packaged example once and creates an offline
report. Open `output/analysis/index.html` locally; keep its directory together
to retain links to full tables and machine evidence.

```python
import prusie
from prusie.plots import plot_convergence
from prusie.report import write_report

example = prusie.load_example()
fit = prusie.susie_rss(**example['inputs'], **example['parameters'],
                       track_fit=True, retain_data=True)
summary = prusie.summarize(fit, example['inputs']['R'],
                           z=example['inputs']['z'])
figure = plot_convergence(fit)
write_report(summary, 'output/analysis', figures={'Convergence': figure})
```

Read the report in order: main candidates and CS resolution; stopping and
refinement; figures; sensitivity and LD checks; complete component details.
The short `analysis_summary.json` links to paths in the complete `evidence.json`.
Both are deterministic outputs and need no model service. CSV is convenient for
reading; tagged JSON/NPZ preserves exact saved results.

## Two worked examples

From a source checkout with the analysis extra installed, run:

```sh
python examples/post_analysis.py --output-dir output/post-analysis --case both
```

The **stable** example uses the packaged 500-variant synthetic toy input. The
**problem** example uses the official susieR 200-SNP `FinemappingConvergence`
sufficient statistics: ordinary IBSS satisfies its stopping rule, and two-step
refinement finds a higher objective. Both export seven figure types, PNG/SVG,
CSV, lossless JSON/NPZ and offline HTML. They need neither R nor a network
connection. This command fits the examples; reading their saved reports does not.

`track_fit=False` and `refine=False` remain the defaults. Tracking does not change
the selected stopping rule. `retain_data=True` owns a read-only copy of the
validated matrix and scale metadata for explicit continuation and refinement;
tracking or initialization also retains these inputs. Ordinary calls do not
retain that matrix. LD eigen decomposition, candidate fits, plotting and network
requests happen only when explicitly requested.

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
fit = prusie.susie_rss(**example['inputs'], convergence=rule, track_fit=True)
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
uniformly recomputed original-model objective improves by **more than** the
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

## Posterior resolution and LD checks

`summarize` returns variant, component and CS tables, the actual CS pipeline
audit, direct LD proxies, CS overlap and prior influence. Audit reasons include
inactive, duplicate, contains-null, low-purity, disabled and unavailable evidence;
more than one reason can apply. Re-extracting coverage/purity uses the same
posterior and never runs IBSS.

The effective number of candidate variants is
$\exp(-\sum_j a_j\log a_j)$, with biological component alpha normalized to sum
one. It measures resolution, not a causal count or calibration. Proxy edges
require a direct pairwise $|r|$ above the requested threshold (default 0.99);
they neither add CS members nor expand by transitive closure. Representatives
use highest alpha, with lexical variant ID tie-breaking. Signed r and r² are
labelled separately. Genotype correlation between representative variants is
different from conditional distinguishability of effects in a multi-component
model: nonzero r alone does not show that the model cannot distinguish them.
Returned CS count does not confirm the true number of causal variants.

Prior influence shows the supplied and effective epsilon-adjusted priors,
conditional component log BF, alpha and odds decomposition. Zero prior weights
are reported explicitly. These BFs depend on the other fitted components and
settings; they are not independent external evidence.

```python
ld = prusie.diagnose_ld(example['inputs']['z'], example['inputs']['R'],
                        example['inputs']['n'], variant_ids=fit.variant_ids)
```

The explicit LD diagnostic implements the RSS null model
$z\sim N(0,(1-s)R+sI)$, including finite-n z adjustment, likelihood-based s
estimation, conditional means/variances and a normal scale mixture for allele-flip
likelihood ratios. `estimate_s_rss` exposes null MLE, partial MLE and pseudo-MLE.
Eigen decomposition is explicit, separately timed and optionally cached. Small
or negative eigenvalues are truncated only inside the diagnostic, with warnings;
the fitting matrix is untouched. Singular s=0 uses the labelled spectral
pseudoinverse convention. Missing statistics and undefined conditional quantities
remain unassessed. The report separates assessed/flagged, assessed/not flagged,
conditional-only and unassessed rows. A finite conditional residual does not
establish that mixture optimization and the flip test succeeded. For missing z,
the diagnostic's working zero is recorded separately from the unavailable
observed statistic. Flags suggest possible mismatch; nothing is flipped or removed.

The scale-mixture likelihood is solved with a full-simplex KKT check. Some pinned
R `mixsqp` default fits stop at suboptimal weights; prusie records this solver
accuracy difference rather than reproducing premature stopping. Conditional
formulas and independently certified R mixture optima are validated separately.

User-provided allele labels, EAF, N, INFO, ancestry/build, LD reference sample
size and external QC flags can be carried in metadata and `input_info`.
Missing information stays `not_assessed`. Provided P values take precedence;
otherwise plotted association values use a labelled two-sided normal
approximation with a stable log-tail calculation.

## Compare explicit settings

`compare_fits` compares existing results without new fits. `run_sensitivity`
executes only a named baseline plus one-factor plan:

```python
sensitivity = prusie.run_sensitivity(example['inputs'], [
    {'name': 'larger_L', 'changes': {'L': 6}},
    {'name': 'fixed_V', 'changes': {'scaled_prior_variance': .05,
                                  'estimate_prior_variance': False}},
    {'name': 'coverage90', 'report': {'coverage': .90}},
], baseline_settings={'L': 5})
```

Supported choices include L, variant priors, genuine fixed effect variance,
explicit LD/window/filter inputs, continuation and refinement. Changing V's
initialization while estimating it is labelled computational initialization
sensitivity. External-LD residual variance is not automatically estimated.
Coverage/purity changes are report filters and reuse the posterior. No Cartesian
grid, implicit window expansion or average posterior is constructed.

An alternative `inputs` mapping supplies its own summary statistics, LD and
explicit variant IDs. Existing prior weights and inherited metadata columns are
mapped by those IDs, including when a previously fitted baseline is supplied.
For a subset, the prior is conditioned on the retained variants and the null
component. If raw biological weights are $w$ and `null_weight` is $\eta$, the
constructor normalizes $[(1-\eta)w,\eta]$ **once**, after selecting the retained
weights. Biological weights are not separately normalized first: $\eta$ is not
necessarily the final null probability when the raw weights do not sum to one.
Zero biological support is rejected, even with a positive null weight.

For new IDs under a nonuniform prior, provide complete alternative weights in
the alternative input order. This belongs to the input-change category:

```python
alternative_inputs = dict(new_inputs, prior_weights=new_variant_weights)
plan = [{'name': 'alternative_window', 'inputs': alternative_inputs}]
```

Explicit `prior_weights=None` requests uniform weights for the alternative.
Without an explicit prior, a uniform baseline extends its existing equal raw
weight to new IDs; the final null probability can therefore change with the
window size. New IDs also need explicit metadata columns if baseline columns
were supplied; explicit alternative metadata may omit unavailable columns.
The plan is checked for IDs, dimensions and valid weights before any fit starts.
Each fit record includes `prior_resolution` (mapping, policy, raw and normalized
weights, effective null probability), and input changes include `input_resolution`.
The cache hashes the resolved data and settings. Alternative fits are cold starts;
baseline arrays and results are not mutated.

Missing variants stay NA. The default comparison shows formal-CS-supported
signals against baseline; complete active-component relations are a separate
view. A diffuse component can overlap another diffuse component almost perfectly
without identifying a localized signal. Unresolved support changes remain visible
and do not become confirmed split/merge events from diffuse graph degrees.
Main candidates include all formal CS members, top candidates and threshold
candidates, so a large CS is retained even when every member has PIP below 0.1.
Signal relations use shared-support posterior overlap
without renormalizing away unobserved mass, CS overlap and explicitly identified
LD evidence. Component numbers are not matched directly. Unmatched,
ambiguous and split/merge relations remain visible; a one-to-one display match
is only an aid. Candidate PIP changes, ranks, thresholds, CS sizes/purity and
high-LD exchanges remain separate metrics rather than a composite robustness
score. Cache keys include inputs/order/alleles, settings, software and
initialization; caches are caller-owned.

`mode='union'` displays the union and computes each pair's PIP and posterior/CS
overlap on that pair's intersection; representative LD uses the original
representatives in the named LD reference. `mode='common'` uses the **global intersection across
all fits** for displayed variants, ranks/top-candidate selection, all PIP metrics,
shared counts, allele alignment, posterior overlap and CS-overlap evidence.
Representative-LD evidence additionally requires both original representatives
to belong to that support. The `support` and `support_scope` fields identify the
policy. Component posteriors, original CS sizes/purities and representative IDs
remain full-fit metadata; they are not refitted or renormalized on the intersection.
Reported omitted/unobserved mass is outside the declared comparison support.
An empty global intersection gives zero shared counts and unavailable PIP metrics,
with components unmatched on that support. It does not imply biological absence.
The legacy field name `full_region_pip_correlation` refers to the declared support
for that mode, not to a union in which missing PIPs have been filled in.

## Figures, persistence and optional interpretation

`prusie.plots` provides `plot_locus`, `plot_signal`, `plot_convergence`,
`plot_refinement`, `plot_ld_diagnostics`, `plot_sensitivity` and
`plot_prior_influence`. Each returns Figure, axes and plotted data, accepts
existing axes, and uses local style. Negative ELBO increments remain signed.
Discrete CS identity and continuous LD have separate visual channels. Absent
physical coordinates use variant index. Gene tracks accept local GTF or explicit
tables with build, source, transcript selection and coordinate convention.
Supply a locus chromosome/start/end to select the relevant annotation. Known
chromosome or build mismatches between variants and genes are rejected; unknown
metadata is labelled instead of inferred from non-position identifiers such as rsIDs.

Use `plot_convergence(before, after=refined)` to compare separate IBSS trajectories
without joining their iteration axes. `plot_refinement` distinguishes candidate
improvements against each round's incumbent from the path of accepted solutions.
For the problem example, `plot_signal(summary, variant_id='33')` selects its
singleton signal; `plot_signal(summary, variant_id='90')` selects the high-LD
pair. `plot_prior_influence` accepts an explicit `component_id` and `variant_ids`
selection. `plot_sensitivity` defaults to baseline comparisons in the signal
layer; `layer='components'` opens the complete component view. Table pagination
with `table_page` and `table_page_size` affects display, not the returned evidence.

`save_analysis` / `load_analysis` in `prusie.report` preserve ndarray bytes,
dtypes, identifiers, scalar NaNs and structure through tagged JSON plus NPZ;
no pickle is used. CSV is a readable export, not the lossless representation.
Saved fits keep diagnostics but omit the retained optimization context: reload
explicit inputs to continue fitting. Offline HTML embeds images and records
checksums, harmonization, settings, initial state, references, versions, seed,
timings and skipped checks. Full scalar facts stay in `evidence.json`, outside
the default HTML DOM. `analysis_summary.json` is a compact entry point with
table paths, selection policy and diagnostic counts; complete records retain
full precision while report tables round numbers for reading. Deterministic
evidence remains separate from AI text.

The optional `CodexCLI` adapter sends only an explicitly reviewed evidence bundle
after `consent=True`. It uses the official installed CLI, existing login, an
isolated read-only context and the requested Astra model. Custom provider identity
`OpenAI` needs explicit non-secret provider configuration matching the user's
profile; missing configuration fails closed. No auth files are read/copied, no
API fallback occurs, and no raw LD/genotypes or fitting tools are supplied.

The response cites stable evidence IDs. Exact numerical facts are rendered from
those references; invented references or generated numbers fail validation.
Timeout, cancellation and bounded retry affect only the adapter's own process.
Qualitative interpretation still requires scientific review. Authentication/model
availability is established only by a real authorized response; offline reports
and deterministic analysis work independently of it.

After reviewing the synthetic example's `evidence.json`, optional interpretation
can be requested explicitly (this sends that kind of evidence to OpenAI):

```bash
python examples/post_analysis.py --case stable --output-dir output/toy-with-ai --interpret
```

The script allows one CLI attempt and records failures without losing the offline
report. Omit `--interpret` for an entirely offline run. Figure table selection
and page controls are recorded in the returned plotting data; full records remain
available in evidence and CSV exports. Select components explicitly when their
scientific questions differ, such as a singleton candidate versus a high-LD pair.
