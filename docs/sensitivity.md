# Sensitivity and LD diagnostics

Use sensitivity analysis to ask which candidates and credible sets persist when
you change a defensible modeling or input choice. Use LD diagnostics to ask
whether signed association statistics agree with the supplied correlation matrix.
They answer different questions: agreement with one diagnostic does not establish
that all modeling assumptions hold.

The following examples continue the fit from the [quick start](example.md).
Install the optional [analysis dependencies](install.md#optional-analysis-tools)
for diagnostic plots and reports.

## Compare explicit settings

`compare_fits` compares existing results without new fits. `run_sensitivity`
executes only a named baseline plus one-factor plan:

```python
sensitivity = prusie.run_sensitivity(example['inputs'], [
    {'name': 'larger_L', 'changes': {'L': 6}},
    {'name': 'fixed_variance', 'changes': {'scaled_prior_variance': .05,
                                  'estimate_prior_variance': False}},
    {'name': 'coverage90', 'report': {'coverage': .90}},
], baseline=fit)
comparison = sensitivity['comparison']
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

## Read the sensitivity figure

```python
from prusie.plots import plot_sensitivity

figure, axes, plotted = plot_sensitivity(
    comparison, layout="compact",
    display_names={"baseline": "Baseline", "larger_L": "L = 6",
                   "fixed_variance": "Fixed scaled prior variance = 0.05",
                   "coverage90": "CS coverage = 90%"},
)
```

Columns begin with baseline. Candidate rows include every formal CS member,
per-component top candidates and variants meeting the candidate PIP threshold.
Missing variants have their own color and remain NA; an observed PIP of zero is
a different value. Small heatmaps label each probability. The accompanying
summary reports the maximum absolute PIP change on the declared comparison
support, even when that variant is outside the candidate view.

`candidates_only=False` includes all comparison variants. `layer="components"`
opens the complete component relations; the default `layer="signals"` emphasizes
formal CS-supported relations. Page controls affect the relation table display,
not the exported evidence. Always check `plotted` and the full variant table when
your question concerns variants outside the displayed candidates.

`scaled_prior_variance` is V divided by the working response variance yty/(n−1);
it is not generally V itself. A positive fixed variance together with
`estimate_prior_variance=False` keeps unused components from shrinking to V=0
through variance estimation. Their posterior mass can raise background PIPs
without changing the leading CS. The [stable example](worked_examples.md#stable-signals-with-a-prior-sensitive-background)
shows this distinction using the saved component tables.

## Diagnose consistency of z and LD

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

The default allele-flip flag requires **log LR > 2 and |adjusted z| > 2**.
A large conditional residual alone is not a flip flag. The LD figure labels a
small number of flagged or extreme-residual variants and records why they were
selected. Inspect both parts of the rule before interpreting a label.

| Report state | Meaning |
| --- | --- |
| Assessed, flagged | The joint rule was evaluated and satisfied |
| Assessed, not flagged | The joint rule was evaluated and not satisfied |
| Conditional only | Conditional quantities exist, but the full flip test is unavailable |
| Not assessed | The required observation or calculation is unavailable |

```python
from prusie.plots import plot_ld_diagnostics
from prusie.report import summary_text, write_report

figure = plot_ld_diagnostics(ld, layout="compact")
summary = prusie.summarize(fit, example['inputs']['R'], z=example['inputs']['z'])
print(summary_text(summary, ld=ld, comparison=comparison))
write_report(summary, "output/diagnostics", ld=ld, comparison=comparison,
             figures={"LD diagnostics": figure})
```

Missing observed z is not plotted at zero. Failed or skipped diagnostics retain
their reason and assessed counts. Keep ancestry, effect alleles, variant order,
window/filter selection and LD reference identity with the report so that these
checks can be interpreted in context.
