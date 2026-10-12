# Stable synthetic example: interpretation for scientific review

**Generated in the current Codex session for human scientific review.** This is an interpretation of existing deterministic outputs, not output from the optional CLI adapter. The configured model was verified as `gpt-6-astra`, reasoning `xhigh`; this is a configuration observation, not a provider-side receipt. No nested model or adapter call was made. All seven PNGs were actually opened. Numerical statements below come from saved JSON/CSV or selected frozen LD entries, not estimates from pixels. See [session provenance](../../session_interpretation.json).

The [500-variant fixture](../../../../../src/prusie/data/teaching/metadata.json) is synthetic (n=1,000, seed 20261007). Its generated effects are at `syn0100` and `syn0350`; they are toy truth, not biological annotations. The report uses zero-based variant index because physical genomic coordinates were not supplied. [Parameters and input hashes](provenance.json) identify the executed analysis.

## Fit and convergence

The baseline and retained final fit stop after **three IBSS iterations** using ELBO, with `tol=0.001`: the last signed increase is **2.8121007744630333e-7**, satisfying 0 ≤ ΔELBO < tol. Last alpha and marginal-PIP distances are both **9.470669224898032e-5**; the standardized-predictor posterior-mean distance is **4.324257407689602e-5**. The fixed-point event is a separate posterior diagnostic, not a different requested stopping rule. There are zero recorded objective decreases or nonfinite ELBOs, and no iteration-limit or short-cycle stop. These statements use `baseline.convergence` and `baseline.trace` in [convergence.json](convergence.json); they are supported by the recorded three-iteration trace.

Residual variance is **fixed at 1**, so its flat trace is a model setting, not evidence that variance estimation converged. The two active component variances finish at **0.08431651034695109** and **0.06199047237827562**; their last changes are 9.91054e-6 and 4.89295e-7. Components 2–4 are inactive/trimmed. Their uniform alpha and large entropy must not be interpreted as three additional signals. The trace phase is `post_sweep_pre_sigma2_update`; finalization is separately recorded. This small example supports local numerical stability, not global optimality or calibrated causal identification.

## Refinement

The search tested **two CS exclusions, four inner fits, one round**, accepting **none**. Excluding `syn0100` and `syn0350` gave restored-prior stage-B improvements of **−6.184563972055912e-11** and **6.821210263296962e-13**, respectively: neither exceeds the independent absolute improvement tolerance 0.001. All four inner fits satisfied their stopping criterion; stage A assigned exactly zero alpha to the excluded variant. See `history[0:2]`, `total_fits` and `config` in [refinement.json](refinement.json).

The incumbent's consistent returned-state objective stays **−1363.854753649087**. Its logged pre-trim ELBO is **−1363.8547312974285**; these are different recorded definitions/phases and should not be interchanged. PIPs and both CSs are unchanged. “No improvement” describes these attempts, not an exhaustive search over all optima. The original-model selection uses stage B only.

## Candidates, resolution and priors

The final PIPs are **0.9696741972541479** for `syn0100` and **0.9999366187170291** for `syn0350`. Each forms a singleton 95% CS, with actual component coverage approximately 0.969674 and 0.999937. Component-alpha and marginal PIP are distinct quantities even where their numerical values nearly coincide. Active-component effective candidate counts are **1.145960** and **1.000678**. See rows identified by `variant_id` in [variants.csv](tables/variants.csv), and `component_id=0,1` in [components.csv](tables/components.csv) and [credible_sets.csv](tables/credible_sets.csv).

The neighboring `syn0101` retains PIP **0.0302806602675062** and has signed r=**0.8578683400236036** with `syn0100`. Thus some local alternative remains despite a singleton CS. The two CS representatives have r=**0.027360870753297715** (r²=0.0007486172483786623); no outside-CS direct proxies meet the configured |r|≥0.99 rule. Singleton purity=1 is a construction property, not a general LD-QC pass. [Selected LD](selected_ld.json), [CS relations](tables/cs_relations.csv), and `proxies` in [posterior_summary.json](posterior_summary.json) support these statements.

The supplied variant prior is uniform, 0.002. Within component 0, prior odds therefore do not explain why `syn0100` dominates; the saved conditional log-BF contrast does. These BFs depend on the other fitted components and settings and are not independent external evidence. The table also exposes the inherited ordinary-SER epsilon floor. See [prior_influence.csv](tables/prior_influence.csv), keyed by component and variant.

## Sensitivity

Increasing L from 5 to 6 and changing reported coverage to 0.90 both leave every PIP exactly unchanged; coverage is posterior re-extraction, not a new fit. A genuinely fixed prior variance (`scaled_prior_variance=0.05`, estimation disabled) retains both main candidates, but gives `syn0143` and `syn0145` PIPs **0.254364932495293** and **0.14297897778579471**, compared with near-zero baseline values. The maximum change is **0.254364932442118**. [comparison.json](comparison.json): `fit_settings`, `pairwise`, `variants`, and `signal_relations` contain the full comparison; [review_values.json](review_values.json) contains the candidate rows.

Those two extra candidates have r≈0.7284 with one another, but only |r|≈0.0086–0.0463 with the two main representatives ([selected_ld.json](selected_ld.json)). This is **not merely redistribution among near-identical proxies of a main hit**. The fixed-variance configuration produces three extra unmatched active components; the saved comparison does not establish that they are independent causal signals. All settings include all 500 variants: low or zero PIP is not a missing-input code. No alternative LD, allele recoding or window was tested here.

## LD diagnostics

All 500 variants were assessed; **none met the joint flip flag** (`logLR>2` and `|adjusted z|>2`). Estimated regularization is **4.2300630066830894e-13**. Nevertheless, `syn0350` has standardized residual **−4.38563687696406**: observed adjusted z=−7.68809587466882 versus conditional expectation −6.437778217060084. Its flip logLR is **−41.00366396647055**, so a large residual does not imply flip evidence. The largest logLR among |adjusted z|>2 is **−13.657166047952451** (`syn0500`). See variant rows and thresholds in [ld_diagnostics.json](ld_diagnostics.json).

This is evidence against a flip under this fitted diagnostic model, not proof of LD calibration. The existing [six-variant reference study](../../numerical/numerical_validation.json) distinguishes raw R `mixsqp` from independently certified mixtures. It does **not** contain paired R vectors for this exact 500-variant input. To establish whether that solver difference changes this example's flags, the missing evidence is case-matched raw/certified R mixture weights, adjusted z, logLR and flags at the same n, s and matrix hash. The wide native margins are descriptive; the smaller study's maximum error is not a bound for this case. No additional validation campaign was run to fill that gap.

## Feedback on the seven saved views

| File actually viewed | What it answers; concrete limitation |
|---|---|
| [figure_1.png](figure_1.png) | Separates marginal association, PIP, formal CSs and LD. The LD track is r² to `syn0350`; it does not show LD to the other signal. Coordinates correctly say variant index. |
| [figure_2.png](figure_2.png) | Shows component 0 alpha, cumulative coverage, singleton signed-r matrix and no proxies. Plotting all 500 ranked variants with sparse tick labels hides the identity of second-ranked `syn0101`; consult the full component row. It does not describe component 1. |
| [figure_3.png](figure_3.png) | Displays the actual three-iteration fit, selected ELBO threshold, diagnostic alpha/PIP thresholds and fixed σ². It retains signed delta-ELBO; exact small endpoint values require the trace. |
| [figure_4.png](figure_4.png) | Confirms unchanged PIPs/CSs and two negligible candidate gains. The 1e-11 axis can visually exaggerate irrelevant changes; compare with the 0.001 acceptance threshold. Its six table rows are complete. |
| [figure_5.png](figure_5.png) | Shows residuals and no flip flags, including the large negative residual. “No flagged variants” must not be read as “no unusual residual”; the point lacks its variant label, supplied above. |
| [figure_6.png](figure_6.png) | Clearly shows fixed-variance sensitivity at `syn0143/145`. All four candidate rows are shown; only 12/21 relationship rows appear. The narrow colorbar label is cramped. Gray means missing, but none are missing in this example. |
| [figure_7.png](figure_7.png) | Separates uniform prior, conditional log BF and component 0 alpha; it cannot summarize all components or demonstrate a nonuniform-prior sensitivity experiment. |

The next useful review is the scientific meaning and presentation of the fixed-variance-only unmatched components, while retaining the robust two leading candidates. No biological mechanism can be inferred from this synthetic example.
