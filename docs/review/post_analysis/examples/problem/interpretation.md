# Local-optimum example: interpretation for scientific review

**Generated in the current Codex session for human scientific review.** The configured model was verified as `gpt-6-astra`, reasoning `xhigh`; this records configuration, not a provider-side receipt. No nested model or optional adapter was called. All seven PNGs were actually opened; numerical values were read from saved files or extracted from the frozen matrix. See [session provenance](../../session_interpretation.json).

The [official susieR fixture](../../../../../src/prusie/data/convergence/metadata.json) contains 200 variants, n=50,000, and sufficient statistics for a simulated response based on real genotypes. It is not a newly simulated study and supplies no individual genotypes or biological annotation. String variant IDs such as `33` are **zero-based positions**, not rsIDs or genomic coordinates. [Input provenance](provenance.json) identifies the same frozen analysis throughout.

## Convergence is not a global-optimum certificate

The initial fit satisfies the requested **ELBO criterion, tol=0.001, after ten IBSS iterations**. Its last signed increase is **0.00036840111715719104**, but alpha and PIP still change by **0.0005930574271182367** and **0.0004130597419563786**. These exceed the separate 1e-4 posterior diagnostic tolerances, which were not selected stopping criteria. The final posterior-mean distance is 2.70815e-5; estimated σ² is **0.994481278879717**, with last change −8.65638e-7. All ten component variances exceed the active threshold, but only three CSs are retained. See `baseline` in [convergence.json](convergence.json).

The accepted refined fit takes **three inner IBSS iterations**, with final ΔELBO **4.220055416226387e-9**, alpha/PIP distance **6.798545404418377e-8**, and mean-effect distance **2.549280071345583e-7**. Its two active variances are **0.006797785826108811** and **0.005683253754036641**; σ² is **0.9940978424504077**, with recorded last change −3.13561e-7. Eight zero-V components are trimmed. Both saved traces have zero decrease/nonfinite counts, neither stops at its iteration limit, and only the refined trace records a fixed point. Neither exhibits a short cycle. Metrics refer to `post_sweep_pre_sigma2_update`; finalization is separate. These are trajectory-supported findings, not reconstructed history.

[figure_3.png](figure_3.png) shows **only the selected refined three-iteration trajectory**, so it cannot on its own demonstrate the initial fit's remaining posterior motion. The saved baseline trace is essential to that comparison.

## What the two-step search found

There were **five candidate attempts, ten A/B fits, two rounds, one accepted candidate**. Every recorded inner fit met its stopping rule; A's excluded alpha is exactly zero. The history in [refinement.json](refinement.json) is complete:

| Round | Excluded IDs | A/B iterations | Restored-prior B improvement over that round's incumbent | Decision |
|---|---|---:|---:|---|
| 1 | `35` | 15 / 3 | +17.45830930196098 | Accepted best |
| 1 | `32` | 9 / 3 | +17.45830926578492 | Not best |
| 1 | `86,90` | 17 / 17 | +0.00011266960063949227 | Not best; below 0.001 acceptance tolerance |
| 2 | `33` | 10 / 3 | −17.458094642162905 | Rejected |
| 2 | `86,90` | 17 / 17 | −17.45819663236034 | Rejected |

The consistent original-prior returned-state objective improves from **−70832.27160472395** to **−70814.81329542199**. The final *logged* ELBO is **−70814.81327158018**, giving a different logged gain, 17.458333143775235. Eight trimmed components and the inherited epsilon-weight convention account for the approximately 2.38418e-5 distinction ([numerical_validation.json](../../numerical/numerical_validation.json), `results.refinement.trim_explanation`). Selection uses the consistent post-trim objective. Stage A changes priors and its objective is not used as an original-model competitor. Round-2 failures preserve the earlier best state.

Thus an ELBO-stopped initial fit can coexist with a substantially better searched solution. Stopping verifies a local iteration criterion; refinement demonstrates that this incumbent was improvable. “Completed/no improvement” means no further acceptable candidate in the recorded search, not global optimality.

## Posterior changes and LD-limited resolution

| Variant ID | Baseline PIP | Refined PIP | L=12 cold-start PIP |
|---|---:|---:|---:|
| `32` | 0.9999999999999998 | 0 | 0.9999999999999998 |
| `33` | 0.03660482803794274 | 1 | 0.04627234298543381 |
| `35` | 0.9999999999288653 | 0 | 0.9999999999295759 |
| `86` | 0.21140440228153057 | 0.17717687823750528 | 0.21923674150269024 |
| `90` | 0.8214266429946268 | 0.8228231130062094 | 0.8232061050196247 |

These are complete candidate rows from [comparison.json](comparison.json), also selected in [review_values.json](review_values.json). All 200 variants occur in all three settings. The zeros for `32/35` are fitted PIPs, not missing variants.

Initial CSs are component 0 `{35}`, component 2 `{32}`, and component 1 `{86,90}`. Final CSs are component 2 `{33}` and component 1 `{86,90}` ([convergence.json](convergence.json): `baseline.sets`, `refined.sets`). The two-variant CS has actual mass **0.9999999912437146**, |r|=**0.9989091776612095**, and entropy-based effective candidate count **1.5953381386626992**. It remains difficult to resolve `86` versus `90` despite strong combined support. Component-alpha mass and marginal PIP are distinct; the baseline's additional active components also contribute to marginal PIP.

The component-1 before/after posterior overlap is **0.9925405245291377**, so its high-LD pair is retained with modest probability redistribution. The replacement of `{32}`/`{35}` by `{33}` is a larger posterior configuration change: r(32,33)=**0.5996648929848706**, r(35,33)=**0.5642377546607126**, and r(32,35)=−0.24077748000190774. These are not interchangeable near-perfect proxies. See [selected_ld.json](selected_ld.json) and all `signal_relations` in [comparison.json](comparison.json). The matcher leaves these components unmatched; it does not establish a biological split/merge. Reuse of component number 2 across fits does not establish signal identity.

Final CS representatives `90` and `33` have r=**0.5321733801522945** (r²=0.2832085065427186), with no shared CS members. Two returned CSs therefore should not be called two independent causal signals. No outside-CS proxies pass |r|≥0.99; for example, `79` has r≈0.90 to the pair but is not a near-perfect proxy. Uniform prior weights 0.005 mean within-component odds contrasts arise from conditional BFs; the component-2 BF plot is conditional on other effects, not external independent evidence. [CS relations](tables/cs_relations.csv), [posterior summary](posterior_summary.json), [prior influence](tables/prior_influence.csv).

## Sensitivity and the comparison's limits

Increasing L from 10 to 12 with a cold start does **not** recover the refined configuration: the original singleton candidates remain near PIP 1. Its largest PIP change from baseline is **0.00969418571478775**. This example supports explicit refinement over assuming that a larger L alone fixes this local solution. Alternative LD, windows, filtering, and priors were not tested in this example.

The comparison includes all active components, including very weak ones without retained CSs. Baseline versus L=12 has **63 ambiguous and three matched relations**; baseline versus refinement has **one matched and ten unmatched relations**. These relation counts are not counts of causal signals. The complete table is scientifically necessary; the figure's twelve-row excerpt is insufficient to summarize it. Sources: [comparison.json](comparison.json), `signal_relations`, `fit_settings`, `pairwise`.

## LD diagnostics and R-reference uncertainty

All 200 variants are assessed and **none satisfy the joint flip flag**. Estimated s is **4.2300630066830894e-13**; the smallest eigenvalue is **3.477986166282123e-8** and reported rank is 200, indicating a nearly singular but retained direction under the recorded r_tol. No matrix repair is implied. Variant `33` has the largest absolute standardized residual, **−3.871962268434617**, but flip logLR **−90.95196456810982**. Variant `0` has positive logLR **3.5595496960231547**, yet |adjusted z|≈0.644 and so fails the second required threshold. For |adjusted z|>2, the largest logLR is **−11.2333341793607**, at `114`. All are in [ld_diagnostics.json](ld_diagnostics.json), keyed by variant ID.

The solver problem shown here is not an observed allele-flip diagnosis. Absence of flags does not certify LD calibration. The existing raw/default-R and certified-R study covers six **different six-variant cases**, not this exact fixture. It preserves a raw maximum logLR discrepancy of 0.871891 and certified discrepancy of 2.03694e-7, but neither is a transferable error bound. Case-matched R adjusted z, s, mixture weights, per-variant logLR/flags and matching input hashes are missing for a direct flag comparison here. The native margins are far from the flag threshold, yet cross-implementation agreement for this case remains unverified. No new fitting campaign was undertaken.

## Feedback on the seven saved views

| File actually viewed | What it answers; concrete limitation |
|---|---|
| [figure_1.png](figure_1.png) | Shows the final selected fit and distinct CS tracks; r² refers to `33`. The strongest marginal association need not be the highest final PIP. Positional IDs are correctly not labeled genomic coordinates. |
| [figure_2.png](figure_2.png) | Shows singleton component 2 (`33`), with flat cumulative mass and no proxies. It misses the scientifically important `{86,90}` resolution problem because the script chooses the first retained CS. |
| [figure_3.png](figure_3.png) | Describes the accepted candidate's three iterations, not the baseline's ten. Variance legends include eight inactive rows; use the saved baseline trace to assess the pre-refinement fit. |
| [figure_4.png](figure_4.png) | Combines PIP changes with all five B-stage gains. Gains are relative to each round's incumbent, so later negative values are not decreases in the returned best objective. Only 12/16 table rows appear; later unmatched relations, including the new component, can be hidden. |
| [figure_5.png](figure_5.png) | Clearly separates observed/expected z and residuals; “No flagged variants” uses a joint flip rule. Unlabeled extremes and the absent logLR panel require the row-level JSON above. |
| [figure_6.png](figure_6.png) | All five candidate rows are visible, making `32/35→33` and L=12's failure to reproduce refinement clear. Only 12/90 signal relations are shown, hiding most ambiguous weak-component relations. Gray is defined as missing; no inputs are missing here. |
| [figure_7.png](figure_7.png) | Aligns uniform priors, conditional component-2 log BFs and alpha. It does not show component 1 or justify interpreting all high conditional BFs as additional independent signals. |

The most useful scientific review targets are returned-state objective semantics, the distinction between ELBO stopping and posterior stability, and the handling/presentation of unmatched weak components. No gene or biological mechanism is supported by the available annotations.
