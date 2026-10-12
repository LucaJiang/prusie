> Public edition of the original `reports/implementation_mapping.md`. Local reference locators describe the retained source run; no external source mirrors are included. See `README.md` for the public file map.

# Official susieR source mapping for post-analysis

This is a development audit of exact frozen upstream source. It does not change the historical numerical reference and is not an assertion that every upstream behavior should be copied.

## Identity and provenance

Official repository: <https://github.com/stephenslab/susieR>. The source download uses the official GitHub commit API and `codeload.github.com`; raw archives, commit responses, full DESCRIPTION text, and SHA-256 for every extracted file are recorded in `reference_identity.json`.

| Snapshot | Full SHA | DESCRIPTION version |
|---|---|---|
| Old architecture | `1f9166cf88ed19c1d6acd4ee53a605ff9505a563` | 0.14.22 |
| SuSiE 2.0 architecture | `84616edd3469f741d75be7a64f10530ce6c87976` | 0.15.57 |
| Requested later snapshot | `8e56a8e038e989856d106d9ca5175cc664fea9d2` | 0.16.6 |

The installed historical library was the separately frozen susieR 0.16.6 reference. This run verifies bodies and formals of the actually used convergence, refinement, tracking and LD functions against the downloaded 8e56a8 source in `reference/reference_environment.json`. The root `upstream.lock.json` instead describes the original CRAN 0.14.2 baseline; it remains unchanged and is a distinct reference.

## Convergence and tracking

| Subject | Source and observed semantics | Implementation decision required |
|---|---|---|
| Old finite ELBO stop | 0.14.22 `R/susie_ss.R` and `R/susie.R`: Δ ELBO < tol, so a negative increment also stops | Preserve ordinary validated finite path but reject negative increments as evidence of ELBO convergence |
| New finite ELBO stop | Both newer `R/model_methods.R::check_convergence.default`: first iteration cannot stop; thereafter 0 ≤ Δ ELBO < tol | Keep explicit phase and strict upper inequality |
| 0.15.57 “pip” | `R/model_methods.R`: max(abs(prev_alpha − alpha)); no marginal-PIP calculation in the standard path | Label this alpha distance, not PIP distance |
| 0.16.6 joint/cycle | `R/susie_utils.R::check_alpha_pip_cycle_convergence`: max of alpha and marginal-PIP distance, fixed point or lag 2..5 cycle | Track separate alpha/PIP metrics; short-cycle diagnosis is not IBSS convergence |
| Activity bug at history initialization | Its helper reconstructs previous PIP from previous alpha using **current** model V | Save actual previous PIP or previous active mask |
| Cycle posterior inconsistency | On a detected cycle it averages alpha alone, without updating mu/mu2/V/KL or fitted products | Retain a complete state; do not return mixed posterior fields |
| Tracking phase | `susie_workhorse`: snapshot before sweep; then all component updates, objective, stop test; variance update only if not converged; post-loop trim and finalize | Trace labels must distinguish these phases and post-trim effects |
| Sparse tracking | `susie_track.R::make_track_snapshot`: keep components with max alpha ≥ 2/p and entries alpha ≥ 1/p; omitted values filled with zero by restore helper | Explicitly lossy display; never reconstruct exact convergence metrics from it |

The phrase “Convergence Diagnosis Metrics” is not a dedicated public convergence function in these snapshots. The concrete stopping checks are listed above. New `susie_track.R::make_track_diagnosis` summarizes `R_finite_diagnostics` and LD mismatch traces when present, not a universal posterior-stability guarantee. Scalar/component summaries, effect-moment distances, signed changes, trim events, and separate inner/outer statuses in prusie are additional deterministic diagnostics.

## Initialization, state, and objective

New `iterative_bayesian_stepwise_selection.R::ibss_initialize.default` validates initialization, uses `extract_model_init_fields`, rebuilds fitted products through `initialize_fitted.ss`, and resets KL/lbf. The extracted fields are alpha, mu, mu2, V, with sigma2 only when residual variance is estimated. The two-step refinement initialization deliberately includes **only alpha, mu, mu2**, so V and sigma2 are reinitialized from fitting settings. `adjust_L` refuses to shrink below retained initial components and expands with zero moments; the public prusie contract must state its own explicit grow/shrink rules and validate IDs, direction and working scale more strictly.

For standard sufficient statistics, write G = XᵀX, t = Xᵀy, B_lj = alpha_lj mu_lj, bbar = sum_l B_l. `sufficient_stats_methods.R::get_ER2.ss` (old `elbo_ss.R::get_ER2_ss`) computes

    ER² = yᵀy − 2 bbarᵀt + bbarᵀG bbar − Σ_l B_lᵀG B_l
          + Σ_lj G_jj alpha_lj mu2_lj.

The expected log likelihood is −n/2 log(2π sigma²) − ER²/(2 sigma²). Upstream objective subtracts cached component KL. For a nondegenerate Gaussian row the direct KL is the alpha-weighted sum of categorical and normal KL:

    Σ_j alpha_lj [log(alpha_lj / w_j)
      + ½ {(mu2_lj/V_l) − 1 − log((mu2_lj−mu_lj²)/V_l)}].

Both old `single_effect_regression_ss.R` and new `susie_utils.R::lbf_stabilization` use **w_j = pi_j + sqrt(machine epsilon)** inside the model BF without renormalizing w. Alpha is normalized, but the BF/KL convention differs from the normalized effective prior by log(sum w). This explains why a mathematically normalized-prior ELBO is not numerically identical to upstream's logged objective. Null predictors have no likelihood information and their conditional posterior equals their Gaussian prior. Degenerate trimmed rows use the explicit upstream KL = 0 convention.

`model_methods.R::trim_null_effects.default` sets V < prior_tol to zero, alpha to the original pi, mu/mu2 and lbf to zero, and KL to zero. Logged ELBO was computed before trim, and on an iteration-limit exit sigma2 may also have been updated after that logged objective. A candidate comparison must recompute likelihood, normal/categorical KL and products from the same returned state, while retaining logged ELBO separately. Merely recomputing likelihood while reusing KL is insufficient.

## Two-stage refinement

Old 0.14.22 `susie_ss.R` and `susie.R` repeat calls to their fitting interfaces. New `refinement.R::run_refine` reuses preprocessed data and calls `susie_workhorse`. Both freeze the round incumbent, exclude each retained CS in turn, cold fit stage A, restore original priors, and warm fit stage B with alpha/mu/mu2. Only B is evaluated against the original model; A changes the model prior and is not comparable.

Old acceptance requires a strictly positive improvement. New acceptance requires best candidate ELBO − incumbent ELBO > params$tol. The new outer `block_coordinate_ascent` stops after step completion or abs(Δ ELBO)/max(1,abs(current ELBO)) < tol, with at most 100 rounds. It overwrites `model$converged` with the outer-loop status, so prusie must preserve the selected fit's IBSS flag separately.

Both upstream implementations add epsilon to zero prior weights during SER, so “exclusion” is approximate. prusie's explicit stage-A hard mask is a documented correction; this can produce boundary differences from raw R. Upstream also uses `break` if excluding one CS removes all prior support; prusie should skip that candidate and continue others. Upstream does not independently reject every unconverged candidate: prusie must record and reject these by default. The old/new R candidate fixtures retain these original semantics without patching the reference.

## LD diagnostics

The relevant methods live in old `R/susie_rss_utils.R` and new `R/susie_rss_utils.R`; this run uses the later pinned methods. With sample size n, observed z is adjusted by sqrt((n−1)/(z²+n−2)). Eigendecomposition R = U diag(d) Uᵀ is explicit; eigenvalues below r_tol become zero in the diagnostic calculation, with a warning for materially negative values. This does not authorize modifying fitting LD.

* `null-mle`: minimize ½ Σ_k [log((1−s)d_k+s) + (Uᵀz)_k²/((1−s)d_k+s)] on [0,1].
* `null-partialmle`: mean squared projection onto the null eigenspace; full rank returns 0. The result can exceed 1 and is not silently clipped by the estimator.
* `null-pseudomle`: minimize the sum of negative univariate conditional Gaussian log likelihoods on [0,1].
* For P = U diag(1/((1−s)d+s)) Uᵀ, conditional variance is 1/P_jj, mean is z_j − (Pz)_j/P_jj, and standardized residual is (z_j−mean_j)/sqrt(var_j). With s=0, upstream replaces infinite reciprocal eigenvalues by zero (pseudoinverse convention); undefined zero precision requires an explicit not-assessed result.
* `kriging_rss` fits a shared scale mixture by `mixsqp` maximum likelihood. Grid minimum is approximately 0.8; maximum is 2 if max residual² <1, otherwise 2 max|residual|; grid is `1.05^(-npoint:0)*amax`, npoint=ceiling(log(amax/.8)/log(1.05)).
* Flip evidence is log mixture density of z+conditional_mean minus that of z−conditional_mean, with the source's `weight+1e-15` convention. Upstream display flags logLR>2 and abs(adjusted z)>2; this is suspicion, not proof of an allele mismatch.
* Upstream replaces supplied s>1 by 0.8. Explicit prusie validation or clearly reported boundary handling is preferable to a hidden change.

`reference/ld_oracles.json` records six 6-SNP deterministic cases, all three estimators, n=200 and absent-n behavior, and estimated/fixed s=0.1/fixed s=0 kriging. To expose exact grid/mixture weights, the pinned function's final return expression alone was replaced; all numerical statements are unchanged. This includes singular consistent and sign-inconsistent cases.

## Licensed bounded refinement example

The official package's `data/FinemappingConvergence.RData` is a 200-SNP sufficient-statistics fixture, documented in `R/data.R` and `man/FinemappingConvergence.Rd`. It contains a toy response generated using real genotypes, with n=50,000; no individual genotypes are distributed in the derived JSON. The official package DESCRIPTION licenses the package BSD_3_clause + file LICENSE, whose attribution is preserved. The larger 1001-SNP vignette fixture is unnecessary for this bounded test.

Inputs and provenance are in `reference/refinement_inputs.json`. Latest reference parameters are L=10, max_iter=100, tol=1e-3, scaled_prior_variance=.2, estimated prior and residual variances, coverage=.95, min_abs_corr=.5. `refinement_oracle_latest.json` stores complete baseline, explicit first-round stage A/B candidates and refined result. It exhibits a genuine positive objective improvement; exact numerical measurements and environment are in `reference_validation.json` and the raw oracle, rather than inferred from the vignette.

The source audit is complete independently of final prusie integration. Any Python/Rust parity results must be recorded by the integration tests and must not be inferred from these R-only fixtures.
