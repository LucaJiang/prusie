# Implementation

prusie has three layers. The Python package (`src/prusie`) provides the public
API: it checks and aligns inputs, prepares the working model, chooses a fitting
route and assembles the result. The binding crate (`rust/prusie-python`,
imported as `prusie._native`) converts between NumPy and Rust and provides
native input checks. The solver crate (`rust/prusie-core`) has no Python
dependency and runs the iterative Bayesian stepwise selection (IBSS) algorithm.
This page follows a call from the public functions to the solver and back.
[Optimization](optimization.md) describes the speed and memory techniques
used along these paths.

## Architecture

![Call-path roadmap. Four public entry points in the Python package prepare their inputs, pass through the shared preparation and fitting functions, call native functions in the binding and reach the Rust solver, whose sweep calls single-effect regression, the Gram operator, the objective and the stopping rule.](assets/architecture.svg){.figure-complex}

Shaded boxes are public entry points and arrows are calls. The left-hand arrow
marks the native input checks used during preparation. Results return along
the same path: the solver returns arrays and diagnostics to `_fit`, which adds
PIPs and credible sets and builds the `SusieResult`.

| Layer | Main modules | Responsibilities |
| --- | --- | --- |
| Python package | `api.py`, `individual.py`, `rss_lambda.py`, `rss_models.py`, `posterior.py`, `result.py`, `refinement.py` | Public functions and argument checks; alignment of identifiers and metadata; `maf` filtering, null column and prior weights; RSS preparation; choice of route; the sweep loop of dynamic RSS models; PIPs, credible sets and `SusieResult`; continuation and refinement |
| Binding | `lib.rs`, `options.rs`, `array_layout.rs`, `summary_helpers.rs`, `prepare_helpers.rs`, `design.rs`, `ser.rs`, `output.rs` | Read-only borrowing of NumPy arrays after layout checks; conversion of options to typed Rust values; native matrix validation, repair and preparation; credible-set purity; conversion of results and errors |
| Solver | see the table below | IBSS sweeps, single-effect regression (SER), prior-variance search, expected residual sum of squares (ER2) and ELBO, stopping rules, trimming and Gram operators |

| Solver module | Contents |
| --- | --- |
| `ibss.rs` | `fit_controlled`, the dense entry point, and `fit_operator`: sweeps, effect updates, residual variance and trimming |
| `ser.rs` | Gaussian and fixed-mixture SER (Bayes factors, posterior, KL) and the standalone `ser_update` |
| `optimizer.rs` | One-dimensional Brent search for the prior variance |
| `residual.rs` | Expected residual sum of squares |
| `diagnostics.rs` | Stopping rules, cycle detection and the convergence trace |
| `operator.rs`, `exact_columns.rs`, `activation.rs` | The `GramOperator` trait, dense products, implicit scaling and exact signed-column classes |
| `design.rs` | Matrix-free operators for dense, sparse and trend-filtering designs |
| `nig.rs`, `slot.rs` | Normal–Inverse-Gamma residual-variance model and effect-activity priors |
| `vector_math.rs`, `numerical.rs`, `input_validation.rs` | Vector exponentials, compensated summation, digamma and finiteness checks |
| `types.rs`, `options.rs`, `model.rs` | Options, model extensions and output types |

## Fitting routes

All public fitting functions build a `_FitRequest` with `_prepare_suff_stat`
and pass it to `_fit`, which follows one of four routes:

| Route | Used for | Native entry | Sweep loop |
| --- | --- | --- | --- |
| Dense | `susie_suff_stat`; `susie_rss` with `R`; `susie_rss` with `X`, `bhat`, `shat` and `var_y`, which builds a correlation matrix | `_native.fit` → `fit_controlled` | Rust, with `DenseGram` |
| Matrix-free design | `susie`, `susie_trendfilter`; `susie_rss` with reference genotypes `X` | `fit_design_dense`, `fit_design_csc`, `fit_design_trend` | Rust, with `Design` or `TrendDesign` |
| Dynamic RSS | `susie_rss` with `R_finite`, `R_mismatch` other than `"none"` or a list of LD panels; `susie_rss_lambda` | `_native.ser_update` for each effect update | Python, `fit_dynamic` |
| Greedy | `L_greedy` | `_fit` repeatedly, with increasing `L` | That of the underlying route |

The dynamic models recompute their working quantities during fitting, so
their sweep loop runs in Python with NumPy matrix products. `susie_ser` calls
`_native.ser_update` once, and `susie_auto` calls `susie` repeatedly with warm
starts.

## Call paths

### `susie_rss`

```text
susie_rss                                        api.py
├─ argument checks; RSSControl
├─ list of LD panels → prepare_multipanel        rss_models.py
│    fit each panel, start from the best, joint dynamic fit
├─ z ← bhat/shat when z is not given; missing z ← 0 with a warning
├─ reference X → _validated_design, _column_stats → design route
├─ validate R: validate_and_repair_rss (fused) | validate_rss_matrix
├─ working statistics: n known | bhat, shat and var_y | n missing
├─ asymmetric R → RuntimeWarning → (M + Mᵀ)/2
├─ R_finite or R_mismatch → dynamic request
├─ _prepare_suff_stat → _FitRequest
└─ _fit
   ├─ dynamic → fit_dynamic → _native.ser_update for each effect
   ├─ design  → _native.fit_design_dense | _native.fit_design_csc
   ├─ dense   → _native.fit → fit_controlled → fit_operator
   ├─ credible_sets → _native.cs_pairwise_abs; marginal_pip
   ├─ SusieResult
   └─ refine  → refine_fit | _refine_by_refitting
```

### `susie`

```text
susie                                            individual.py
├─ type checks; _validated_design
│    dense → F-ordered float64, sparse → CSC copy, TrendDesign kept implicit
├─ _column_stats: column means and scales (centering stays implicit)
├─ working y, Xᵀy and diagonal → _DesignData
├─ _prepare_suff_stat(XtX=None, Xty, yty, n, design) → _FitRequest
└─ _fit → _native.fit_design_dense | fit_design_csc | fit_design_trend
        → credible_sets (correlations of members computed from X)
        → SusieResult with fitted values
```

### Inside the native fit

```text
_native.fit                                      rust/prusie-python/src/lib.rs
├─ parse options → FitOptions, FitControl, model extensions
├─ check NumPy layout and borrow read-only views
├─ fit_controlled                                rust/prusie-core/src/ibss.rs
│  ├─ check sizes, scalars and scale factors
│  ├─ implicit scaling, or a materialized working matrix
│  ├─ diagonal of Q (must be nonnegative) → DenseGram
│  └─ fit_operator
│     ├─ each sweep, for ℓ = 1, …, L in order:
│     │  ├─ single_effect → minimize             ser.rs, optimizer.rs
│     │  └─ GramOperator::apply                  operator.rs, exact_columns.rs
│     ├─ after the sweep: expected_residual_sum_squares, ELBO,
│     │  detect_cycle, stopping_decision          residual.rs, diagnostics.rs
│     └─ trimming of null effects
└─ fit_result → dict of NumPy arrays, warnings and diagnostics
```

The design entry points build a `Design` or `TrendDesign` operator and call the
same `fit_operator`. The binding holds the Python global interpreter lock and
the read borrows of the input arrays for the duration of the solver call.

### Refinement, continuation and warm starts

With `retain_data`, `track_fit`, `refine` or `model_init`, preparation stores
read-only copies of the inputs and a signature that identifies them.
`continue_fit` and `refine_fit` call `_native.fit` again on those copies with a
warm start, then recompute PIPs and credible sets. For each credible set,
`refine_fit` first fits with that set's prior weights set to zero, then refits
from that state with the original weights. It accepts a refit whose ELBO
exceeds the current one by more than the tolerance. Models whose inputs cannot
be retained (designs, dynamic RSS, mixture, NIG and slot priors, the PIP
criterion, `L_greedy` and nondefault credible-set controls) use
`_refine_by_refitting`, which repeats `_fit` as susieR does. A warm start
supplies `alpha`, `mu` and `mu2` (and optionally `V` and `sigma2`); the solver
recomputes every product $\mathbf Q\mathbf m_\ell$ before the first sweep.
[Convergence and refinement](post_analysis.md) describes these interfaces.

## Pseudocode

The pseudocode follows the Gaussian model with the default options. Fixed
mixture priors replace `single_effect` by `single_effect_mixture`. The NIG
model uses `single_effect_nig`, EM prior updates, the PIP rule when $L>1$ and
its own objective and residual-variance update. Slot priors weight each stored
product by the effect's activity probability and skip inactive effects.

### Preparation

```text
function susie_rss(z or (bhat, shat), R or X, n, var_y, …):
    check argument combinations and option types
    z ← bhat / shat if z is not supplied;  missing z ← 0 with a warning
    validate R: finite, diagonal ≥ 0, exact symmetry ((n − 1)R for z-scores with known n)
    if n is missing:            XtX ← R, Xty ← z, yty ← 1 (with a warning)
    else if bhat, shat, var_y:  reconstruct XtX (always averaged with its transpose),
                                Xty and yty on the response scale
    else:                       XtX ← (n − 1)R, Xty ← √(n − 1)·z̃, yty ← (n − 1)·var_y
                                (z̃ is the Wald-adjusted z when z_method = "wald";
                                 var_y is 1 when not supplied)
    if the z-score matrix was asymmetric: warn and use (M + Mᵀ)/2
    return _fit(_prepare_suff_stat(XtX, Xty, yty, n, …))

function _prepare_suff_stat(XtX, Xty, yty, n, options):
    check scalars, tolerances, priors and residual-variance bounds
    align identifiers and metadata; apply the maf filter
    if null_weight > 0: append a null column; normalize the prior weights
    L ← min(L, p)
    d_j ← sqrt(XtX_jj / (n − 1)) if standardize else 1;  zero scales ← 1
    pass XtX with scale factors (implicit scaling), or prepare D⁻¹ XtX D⁻¹
    build native options: prior variance × yty / (n − 1), convergence, warm start
    keep read-only copies if continuation or refinement must be supported
    return _FitRequest

function _fit(request):
    native ← solver result for the request's route
    re-emit native warnings; warn if not converged
    pip  ← marginal_pip(alpha, V)
    sets ← credible_sets(alpha, V, correlation or X)
    result ← SusieResult(native, pip, sets, …)
    return refine(result) if refinement was requested else result
```

### IBSS

```text
function fit_operator(Q, g = Xᵀy, yᵀy, n, options):
    α_ℓ ← π,  μ_ℓ = μ2_ℓ ← 0,  V_ℓ ← V₀,  σ² ← σ₀²
    u_ℓ ← 0,  f ← 0              # warm start: u_ℓ ← Q(α_ℓ ∘ μ_ℓ), f ← Σ_ℓ u_ℓ
    for t = 1, …, max_iter:
        for ℓ = 1, …, L:            # sequential: each update uses the previous ones
            f ← f − u_ℓ;  r ← g − f # residual crossproduct without effect ℓ
            (α_ℓ, μ_ℓ, μ2_ℓ, V_ℓ, lbf_ℓ) ← single_effect(r, diag Q, σ², V_ℓ, π)
            KL_ℓ ← −lbf_ℓ + E_q[log-likelihood of the single-effect model]
            u_ℓ ← Q(α_ℓ ∘ μ_ℓ);  f ← f + u_ℓ     # one product with Q per update
        if check_prior and some V_ℓ > 100 · max_j g_j² / (Q_jj σ²): error
        ER2  ← expected_residual_sum_squares(g, yᵀy, diag Q, α, μ, μ2, u)
        ELBO ← −(n/2) log(2πσ²) − ER2 / (2σ²) − Σ_ℓ KL_ℓ
        decision ← stopping_decision(ELBO history, α and PIP history)
        if decision ≠ continue: break
        if estimate_residual_variance: σ² ← clamp(ER2 / n, lower, upper)
    for ℓ with V_ℓ < prior_tol:     # trimming, also at the iteration limit
        α_ℓ ← π;  μ_ℓ, μ2_ℓ, lbf_ℓ, KL_ℓ ← 0;  V_ℓ ← 0
    return α, μ, μ2, V, σ², ELBO history, iterations, converged, diagnostics
```

Because the stopping test precedes the residual-variance update, a converged
fit keeps the $\sigma^2$ used in its last sweep, whereas a fit that reaches
`max_iter` receives one more update.

### Single-effect regression

```text
function single_effect(r, d, σ², V, π, method):
    b̂_j ← r_j / d_j;   s_j² ← max(σ² / d_j, ε)        # rebuilt when σ² changes
    ℓ(V) ← log Σ_j π_j BF_j(V)                         # SER log marginal likelihood
    if method = "optim":
        V ← argmax of ℓ(V) over log V ∈ [−30, 15]      # Brent search on log V
        keep the previous V if the search result is worse
    if method ∈ {"optim", "simple"} and ℓ(0) + check_null_threshold ≥ ℓ(V): V ← 0
    w_j ← log BF_j(V) + log(π_j + √ε)
    α_j ← exp(w_j − max_k w_k) / Σ_k exp(w_k − max_k w_k)
    v_j ← V s_j² / (V + s_j²);  μ_j ← v_j b̂_j / s_j²;  μ2_j ← v_j + μ_j²
    if method = "EM": V ← Σ_j α_j μ2_j
    lbf ← max_k w_k + log Σ_k exp(w_k − max_k w_k)
    return α, μ, μ2, V, lbf
```

### Stopping rule

```text
function stopping_decision(t, ELBO history, Δα, ΔPIP, cycle):
    infinite ELBO, or NaN ELBO with criterion "joint"   → numerical failure
    t < min_iter                                        → continue
    criterion "elbo":      converged if 0 ≤ ELBO(t) − ELBO(t − 1) < tol
    criterion "alpha_pip": converged if Δα < alpha_tol and ΔPIP < pip_tol
    criterion "joint":     converged if both hold
    criterion "pip":       converged if ΔPIP < tol and Δα < tol
    NaN ELBO with "elbo":  apply the PIP rule instead, recorded as a fallback
    converged                                           → stop
    cycle of length 2–5 under an α/PIP rule             → stop
    otherwise                                           → continue
```

$\Delta\alpha$ and $\Delta\mathrm{PIP}$ are maximum absolute changes from an
earlier sweep. Under the PIP rule a detected cycle counts as convergence and
alpha is averaged over the cycle; under `"alpha_pip"` and `"joint"` it stops
with `stop_reason="cycle_detected"`. Slot priors, NIG with $L>1$ and
`convergence_method="pip"` use the PIP rule. The Python sweep loop of the
dynamic RSS models implements only the ELBO and PIP rules, with the default
`min_iter` and cycle lag; `susie_rss` rejects other `ConvergenceConfig`
settings for these models before fitting.

## Boundaries and exceptions

### Errors

- `ValueError`: invalid values, shapes and option combinations, and every
  error from the native layer, whose message is passed through unchanged.
- `TypeError`: wrong types of boolean flags, `slot_prior`, `convergence`,
  `RSSControl` options, `refine` and `model_init`.
- `KeyError`: a missing key of the `SusieInputs` mapping returned by
  `init_only=True`.

Native functions return errors as messages, and the binding converts every
message into a `ValueError`. Warnings are `RuntimeWarning`s. Those issued by
`susie_rss`, by the shared preparation and fitting functions and by the solver
are also stored in `fit.warnings`.

### Input checks

| Condition | Behavior |
| --- | --- |
| Matrix of the wrong shape, nonfinite entry or negative diagonal | `ValueError` |
| Exactly asymmetric `R` or `XtX` | Warning; $(\mathbf M+\mathbf M^T)/2$ is used. There is no symmetry tolerance |
| LD diagonal not one, or $\lvert r_{ij}\rvert>1$ | Accepted; only finiteness, the diagonal sign and symmetry are required |
| $n\le1$, negative `yty`, or zero `yty` with nonzero `Xty` or an estimated residual variance | `ValueError` |
| `shat` missing or not positive | `ValueError` |
| Missing z-scores | Replaced with zero with a warning |
| Missing `Xty` entries | Replaced with zero with a warning; `ValueError` with `check_input=True` |
| Infinite summary statistics | `ValueError` |
| Negative or nonfinite prior weights, a zero sum, or `null_weight` outside $[0,1)$ | `ValueError` |
| `L` larger than the number of columns | `L` is reduced to that number, including any null column |
| Column with zero variance | Scale set to one; its Bayes factor is one, so it carries no information |
| `maf` filter that removes every variant | `ValueError` |
| `check_input=True` | Eigendecomposition: a negative eigenvalue raises `ValueError`; `Xty` outside the column space gives a warning |
| Infinite `y`, or missing `y` without `na_rm` | `ValueError` |

[Inputs and priors](inputs.md) gives the complete input rules.

### Numerical safeguards

Posterior weights are normalized after subtracting the largest log weight.
This prevents exponential overflow without changing relative probabilities.
For log weights $(1000, 999)$, $e^{1000}$ overflows in float64, whereas the
shifted form does not:

<!-- calculated:logsum:start -->
A 60-digit decimal calculation gives **1000.313261687518**; the first normalized weight is **0.731058578630**.
<!-- calculated:logsum:end -->

- **Floors.** Sampling variances are at least machine epsilon, log prior
  weights use $\log(\pi_j+\sqrt\epsilon)$ as in susieR, and slot log-odds are
  limited to $[-20,20]$.
- **Columns without information.** A variant with zero diagonal or a nonfinite
  sampling variance gets Bayes factor one and is excluded from the expected
  log-likelihood and from the `check_prior` bound.
- **Prior-variance search.** A nonfinite trial objective or more than 1000
  steps is an error.
- **Posterior and objective.** A nonfinite posterior quantity, an infinite
  ELBO or a nonpositive or nonfinite residual-variance estimate is an error.
  The estimate is otherwise limited to [`residual_variance_lowerbound`,
  `residual_variance_upperbound`].
- **Prior variance.** With `check_prior`, an estimated prior variance more than
  100 times the largest squared z-score is an error, which usually indicates
  inconsistent summary statistics and LD.
- **Fast paths.** Grouped, vectorized and implicitly scaled computations fall
  back to the general formula whenever an intermediate is nonfinite or
  subnormal; see [Optimization](optimization.md).

### Convergence outcomes

| `stop_reason` | `converged` | Meaning |
| --- | --- | --- |
| `criterion_satisfied` | `True` | The selected criterion was met |
| `iteration_limit` | `False` | `max_iter` sweeps ran without meeting it; a warning reports the iteration count |
| `cycle_detected` | `False` | Alpha and PIPs alternate among 2–5 states under `"alpha_pip"` or `"joint"` |
| `numerical_failure` | `False` | A NaN ELBO under `"joint"` |

An ELBO decrease larger than `tol` produces a warning and is counted, and the
fit continues. A NaN ELBO with a finite posterior under `"elbo"` switches that
sweep to the PIP rule and produces a warning. In all cases the fit returns its
current state. [Reading results](results.md#iterations-and-finalization)
describes the returned diagnostics.

### Trimming and output shapes

After the final sweep, including at the iteration limit, effects with
$V_\ell<$ `prior_tol` are reset: alpha to the prior weights and the moments,
log Bayes factors and KL terms to zero. The stored ELBO is not recomputed. With
a null column, `alpha`, `mu`, `mu2` and `lbf_variable` have one more column
than `pip`, and credible sets that contain the null column are rejected. If no
credible set qualifies, the set collections are empty and PIPs are still
returned. PIPs use effects with $V_\ell>$ `prior_tol`; credible sets use
$V_\ell>10^{-9}$.

## Numerical reproducibility

Agreement with susieR is validated with relative tolerance zero: complete final
component alpha matrices are compared with pinned susieR 0.16.6 at absolute
tolerance `1e-5`, with original component and variant order retained. PIPs and other summaries
are separate diagnostics; see [numerical accuracy](numerical_accuracy.md).

The solver uses float64 rather than long-double C helpers or R-specific
accumulation order. Small platform-dependent differences are assessed using the
complete-alpha tolerance above, not bitwise equality. The compile-time
`portable-math` feature disables the optional GNU/Linux vector-math shim and
uses the existing scalar fallback. It leaves model settings unchanged and makes
no promise of identical floating-point results across platforms. There is no
runtime environment switch selecting an experimental numerical algorithm.

## Reference source mapping

The reference is [susieR 0.16.6](https://github.com/stephenslab/susieR/tree/8e56a8e038e989856d106d9ca5175cc664fea9d2).
Corresponding functions at that commit are:

| prusie | susieR 0.16.6 |
| --- | --- |
| `ibss.rs`: `fit_operator` | `susie_workhorse` (`R/susie_workhorse.R`); `ibss_fit` (`R/iterative_bayesian_stepwise_selection.R`) |
| `ser.rs`: `single_effect`; `optimizer.rs`: `minimize` | `single_effect_update`, `single_effect_regression`, `optimize_prior_variance` (`R/single_effect_regression.R`) |
| `residual.rs`: `expected_residual_sum_squares`; ELBO in `ibss.rs` | `get_ER2`, `Eloglik` (`R/sufficient_stats_methods.R`); `get_objective` (`R/model_methods.R`) |
| `diagnostics.rs`: `stopping_decision`, `detect_cycle` | `check_convergence` (`R/model_methods.R`); `check_alpha_pip_cycle_convergence` (`R/susie_utils.R`) |
| `ibss.rs`: residual update and trimming | `update_model_variance`, `trim_null_effects` (`R/model_methods.R`) |
| `individual.py`: `susie`; `api.py`: `susie_suff_stat`, `susie_rss`; `rss_lambda.py`: `susie_rss_lambda` | `susie`, `susie_ss`, `susie_rss`, `susie_rss_lambda` (`R/susie.R`) and their constructors (`R/susie_constructors.R`) |

The Brent search in `optimizer.rs` is adapted from `Brent_fmin` in R's
`src/library/stats/src/optimize.c` (GPL-2.0-or-later). Attribution and licenses
are retained in the [third-party notices](https://github.com/LucaJiang/prusie/blob/main/THIRD_PARTY_NOTICES.md).
