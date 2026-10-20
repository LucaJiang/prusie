# Implementation

Python prepares the model, validates aligned inputs and assembles the result.
A thin PyO3 crate checks NumPy metadata and converts options to typed Rust enums.
The solver crate has no Python dependency. It performs sequential IBSS, single-effect
regression (SER), variance estimation and objective evaluation.

## Solver structure

The Cargo workspace contains `rust/prusie-core` and `rust/prusie-python`.
Within the solver, `ibss.rs` controls component updates; `ser.rs` evaluates
single-effect posteriors; `optimizer.rs` implements the one-dimensional prior-variance
search; `residual.rs` computes expected residual sums of squares. `operator.rs`
handles crossproducts and `design.rs` applies individual-level design matrices.
The binding crate owns NumPy borrowing, Python option parsing and result conversion.

Individual-level fits apply dense or sparse design matrices directly, without
constructing a predictor-by-predictor Gram matrix. Centering and scaling are
represented by column means and scales. Sparse input remains sparse. Ordinary
single-panel RSS fits with reference `X` also use this representation. RSS routes
that reconstruct a known response scale from `bhat`, `shat` and `var_y`, apply
finite-reference or mismatch corrections, combine reference panels, or use
`susie_rss_lambda` construct dense predictor-by-predictor correlation matrices.
Dense sufficient-statistic and LD inputs retain their dense representation.

## Retained component products

Write $\mathbf m_\ell=\boldsymbol\alpha_\ell\odot\boldsymbol\mu_\ell$,
$\mathbf u_\ell=\mathbf Q\mathbf m_\ell$ and
$\mathbf f=\sum_\ell\mathbf u_\ell$. A component update uses

$$
\mathbf r_\ell=\mathbf g-(\mathbf f-\mathbf u_\ell),\qquad
\mathbf f\leftarrow\mathbf f-\mathbf u_\ell+\mathbf Q\mathbf m_\ell^{\mathrm{new}}.
$$

Retaining each component product costs $O(Lp)$ storage and avoids recalculating
its old contribution. A dense sweep still needs $O(Lp^2)$ work. For example,
$\mathbf Q=((2,1),(1,2))$, $\mathbf g=(1,0.8)$,
$\mathbf m_1=(0.1,0)$ and $\mathbf m_2=(0,0.2)$ give:

<!-- calculated:cache:start -->
The products are $\mathbf u_1=(0.2, 0.1)^T$ and $\mathbf u_2=(0.2, 0.4)^T$, giving $\mathbf f=(0.4, 0.5)^T$. Component 1 uses residual **(0.8, 0.4)**. Replacing its mean by $\mathbf m_1^{\mathrm{new}}=(0.2, 0.1)^T$ changes the fitted crossproduct to **(0.7, 0.8)**.
<!-- calculated:cache:end -->

Posterior uncertainty enters the objective through

$$
\mathrm{ER2}=y^Ty-2\mathbf g^T\mathbf m+\mathbf m^T\mathbf Q\mathbf m
+\sum_\ell\left\{\sum_j Q_{jj}\alpha_{\ell j}\mu^{(2)}_{\ell j}
-\mathbf m_\ell^T\mathbf Q\mathbf m_\ell\right\}.
$$

The retained products also supply these quadratic forms. Compensated float64
summation reduces cancellation in reductions; it does not reproduce R's
accumulation order or use extended precision to mimic R.

## Single-effect arithmetic

For marginal estimate $\widehat b_j$, sampling variance $s_j^2$ and prior
variance $V$, the Gaussian log Bayes factor is

$$
\log\mathrm{BF}_j=-\tfrac12\log(1+V/s_j^2)
+\tfrac12\widehat b_j^2\frac{V}{s_j^2(V+s_j^2)}.
$$

<!-- calculated:ser:start -->
For $s^2=0.1$, $V=0.2$ and marginal estimates $(0.1,0.2,0.3)$, the common log term is -0.549306 and the denominator is 0.03. The log Bayes factors are **(-0.515973, -0.415973, -0.249306)**.
<!-- calculated:ser:end -->

Posterior weights are normalized after subtracting the largest log weight.
This prevents exponential overflow without changing relative probabilities.

<!-- calculated:logsum:start -->
A 60-digit decimal calculation gives **1000.313261687518**; the first normalized weight is **0.731058578630**.
<!-- calculated:logsum:end -->

The implementation uses ordinary float64 arithmetic. On native GNU/Linux
x86-64 builds, shifted exponentials use the system vector math library when
available and supported by the CPU; scalar evaluation handles unavailable
symbols, tails and nonstandard caller rounding modes. Initialization is cached,
and the caller's floating-point environment is preserved. Existing variances
are used directly without log/exp round trips. Complete final component alpha
matrices are compared with pinned susieR 0.16.6 at absolute tolerance `1e-5`,
with original component and variant order retained. PIPs and other summaries
are separate diagnostics; see [numerical accuracy](numerical_accuracy.md).

## Dense storage and matrix products

<!-- calculated:memory:start -->
At $p=5000$, $8p^2=200,000,000$ bytes, or approximately **190.7 MiB**.
<!-- calculated:memory:end -->

The dense operator applies diagonal scaling implicitly, avoiding a separate
scaled matrix when the factors permit safe evaluation. Its products use a four-column kernel
with runtime-checked AVX2 support and a portable fallback. Gaussian single-effect
updates reuse groups of equal sampling variance. Default optimizations are
selected using numerical validation and lower median complete-call runtime on
a strict majority of a fixed case panel; this does not imply every input is faster.
There are no environment variables for selecting alternative mathematical paths.
`PRUSIE_NUM_THREADS` controls native matrix-check threads (1–32), is read once
per process, and should be set before importing prusie. Component updates are
sequential. The primary comparisons use one numerical thread throughout.

## Reference source mapping

The reference is [susieR 0.16.6](https://github.com/stephenslab/susieR/tree/8e56a8e038e989856d106d9ca5175cc664fea9d2).
The Gaussian updates correspond to `single_effect_regression`,
`update_each_effect` and `susie_workhorse`; preprocessing follows the corresponding
individual-data and summary-statistic entry points. Attribution and licenses are
retained in the [third-party notices](https://github.com/LucaJiang/prusie/blob/main/THIRD_PARTY_NOTICES.md).
