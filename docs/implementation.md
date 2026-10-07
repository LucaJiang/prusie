# Implementation and optimizations

For dense LD, the main repeated cost is multiplying the predictor crossproduct
matrix by a component's posterior mean. Prior-variance searches add many
single-effect likelihood evaluations, and input preparation can allocate another
dense matrix. prusie reduces these costs by retaining component products,
reusing SER workspaces, and applying matrix structure when its conditions hold.
The statistical update remains sequential IBSS as described in
[Statistical model](model.md).

## From the public call to a result

Python validates options and defines the working sufficient statistics. It also
normalizes priors, chooses predictor scales and assembles `SusieResult`. Matrix
validation and some preparation run in Rust binding helpers; Python does not
perform every matrix operation itself. The PyO3 boundary checks array metadata
before borrowing NumPy storage. Rust then executes IBSS, including SER, variance
updates and the objective. Finally, Python computes marginal PIPs and controls
credible-set selection, while a Rust helper extracts complete pairwise purity.

```text
susie_rss ── RSS working quantities ─┐
                                  ├─ _susie_suff_stat ─ fit binding
susie_suff_stat ─ public checks ────┘                     │
                                    fit_layout_operator / single_effect
                                                        │
                                     credible_sets / marginal_pip
                                                        │
                                                   SusieResult
```

The private `_rss_matrix_validated` flag belongs only to the RSS call path,
which has already checked raw and scaled matrix entries together. Direct public
`susie_suff_stat` calls always validate their matrix. Binding checks additionally
protect dtype, dimensions, alignment and strides after option conversion,
because Python conversions can execute callbacks that mutate array metadata.
These boundaries serve different purposes.

## Retaining component matrix products

Write $\mathbf m_\ell=\boldsymbol\alpha_\ell\odot\boldsymbol\mu_\ell$,
$\mathbf u_\ell=\mathbf Q\mathbf m_\ell$ and
$\mathbf f=\sum_\ell\mathbf u_\ell$. The update can use a retained old product:

$$
\begin{aligned}
\mathbf r_\ell&=\mathbf g-(\mathbf f-\mathbf u_\ell),\\
\mathbf u_\ell^{\mathrm{new}}&=\mathbf Q\mathbf m_\ell^{\mathrm{new}},\\
\mathbf f&\leftarrow\mathbf f-\mathbf u_\ell^{\mathrm{old}}
+\mathbf u_\ell^{\mathrm{new}}.
\end{aligned}
$$

`component_products` stores the $\mathbf u_\ell$ vectors and `output.xtxr`
stores their running sum. A product remains valid until its component mean
changes; the matrix and its scaling remain fixed throughout the fit. Initial
means and products are zero. Updating one component needs one new matrix–vector
product, while removing its old contribution needs only vector subtraction.
This avoids recomputing the old product or rebuilding the fitted contribution
from all other components. Maintaining fitted contributions is also used in
susieR; the retained per-component products make their reuse explicit here.

For a hand-sized example, take

$$
\mathbf Q=\begin{pmatrix}2&1\\1&2\end{pmatrix},\quad
\mathbf g=\begin{pmatrix}1\\0.8\end{pmatrix},\quad
\mathbf m_1=\begin{pmatrix}0.1\\0\end{pmatrix},\quad
\mathbf m_2=\begin{pmatrix}0\\0.2\end{pmatrix}.
$$

<!-- calculated:cache:start -->
The products are $\mathbf u_1=(0.2, 0.1)^T$ and $\mathbf u_2=(0.2, 0.4)^T$, giving $\mathbf f=(0.4, 0.5)^T$. Component 1 uses residual **(0.8, 0.4)**. Replacing its mean by $\mathbf m_1^{\mathrm{new}}=(0.2, 0.1)^T$ changes the fitted crossproduct to **(0.7, 0.8)**.
<!-- calculated:cache:end -->

The example's new mean is supplied solely to illustrate the replacement
arithmetic. In a fit, SER computes that mean from the residual and variances.
The storage for retained products is $O(Lp)$; a dense sweep still costs
$O(Lp^2)$ for the new products. `multiply_columns_grouped` accumulates up to
four nonzero columns while keeping output entries in registers. It reduces
output loads and stores, without dropping small nonzero coefficients.

### Expected residual sum of squares

Posterior uncertainty contributes to the expected squared residual as well as
the total posterior mean:

$$
\begin{aligned}
\mathrm{ER2}
&=\mathbb E_q\|\mathbf y-\mathbf X\mathbf b\|^2\\
&=s_y-2\mathbf g^T\mathbf m+\mathbf m^T\mathbf Q\mathbf m\\
&\quad+\sum_\ell\left\{\sum_jQ_{jj}\alpha_{\ell j}M^{(2)}_{\ell j}
-\mathbf m_\ell^T\mathbf Q\mathbf m_\ell\right\}.
\end{aligned}
$$

The first line of crossproduct terms is the residual sum of squares at the
posterior mean. The component corrections restore the uncertainty lost by using
that mean alone, which is why `mu2` must contain a conditional second moment.
`expected_residual_sum_squares` reuses `component_products` both in the total
quadratic form and in each component's quadratic correction. No extra dense
product is required for ER2.

`ResidualScratch` holds length-$p$ mean and fitted vectors plus $L$ component
quadratics. The conditional-second-moment terms stream through the compensated
sum in component-major, then variant-major order, retaining the parentheses
`diagonal[j] * (alpha[idx] * mu2[idx])`. An $L\times p$ array of those terms is
unnecessary. The fitted vector is freshly summed from component products for
the objective: replacing it with the repeatedly subtracted and added `XtXr`
cache would change floating-point rounding. The small component-quadratic
workspace keeps the order of the final compensated sum explicit.

## Shared SER terms and reusable storage

A prior-variance search changes $V$ while the crossproduct residual, marginal
estimate $\widehat b_j$, sampling variance $s_j^2=\sigma^2/Q_{jj}$ and prior
weights remain fixed. For variants with equal $s_j^2$, their log Bayes factors
share the logarithm and denominator:

$$
\log\mathrm{BF}_j(V)=
\underbrace{-\tfrac12\log(1+V/s^2)}_{\text{shared}}
+\tfrac12\widehat b_j^2
\underbrace{\frac{V}{s^2(V+s^2)}}_{\text{shared}}.
$$

<!-- calculated:ser:start -->
For $s^2=0.1$, $V=0.2$ and marginal estimates $(0.1,0.2,0.3)$, the common log term is -0.549306 and the denominator is 0.03. The log Bayes factors are **(-0.515973, -0.415973, -0.249306)**.
<!-- calculated:ser:end -->

Thus one trial variance needs one shared logarithm and denominator for this
group, plus each variant's multiply and add. The example counts algebraic
subexpressions, not measured speedup. Across updates, the fixed matrix diagonal
is retained. `SerSampling` also retains sampling variances and their group
layout until `sigma2` changes. A residual-variance update invalidates it.
Residuals, marginal estimates and their squared values are rebuilt for each
component.

`SerBuffers` retains capacities for squared estimates, group indices,
variance groups, group terms and group maxima. `SerLikelihood` borrows the
sampling layout when possible and takes then returns reusable buffers. Its
contents always correspond to the current component. Search scratch is filled
for every likelihood evaluation. With uniform weights, the maximum squared
estimate within each variance group locates the largest log weight without
another full maximum scan.

`single_effect` constructs a search context lazily for `optim` or `simple`.
Fixed-V and EM updates only need posterior evaluation, so they do not prepare
unused search invariants. The optimized winning objective is retained for the
null comparison, while the previous-V comparison follows the reference's
log/exp transformation. Posterior probabilities and moments are then written
into the component's existing result row. No other component is overwritten.

<details markdown="1">
<summary>Grouping limits and exceptional values</summary>

Groups require exactly equal finite sampling variances after the likelihood's
machine-epsilon floor. There is no rounding into bins. More than 16 distinct
values selects the generic path. A nonfinite marginal estimate takes a path
that rebuilds no-information groups explicitly. When a shared denominator or
coefficient has an exceptional floating-point scale, likelihood evaluation
falls back to the original per-variant division expression. The optional AVX2
four-lane group-fill path additionally requires at most four groups, uniform
weights, the selected libmvec AVX2 backend and a compatible floating-point
control state. Generic and scalar oracles remain independently tested.

</details>

## Applying matrix scaling implicitly

Preparing a scaled crossproduct can double dense matrix storage. For an input
matrix $\mathbf M$, positive scalar $a$ and predictor scales in $\mathbf D$,
the working operator can instead apply

$$
\begin{aligned}
\mathbf Q&=a\mathbf D^{-1}\mathbf M\mathbf D^{-1},\\
\mathbf Q\mathbf v
&=a\mathbf D^{-1}\left[\mathbf M(\mathbf D^{-1}\mathbf v)\right].
\end{aligned}
$$

`MatrixScaling` carries the scalar, scales and reciprocals. `scaled_multiply`
scales the vector, multiplies by the retained matrix, and scales the result.
This uses $O(p)$ scratch instead of materializing another $p\times p$ matrix
when the operator conditions hold. For comparison:

<!-- calculated:memory:start -->
At $p=5000$, $8p^2=200,000,000$ bytes, or approximately **190.7 MiB**.
<!-- calculated:memory:end -->

This is the size of **one** float64 matrix, not a prediction of process peak
memory. Input normalization, null-column augmentation or a fallback can still
require dense storage.

The operator certificate requires $1\leq a\leq10^9$, scales and their inverses
in $[0.5,2]$, and every raw matrix entry satisfying $|M_{ij}|\leq2$. The full
entry check also rules out NaNs and infinities. Outside these bounds,
`materialize_scaled` performs the ordered entry calculation
`((M[i,j] * a) * inverse[j]) / scales[i]` and checks finiteness. If an implicit
product creates a nonfinite intermediate, it also uses the materialized route.
These are execution bounds, not changes to the accepted statistical model.

## Identical and sign-reversed matrix columns

Perfect LD can make some columns redundant for multiplication. If

$$
\mathbf Q_{:j}=s_j\mathbf Q_{:g(j)},\quad s_j\in\{-1,+1\},
\qquad
\mathbf Q\mathbf v=\sum_g\mathbf Q_{:g}
\sum_{j:g(j)=g}s_jv_j,
$$

then signed coefficients can be combined before multiplying representative
columns. Here $s_j$ is a column sign, unrelated to the SER sampling variance
$s_j^2$ above. For example,

$$
\mathbf Q=\begin{pmatrix}1&1&-1\\1&1&-1\\-1&-1&1\end{pmatrix},
\qquad\mathbf v=\begin{pmatrix}0.2\\0.3\\-0.1\end{pmatrix}.
$$

<!-- calculated:columns:start -->
The signed coefficient is $0.2+0.3-(-0.1)=0.6$, and the product is **(0.6, 0.6, -0.6)**.
<!-- calculated:columns:end -->

All three original variants retain their individual priors and posterior
entries. Only their matrix contributions are combined. `ExactColumns` uses
hashes to find candidates, then compares entire columns to establish equality,
including sign. A hash collision never establishes a relationship by itself.
If additional full row checks succeed, a compact representative matrix can
also reduce the product's output dimension; the result is then expanded in
original order with the recorded signs. Otherwise the full-length
representative columns remain in use. With implicit scaling, these relations
are discovered in the retained matrix $\mathbf M$ and applied to the
scale-adjusted vector.

The three-variant example explains the algebra. Actual discovery is lazy and
requires $p\geq1024$ and at least $\lceil p/8\rceil$ nonzero component means.
A plan is used only with at most $7p/8$ representatives. Small, sparse or
insufficiently redundant inputs use the full matrix path. Tests at the actual
size threshold cover discovery and its compact/full-column alternatives,
separately from the hand-sized example.

## Layout, ownership and memory

NumPy C-contiguous matrices store rows together; F-contiguous matrices store
columns together. A float64 $3\times3$ C array has byte strides `(24, 8)`;
its transposed view has `(8, 24)` and shares the same storage. `_matrix`
normalizes dtype and incompatible alignment/strides, validates all entries,
then returns C-contiguous storage. An ordinary C input can be reused;
an F input or a strided slice is copied at this public preparation boundary.
Read-only compatible arrays can be borrowed because the fit never writes to
input storage. Misaligned data are copied before constructing a Rust view.

The API passes a transposed F-contiguous view for column traversal. The direct
Rust binding accepts either C or F contiguous storage; a C matrix is converted
to columns inside the core. `prepare_crossproduct` handles applicable
contiguous preparation with runtime SIMD, with a NumPy fallback. The capability
check selects the matrix-operator interface only when supported; tests also use
the materialized route as a comparison oracle. `IntoPyArray` transfers owned
Rust result buffers into NumPy, avoiding a second copy of those outputs.

| Storage | Allocations and lifetime |
| --- | --- |
| $O(p^2)$ | Caller/input matrix; copies needed for normalization, null augmentation, known-var_y reconstruction or materialized scaling; optional compact representative matrix |
| $O(Lp)$ | Four component output matrices (`alpha`, `mu`, `mu2`, `lbf_variable`), retained component products and previous alpha |
| $O(p)$ | Component, product and residual vectors; total mean/fitted scratch; diagonal and scales; likelihood scratch, sampling variances and SER buffers; optional scaled-vector scratch |
| $O(L)$ and $O(\mathtt{max\_iter})$ | Component quadratics, V/BF/KL and objective trace |
| Optional/data dependent | Eigenvectors and eigensolver workspace for `check_input`; up to five alpha/PIP states for nonfinite convergence; $k(k-1)/2$ pair values for a retained $k$-variant set |

Credible-set ordering also uses per-component length-$p$ arrays and stores
candidate member lists. Its pairwise helper can reject a failing set before
allocating all pair values; retained sets still receive complete purity
summaries. The [performance measurements](performance.md) include interpreter,
imports and inputs as well as these algorithm arrays.

## Numerical operations and parallel work

SER normalizes log weights after subtracting their maximum. With log weights
1000 and 999, the corresponding calculation is

$$
\log(e^{1000}+e^{999})=1000+\log(1+e^{-1}).
$$

<!-- calculated:logsum:start -->
A 60-digit decimal calculation gives **1000.313261687518**; the first normalized weight is **0.731058578630**.
<!-- calculated:logsum:end -->

This avoids forming either overflowing exponential. The objective's positive
exponential terms use balanced pairwise reduction. Posterior normalization and
signed dot products/KL/ER2 reductions use the compensated `sum` helper where
called. Dense column products have their own fixed traversal. Algebraic
identities such as implicit scaling and signed-column grouping can change
floating-point association; bitwise equality to R is not implied. The
[version-specific numerical comparison](numerical_accuracy.md) reports actual
differences with the reference.

The `vector-math` feature builds a small C helper on native GNU/Linux x86-64.
It loads available libmvec symbols and selects AVX2 or SSE2 at runtime. Missing
symbols or unsupported builds retain scalar exponentials. Nondefault rounding
or flush controls also require scalar evaluation; scalar tails preserve the
same exceptional-value handling. `PRUSIE_SER_MATH=scalar`, set before import,
selects the portable exponential path. It does not disable every SIMD matrix
helper. Generic builds do not require an AVX2 processor to import or fit.

Ordinary use requires no thread tuning. IBSS components always update
sequentially. For sufficiently large matrices, `PRUSIE_NUM_THREADS` permits
parallel RSS validation, signed-column discovery and representative row
verification. It defaults to one and accepts 1–32; invalid values fall back to
one. NumPy's optional eigendecomposition may use BLAS threads configured
separately. The binding retains the GIL during borrowed-array fitting;
independent processes, rather than Python threads, can fit separate regions
concurrently. Budget their dense matrices and numerical threads together.
The retained R comparison uses one numerical thread per fit.

## Source navigation

| Responsibility | Source and functions |
| --- | --- |
| Public contract, RSS reconstruction and assembly | [api.py](https://github.com/LucaJiang/prusie/blob/main/src/prusie/api.py): `susie_rss`, `susie_suff_stat`, `_susie_suff_stat` |
| Array alignment and normalization | [_array_layout.py](https://github.com/LucaJiang/prusie/blob/main/src/prusie/_array_layout.py): `native_array`; [array_layout.rs](https://github.com/LucaJiang/prusie/blob/main/bindings/array_layout.rs): `ensure_layout` |
| Entry validation and preparation | [summary_helpers.rs](https://github.com/LucaJiang/prusie/blob/main/bindings/summary_helpers.rs): `validate_matrix`, `validate_rss_matrix`; [prepare_helpers.rs](https://github.com/LucaJiang/prusie/blob/main/bindings/prepare_helpers.rs): `prepare_crossproduct` |
| Borrowed inputs and owned outputs | [bindings/lib.rs](https://github.com/LucaJiang/prusie/blob/main/bindings/lib.rs): `fit` |
| IBSS, SER and matrix products | [core.rs](https://github.com/LucaJiang/prusie/blob/main/rust/core.rs): `fit_layout_operator`, `single_effect`, `SerSampling`, `SerLikelihood`, `ExactColumns`, `expected_residual_sum_squares` |
| Brent search | [optimizer.rs](https://github.com/LucaJiang/prusie/blob/main/rust/optimizer.rs): `minimize` |
| Optional exponentials | [vector_math.rs](https://github.com/LucaJiang/prusie/blob/main/rust/vector_math.rs), [vector_math.c](https://github.com/LucaJiang/prusie/blob/main/rust/vector_math.c) |
| Posterior summaries | [posterior.py](https://github.com/LucaJiang/prusie/blob/main/src/prusie/posterior.py): `marginal_pip`, `credible_sets`; [summary_helpers.rs](https://github.com/LucaJiang/prusie/blob/main/bindings/summary_helpers.rs): `cs_pairwise_abs` |

## Reference source mapping

The statistical reference is susieR 0.16.6, commit
`8e56a8e038e989856d106d9ca5175cc664fea9d2`. The corresponding source locations are:

| Calculation | Pinned susieR source |
| --- | --- |
| RSS working quantities and predictor scaling | [susie_constructors.R](https://github.com/stephenslab/susieR/blob/8e56a8e038e989856d106d9ca5175cc664fea9d2/R/susie_constructors.R): `summary_stats_working_quantities`, `summary_stats_constructor`, `sufficient_stats_constructor` |
| SER Bayes factors and moments | [single_effect_regression.R](https://github.com/stephenslab/susieR/blob/8e56a8e038e989856d106d9ca5175cc664fea9d2/R/single_effect_regression.R): `gaussian_ser_lbf`, `gaussian_ser_moments` |
| Prior updates and fitting schedule | [generic_methods.R](https://github.com/stephenslab/susieR/blob/8e56a8e038e989856d106d9ca5175cc664fea9d2/R/generic_methods.R), [susie_workhorse.R](https://github.com/stephenslab/susieR/blob/8e56a8e038e989856d106d9ca5175cc664fea9d2/R/susie_workhorse.R) |
| Sufficient-statistic objective and residual updates | [sufficient_stats_methods.R](https://github.com/stephenslab/susieR/blob/8e56a8e038e989856d106d9ca5175cc664fea9d2/R/sufficient_stats_methods.R), [model_methods.R](https://github.com/stephenslab/susieR/blob/8e56a8e038e989856d106d9ca5175cc664fea9d2/R/model_methods.R) |
| PIP and credible sets | [susie_get_functions.R](https://github.com/stephenslab/susieR/blob/8e56a8e038e989856d106d9ca5175cc664fea9d2/R/susie_get_functions.R), [susie_utils.R](https://github.com/stephenslab/susieR/blob/8e56a8e038e989856d106d9ca5175cc664fea9d2/R/susie_utils.R) |

The scalar optimizer derives from R's `stats/src/optimize.c`; the
[third-party notices](https://github.com/LucaJiang/prusie/blob/main/THIRD_PARTY_NOTICES.md)
retain its attribution and license. [Contributing](contributing.md) gives
commands for the independent reference, scalar/vector and layout tests.
