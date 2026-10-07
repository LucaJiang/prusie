# Implementation and optimizations

The public call performs input validation, summary-statistic preparation,
sequential native inference and posterior summaries. Python owns the API and
result objects; Rust owns the IBSS loop and single-effect regression (SER).
Several numerical helpers are also implemented in Rust. These are engineering
implementations of the established SuSiE updates.

## Source map

| Phase | Executed functions | Responsibility |
| --- | --- | --- |
| Python API | [susie_rss and _susie_suff_stat](https://github.com/LucaJiang/prusie/blob/main/src/prusie/api.py) | Validate options, transform summaries, normalize priors, choose matrix representation and assemble SusieResult |
| Array preparation | [_array_layout.py](https://github.com/LucaJiang/prusie/blob/main/src/prusie/_array_layout.py), [prepare_helpers.rs](https://github.com/LucaJiang/prusie/blob/main/bindings/prepare_helpers.rs) | Normalize incompatible arrays; scan matrix entries; prepare scaled crossproducts when needed |
| NumPy boundary | [bindings/lib.rs::fit](https://github.com/LucaJiang/prusie/blob/main/bindings/lib.rs) | Validate dtype, dimensions, alignment and strides after option conversion; borrow input arrays; transfer output buffers |
| Inference | [core.rs::fit_layout_operator and single_effect](https://github.com/LucaJiang/prusie/blob/main/rust/core.rs) | Sequential IBSS, SER, prior/residual variance and ELBO |
| Prior variance search | [optimizer.rs::minimize](https://github.com/LucaJiang/prusie/blob/main/rust/optimizer.rs) | Brent search adapted from R's optimizer, with retained attribution |
| Posterior summaries | [posterior.py](https://github.com/LucaJiang/prusie/blob/main/src/prusie/posterior.py), [summary_helpers.rs](https://github.com/LucaJiang/prusie/blob/main/bindings/summary_helpers.rs) | Python PIP and CS ordering/filtering; native complete pairwise purity extraction |

The path is **Python API → validated arrays/working quantities → PyO3/NumPy
boundary → Rust IBSS/SER → Python posterior summaries → SusieResult**.
No R interpreter participates in fitting.

## Sequential updates and matrix products

Let Q = XᵀX, g = Xᵀy and mℓ = αℓ ⊙ μℓ. The crossproduct residual for component ℓ is

**rℓ = g − Q ∑k≠ℓ mk**.

`component_products` stores Qmℓ for each component, and `output.xtxr` stores
their running sum. At each update the old Qmℓ is subtracted, SER is evaluated
on rℓ, and the new Qmℓ is added. Initially all coefficient means and products
are zero. A product remains valid until that component's mean changes. The
matrix and its scaling stay fixed during one fit; no cache survives a public
call. Thus each component needs one new matrix-vector product, without a second
product to remove its old contribution.

```python
# Mathematical pseudocode for the native sequential loop.
fitted -= component_products[ell]
residual = g - fitted
alpha[ell], mu[ell], mu2[ell] = SER(residual, diagonal, sigma2, V[ell])
component_products[ell] = Q @ (alpha[ell] * mu[ell])
fitted += component_products[ell]
```

`expected_residual_sum_squares` reuses those products for the total quadratic
form and component corrections, with `ResidualScratch` vectors for total means,
fitted values, quadratic terms and second moments. This uses matrix linearity
without refitting components. Dense multiplication accumulates nonzero columns,
four at a time where possible, to reduce repeated output loads and stores;
it does not prune small nonzero coefficients.

For dense Q, a sweep is O(Lp²) plus SER searches and summaries. The cached
products cost O(Lp). Iteration count and prior-variance search evaluations remain
data dependent.

## SER and prior-variance workspace

In `single_effect`, per-variant marginal estimates are rebuilt from the current
residual. `SerSampling` holds σ²/Qjj and exact-equality groups of the finite
sampling variances (clipped to machine epsilon for the likelihood calculation).
At most 16 distinct groups are used; otherwise the generic calculation applies.
The cache is rebuilt whenever σ² changes. The diagonal stays fixed within a fit.

`SerBuffers` reuses the capacity of `half_beta_squared`, `group_index`,
`safe_groups`, `terms` and `group_max_half_beta_squared`. Each SER rebuilds the
residual-dependent terms and maxima. Equal sampling variances share logarithms
and denominators during the search over V; no rounding or binning forms groups.
Fixed-V and EM paths avoid constructing an unused search workspace. The `optim`
path retains the already evaluated winning objective for the null comparison,
while preserving the reference's comparison with the initial V.

These allocations and common subexpressions concern the same SER likelihood.
They do not reuse another component's posterior or alter its prior weights.
The Rust tests compare reused and newly constructed likelihood workspaces;
Python reference tests exercise fixed, optimized and EM variance paths.

## Matrix scaling, exact redundancy and fallback

The Python preparation can pass the validated unscaled matrix with scalar and
diagonal factors through `MatrixScaling`. Products apply the factors to the
coefficient vector and output, avoiding a second dense scaled matrix. This
route requires a global multiplier between 1 and 10⁹, scales and inverse scales
between 0.5 and 2, and every raw entry satisfying |Mij| ≤ 2 in
`operator_entries_bounded`. Outside these bounds, `materialize_scaled` evaluates
the ordered dense preparation. A nonfinite intermediate in a factored product
also uses this dense fallback. The factorization is algebraically equivalent;
floating-point association can differ.

For p ≥ 1024 and a component with at least ⌈p/8⌉ nonzero means, lazy discovery
can use exact signed matrix-column equality. Sampled hashes propose candidate
classes; full float64 equality verifies every accepted relation. The plan is
used only when the number of representatives is at most 7p/8. In a class with
column j = sj times its representative, multiplication first sums sj vj,
then multiplies representative columns. Additional full row checks determine
whether a compact representative matrix can be used; otherwise full-length
representative columns remain available.

All original variants, their priors and their posterior arrays remain in the
SER. This compresses a matrix operation, not the statistical model. It is
neither an approximate low-rank method nor an LD threshold. Matrices with no
useful exact redundancy retain dense multiplication. Tests include sign flips,
zeros, hash collisions, dense fallback and the original variant order.

## Layout and memory

Compatible aligned arrays can be borrowed at the binding. Public normalization
converts dtype/layout when required; C-contiguous inputs have a transposed
column-major view for the native column traversal. Direct native row-major
inputs can require a column-major copy. Null-column augmentation, incompatible
strides, known-phenotype-variance RSS and scaled-matrix fallback can allocate
dense arrays. This is not a zero-copy public call.

| Memory category | Main objects |
| --- | --- |
| O(p²) | Retained input matrix; any required prepared matrix; optional exact representative matrix; optional eigendecomposition for check_input |
| O(Lp) | alpha, mu, mu2, variant log BFs, cached component products and previous alpha |
| O(p) | Residuals, one-component means/products, SER scratch, sampling quantities and predictor scales |
| Data dependent | CS pair correlations (O(k²) for a k-member set), optional nonfinite-convergence history and iteration trace |

Rust-owned result vectors are transferred into NumPy arrays through
`IntoPyArray`. The result retains scientific summaries rather than Q itself.
The [performance](performance.md) measurements are full-process peak memory,
which also includes interpreters, imports and input loading.

## Floating-point calculations and portability

SER log weights use max-shifted exponentials before normalization. Likelihood
log-sum-exp uses balanced reduction of nonnegative exponential terms; signed
posterior, KL and residual sums use the compensated `sum` helper where called.
These operations reduce numerical loss without requiring R's accumulation order.

The optional `vector-math` feature builds a small C helper only for native GNU
Linux x86-64. [vector_math.rs](https://github.com/LucaJiang/prusie/blob/main/rust/vector_math.rs)
and [vector_math.c](https://github.com/LucaJiang/prusie/blob/main/rust/vector_math.c)
load system libmvec symbols and detect AVX2 at runtime; SSE2 is available on
x86-64. Missing symbols, unsupported builds or `PRUSIE_SER_MATH=scalar` select
scalar exponentials. Nondefault floating-point rounding or flush modes also use
scalar behavior. Tail elements and representable subnormals have explicit tests.
Generic builds do not require an AVX2 CPU to import or fit.

## Threads and the Python boundary

The IBSS component loop is serial. `PRUSIE_NUM_THREADS` defaults to one and accepts
integer values from 1 to 32 (invalid values fall back to one); it controls independent exact-column discovery and representative
verification work on sufficiently large matrices. It does not parallelize
component updates or launch a fit cache. NumPy's optional eigenvalue checks may
use its loaded BLAS threads, configured separately.

The binding **retains the GIL while borrowing NumPy arrays and running the
native fit**. Python threads therefore do not provide concurrent public fits
through this binding. Independent processes can run separate regions, subject
to memory and thread budgeting. [Performance](performance.md) uses one numerical
thread. Whole-call speedups compare complete implementations; the retained
measurements do not isolate the contribution of an individual mechanism.
