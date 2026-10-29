# Speed and memory optimization

Large linkage disequilibrium (LD) matrices increase both the computational work
and memory needed for fine-mapping. prusie reduces repeated work and storage by
reusing component products, limiting passes over the input and avoiding extra
matrix allocations where possible.

These optimizations preserve the chosen [statistical model](model.md), priors,
stopping rules and variant order. Some regroup floating-point operations, so
agreement with susieR is assessed numerically; see
[computational compatibility](model.md#computational-compatibility).
[Implementation](implementation.md) describes the fitting routes, and
[Performance](performance.md) reports runtimes and memory for the inputs,
settings and hardware measured there.

The [benchmark measurements](performance.md#runtime) compare prusie and
susieR 0.16.6 on 200 FinnGen T2D GWAS loci and 14,157 GTEx Whole_Blood eQTL
genes, with three retained complete calls per case on exclusive EPYC 7742
nodes. For these evaluated cases, the cumulative runtime of susieR was 49.2
times that of prusie for GWAS and 26.6 times for eQTL (55.4 and 29.8 for the
fitting stage), and prusie was faster in every case. Ratios for other regions
and references may differ. The performance page also reports memory and
thread scaling.

## Principles

The optimizations reduce the work needed to evaluate the SuSiE updates while
retaining float64 arithmetic. They introduce no LD thresholding, variant
pruning or low-rank approximation. Grouping exactly equal or opposite matrix
columns changes only the matrix calculation: posterior probabilities remain
separate for every variant. Single-effect components are still updated in
sequence, using the latest residual.

Specialized calculations check input layout, dimensions, numerical bounds
and exact equality where required; vector instructions also require CPU
support. Implicit scaling and grouped matrix products fall back to the
general calculation when numerical checks fail. The decision to discover
exact-column classes uses deterministic work counts rather than timings
collected during the fit.

Equivalent formulas can have different floating-point results when they
regroup products or sums. Numerical validation compares the complete final
`alpha` matrix with pinned susieR under matched inputs and fitting settings,
in the original component and variant order. The
[computational compatibility](model.md#computational-compatibility) section
defines the comparison; [numerical accuracy](numerical_accuracy.md) reports the
tested configurations and results.

## Where the time and memory go

Let $p$ be the number of variants, $L$ the maximum number of single-effect
components, $T$ the number of completed sweeps and $m$ the number of members
in a candidate credible set. The table gives operation counts and array
storage for the core Gaussian fit from a dense LD or `XtX` matrix.

| Stage | Work | Array storage |
| --- | --- | --- |
| Matrix validation and preparation | $O(p^2)$ per full matrix pass | $O(p^2)$ per dense array; copies depend on [layout and options](#memory) |
| Component matrix product | $O(p^2)$ per update with the ordinary dense kernel | $O(p)$ workspace and $O(Lp)$ retained products |
| Single-effect regression (SER) | $O(p)$ for the posterior and each prior-variance objective evaluation | $O(p)$ scratch and $O(Lp)$ posterior arrays |
| Expected residual sum of squares and objective | $O(Lp)$ per sweep | $O(p+L)$ scratch, reusing retained products |
| Credible-set purity from the supplied matrix | $O(m^2)$ pair evaluations | $O(m^2)$ when pair correlations are collected |

Across $T$ sweeps, ordinary dense products require $O(TLp^2)$ work.
[Exact signed-column classes](#exact-signed-column-classes) can reduce this
work when their conditions hold. With `estimate_prior_method="optim"`,
each component update evaluates the SER likelihood repeatedly. Purity
screening can stop before visiting every pair if a pair already proves that
the set fails `min_abs_corr`.

For individual-level data with $n$ samples, products use the design directly.
A dense product costs $O(np)$; a sparse product costs
$O(\mathrm{nnz}+n+p)$, where $\mathrm{nnz}$ is the number of stored design
entries. Both use $O(n+p)$ product workspace; see
[Individual-level designs](#individual-level-designs).

## Fewer passes over the input matrix

### Reusing the input array

With the default `overwrite_ld=False`, preparation preserves the caller's LD
array. An aligned, native float64 C-contiguous `R` or `XtX` can be reused by
the dense solver: its transpose is a column-major view of the same storage.
Dtype conversion, alignment repair and conversion to C order can require
copies. Further arrays may be needed for symmetry repair, scaling or
retained analysis; see [Memory](#memory).

### Combined validation

The validation-only traversal checks finiteness, nonnegative diagonal entries
and exact symmetry together, visiting transposed pairs in tiles without a
$p\times p$ temporary. For ordinary RSS with LD matrix $\mathbf R$ and
known sample size $n$, including `bhat`/`shat` without `var_y`, the
symmetry comparison uses
$(n-1)R_{ij}$ and $(n-1)R_{ji}$. Without $n$, it compares the raw LD entries.
[Inputs](inputs.md#shapes-and-numerical-checks) describes the required checks
and the additional diagnostics enabled by `check_input=True`.

On x86-64, a vectorized proof can accept contiguous LD whose diagonal is
within $10^{-8}$ of one, whose entries have magnitude at most $1+10^{-8}$
and which is exactly symmetric. A guard on the RSS multiplier protects
off-diagonal scaling from overflow, and the scaled diagonal is checked
explicitly. A successful proof establishes the validation result without a
further general traversal. If it fails, the general checks decide whether
the matrix is valid.

### Fused validation and repair

If the pair comparisons detect asymmetry, prusie emits a `RuntimeWarning`
and averages the checked matrix with its transpose. For known-$n$ RSS,
scaling by $n-1$ precedes averaging. Repair writes the same computed value
to both triangles and checks the averaged entries for finiteness and a
nonnegative diagonal during that traversal.

With `overwrite_ld=False`, `susie_rss` with known `n`, `var_y=None` and a
plain, aligned, C-contiguous float64 NumPy `R` can combine validation and
repair. Unless the symmetry proof accepts the input, this route allocates a
private repair buffer and fills it while checking the raw and scaled inputs.
The buffer is retained for asymmetric input and discarded for symmetric
input. When fusion is unavailable, validation and required repair use
separate traversals.
The asymmetry warning precedes any error found in the repaired values.

`n_threads` defaults to 1. For eligible RSS inputs with $p\ge1024$, a larger
value can parallelize the proof and owned repair. Preparation workers join
before subsequent Python transformations or warnings. Warning display,
recording, logging capture and tracing do not determine eligibility.
Unsupported parallel preparation uses the serial calculation.
Caller callbacks must not modify input arrays
during a fit; see [Threads](#threads).

The opt-in `overwrite_ld=True` can instead use the caller's `R` for repair
after finding a finite asymmetric pair in the scaled matrix. Once writes
begin, later errors or warnings promoted to exceptions can leave `R` and
its aliases changed. Reload the original LD before reuse, including after a
failed call. [Optional LD workspace](api.md#optional-ld-workspace) describes
the supported storage and options.

### Implicit scaling

Let $\mathbf M$ be the input matrix after any repair and
$\mathbf D=\operatorname{diag}(d_1,\ldots,d_p)$ contain the
[predictor scale factors](model.md#data-notation-and-scale). The working
matrix is $\mathbf Q=a\,\mathbf D^{-1}\mathbf M\mathbf D^{-1}$.
For ordinary RSS with known $n$ and unrepaired $\mathbf M=\mathbf R$,
$a=n-1$; otherwise $a=1$, including when repair has already incorporated
that factor. For a vector $\mathbf v$ of length $p$, the solver can apply

$$
\mathbf Q\mathbf v
=a\,\mathbf D^{-1}\{\mathbf M(\mathbf D^{-1}\mathbf v)\}
$$

without storing $\mathbf Q$. This adds $O(p)$ vector scaling per product.
The native guard admits this calculation when $1\le a\le10^9$, every
$d_j$ and stored reciprocal $1/d_j$ lies in $[0.5,2]$, and every
$\lvert M_{ij}\rvert\le2\times10^9$. The entry checks also establish
finiteness. Outside these bounds, the solver materializes the scaled
matrix once and checks that its entries remain finite.

The bounds protect prepared matrix entries, while individual products still
need finiteness checks. A nonfinite intermediate or result in the factored
calculation causes that product to be recomputed with a temporary
materialized matrix. The formulas agree in exact arithmetic, but regrouping
the operations can change floating-point results. When Python must prepare
the full working matrix, its fallback uses one output array, with an
AVX-512 helper where supported or NumPy operations writing into that array.

## Matrix products in the sweep

### Retained component products

Each IBSS update needs the fitted contribution of all components except the
one being updated. In the ordinary Gaussian SuSiE fit, let
$\mathbf m_\ell=\boldsymbol\alpha_\ell\odot\boldsymbol\mu_\ell$ be component
$\ell$'s posterior mean, formed from its rows of `alpha` and `mu`, where
$\odot$ denotes elementwise multiplication. The solver retains
$\mathbf u_\ell=\mathbf Q\mathbf m_\ell$ and their sum
$\mathbf f=\sum_\ell\mathbf u_\ell$. With working predictor-response
crossproduct $\mathbf g$, the residual crossproduct and fitted update are

$$
\mathbf r_\ell=\mathbf g-(\mathbf f-\mathbf u_\ell),\qquad
\mathbf f\leftarrow\mathbf f-\mathbf u_\ell+\mathbf Q\mathbf m_\ell^{\mathrm{new}}.
$$

Keeping every $\mathbf u_\ell$ costs $O(Lp)$ storage. Each update then needs
only the new product with $\mathbf Q$; the old contribution is subtracted
from the cache. For a toy example, take

$$
\mathbf Q=\begin{pmatrix}2&1\\1&2\end{pmatrix},\qquad
\mathbf g=\begin{pmatrix}1\\0.8\end{pmatrix},\qquad
\mathbf m_1=\begin{pmatrix}0.1\\0\end{pmatrix},\qquad
\mathbf m_2=\begin{pmatrix}0\\0.2\end{pmatrix}.
$$

<!-- calculated:cache:start -->
The products are $\mathbf u_1=(0.2, 0.1)^T$ and $\mathbf u_2=(0.2, 0.4)^T$, giving $\mathbf f=(0.4, 0.5)^T$. Component 1 uses residual **(0.8, 0.4)**. Replacing its mean by $\mathbf m_1^{\mathrm{new}}=(0.2, 0.1)^T$ changes the fitted crossproduct to **(0.7, 0.8)**.
<!-- calculated:cache:end -->

The same products supply the Gaussian objective. Let $s_y$ be the working
response sum of squares and $M^{(2)}_{\ell j}$ the conditional posterior
second moment stored in `mu2`, as defined in the
[statistical model](model.md#single-effect-posteriors-and-effect-means).
The expected residual sum of squares is

$$
\mathrm{ER2}=s_y-2\mathbf g^T\mathbf m+\mathbf m^T\mathbf Q\mathbf m
+\sum_\ell\left\{\sum_j Q_{jj}\alpha_{\ell j}M^{(2)}_{\ell j}
-\mathbf m_\ell^T\mathbf Q\mathbf m_\ell\right\},
$$

with $\mathbf m=\sum_\ell\mathbf m_\ell$. Because
$\mathbf Q\mathbf m=\sum_\ell\mathbf u_\ell$ and
$\mathbf m_\ell^T\mathbf Q\mathbf m_\ell=\mathbf m_\ell^T\mathbf u_\ell$, every
quadratic form is an $O(p)$ dot product. Together with the diagonal terms,
this takes $O(Lp)$ work per sweep and requires no further matrix product.
The dot products and variance correction use compensated float64 summation
to reduce cancellation.

### Dense matrix–vector kernel

The ordinary column-major kernel adds matrix columns in ascending order,
skipping column $j$ when its coefficient $v_j$ is zero. An all-zero vector
therefore reads no matrix columns, although clearing the output and scanning
the coefficients still take $O(p)$ work. Groups of four nonzero columns update
each output element together, reducing output-vector loads and stores. Each
element receives the same sequence of separate float64 multiplications and
additions as in a column-by-column product.

On x86-64 an AVX2 build of this kernel is selected at run time when supported;
otherwise the portable build is used. `n_threads` defaults to 1. Eligible
large products can partition the output rows among threads while preserving
each row's column order. Component updates remain sequential; see
[Threads](#threads).

### Exact signed-column classes

Exactly equal or opposite LD columns allow repeated contributions to be
combined within a product. The grouping uses the matrix actually multiplied:
$\mathbf M$ under [implicit scaling](#implicit-scaling), otherwise the
materialized $\mathbf Q$. If its $p$ columns fall into $g$ signed classes,
prusie adds each class's signed coefficients with compensated summation and
multiplies only its representative column. This costs $O(pg)$ work. When a
separate check proves the corresponding row relations, a stored $g\times g$
matrix reduces the work to $O(g^2+p)$, including coefficient grouping and
expansion to all $p$ output entries.

Discovery hashes 16 sampled entries per column, then complete columns within
crowded buckets, to propose comparisons. Every accepted column relation is
verified over all entries with exact float64 equality. The compact form also
requires exact row relations. If the proved class count exceeds $7p/8$, no
plan is retained. These checks preserve separate posterior coordinates for
every variant; no LD threshold or rounding defines a class.

Full discovery is delayed by a deterministic work-count rule. A product is
*dense* for this rule when its coefficient vector has at least
$\lceil p/8\rceil$ nonzero entries. At the first dense product, sampled
hashes give a lower bound on the class count. If this already exceeds
$7p/8$, or the columns are not contiguous, the operator abandons discovery.
Otherwise, each ordinary dense product adds any positive estimated saving
over a compact product to a running total, using its actual nonzero count.
Once that total covers the estimated setup cost, the next dense product attempts
full discovery. The fixed operation-count weights are defined in
[`activation.rs`](https://github.com/LucaJiang/prusie/blob/main/rust/prusie-core/src/activation.rs);
the decision uses no timings collected during the fit. A fit that finishes
before reaching this threshold avoids full discovery. Sparser products
always use the ordinary kernel and contribute nothing to the total.

Full discovery is attempted at most once per native operator, including when
it returns no plan. Refinement can run several native fits, each with its own
operator and discovery decision. A nonfinite class sum, scaled intermediate
or grouped result causes that product to use the ordinary calculation.
Regrouping coefficients changes floating-point arithmetic, so exact column
relations do not guarantee bitwise equality with an ordinary product.

The full matrix remains allocated. A retained plan adds $O(p)$ storage,
plus $g^2$ values when the compact form is available. Individual-level designs
and RSS models fitted with the Python sweep loop do not use this plan.

## Single-effect regression

Searching for a component's prior variance repeatedly evaluates the same
per-variant evidence. For component $\ell$, ordinary Gaussian single-effect
regression (SER) uses marginal estimates
$\widehat b_{\ell j}=r_{\ell j}/Q_{jj}$ and sampling variances
$s_j^2=\sigma^2/Q_{jj}$ when $Q_{jj}>0$; see the
[SER equations](model.md#one-single-effect-regression).
With `estimate_prior_variance=True` and `estimate_prior_method="optim"`
(the defaults), a scalar search evaluates the SER log marginal likelihood
at multiple candidate prior variances $V_\ell$. Each evaluation takes
$O(p)$ work. The calculations below reuse quantities that remain fixed
during that search.

### Grouped sampling variances

Within a native Gaussian fit, the sampling variances and their groups are
reused until $\sigma^2$ changes. For Bayes factors, finite sampling variances
are floored at float64 machine epsilon. Grouping compares these floored
values by exact equality. With at most 16 distinct values, the log term
$-\tfrac12\log(1+V_\ell/s^2)$ and factor
$V_\ell/\{s^2(V_\ell+s^2)\}$ are computed once per group for each tested
$V_\ell$. Ordinary z-score
RSS supplied with a unit-diagonal `R` has one group of informative variants.
For a toy example with a common sampling variance:

<!-- calculated:ser:start -->
For $s^2=0.1$, $V=0.2$ and marginal estimates $(0.1,0.2,0.3)$, the common log term is -0.549306 and the denominator is 0.03. The log Bayes factors are **(-0.515973, -0.415973, -0.249306)**.
<!-- calculated:ser:end -->

Grouping supplies the posterior log Bayes factors and the objective
evaluations for `"optim"` and `"simple"`. More than 16 distinct finite,
floored variances select the per-variant formula. The search objective also
uses that formula if a marginal estimate is nonfinite, a group denominator
is nonfinite, or a scale factor is nonfinite, subnormal or zero at positive
$V_\ell$. Zero scale is allowed at $V_\ell=0$.

### Search invariants

With grouping available and finite marginal estimates,
$\tfrac12\widehat b_{\ell j}^2$ is computed once per component update and
reused during its search. With equal log prior weights and finite squared
estimates, the nonnegative group factors let per-group maxima determine the
largest log weight needed for log-sum-exp. This avoids a separate scan over
variants for that maximum. In this guarded case, the null objective at
$V_\ell=0$ is the common log prior weight plus $\log p$.

The buffers belong to the current fit, and the estimate-dependent quantities
are refreshed before each component's search. Fixed prior variances and
`estimate_prior_method="EM"` skip preparation of these search invariants.

### Vector exponentials

The search objective and posterior construction evaluate shifted exponentials
$\exp(x_j-\max_k x_k)$, where $x_j$ is a log Bayes factor plus its log prior
weight. Native GNU/Linux x86-64 builds can use glibc `libmvec`: four float64
values per AVX2 library call or two per SSE2 call, subject to symbol
availability and CPU support. Scalar evaluation handles unavailable vector
functions, tail values, nondefault rounding or enabled flush-to-zero and
denormals-are-zero modes. Backend selection is cached; the dispatch checks
the caller's control modes without changing them.

With the AVX2 exponential available, compatible control modes, equal prior
weights and at most four sampling-variance groups, the search also fills
log weights four at a time using separate multiplications and additions.
The compile-time `portable-math` feature excludes the vector-math shim; see
[numerical reproducibility](implementation.md#numerical-reproducibility).

### Summation of likelihood weights

The search objective adds its nonnegative exponentiated weights with a
balanced pairwise tree. Each term passes through at most
$\lceil\log_2 p\rceil$ additions, and the tree overwrites scratch storage
that the next objective evaluation reuses. Posterior weight normalization
and signed objective reductions, including the ER2 dot products, retain
compensated float64 summation.

## Individual-level designs

During IBSS, `susie` uses the working design matrix $\mathbf X$ directly,
avoiding storage of its $p\times p$ crossproduct. For a coefficient vector
$\mathbf v$, it applies

$$
\mathbf Q\mathbf v=\mathbf X^T(\mathbf X\mathbf v).
$$

Column means and scales implement centering (`intercept=True`) and
standardization (`standardize=True`) within these products. Sparse predictors
use compressed sparse column (CSC) storage, and a positive `null_weight`
adds a virtual zero column. A dense product takes $O(np)$ work; a sparse
product takes $O(\mathrm{nnz}+n+p)$, including the vector operations, where
$\mathrm{nnz}$ counts stored design entries. Both use $O(n+p)$ product
workspace in addition to the design and the $O(Lp)$ posterior arrays and
[retained component products](#retained-component-products).

Preparation can allocate dtype or layout copies and temporary arrays for
column statistics. Credible-set construction separately forms requested
correlation blocks, which can be dense even for sparse predictors.

For `susie_trendfilter`, the uncentered design is $(-\mathbf U)^{k+1}$,
where $U_{ij}=1$ for $i\le j$ and zero otherwise, and $k$ is `order`.
Repeated cumulative sums apply the $n\times n$ design and its transpose in
$O((k+1)n)$ work per product with $O(n)$ product workspace.

`susie_rss` also uses a design operator for a single reference genotype
matrix `X`, except with dynamic RSS models or reconstruction from `bhat`,
`shat` and `var_y`, which form dense matrices. Product work depends on the
reference row count; see
[supported input routes](inputs.md#supported-model-and-input-routes).

## Memory

A dense $p\times p$ float64 matrix uses $8p^2$ bytes for its values. Memory
planning also needs to account for copies, temporary workspaces, posterior
arrays and optional diagnostics.

<!-- calculated:memory:start -->
At $p=5000$, $8p^2=200,000,000$ bytes, or approximately **190.7 MiB**.
<!-- calculated:memory:end -->

[Input reuse](#reusing-the-input-array) and
[implicit scaling](#implicit-scaling) avoid extra full-matrix allocations
when their conditions hold. Common sources of additional matrix storage are:

- dtype, alignment or C-order conversion of the input;
- owned [validation and repair](#fused-validation-and-repair), including a
  temporary buffer that can be discarded after proving the input symmetric;
- materialized scaling when the implicit-scaling guard fails, or a temporary
  scaled matrix when a factored product has a nonfinite intermediate or result;
- `maf` filtering or a positive `null_weight` on a materialized-matrix route;
  [design operators](#individual-level-designs) use a virtual null column;
- reconstructed or updated RSS matrices, including the `bhat`/`shat`/`var_y`
  route, finite-reference and mismatch models, multiple reference panels and
  `susie_rss_lambda`; `check_input=True` also allocates eigenvectors on
  matrix-based routes;
- read-only input snapshots for [supported retained analyses](api.md#convergence-and-post-analysis).
  Within that scope, `retain_data=True`, tracking, refinement and
  `model_init`/`s_init` can retain matrix copies.

The core `alpha`, `mu`, `mu2`, `lbf_variable` and retained component-product
arrays use $O(Lp)$ storage, with $O(p)$ vector workspace. An
[exact-column plan](#exact-signed-column-classes) adds $O(p)$ storage plus
$g^2$ values for its compact form, where $g$ is the number of signed classes;
the full matrix remains allocated. In native IBSS fits, `track_fit="dense"`
stores `alpha`, `mu` and `mu2` for each recorded sweep, adding $O(TLp)$
storage over $T$ sweeps.
Other trace contents are described under
[convergence diagnostics](post_analysis.md#convergence-diagnostics).

For a candidate credible set of $m$ members, purity from a supplied matrix
reads pair entries directly. With a `min_abs_corr` threshold and
`median_abs_corr=None`, the first failing pair rejects the set before the
pair-value vector is allocated. Sets that pass collect all $m(m-1)/2$
absolute correlations to report the minimum, mean and median,
using $O(m^2)$ storage. The optional `cs_extension_corr` step also forms
candidate-by-all-variant correlation blocks. Purity from a design matrix
forms the requested blocks as described under
[Individual-level designs](#individual-level-designs).

In the [real-data overwrite series](performance.md#ld-overwrite), enabling
`overwrite_ld` on 200 GWAS cases produced a material fresh-process peak-RSS
reduction in 173 cases and no material saving in 27. The median within-case
False/True peak ratio was 1.72. This is a measured process peak, including
inputs and interpreter, rather than the matrix-size arithmetic above. The
[LD workspace contract](api.md#optional-ld-workspace) describes ownership
requirements and when the caller must reload its array.

## Threads

IBSS components remain sequential: each update uses the latest residual
from earlier updates in the sweep. Parallelism within a dense matrix product
instead assigns disjoint output rows to workers. Each row retains its
source-column accumulation order.

`n_threads` defaults to 1 and accepts integers from 1 to 32. With a larger
value, eligible dense products use a private native pool. Dispatch requires
at least 1024 output rows and $1024^2$ matrix entries; compact products from
[exact signed-column classes](#exact-signed-column-classes) use their reduced
dimensions for this check. Smaller products,
[design operators](#individual-level-designs) and the Python dynamic RSS
sweep are not parallelized by this option.

Eligible `susie_rss` calls can also parallelize the
[symmetry proof and owned repair](#fused-validation-and-repair).
Preparation workers join before later Python transformations or warnings;
`overwrite_ld=True` preparation is serial. The
[native threading contract](implementation.md#native-threading) describes
the storage, build and floating-point requirements. Caller code, including
callbacks and other threads, must not modify input arrays or their aliases
during a fit.

`PRUSIE_NUM_THREADS` does not override the public `n_threads` option.
The option leaves process-wide Rayon and BLAS settings unchanged. NumPy
matrix operations, including those in dynamic RSS, follow their own BLAS
threading settings, so `n_threads=1` does not limit every operation in the
Python process to one thread.

For an eligible dense fit, 2–4 threads are useful starting points; measure
on the intended workload and keep BLAS at one thread when assessing this
option. In the controlled
[thread scaling](performance.md#thread-scaling) experiment on 87 fixed cases,
median paired across-thread/one-thread complete-call ratios were 0.799 at
two threads, 0.735 at four threads and 0.783 at eight threads. Four threads
had the lowest ratio and improved every case; eight threads were slower than
one in 8 cases and slower than four in 52.
When running several fits concurrently, account for each fit's worker threads.

## What prusie does not do

The optimizations described here retain float64 arithmetic. They do not
threshold LD, prune variants or substitute a low-rank approximation.
Variant filtering through `maf` and
[model options](model.md#additional-model-options) are explicit analysis
choices. IBSS components retain their
[sequential update order](#threads).

prusie does not guarantee R's accumulation order or bitwise equality with
susieR or across platforms. Regrouped operations and math-library
implementations can change results. Numerical agreement is assessed on the
complete final `alpha` matrix under matched inputs and fitting settings.
The [computational compatibility](model.md#computational-compatibility) section
defines the comparison, and [numerical accuracy](numerical_accuracy.md)
reports the tested configurations.
