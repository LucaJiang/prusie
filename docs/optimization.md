# Speed and memory optimization

prusie fits the same model as susieR 0.16.6. The techniques on this page
change how the required quantities are computed: which arrays are stored, how
many passes are made over the data and in which order floating-point
operations are carried out. They do not change the model, priors, stopping
rules or the set of variants. [Implementation](implementation.md) describes the
call structure in which they operate; [Performance](performance.md) reports
measured runtimes and memory.

## Principles

- **Mathematical equivalence.** Every technique computes the same quantity as
  the direct formula in exact arithmetic. Some leave the floating-point result
  bitwise unchanged (for example, removing a repeated pass over the input).
  Others reassociate sums (for example, grouping exactly equal columns), so
  results can differ at rounding level.
- **No approximation.** There is no LD thresholding, pruning, low-rank
  approximation or reduced-precision arithmetic. No variant is merged, removed
  or reordered, and the effects are updated one at a time.
- **Automatic, deterministic dispatch.** A fast path is chosen from the input
  (layout, size, value bounds, exact equality) and the CPU's instruction set.
  The choice never depends on timers, and there is no environment variable or
  option that selects an alternative mathematical path. Each fast path checks
  its preconditions and falls back to the general computation when they fail.
- **Measured acceptance.** A change must keep the complete final component
  alpha matrices within absolute error `1e-5` of pinned susieR 0.16.6
  ([numerical accuracy](numerical_accuracy.md)). It must also be faster on more
  than half of a fixed benchmark panel without slowing any predeclared group
  of cases (data type, variant count or iteration count); see
  [contributing](contributing.md#benchmark-scripts).

## Where the time and memory go

Write $p$ for the number of variants, $L$ for the number of effects, $T$ for
the number of sweeps and $n$ for the number of individuals in
individual-level data. For a dense LD or `XtX` input:

| Stage | Work | Storage |
| --- | --- | --- |
| Validation and preparation, once per call | Reads the $p\times p$ matrix | The input matrix, plus at most one further $p\times p$ array when required |
| One matrix product per effect update | $O(p^2)$ dense; $O(np)$ for a dense design; $O(\mathrm{nnz})$ for a sparse design | Two vectors of length $p$ |
| Single-effect regression per effect update | $O(p)$ per objective evaluation; the prior-variance search evaluates the objective many times | $O(p)$ scratch |
| Objective and residual variance, once per sweep | $O(Lp)$ | $L\times p$ posterior arrays and $L\times p$ stored products |
| Credible-set purity | $O(m^2)$ for a candidate set of $m$ variants | $O(m^2)$ for a retained set |

The products cost about $TLp^2$ in total and dominate large dense fits. In
small fits, the arithmetic of the single-effect regression (SER) dominates.
Both parts are optimized below.

## Fewer passes over the input matrix

### Zero-copy input

A C-contiguous float64 `XtX` or `R` is validated in place. Python passes its
transpose, a view that is column-major in the same memory, so the solver reads
it without copying. Other dtypes or layouts are first converted to a C-ordered
float64 array; this is the only copy of an ordinary input.

### One validation traversal

Validation needs three properties: every entry is finite, the diagonal is
nonnegative and the matrix is exactly symmetric (or must be repaired). One
native pass over 32 × 32 tiles checks all three, keeping both sides of each
transposed comparison in cache, and creates no $p\times p$ temporary. For
z-score RSS input (`susie_rss` with z-scores, or `bhat` and `shat` without
`var_y`, and known $n$), the comparison uses $(n-1)R_{ij}$ and $(n-1)R_{ji}$,
the values the model actually uses.

On x86-64, a vectorized proof first tries to accept contiguous LD matrices
whose diagonal is within $10^{-8}$ of one, whose entries do not exceed
$1+10^{-8}$ in magnitude and which are exactly symmetric. If the proof
succeeds, the complete checks are already established; if it fails, the
complete checks decide. `PRUSIE_NUM_THREADS` (default 1, limited to 1–32) sets
the number of threads for this proof when $p\ge1024$. It is read once per
process and does not change any result.

### Fused validation and repair

As in susieR's dense path, an exactly asymmetric matrix produces a warning and
is replaced by $(\mathbf M+\mathbf M^T)/2$, after any factor $n-1$ is applied.
The repair writes both triangles from one computed value into a single new
$p\times p$ array, tile by tile, and checks finiteness and the diagonal in the
same pass. The result is symmetric by construction, so it is not validated
again.

For z-score RSS input supplied as an aligned, C-contiguous float64 array, with
`z_method` given as a string and standard Python warning handling (display,
filters, recording or logging capture), validation and repair are fused into
one traversal. That traversal computes the averaged
matrix while it checks the input, and discards it if the input is exactly
symmetric. The warning is still emitted at its usual point, before any repair
error. Otherwise validation, the warning and the repair run in that order, so a
matrix that needs repair is read twice. Error messages and their priority are
the same on both paths. prusie assumes that input arrays are not modified while
a call is running; warning callbacks that modify inputs, and replacing
`warnings.warn` before importing prusie, are not supported.

### Implicit scaling

The working matrix is $\mathbf Q=a\,\mathbf D^{-1}\mathbf M\mathbf D^{-1}$,
where $\mathbf M$ is the validated matrix and $\mathbf D$ holds the predictor
scales. The factor is $a=n-1$ when $\mathbf M$ is the LD matrix of z-score RSS
input, and $a=1$ otherwise, including for a repaired matrix, which already
contains the factor.

Rust scans every entry of $\mathbf M$ once more. If $1\le a\le10^9$, every
$d_j$ and $1/d_j$ lies in $[0.5,2]$ and every $\lvert M_{ij}\rvert\le2\times10^9$,
each product is evaluated as $a\,\mathbf D^{-1}\{\mathbf M(\mathbf D^{-1}\mathbf v)\}$
without forming $\mathbf Q$. These bounds keep every scaled entry below
$8\times10^{18}$ in magnitude, and the same comparisons establish that
$\mathbf M$ is finite. For z-score RSS input with unit LD diagonal, $d_j=1$ and
$a=n-1$, so $(n-1)\mathbf R$ is never stored. Each product costs two extra
$O(p)$ scalings instead of a $p\times p$ allocation.

Outside these bounds $\mathbf Q$ is materialized once, entry by entry as
$((M_{ij}a)/d_j)/d_i$, in a new $p\times p$ array that must be finite. If an
implicit product has a nonfinite intermediate or result, that product is
recomputed with a temporary materialized $\mathbf Q$. The two evaluation orders
are equal in exact arithmetic and differ only by rounding. If the scales
themselves are not finite and positive, Python prepares $\mathbf Q$ before
fitting in one allocation, using AVX-512 when the CPU supports it.

## Matrix products in the sweep

### Retained component products

Write $\mathbf m_\ell=\boldsymbol\alpha_\ell\odot\boldsymbol\mu_\ell$,
$\mathbf u_\ell=\mathbf Q\mathbf m_\ell$ and
$\mathbf f=\sum_\ell\mathbf u_\ell$. An effect update needs the residual
crossproduct that excludes its own contribution, and then replaces that
contribution:

$$
\mathbf r_\ell=\mathbf g-(\mathbf f-\mathbf u_\ell),\qquad
\mathbf f\leftarrow\mathbf f-\mathbf u_\ell+\mathbf Q\mathbf m_\ell^{\mathrm{new}}.
$$

Keeping every $\mathbf u_\ell$ costs $O(Lp)$ storage. In return each update
needs one product with $\mathbf Q$ instead of two, because the old contribution
is subtracted rather than recomputed. For example,
$\mathbf Q=((2,1),(1,2))$, $\mathbf g=(1,0.8)$,
$\mathbf m_1=(0.1,0)$ and $\mathbf m_2=(0,0.2)$ give:

<!-- calculated:cache:start -->
The products are $\mathbf u_1=(0.2, 0.1)^T$ and $\mathbf u_2=(0.2, 0.4)^T$, giving $\mathbf f=(0.4, 0.5)^T$. Component 1 uses residual **(0.8, 0.4)**. Replacing its mean by $\mathbf m_1^{\mathrm{new}}=(0.2, 0.1)^T$ changes the fitted crossproduct to **(0.7, 0.8)**.
<!-- calculated:cache:end -->

The stored products also supply the objective. Posterior uncertainty enters
it through the expected residual sum of squares,

$$
\mathrm{ER2}=y^Ty-2\mathbf g^T\mathbf m+\mathbf m^T\mathbf Q\mathbf m
+\sum_\ell\left\{\sum_j Q_{jj}\alpha_{\ell j}\mu^{(2)}_{\ell j}
-\mathbf m_\ell^T\mathbf Q\mathbf m_\ell\right\},
$$

with $\mathbf m=\sum_\ell\mathbf m_\ell$. Because
$\mathbf Q\mathbf m=\sum_\ell\mathbf u_\ell$ and
$\mathbf m_\ell^T\mathbf Q\mathbf m_\ell=\mathbf m_\ell^T\mathbf u_\ell$, every
quadratic form is an $O(p)$ dot product, and the objective requires no further
matrix product. Compensated float64 summation reduces cancellation in these
reductions.

### Dense matrix–vector kernel

The column-major product $\mathbf Q\mathbf v$ adds the columns $j$ with
$v_j\ne0$ in ascending order. Zero coefficients are skipped, so an effect whose
prior variance is zero, with $\mathbf m_\ell=\mathbf 0$, costs no matrix work.
Four nonzero columns are accumulated into each output element while it is held
in a register, so the output vector is loaded and stored once per four columns
instead of once per column. Each element still receives its additions in
ascending column order, with separate float64 multiplies and adds, so this
reduces memory traffic without changing the arithmetic of a column-by-column
product.

On x86-64 the same kernel is also compiled for AVX2 and chosen at run time
when the CPU supports it; other CPUs use the portable build. Component updates
are sequential, and each product uses one thread.

### Exact signed-column classes

An LD matrix can contain columns that are exactly equal, or exact negatives of
one another, for example for variants with identical genotypes in the LD
sample. If the $p$ columns fall into $g$ such classes, a dense product can add
the signed coefficients of each class first and multiply only $g$ columns.

- **Product.** The signed coefficients of each class are added with
  compensated summation, and $\mathbf M$ is multiplied by the $g$ class sums
  placed at representative columns, costing $O(pg)$ instead of $O(p^2)$. If
  every row is also proved to satisfy the same relations, a $g\times g$ matrix
  of representative entries is stored and the product costs $O(g^2)$, plus
  $O(p)$ for grouping and expansion.
- **Discovery.** Hashes of 16 sampled entries per column, and of complete
  columns within crowded buckets, only propose candidates. A relation
  $\mathbf M_{\cdot k}=\pm\mathbf M_{\cdot j}$ is accepted only when all $p$
  entries compare equal in float64, with no tolerance or rounding. If the
  proved number of classes $g$ exceeds $7p/8$, no plan is built and the
  ordinary kernel is used for the rest of the call.
- **Fallback.** A nonfinite class sum, intermediate or result sends that
  product to the ordinary kernel.

Discovery reads the whole matrix and costs order $p^2$ comparisons, so it pays
off only when enough products follow. Since the number of remaining sweeps is
unknown, prusie decides with a deterministic rent-or-buy rule. A product is
*dense* when its coefficient vector has at least $\lceil p/8\rceil$ nonzero
entries; sparser products always use the ordinary kernel and are not counted.

1. At the first dense product, a screen hashes the 16 sampled entries of every
   column. The number of distinct hashes $\hat g$ is a lower bound on $g$. If
   $\hat g>7p/8$, or the matrix is not stored column-contiguously, discovery is
   abandoned for the rest of the call.
2. Each ordinary dense product with $k$ nonzero coefficients, including the
   first, adds the work a plan would have saved,
   $\max\{0,(c_1pk+c_2p)-(c_3\hat g^2+c_4p)\}$, to a running total $R$.
3. Before a dense product, discovery runs once if
   $R\ge c_5p^2+c_6\hat g^2$, the estimated cost of discovery and of building
   the compact matrix.

The weights $c_1,\dots,c_6$ are fixed operation-count constants, calibrated on
real LD matrices separate from those used to evaluate the rule; they are not
timings measured during a call. Their current values in
[`activation.rs`](https://github.com/LucaJiang/prusie/blob/main/rust/prusie-core/src/activation.rs)
are approximately $c_1=1.453$, $c_2=7161$, $c_3=2.226$, $c_4=419.8$,
$c_5=5.058$ and $c_6=18.15$. A fit that stops early never pays for discovery;
a long fit buys the plan once the work it would already have saved covers its
cost. Discovery runs at most once per native operator, and a failed screen or
proof is never retried.

The plan is built for the matrix actually multiplied: $\mathbf M$ under
implicit scaling, otherwise the materialized $\mathbf Q$. Posterior
coordinates and scaling are unchanged; no variant is merged or removed.
Grouping changes only the order of floating-point additions, so exact matrix
relations do not imply bitwise equality with a dense product. Refinement can
perform several native fits, each with its own operator and discovery
decision. The plan saves work, not memory: it adds $O(p)$ storage, and $g^2$
values for the compact form, while the full matrix is retained.
Individual-level designs and the RSS models with a Python sweep loop do not
use it.

## Single-effect regression

For marginal estimate $\widehat b_j$, sampling variance $s_j^2=\sigma^2/Q_{jj}$
and prior variance $V$, the Gaussian log Bayes factor is

$$
\log\mathrm{BF}_j=-\tfrac12\log(1+V/s_j^2)
+\tfrac12\widehat b_j^2\frac{V}{s_j^2(V+s_j^2)}.
$$

With `estimate_prior_method="optim"` (the default), a one-dimensional search
evaluates the SER log marginal likelihood
$\log\sum_j\pi_j\mathrm{BF}_j$ for many values of $V$ within each effect
update. Each evaluation is $O(p)$, with one logarithm, one division and one
exponential per variant in the direct form.

### Grouped sampling variances

The sampling variances are recomputed only when $\sigma^2$ changes. When they
take at most 16 distinct values, the log term and the factor
$V/\{s^2(V+s^2)\}$ are computed once per distinct value. z-score RSS input
with unit LD diagonal has a single value.

<!-- calculated:ser:start -->
For $s^2=0.1$, $V=0.2$ and marginal estimates $(0.1,0.2,0.3)$, the common log term is -0.549306 and the denominator is 0.03. The log Bayes factors are **(-0.515973, -0.415973, -0.249306)**.
<!-- calculated:ser:end -->

Each variant then needs one multiply and one add. Grouping applies to the
posterior and, with `"optim"` or `"simple"`, to every evaluation during the
prior-variance search. More than 16 distinct sampling variances, a nonfinite
estimate or a nonfinite or subnormal group term fall back to the per-variant
formula.

### Search invariants

Within one effect update only $V$ changes during the search, so
$\tfrac12\widehat b_j^2$ is computed once and reused by every evaluation. With
equal prior weights, the largest log weight, which the log-sum-exp needs, is
taken from per-group maxima of $\tfrac12\widehat b_j^2$ instead of a separate
pass over the variants, because each group's factor is nonnegative. At $V=0$
every Bayes factor is one, so the value is the common log prior weight plus
$\log p$.
These quantities are rebuilt within each fit; nothing is carried between fits.

### Vector exponentials

On native GNU/Linux x86-64 builds, the shifted exponentials
$\exp(x_j-\max_k x_k)$ use the system vector math library (glibc `libmvec`,
four values per AVX2 instruction or two per SSE2 instruction) when it is
available. Scalar evaluation handles unavailable symbols, the remaining tail
values and nondefault caller floating-point modes (rounding direction,
flush-to-zero or denormals-are-zero). Initialization is cached, and the
caller's floating-point environment is preserved. With the AVX2 exponential,
equal prior weights and at most four sampling-variance groups, the per-variant
log weights of the search are also filled four at a time, with separate
multiplies and adds. The compile-time `portable-math` feature removes the
vector library path; see
[numerical reproducibility](implementation.md#numerical-reproducibility).

### Summation of likelihood weights

The search objective adds $p$ nonnegative exponentiated weights. These are
added by a balanced pairwise tree, whose rounding error grows with
$\log_2 p$ rather than $p$, and which is cheaper than compensated summation.
Because every term is nonnegative there is no cancellation. Signed sums, such
as the KL terms and ER2, and the normalization of the posterior weights keep
compensated summation.

## Individual-level designs

`susie` never forms the $p\times p$ matrix $\mathbf X^T\mathbf X$. Each product
is evaluated as $\mathbf X^T(\mathbf X\mathbf v)$, costing $O(np)$ for a dense
design and $O(\mathrm{nnz})$ for a sparse one. Centering and scaling are applied
as column means and scales inside the product, so the centered design is not
stored and sparse input stays sparse. The optional null column is represented
virtually. The trend-filtering design
$\mathbf X=(-\text{upper-triangular ones})^{\text{order}+1}$ is applied with
cumulative sums in $O(n)$ storage, without constructing $\mathbf X$. Ordinary
single-panel RSS fits with a reference genotype matrix `X` use the same
matrix-free representation.

## Memory

<!-- calculated:memory:start -->
At $p=5000$, $8p^2=200,000,000$ bytes, or approximately **190.7 MiB**.
<!-- calculated:memory:end -->

A dense float64 matrix dominates memory, so prusie avoids further $p\times p$
arrays where it can. An ordinary fit with a C-contiguous float64 matrix,
symmetric input and implicit scaling holds only the input matrix at this size;
everything else is $O(Lp)$. Additional $p\times p$ arrays are created only:

- when the input is not C-contiguous float64 (one converted copy);
- by asymmetry repair (one repaired array), and on the fused path also as a
  discarded buffer when a symmetric input is not accepted by the fast proof;
- by materialized scaling, including the rare temporary for an implicit
  product that overflows;
- by `maf` filtering or a null column (`null_weight > 0`), which build a
  modified matrix;
- by RSS routes that construct their own matrix: the `bhat`/`shat`/`var_y`
  route, finite-reference and mismatch corrections, multiple reference panels
  and `susie_rss_lambda`;
- by retained analysis (`retain_data`, `track_fit`, `refine` or `model_init`
  for supported models), which keeps owned read-only copies of the inputs.

Credible-set purity reads only the correlation entries among each candidate
set's members. While checking `min_abs_corr`, it stops at the first pair below
the threshold, because that pair already proves rejection; retained sets visit
every pair to report the minimum, mean and median.

## Threads

The solver uses one thread. Effects are updated sequentially because each
update uses the residual left by the previous one (a Gauss–Seidel scheme);
updating them in parallel would be a different algorithm. Parallelism comes
from vector instructions within a thread. The only native threads are the
optional `PRUSIE_NUM_THREADS` workers of the symmetry proof described above.
NumPy operations in Python follow NumPy's own BLAS threading settings.

## What prusie does not do

- It does not reproduce R's accumulation order or use extended precision to
  imitate R.
- It does not approximate LD, prune variants, or switch to float32.
- It does not update effects in parallel or reorder them.
- It does not offer switches that select alternative mathematical paths.

The reassociations above, together with platform vector math libraries, mean
that results can differ from susieR, and between platforms, at rounding level.
These differences are assessed with the complete-alpha tolerance, not bitwise
equality; see [numerical accuracy](numerical_accuracy.md).
