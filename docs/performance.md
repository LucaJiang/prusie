# Performance and its limits

A historical comparison of fit-local single-effect-regression storage reuse
against an **earlier native implementation** found **118 faster, 0 tied and
82 slower cases out of 200**. This is not a speed comparison with R, and the
current checkout has not been timed again. The evidence identifies the original
binaries below rather than relabeling them as a new measurement.

## Measured complete-call runtimes

The fixed panel contains 100 real regions and two traits per region. Each case
used one warmup and seven retained complete calls, one numerical thread and
`L=5`, `max_iter=100`, `tol=0.001`, with fixed residual variance. Two independent
regions could run concurrently; each region's backends ran sequentially. Imports,
input loading and exports were outside clocks; input validation, matrix setup,
IBSS and result/credible-set construction were inside. No cases were removed
after observing outcomes.

| Stratum | Cases | Faster / tied / slower | Baseline summed medians (s) | Storage-reuse summed medians (s) |
| --- | --- | --- | --- | --- |
| All fixed cases | 200 | 118 / 0 / 82 | 1.839772626 | 1.707386992 |
| Both fits converged | 199 | 117 / 0 / 82 | 1.494473075 | 1.461272684 |
| Same iteration count | 200 | 118 / 0 / 82 | 1.839772626 | 1.707386992 |

[All 200 case medians and repeat variability](assets/performance_cases.tsv)
include every regression. All PIP/input/model checks passed at absolute PIP error
≤1e-5, rtol=0; one case reached the 100-iteration cap without convergence.
Matching PIPs does not imply identical intermediate arrays, credible sets,
Bayes factors or downstream posteriors.

A win means a lower per-case median, with no minimum percentage or significance
threshold. The observed majority applies to this complete change versus its
baseline, not to every constituent optimization. Shared-host variability and
private, nonredistributed input data limit independent reproduction; the public
500-SNP fixture is a separate software example. The anonymized table preserves
repeat variability without disclosing study inputs or locations.

## Threads and memory

`PRUSIE_NUM_THREADS` requests 1–32 native threads (default 1). Small matrices stay
serial; setup and validation may use scoped workers. Set NumPy/BLAS/OpenMP thread
counts separately before importing numerical libraries. More threads can cost
additional memory and slow small regions; no universal speedup is claimed.
IBSS components remain sequential and computations use float64.

Each call validates its input and constructs complete retained credible sets.
Fit-local reuse and verified exact signed matrix redundancy preserve all original
SNPs; dense fallback handles matrices without useful redundancy. No fitted-result
cache crosses calls. Gains on redundant LD need not generalize to other regions.
Optional runtime vector math has scalar and missing-library fallbacks.

The extension keeps the Python GIL while borrowing arrays. Use independent
processes for independent regions, budget their total threads/memory, and do not
mutate input arrays during a call. The package does not control whole-host load.
Any separate R benchmark must identify loaded BLAS, R and dependency versions:
a Netlib baseline does not predict performance against optimized BLAS.

## Evidence identities

| Item | Historical identity and scope |
| --- | --- |
| Baseline | `pyrsusie 0.2.2rc17`, preceding native implementation |
| Measured change | `pyrsusie 0.2.3rc4`, A02-only fit-local SER storage; before the public rename |
| Numerical source | [Preserved source hashes](assets/numerical_source.json) |
| Measurements | [Full case table](assets/performance_cases.tsv), one warmup and seven repetitions |
| Incremental evidence | A02 versus the intermediate A01 change was only an eight-case pilot (5 wins); no isolated 200-case A01 ablation |
| Excluded experiment | B01 reused known-phenotype-variance RSS allocations; memory/correctness checks did not establish its runtime gate |

The earlier combined A02+B01 experiment had 135 wins and 65 regressions on the
z-RSS panel, which did not exercise the changed known-variance branch. That
result cannot qualify B01 separately. The current source excludes it.
See [contributing](contributing.md) for the per-optimization majority criterion.
