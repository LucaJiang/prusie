# Runtime and peak process RSS

## New fixed-R measurements

These full-call measurements use pycoloc 0.2.2rc3 and prusie 0.2.3rc4, R 4.4.0 with OpenBLAS 0.3.20, coloc 6.0.3 and susieR 0.16.6. Each backend uses one numerical thread. The fixed real panel has 100 regions, 200 traits and 103–4162 SNPs per region. It consists of previously selected association windows from one disease/cell-type pairing; it is not a random sample of all loci.

| Task | Comparable cases | Median time, R → Python (ms) | Paired speedup¹ | Faster / tied / slower |
| --- | ---: | ---: | ---: | ---: |
| SuSiE-RSS fine-mapping | 199/200 | 86.9 → 2.52 | 35.2× | 199 / 0 / 0 |

¹ Geometric mean of per-case median R / median Python runtime. The two displayed time columns are separate medians across cases, so their ratio need not equal the paired summary.

| Task | Median peak process RSS, R → Python (MiB) | Lower / tied / higher in Python |
| --- | ---: | ---: |
| SuSiE-RSS fine-mapping | 266 → 37.7 | 199 / 0 / 0 |

The primary subset requires both calls to succeed and converge, with an available pairing result. Every excluded state remains in the complete table. ABF includes all successful comparisons. Fitting is excluded on both sides of the BF/CS pairing comparisons.

| Task / subset | Count | Speedup median [Q1, Q3]; range | Relative repeat IQR, R / Python (median) |
| --- | ---: | --- | ---: |
| SuSiE-RSS fine-mapping / primary comparable | 199 | 33.1 [27.6, 43.6]; 16.6–139 | 0.0867 / 0.0685 |
| SuSiE-RSS fine-mapping / all measured | 200 | 33.1 [27.6, 43.9]; 16.6–139 | 0.0863 / 0.0685 |

Relative repeat IQR is (Q3−Q1)/median across the seven retained times. A runtime ratio below one favors R; ties use exact median equality. These are observed counts without a significance or minimum-percentage threshold.

### Public 500-SNP fixture

| Task | Comparable cases | Median time, R → Python (ms) | Paired speedup¹ | Faster / tied / slower |
| --- | ---: | ---: | ---: | ---: |
| SuSiE-RSS fine-mapping | 4/4 | 120 → 4.1 | 32.7× | 4 / 0 / 0 |

| Task | Median peak process RSS, R → Python (MiB) | Lower / tied / higher in Python |
| --- | ---: | ---: |
| SuSiE-RSS fine-mapping | 267 → 39.3 | 4 / 0 / 0 |

These are the full existing synthetic teaching datasets, separately reported from the real panel. D1–D4 are the fine-mapping timing panel; D1/D2 and D3/D4 are the pairing panel. The three additional diagnostic fits are numerically checked but not timed.

### Complete records and counterexamples

The per-case table ([TSV](assets/r_comparison_cases.tsv), [JSON](assets/r_comparison_cases.json)), [all retained repeats](assets/r_comparison_repeats.json), [summary](assets/r_comparison_summary.json) and [parameters/protocol](assets/r_comparison_protocol.json) preserve all workloads and statuses. Per-repeat rows link to separate numerical validation and separate case/backend memory measurements; those quantities are not measured inside every timing repetition. Private genotypes, association files, variant IDs and input paths are not redistributed.

- **SuSiE-RSS fine-mapping:** 0/199 slower primary cases (first IDs: none); 0/199 higher-RSS cases (first IDs: none). Lowest speed ratio: region77/gwas, 16.6×; lowest memory ratio: region08/gwas, 5.1×. Paired memory ratio median 6.89 [Q1 6.65, Q3 7.04], range 5.1–7.52; median per-case RSS reduction 85.5%. Summed case medians: R 49.5 s, Python 1.06 s.

An OS peak includes imports, inputs, the full return and native allocations; it is not incremental algorithm memory. BF/CS consumers load matched complete BF matrices, variant IDs, original CS/member IDs and convergence metadata, without the unused coefficient arrays of an upstream fitting workspace. Fine-mapping retains its full standard result. Runtime and memory conclusions apply to this panel and environment.

All formal calls ran sequentially in one reserved measurement window. The audit recorded 0 pressure-contaminated attempts and 0 process-failed attempts. The predeclared rule retains contaminated attempts and repeats both backends; all original records remain available. Shared-host load monitoring is best effort. The default NumPy coloc path was measured; no fresh Numba compilation or warm-JIT performance claim is made.


## Reproduce with R

The manual scripts in `tools/benchmark/` compare the installed prusie with
**susieR 0.16.6** using full public `susie_rss` calls. They separate numerical
validation, warm timing and fresh-process peak memory. Use the optimized BLAS
installation and fixed R library documented with the results.

From the repository root, after installing R, the pinned susieR and jsonlite:

```sh
mkdir -p benchmark-work
cp tools/benchmark/blas_probe.c benchmark-work/
(cd benchmark-work && R CMD SHLIB blas_probe.c -o blas_probe.so -ldl)
python tools/benchmark/compare_r.py --rscript Rscript \
  --blas-probe benchmark-work/blas_probe.so --output-dir benchmark-results
```

This uses all 500 SNPs in each of the four public teaching datasets. All seven
frozen cases are numerically checked; the three edge-case diagnostics are
separate from the primary timing panel. Inputs and the case order are frozen
before fitting. Every backend uses one warmup, seven retained full calls and
one numerical thread. Backend order alternates across datasets. Imports, reads,
explicit garbage collection and exports are outside the clock; validation,
RSS transformations, IBSS and standard result/CS construction are inside.

Memory uses a separate fresh process for each case/backend and one complete
fit. **Peak process RSS** includes imports, inputs and native allocations; it
is recorded before numerical export. It is a process high-water mark, with
no historical-maximum subtraction. Both backends retain one LD matrix in
their native storage layout and all normal fit outputs. Dense float64 LD alone
requires 8m² bytes for m variants; fitting workspace and results add to this.

Raw per-repeat records, actual backend/library identities, failures and
convergence statuses are retained. Speedup is median R time / median Python
time; memory ratio is peak R / peak Python. A ratio below one favors R.
Report the fixed denominator, geometric mean, median/IQR, range and
faster/tied/slower counts; read memory and runtime as separate outcomes.

## Historical native comparison

A historical comparison of fit-local single-effect-regression storage reuse
against an **earlier native implementation** found **118 faster, 0 tied and
82 slower cases out of 200**. This is not a speed comparison with R, and the
historical binaries were measured separately from the new R comparison above. The evidence identifies the original
binaries below rather than relabeling them as a new measurement.

### Recorded complete-call runtimes

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

### Historical evidence identities

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
