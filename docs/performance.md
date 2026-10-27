# Performance

These measurements describe the source revision and settings identified on this
page. They do not measure later revisions of the solver. See
[computational compatibility](model.md#computational-compatibility) for the
current numerical validation criteria.

These measurements compare prusie 0.2.4.dev2 with susieR 0.16.6 on 100 GWAS and
100 eQTL locus analyses. They measure the runtime of one fitting call and the
peak resident memory (RSS) of a new process for each locus and backend.

<!-- current:highlights:start -->
Measured **prusie 0.3.0** against fresh **susieR 0.16.6** fits.

| Analysis | Cases | Timing pairs | Total-time speed-up¹ | Per-case geometric mean² | Peak RSS ratio³ |
| --- | --- | --- | --- | --- | --- |
| GWAS | 200 | 200 | 27.7× | 35× | 3.66× |
| eQTL | 14157 | 14157 | 19.1× | 22.7× | 3.77× |

¹ Sum of all seven retained R fit times divided by the corresponding prusie total. ² Geometric mean of within-case median-time ratios. ³ Median paired R/prusie fresh-process peak RSS, including interpreter and inputs. The two speed-up summaries use different weighting.
<!-- current:highlights:end -->

**Environment:** AMD EPYC 9754; Linux x86-64, glibc 2.35; CPython 3.12.2,
NumPy 2.2.6; R 4.4.0 with OpenBLAS 0.3.20 (Zen); one numerical thread.
NumPy uses its packaged OpenBLAS; the Rust single-effect updates use the
optional glibc libmvec AVX2 path. The [environment record](assets/benchmark/protocol.json)
identifies the loaded libraries and binary hashes.

## Data

<!-- generated:panel:start -->
The fixed panel contains **200 FinnGen R13 type 2 diabetes GWAS loci** (n = 486,019; 89,727 cases and 396,292 controls) and **14,157 GTEx v11 Whole_Blood genes** (n = 800). GWAS is fitted with the frozen Gaussian RSS approximation. All cases remain accounted for, including iteration-limit outputs, empty credible sets, numerical discrepancies and operational failures. Gene windows and study participants overlap; cases are not independent biological replications.

| Analysis | Cases | Variants per case | Alpha passes / failures / unavailable | Operational failures |
| --- | --- | --- | --- | --- |
| GWAS | 200 | 1575 to 19221 | 200 / 0 / 0 | 0 |
| eQTL | 14157 | 7 to 6862 | 14156 / 1 / 0 | 0 |
<!-- generated:panel:end -->

Each analysis contains 103–4,162 aligned variants. Signed LD comes from
external 1000 Genomes samples: 99 FIN founders for GWAS and 525 EUR founders
for eQTL. Both analyses use the same variant intersection within each window,
with the respective LD matrix. See the [input sources](reproducibility.md#real-gwas-and-eqtl-inputs)
for study pages and harmonization details.

## Methods

Timing surrounds the public `susie_rss` call, including input validation, RSS
preparation, IBSS, posterior summaries and credible-set construction. Imports,
file reads, explicit garbage collection and exports are outside the clock.
One warm-up precedes seven repeats; the median represents each locus/backend.
Backends run sequentially, with their order alternating between cases. The
machine is shared, so the results include repeat variability.

The speed-up for a locus is its median susieR runtime divided by its median
prusie runtime. The main table reports the geometric mean of these ratios.
The memory ratio is susieR peak RSS divided by prusie peak RSS for each locus;
the table reports their median. Neither ratio is a ratio of the group medians.

Memory is measured separately from timing in a new process for each
locus/backend. The process fits once and retains the result; its high-water
mark includes the interpreter, imports and input loading. Current RSS is also
sampled before and during fitting. The sampled increment subtracts the RSS
immediately before fitting from the sampled fit peak. Sampling targets 2 ms;
short allocations can be missed, making sampled peaks and increments lower
bounds. The largest observed gaps are 35.1 ms for prusie and 404.5 ms for R.
The process high-water mark is the primary memory measure. KiB is divided by
1024 to obtain MiB.

All 200 analyses are included, including iteration-limit and empty-CS results.
Settings are L=5, `tol=0.001`, `max_iter=100`, `scaled_prior_variance=0.2`,
fixed residual variance 1, estimated prior variance, uniform variant priors,
no null column, standardization, coverage 0.95 and minimum absolute correlation
0.5. Purity uses every pair of variants in each set. Both implementations
skip optional eigenvalue/projection diagnostics; prusie's `check_input=False`
retains finite-value, symmetry, diagonal and range checks. Process address
space is limited to 16 GiB. The results apply to these inputs, binaries and
hardware; implementation agreement does not establish model validity.

## Results

### Runtime

<!-- current:runtime:start -->

| Analysis | prusie median (ms) | R median (ms) | Faster / tied / slower | prusie repeat IQR/median | R repeat IQR/median |
| --- | --- | --- | --- | --- | --- |
| GWAS | 2.13e+03 | 6.81e+04 | 200 / 0 / 0 | 0.16 | 0.163 |
| eQTL | 328 | 8.32e+03 | 14157 / 0 / 0 | 0.143 | 0.13 |
<!-- current:runtime:end -->

Runtime columns are medians across loci of the seven-repeat medians. The
variability columns summarize each locus's interquartile range divided by its
median. Faster/tied/slower counts compare prusie with susieR at every locus.
Downloads below retain individual repeats and convergence/CS groups.

![Median fitting runtime by variant count for GWAS and eQTL loci.](assets/benchmark/runtime_by_variants.svg)

### Memory

<!-- current:memory:start -->

| Analysis / backend | Fresh-process peak RSS, across-case median (MiB) | Measured cases |
| --- | --- | --- |
| GWAS / prusie | 478 | 200 |
| GWAS / susieR | 1.76e+03 | 200 |
| eQTL / prusie | 229 | 14157 |
| eQTL / susieR | 883 | 14157 |
<!-- current:memory:end -->

Each column is a separate median across loci. The sampled increment is
calculated within each process before summarization.

![Peak resident memory by variant count for GWAS and eQTL loci.](assets/benchmark/memory_by_variants.svg)

### Dense-matrix scaling

A separate deterministic computational fixture uses L=5, n=5000, AR(1)
correlation ρ=0.4, effects [8,7,6] at [p//5,p//2,4p//5], and a
0.15 cos(0.73j) background. It uses `tol=0.001`, `max_iter=100`, estimated
prior variance and fixed residual variance 1.

All six sizes meet the PIP comparison criterion; the largest absolute
difference is 2.62e-11, with matching convergence flags and CS membership.
At p=20,000, prusie takes three sweeps and susieR four. The timings include
this iteration-count difference.

<!-- current:scaling:start -->

| Leading variants | Backend | Median fit time (s) | Observed iterations | Converged | Peak RSS (MiB) |
| --- | --- | --- | --- | --- | --- |
| 500 | prusie | 0.0162 | 14 | True | 49.8 |
| 500 | susieR | 0.997 | 14 | True | 282 |
| 1000 | prusie | 0.057 | 15 | True | 55.4 |
| 1000 | susieR | 3.38 | 15 | True | 319 |
| 2000 | prusie | 0.0783 | 5 | True | 80.6 |
| 2000 | susieR | 2.75 | 5 | True | 413 |
| 4000 | prusie | 0.703 | 11 | True | 180 |
| 4000 | susieR | 22.6 | 11 | True | 1.29e+03 |
| 10000 | prusie | 1.88 | 17 | True | 863 |
| 10000 | susieR | 115 | 17 | True | 3.91e+03 |
| 19221 | prusie | 57.6 | 28 | True | 5.95e+03 |
| 19221 | susieR | 648 | 28 | True | 1.72e+04 |
<!-- current:scaling:end -->

![Runtime, process memory, iteration count and sampled RSS across six matrix sizes.](assets/benchmark/controlled_scaling.svg)

All sizes use dense arrays materialized before fitting from lossless gzip
files. Peak process RSS includes loading costs. Curves connect measured sizes.

## Reproducibility

<!-- current:identity:start -->
- Measured compute commit: `a4bddf00dbb6b19e7e13d4267fc739f928ff0c5a`
- Measured CPython 3.12 wheel SHA-256: `b4627f9116b4fbbbe6338c40d965d21e1e0e63e8ecaf4c148af7b4cd7bac65f3`
- Native extension SHA-256: `a9d377e45583b654ddfbe8323a783e32a61bf042c755cb26c3873f2065ed0b3e`
- Statistical reference: susieR 0.16.6, `8e56a8e038e989856d106d9ca5175cc664fea9d2`.
<!-- current:identity:end -->

Download the [locus summaries](assets/benchmark/real_summary.json),
[locus and repeat CSV](assets/benchmark/real_cases.csv),
[scaling summaries](assets/benchmark/scaling_summary.json) and
[scaling CSV](assets/benchmark/scaling_cases.csv). These records contain
anonymous case labels and measurements; they contain no association inputs
or individual genotypes. Values retain full precision in the downloads.

`python tools/generate_results.py` renders the tables from these records.
The [reproducibility guide](reproducibility.md) describes figure generation
and input-level reruns; [comparison scripts](https://github.com/LucaJiang/prusie/tree/main/tools/benchmark)
implement the timing and memory protocols.

## Earlier measurements

Measurements of prusie 0.2.3rc4 on the same loci, made on 6 October 2026:

<!-- generated:earlier:start -->

| New M1 comparison | Fixed panel | v10 decision | Retained rounds | Applicable / all cases | v10 paired-round runtime W / T / R | Case-median runtime W / T / R | Incremental-memory W / T / R | Role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1_V030_vs_BASE | Primary (87) | DIAGNOSTIC_ONLY | 8 | 87 / 87 | 75 / 0 / 12 | 73 / 0 / 14 | 27 / 0 / 60 | diagnostic |
| C1_V030_vs_BASE | Confirmation (80) | DIAGNOSTIC_ONLY | 8 | 80 / 80 | 71 / 0 / 9 | 71 / 0 / 9 | 36 / 0 / 44 | diagnostic |
| C1_V030_vs_BASE | Large-p (17) | DIAGNOSTIC_ONLY | 8 | 17 / 17 | 13 / 0 / 4 | 13 / 0 / 4 | 6 / 0 / 11 | diagnostic |
| C2_V030_vs_MAIN | Primary (87) | DIAGNOSTIC_ONLY | 8 | 87 / 87 | 87 / 0 / 0 | 87 / 0 / 0 | 28 / 0 / 59 | diagnostic |
| C2_V030_vs_MAIN | Confirmation (80) | DIAGNOSTIC_ONLY | 8 | 80 / 80 | 78 / 0 / 2 | 78 / 0 / 2 | 35 / 1 / 44 | diagnostic |
| C2_V030_vs_MAIN | Large-p (17) | DIAGNOSTIC_ONLY | 8 | 17 / 17 | 17 / 0 / 0 | 17 / 0 / 0 | 7 / 0 / 10 | diagnostic |
| C3_V030_vs_ENG01 | Primary (87) | ACCEPTED | 16 | 87 / 87 | 36 / 0 / 51 | 34 / 0 / 53 | 62 / 0 / 25 | default-veto |
| C3_V030_vs_ENG01 | Confirmation (80) | ACCEPTED | 16 | 80 / 80 | 34 / 0 / 46 | 32 / 0 / 48 | 51 / 0 / 29 | default-veto |
| C3_V030_vs_ENG01 | Large-p (17) | UNRESOLVED_QUALITY | 16 | 17 / 17 | 5 / 0 / 12 | 5 / 0 / 12 | 17 / 0 / 0 | default-veto |
| C4_t2_vs_t1 | Primary (87) | DIAGNOSTIC_ONLY | 8 | 87 / 87 | 87 / 0 / 0 | 87 / 0 / 0 | 56 / 1 / 30 | diagnostic |
| C4_t2_vs_t1 | Confirmation (80) | DIAGNOSTIC_ONLY | 8 | 78 / 80 | 78 / 0 / 0 | 78 / 0 / 0 | 52 / 1 / 25 | diagnostic |
| C4_t2_vs_t1 | Large-p (17) | DIAGNOSTIC_ONLY | 8 | 17 / 17 | 17 / 0 / 0 | 17 / 0 / 0 | 11 / 0 / 6 | diagnostic |
| C4_t4_vs_t1 | Primary (87) | ACCEPTED | 8 | 87 / 87 | 87 / 0 / 0 | 87 / 0 / 0 | 52 / 0 / 35 | thread-option |
| C4_t4_vs_t1 | Confirmation (80) | ACCEPTED | 8 | 78 / 80 | 78 / 0 / 0 | 78 / 0 / 0 | 56 / 0 / 22 | thread-option |
| C4_t4_vs_t1 | Large-p (17) | ACCEPTED | 8 | 17 / 17 | 17 / 0 / 0 | 17 / 0 / 0 | 12 / 0 / 5 | thread-option |
| C4_t8_vs_t1 | Primary (87) | DIAGNOSTIC_ONLY | 8 | 87 / 87 | 79 / 0 / 8 | 79 / 0 / 8 | 58 / 2 / 27 | diagnostic |
| C4_t8_vs_t1 | Confirmation (80) | DIAGNOSTIC_ONLY | 8 | 78 / 80 | 69 / 0 / 9 | 69 / 0 / 9 | 55 / 3 / 20 | diagnostic |
| C4_t8_vs_t1 | Large-p (17) | DIAGNOSTIC_ONLY | 8 | 17 / 17 | 14 / 0 / 3 | 14 / 0 / 3 | 10 / 0 / 7 | diagnostic |
| C5_true_vs_false | Primary (87) | ACCEPTED | 8 | 87 / 87 | 75 / 0 / 12 | 78 / 0 / 9 | 80 / 6 / 1 | memory-only |
| C5_true_vs_false | Confirmation (80) | ACCEPTED | 8 | 80 / 80 | 70 / 0 / 10 | 69 / 0 / 11 | 77 / 1 / 2 | memory-only |
| C5_true_vs_false | Large-p (17) | ACCEPTED | 8 | 17 / 17 | 16 / 0 / 1 | 17 / 0 / 0 | 14 / 2 / 1 | memory-only |

These are new paired Burgundy measurements. Cumulative totals do not replace the fixed-panel majority, stratified veto and robustness gates. The v10 runtime count uses the median of paired-round time ratios; the additional case-median count compares the two within-case median times. These can differ. Memory counts compare within-case medians of incremental peaks. W / T / R means lower, equal and higher candidate measurements. A comparison with 16 retained rounds uses the protocol's single quality retry, pooling both complete windows only for comparisons whose first-window decision was unresolved. Other comparisons retain the original eight rounds. Initial decisions remain in the public M1 records.
<!-- generated:earlier:end -->

These use the same definitions of speed-up and peak process memory. Full
[summary](assets/r_comparison_summary.json),
[case table](assets/r_comparison_cases.tsv),
[repeats](assets/r_comparison_repeats.json) and
[environment](assets/r_comparison_protocol.json) identify the measured binary.
