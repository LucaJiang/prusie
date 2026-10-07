# Reproduce the R comparison

These manual scripts compare the installed prusie with official **susieR 0.16.6**
at commit `8e56a8e038e989856d106d9ca5175cc664fea9d2`. Use an R installation with
optimized BLAS and a library containing that pinned susieR and jsonlite. Check the
installed source provenance as well as the version: a version string alone does
not identify a Git revision. The exported environment records identify the
loaded R package, BLAS, thread settings and numerical libraries.

Run from the repository root after installing prusie. The input is the complete,
licensed 500-SNP synthetic teaching fixture in `examples/data/`; no network access
or private data is needed after the software is installed.

## Prepare a clock and BLAS probe

Use `R` and `Rscript` from the same installation. Build this small measurement
helper once, outside measurement. It uses Linux `CLOCK_MONOTONIC` and records the
loaded BLAS thread count and symbol owners; it performs no statistical work.

```sh
mkdir -p benchmark-work
cp tools/benchmark/blas_probe.c benchmark-work/
(cd benchmark-work && R CMD SHLIB blas_probe.c -o blas_probe.so -ldl)
python tools/benchmark/compare_r.py --rscript Rscript \
  --blas-probe benchmark-work/blas_probe.so --output-dir benchmark-results
```

The output directory must be new. Configure `R_LIBS_USER` first when using a
separate library. This is a manual scientific comparison, outside default CI.
Run with no competing heavy computation and enough memory for both input and
full returned results. The script runs backends sequentially and sets one
numerical/native/BLAS thread. Inspect the recorded actual BLAS identity; an
unknown thread probe or Netlib baseline needs separate qualification.

## Fixed cases and measurement boundary

The script freezes its case list, hashes and execution order before any fit.
It times D1–D4, alternating R/Python order across cases. The partial-zero-prior,
strict-purity/no-CS and one-iteration cases are separate numerical diagnostics.
All seven cases retain their existing `parameters.json` arguments. Default
timing is one warmup plus seven retained full public calls, with float64,
ordinary input validation, sequential IBSS and complete standard results.
Imports, input reads, explicit garbage collection and export are outside clocks.
Only one fit object is held between calls. There is no fitted-result cache.

Peak **process RSS** is measured in a separate fresh child for each case/backend,
using one fit and retaining its full return. It includes the interpreter,
imports, input loading and native/BLAS allocations. It is captured before
large numerical export. Linux OS high-water values are converted from KiB to
MiB; they are not an estimate obtained by subtracting a historical baseline.
Both backends load one signed LD matrix in their native storage order. The
column-major R file represents exactly the same matrix as the row-major Python
file; preparation never rounds, filters or changes SNP order.

`cases.json` retains all errors/nonconvergence, seven individual times, case
medians and independent memory peaks. Missing measurements are null, never zero.
`summary.json` gives geometric-mean paired speedup, median, IQR, range and
wins/ties/losses for the predefined mutually converged subset. Speedup is
median R time / median Python time; memory ratio is peak R / peak Python and
reduction is 1 − peak Python / peak R. A ratio below one favors R.

## Numerical records and independent use

Each backend's separate validation call exports PIP, alpha, moments, component
log Bayes factors, variances, ELBO, original CS component/member IDs, actual
coverage, purity, convergence and iteration count. The R process also saves the
original `.fit.rds`, suitable for independent downstream coloc comparisons.
PIP acceptance is absolute error ≤1e-5, rtol=0, plus input/model/probability
validity. Other differences are recorded by original component ID; a matching
PIP does not certify downstream colocalisation posteriors. The one-iteration
diagnostic remains nonconverged.

To perform numerical validation without timing, use `--validation-only`;
`--blas-probe` is then optional. To adapt a legally held frozen case, call
`fine_mapping.py` or `fine_mapping.R` with `--case CASE.json --mode
validate|time|memory --output RESULT.json`. The JSON contains `case_id`,
`n_variants`, `parameters` (including known `n`), `z_path`, `R_path`,
`R_column_major_path` and `variant_ids_path`. Doubles are little-endian float64;
IDs are one per line. Relative paths resolve against the case JSON. Launch
each memory call in its own fresh process. Keep private input files and paths
local; audit any exported records before sharing them.

The Python adapter also exposes `load_case(case)` and `run_case(inputs, stage='C')`
for a separately frozen study manifest. Real-data settings and inclusion rules
must be recorded independently; the teaching fixture is not a surrogate for a
real-data benchmark. See [performance methods](../../docs/performance.md).
