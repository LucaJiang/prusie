# Compare complete fine-mapping calls

These scripts compare installed prusie with susieR **0.16.6**, commit
`8e56a8e038e989856d106d9ca5175cc664fea9d2`. Use R with the pinned package and
jsonlite; retain package source provenance as well as its version.

## Independent toy example

From the checkout, prepare a Linux high-resolution clock/BLAS probe outside
measurement and run the comparison:

```sh
mkdir -p benchmark-work
cp tools/benchmark/blas_probe.c benchmark-work/
(cd benchmark-work && R CMD SHLIB blas_probe.c -o blas_probe.so -ldl)
python tools/benchmark/compare_r.py --rscript Rscript \
  --blas-probe benchmark-work/blas_probe.so --output-dir benchmark-results
```

The default input is the independently generated 500-variant toy example
in `src/prusie/data/teaching`. Set `R_LIBS_USER` for your intended R library.
The output directory must be new. `--validation-only` skips clocks and memory
measurement and needs no BLAS probe. `--fixture regression` instead uses the
older attributed fixture, whose existing inputs/references are unchanged.

The script freezes inputs and order before fitting, runs backends sequentially
with one numerical thread and retains seven complete-call timings after one
warmup. Imports, reads, explicit GC and export are outside each clock. Run
without competing builds or numerical tests. Peak memory is a separate fresh
process per case/backend and includes interpreter, imports, inputs and result.
Linux high-water KiB is converted to MiB.

Every completed settings-matched timing pair with valid positive times is
included, independently of numerical differences, convergence and CS presence.
`max_iter` is an iteration budget, not a requirement to execute all iterations.
All failed, missing and iteration-limit statuses remain recorded. Numerical
validity has its own outcome. Runtime speedup is the geometric mean of per-case
median-susieR/median-prusie ratios. Memory ratio is peak-susieR/peak-prusie;
reduction is 1 − peak-prusie/peak-susieR.

## Portable single-case format

`fine_mapping.py` and `fine_mapping.R` accept `--case CASE.json --mode
validate|time|memory --output RECORD.json`. A case identifies signed float64 z,
signed LD and ordered variant IDs, with explicit fitting parameters:

```json
{
  "case_id": "my_fine_mapping_case",
  "n_variants": 500,
  "z_path": "z.bin",
  "R_path": "R.bin",
  "R_column_major_path": "R_F.bin",
  "variant_ids_path": "ids.txt",
  "parameters": {"n": 1000, "L": 5, "max_iter": 100, "tol": 0.001}
}
```

The three numerical files are little-endian float64. `R.bin` is row-major;
`R_F.bin` is column-major for exactly the same matrix. IDs are distinct strings,
one per line. Relative paths resolve against the case JSON. No preprocessing,
filtering or allele changes happen in the fitting driver. A complete reproducible
case records all nondefault options, provenance and hashes before execution.

Validation exports alpha, moments, PIPs, natural-log BFs, variances, ELBO, KL and
original CS component/member IDs, returned mass and purity. The R adapter also
retains its original RDS fit. `compare_fits.py` checks PIP absolute error ≤1e-5,
rtol=0, plus input/model/probability validity, and retains all other diagnostics.

## Retained-record analysis

`summary.py` defines stage-C selection, grouping, inclusion and paired summaries;
its unit tests cover empty CSs, both-max-iteration fits, failures and true error
maxima. `import_records.py` can import the original controlled evidence archive
using `--evidence-root`, `--output` and a run-local `--private-receipt`.
The public `tools/generate_results.py` needs only retained normalized records.
See [reproducibility](../../docs/reproducibility.md) for the distinction between
record regeneration and input-level reruns.
