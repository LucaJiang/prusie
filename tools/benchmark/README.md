# Fine-mapping comparisons

These scripts compare installed prusie with susieR **0.16.6**, commit
`8e56a8e038e989856d106d9ca5175cc664fea9d2`. Use R with the pinned package and
jsonlite; retain package source provenance as well as its version.

## Recorded 0.3.0 measurements

The current runtime comparison includes all 200 FinnGen R13 T2D loci,
14,157 GTEx v11 Whole_Blood genes and six nested windows. It runs on exclusive
AMD EPYC 7742 nodes with one numerical thread, one pinned core per L3
domain and NUMA-local memory. Each fresh backend process makes one warmup and
three retained calls. Complete-call and fitting-stage times describe the same
fits; backends run sequentially within each case in a seeded balanced order.
LD is reconstructed untimed with PLINK from the frozen genotype/sample source,
variant order and allele alignment. It is not required to be bitwise identical
to earlier local LD.

`benchmark_records.jsonl.gz`, `benchmark_summary.json` and `benchmark_protocol.json`
under `docs/assets/benchmark/` supply the current runtime and numerical results.
The [protocol guide](../../docs/reproducibility.md#benchmark-protocol) explains
instrumentation, the adoption of 24 completed seven-repeat cases, operational
handling and the direct fitting-stage boundaries. Exact cluster scripts and
patches are [archived](../../docs/assets/benchmark/archive/README.txt).

`v030_records.jsonl.gz` and `v030_summary.json` retain the earlier local
workstation results, which still supply published fresh-process peak RSS.
Their runtime fields are historical. `v030_m1.json` supplies the separate
controlled optimization/thread results on the same AMD EPYC 7742 nodes and
unchanged acceptance decisions. Dense figure
layers are rasterized within SVG/PDF; axes and labels remain vector graphics.

These commands redraw current figures and verify generated text without R,
association inputs or new fits:

```sh
python tools/benchmark/v030.py figures
python tools/generate_results.py --check
python tools/model_examples.py --check
python tools/build_docs.py
```

Use `python tools/generate_results.py` to update generated text. Its `--pages`
option accepts comma-separated repository-relative paths. Figure generation is
separate so that checking text does not redraw all plots. The legacy `import`
and `bind-release` actions of `v030.py` apply only to the original private v030
archive and its memory/software provenance; they do not generate the benchmark
runtime campaign. Current rendering requires both published archives.

For each timing metric, headline speed-up is the sum of all three retained R
times divided by the corresponding prusie total. The ratio of sums of case
medians and the geometric mean of case-median ratios are separate summaries.
All fixed cases enter, including alpha discrepancies, iteration-limit fits
and empty credible sets. Operational retries are recorded separately.

## Independent toy example

From an installed checkout, one command prepares the probe and runs the comparison:

```sh
python tools/benchmark/reproduce.py --rscript Rscript --r-build R \
  --output-dir benchmark-work
```

Use matching R/Rscript executables with the pinned susieR installation.
All generated inputs, build files and records go below the new output directory.
Add `--validation-only` to skip clocks, memory measurements and probe compilation;
add `--panel regression` or `--panel scaling` for the other predefined panels.
Preparation, fitting and record-analysis helpers live in `_internal/`; they are
implementation details rather than additional public entry points.

The default input is the independently generated 500-variant toy example
in `src/prusie/data/toy`. Set `R_LIBS_USER` for your intended R library.
The output directory must be new. `--validation-only` skips clocks and memory
measurement and needs no BLAS probe. `--panel regression` instead uses the
older attributed fixture in `tests/fixtures/susier_regression`, whose existing
inputs/references are unchanged. See its [provenance](../../tests/fixtures/susier_regression/README.md).

The script freezes inputs and order before fitting, runs backends sequentially
with one numerical thread and retains seven timings of the fitting call after one
warmup. Imports, reads, explicit GC and export are outside each clock. Run
without competing builds or numerical tests. Peak memory is a separate fresh
process per case/backend and includes interpreter, imports, inputs and result.
Linux high-water KiB is converted to MiB.

`--sample-rss` additionally records current resident memory after imports, input
loading and GC, then samples through the returned fit before export. The target
interval is 2 ms; actual sample count and largest gap are recorded. Sampled peak
minus this current baseline is an incremental **lower bound**: short allocation
spikes may fall between samples. The process high-water mark is retained as
a separate metric. No historical high-water marks are subtracted. The readiness
handshake is used only in fresh memory processes, never timed calls.

## Controlled dense scaling

```sh
python tools/benchmark/reproduce.py --rscript Rscript --r-build R \
  --panel scaling --sample-rss --output-dir scaling-work
```

The fixed sizes are 500, 1,000, 2,000, 4,000, 10,000 and 20,000 variants. Inputs
use dense AR(1) correlation (ρ = 0.4), three fixed effects plus a small deterministic
oscillation, L = 5, fixed residual variance, estimated prior variance, tol = 0.001
and max_iter = 100. These performance fixtures are not a simulation study.
Actual iterations and convergence remain in each record. Generation streams rows
to disk; a 20,000-variant matrix alone occupies 3.2 GB. Each case records a rough
four-matrix-plus-2-GiB allocation estimate. Check available address space before
running and retain allocation failures; the estimate does not guarantee a fit.

Every completed settings-matched timing pair with valid positive times is
included, independently of numerical differences, convergence and CS presence.
`max_iter` is an iteration budget, not a requirement to execute all iterations.
All failed, missing and iteration-limit statuses remain recorded. Numerical
validity has its own outcome. Runtime speedup is the geometric mean of per-case
median-susieR/median-prusie ratios. Memory ratio is peak-susieR/peak-prusie;
reduction is 1 − peak-prusie/peak-susieR.

## Portable single-case format

For an independently frozen panel of portable cases, use the same command with
`--case-manifest PATH/tasks.json`. The manifest is a JSON list of `case_path`
records, resolved relative to that manifest. Inputs remain read-only; generated
records go under `--output-dir`. Fix the full case panel before looking at results.

The internal `fine_mapping.py` and `fine_mapping.R` drivers accept `--case CASE.json --mode
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

For large, bitwise symmetric controlled-scaling matrices, an optional
`R_gzip_path` can identify a lossless gzip copy of the same byte stream. Verify
its complete uncompressed SHA256 against the frozen `R.bin` before use. Both
drivers materialize the full dense matrix before fitting; no sparse or
approximate LD operation is introduced. Python decompresses in bounded buffers
into a writable contiguous array; R reads into its dense column-major vector.
Record this transport choice because input loading affects process RSS,
although it is outside the fitting clock. Use the ordinary separate row/column
files for matrices that are not bitwise symmetric.

Validation exports alpha, moments, PIPs, natural-log BFs, variances, ELBO, KL and
original CS component/member IDs, returned mass and purity.
The internal comparator checks complete final alpha matrices in their original component
row order, with absolute error ≤1e-5 and rtol=0, plus input/model/probability
validity. No component is filtered or permuted. Rows must sum to one within
1e-10. PIP, CS and intermediate differences remain diagnostics.

## Retained-record analysis

The internal `summary.py` defines case selection, grouping and paired summaries;
its unit tests cover empty CSs, both-max-iteration fits, failures and true error
maxima. The internal `import_records.py` can import the original controlled evidence archive
using `--evidence-root`, `--output` and a run-local `--private-receipt`.
The public `tools/generate_results.py` needs only retained normalized records.
To redraw the current retained runtime, numerical and earlier memory figures
without fitting or R:

```sh
python tools/benchmark/reproduce.py --render-records --output-dir benchmark-figures
```

Use `--assets DIRECTORY` to select a complete set of current public record files.
See [reproducibility](../../docs/reproducibility.md) for the distinction between
record regeneration and input-level reruns.

The independent NumPy oracle in `tests/reference_numpy.py` is included in source distributions for numerical regression tests.
