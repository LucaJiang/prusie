# Reproducibility

The [prusie-analysis repository](https://github.com/LucaJiang/prusie-analysis)
contains the manuscript input-preparation code, benchmark harness, exact archived
scripts, published measurements and figure/table generators.

## Tables and figures

Level A in the analysis repository regenerates the manuscript outputs from
published records alone, without R, association data or refitting:

```sh
git clone https://github.com/LucaJiang/prusie-analysis.git
cd prusie-analysis
python -m pip install -r environment/requirements-level-a.txt
make all
```

This package retains the figures displayed in its documentation and a compact
[documentation summary](assets/benchmark/documentation_summary.json). From the
package checkout, verify the tables and render the site with:

```sh
python tools/generate_results.py --check
python tools/build_docs.py
```

## Benchmark protocol

The measurements compare prusie 0.3.0 with susieR 0.16.6 at commit
`8e56a8e038e989856d106d9ca5175cc664fea9d2`. Complete-call timing accepts loaded
arrays through returned results; imports, input reads, LD construction and
serialization are excluded. Fitting-stage timers were added to both implementations.
The instrumented native binary differs from the release wheel.

The [protocol and exact scripts](https://github.com/LucaJiang/prusie-analysis/tree/main/02_benchmark)
record the timing boundaries, numerical settings, repetition counts and artifact
identities. The analysis repository distinguishes the original nested-window
measurements from the dedicated ten-call R replay at 10,000 variants.

Runtime ratios and process-memory boundaries are described in
[Performance](performance.md). Numerical differences, nonconverged fits and the
component-activity PIP discrepancy are described in
[Numerical accuracy](numerical_accuracy.md).

## Real GWAS and eQTL inputs

Level B in [prusie-analysis](https://github.com/LucaJiang/prusie-analysis)
provides the input preparation, strict SNP QC, allele alignment and PLINK LD
reconstruction for FinnGen R13 T2D and GTEx v11 Whole Blood, with 1000 Genomes
reference genotypes. It lists the exact data releases, access terms and which
steps were re-executed locally. Raw association statistics, genotypes and LD
matrices are not distributed here.

The GWAS analysis sample size is 486,019; its external LD reference has 99 FIN
individuals. These benchmarks measure computation and agreement between software
implementations; they are not biological fine-mapping conclusions.

## Toy R comparison

The package keeps a separate, small comparison tool:

```sh
python tools/benchmark/reproduce.py --rscript Rscript --validation-only --output-dir toy-comparison
```

Use pinned susieR 0.16.6 and jsonlite in the selected R library. See the
[comparison guide](https://github.com/LucaJiang/prusie/blob/main/tools/benchmark/README.md)
for timing and BLAS-probe options. This tool is independent of the manuscript
campaign and requires a new output directory.
