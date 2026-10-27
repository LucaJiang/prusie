# Reproducibility

Reproducing a fine-mapping fit from summary statistics requires the same
aligned association summaries, LD matrix, variant order and fitting settings.
The [real GWAS and eQTL inputs](#real-gwas-and-eqtl-inputs) must be obtained
separately from their sources.

The repository includes [measurement records](#tables-and-figures) for
regenerating the published benchmark tables and figures, and a
[packaged toy example](#toy-example) that runs offline.

## Tables and figures

The published benchmark tables and figures can be regenerated from retained
measurements without R or the association inputs. From the repository root,
with Node.js available for
[documentation rendering](contributing.md#editing-and-building-documentation),
run:

```sh
python -m pip install -r tools/docs-requirements.txt
python tools/model_examples.py
python tools/benchmark/v030.py figures
python tools/generate_results.py
python tools/build_docs.py
```

`tools/model_examples.py` updates the explanatory calculations. The figure
command writes SVG and PDF files to `docs/assets/benchmark/`.
`tools/generate_results.py` updates the generated measurement tables and text.
Add `--check` to `tools/model_examples.py` or
`tools/generate_results.py` to check its generated Markdown without rewriting it.

The current 0.3.0 records are in `docs/assets/benchmark/`:

| File | Contents |
| --- | --- |
| [v030_summary.json](assets/benchmark/v030_summary.json) | Aggregate results, protocol and software identities. |
| [v030_records.jsonl.gz](assets/benchmark/v030_records.jsonl.gz) | Per-case timing repeats, memory measurements, numerical and convergence diagnostics, and host-load summaries. |
| [v030_m1.json](assets/benchmark/v030_m1.json) | Separate paired optimization measurements and acceptance decisions. |

Compact per-case summaries are provided in
[v030_primary.tsv](assets/benchmark/v030_primary.tsv),
[v030_threads.tsv](assets/benchmark/v030_threads.tsv),
[v030_overwrite.tsv](assets/benchmark/v030_overwrite.tsv) and
[v030_windows.tsv](assets/benchmark/v030_windows.tsv).

These public records contain case identifiers and measurement results;
association vectors and LD matrices are not included. The
[performance page](performance.md#reproducibility) gives the measured binary
hashes and the separately qualified release identity.

## Toy example

Use the packaged synthetic toy example to repeat a small fit offline. The
wheel includes inputs, metadata, checksums and a separate susieR reference
under `prusie/data/toy`. The [toy example tutorial](example.md) describes the
data and analysis.

`prusie.load_example()` verifies the SHA-256 hashes of `inputs.npz` and
`metadata.json`, then returns inputs, fitting parameters and generation
metadata. Inspect the reference metadata separately:

```python
import json
from importlib.resources import files

resources = files("prusie").joinpath("data/toy")
reference = json.loads(resources.joinpath("reference.json").read_text())
print(reference["susieR_version"], reference["susieR_commit"])
```

From a checkout with prusie and pytest installed:

```sh
python examples/quickstart.py
python -m pytest tests/test_packaged_example.py
python tools/generate_example.py --check
```

These commands need no R installation.

### Regeneration checks

`tools/generate_example.py --check` regenerates the checkout's arrays in memory
and checks their names, shapes, dtypes, finiteness and values. It uses zero
relative tolerance and the following absolute tolerances:

| Array | Absolute tolerance |
| --- | --- |
| z-scores (`z`) | `8e-14` |
| LD correlations (`R`) | `5e-15` |
| Marginal effects (`bhat`) | `3e-15` |
| Standard errors (`shat`) | `1e-16` |

True effects (`true_beta`) and synthetic coordinates (`position`) must match
exactly. Generation metadata must also match, apart from the running NumPy
version. The frozen metadata records NumPy 2.2.6 and is left unchanged.

To regenerate the inputs in a separate directory, run:

```sh
python tools/generate_example.py --output-dir new-toy-inputs
```

New metadata records the NumPy version used. The check compares array content
independently of compressed archive encoding. Preserve the original inputs
and reference files.

### R comparison

For a fresh comparison, install jsonlite and susieR 0.16.6 at the commit
recorded in `reference.json`. Verify both the package version and source
commit, then choose a new output directory:

```sh
python tools/benchmark/reproduce.py --rscript Rscript \
  --validation-only --output-dir new-reference-comparison
```

`--validation-only` fits both implementations on the toy inputs and writes
results and numerical comparisons below the new directory. It omits the
timing and memory series and requires no BLAS-probe compilation. The comparison
applies the
[complete-alpha validation criterion](model.md#computational-compatibility);
PIP and credible-set differences remain diagnostics. The
[comparison guide](https://github.com/LucaJiang/prusie/blob/main/tools/benchmark/README.md)
describes the output records, timing and memory protocols, and portable input
format.

## Real GWAS and eQTL inputs

The primary real-data comparison uses the following association summaries on
GRCh38. Here $n$ is the study sample size, separate from the LD reference
sample count.

| Association source | Analyses | Study $n$ | LD reference samples |
| --- | --- | --- | --- |
| [FinnGen R13](https://finngen.gitbook.io/documentation/data-download), type 2 diabetes (`T2D`) | 200 loci | 486,019 | 99 Finnish (FIN) |
| [GTEx v11](https://gtexportal.org/home/downloads/adult-gtex/qtl), Whole_Blood | 14,157 genes | 800 | 525 European (EUR) |

For the binary GWAS, $n$ is the total number of cases and controls, and RSS
uses a Gaussian approximation. The GTEx inputs use the released all-pairs
nominal cis-eQTL associations for each selected gene before variant QC.

LD was computed from the
[IGSR high-coverage collection](https://www.internationalgenome.org/data-portal/data-collection/1000genomes_30x/),
release `20220422_3202_phased_SNV_INDEL_SV`, after excluding first- and
second-degree relatives using the release-specific sample exclusion list.
EUR describes the LD reference; the GTEx association cohort was not restricted
to European ancestry. External LD can differ from study LD. Gene windows and
study participants overlap, so cases are not independent biological replications.

Preparation retained nonpalindromic SNPs with minor allele frequency at least
0.05 in both study and reference, reference missingness at most 0.02 and
reference Hardy–Weinberg mid-p value at least `1e-6`, together with sequence,
allele and duplicate checks. Signed effect estimates divided by their standard
errors were aligned to the reference counted ALT allele to form `z`.
[Variant order and alleles](inputs.md#variant-order-and-alleles) describes the
alignment required by the fitting API.

To repeat a fit, preserve the source release, genome build, sample selection,
variant filters, allele orientation, ordered variant IDs, input hashes and
fitting parameters. The
[portable case format](https://github.com/LucaJiang/prusie/blob/main/tools/benchmark/README.md#portable-single-case-format)
records signed summaries, LD and ordered IDs with explicit fitting settings.
Downloaded revisions or newly constructed LD panels define different inputs.

## Software and references

Keep the software identity with each fit. The
[public summary](assets/benchmark/v030_summary.json) records Python, NumPy and
R versions, plus identities of loaded numerical libraries, for the published
measurements. Both builds below report prusie 0.3.0:

| Build | Code commit | Wheel SHA-256 |
| --- | --- | --- |
| Measured | `a4bddf00dbb6b19e7e13d4267fc739f928ff0c5a` | `b4627f9116b4fbbbe6338c40d965d21e1e0e63e8ecaf4c148af7b4cd7bac65f3` |
| Qualified release | `6d60d5e5723461c3e1fc5a46fd0a0e7b6378cfa7` | `85285c26c1f6ae3caaed99c4d7766ede7f0083ea98eb2d26b7763d8654840b59` |

All measured fits used `refine=False`. The two builds have byte-identical
native extensions and ordinary fitting modules. The release wheel was
qualified separately and was not separately timed or measured for
fresh-process peak memory. The [performance page](performance.md#reproducibility)
gives the native extension hash; [numerical accuracy](numerical_accuracy.md)
describes the tested configurations and their limitations.

The numerical reference is susieR 0.16.6 at commit
`8e56a8e038e989856d106d9ca5175cc664fea9d2`; the
[model page](model.md#computational-compatibility) defines the comparison
criterion. The [packaged toy reference](#toy-example) records this R source
and SHA-256 hashes for the raw reference record and reference arrays.

The [regression fixtures](https://github.com/LucaJiang/prusie/blob/main/tests/fixtures/susier_regression/README.md)
retain synthetic inputs from coloc 6.0.3, attributed to Chris Wallace and the
coloc contributors, with frozen susieR outputs, source identities and
checksums. Preserve their upstream GPL declaration and included GPL-3 license
text.

The [convergence and refinement example](worked_examples.md#sources-and-reproducibility)
uses the official susieR `FinemappingConvergence` data under BSD-3-Clause.
Its guide describes the source, attribution and license, with commands to
repeat the fit or render saved posterior results.
