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

The records in `docs/assets/benchmark/` are:

| File | Contents |
| --- | --- |
| [benchmark_records.jsonl.gz](assets/benchmark/benchmark_records.jsonl.gz) | All 14,363 cases: three retained complete-call and fitting-stage times per backend, timestamps, peak memory, iteration and convergence outcomes and numerical differences. |
| [benchmark_summary.json](assets/benchmark/benchmark_summary.json) | Panel summaries, variant-count quintiles and repeat variability. |
| [benchmark_protocol.json](assets/benchmark/benchmark_protocol.json) | Measurement protocol and artifact hashes. |
| [cs_membership.tsv.gz](assets/benchmark/cs_membership.tsv.gz) | Per-case credible sets of both implementations, derived from the saved final alpha with the susieR 0.16.6 rule (coverage 0.95, minimum absolute LD correlation 0.5), and whether their members agree. |
| [window10000_retime.json](assets/benchmark/window10000_retime.json) | Dedicated susieR replay for the 10,000-variant nested window. |
| [benchmark_overwrite_records.jsonl.gz](assets/benchmark/benchmark_overwrite_records.jsonl.gz) | Paired single-call peak memory with `overwrite_ld` disabled and enabled for 200 GWAS loci. |
| [thread_scaling.json](assets/benchmark/thread_scaling.json) | Paired runtimes with `n_threads` 2, 4 and 8 against one thread for 87 cases. |
| [release_qualification.json](assets/benchmark/release_qualification.json) | Release numerical checks against fixed susieR references on 184 cases. |

These records contain case identifiers and measurements; association
statistics, LD matrices and individual genotypes are not included.

## Benchmark protocol

The full panel is 200 GWAS loci, 14,157 eQTL genes and six nested windows of one
GWAS locus. Each backend starts a fresh process, loads inputs untimed, makes
one warmup and three retained calls, and records both timing metrics from each
call. Backends run sequentially on one pinned physical core, with seeded,
balanced order. Exclusive nodes use AMD EPYC 7742 processors. One lane per L3
domain leaves the other cores in that domain idle; a separate domain serves
the coordinator. NUMA binding, topology, node identity, host samples and
per-call timestamps are retained. One numerical thread is enforced in both
backends, and the R BLAS probe verifies one thread.
The measured nodes were cpunode-011, cpunode-015 and cpunode-029, with 31 timing
lanes each. All lane probes used NUMA memory binding. Frequency and governor
sysfs readings were unavailable, so no fixed-frequency claim is made.

The signed LD is built on the compute node with PLINK 1.9.0 (27 September
2026), `--r square bin`, `--threads 1`, frozen variants/order and counted
alleles. The reference has no missing genotypes for the selected samples.
The EUR sample list contains one nonfounder; `--make-founders` includes all
525 selected samples in this calculation. Genotype/reference receipts are
verified before staging. The 24-case fidelity sample checks signs and the
maximum absolute difference from the prepared LD; its observed maximum is
$2.15\times10^{-14}$. This establishes rounding-level agreement on the
sample, not bitwise equality or an unmeasured full-panel LD comparison.

Complete-call timing includes validation and preparation. Fitting-stage
measurement starts after input/LD preparation and includes initialization,
IBSS/SER, convergence/ELBO, and final posterior/credible-set construction.
R measures from immediately before `susie_workhorse` to the final public
return. Native timing combines directly measured initialization/solver,
output-export and Python-finalization spans. It includes implicit Gram
scaling during matrix-vector products; R constructs its scaled Gram matrix
in preparation. Minor scalar checks and final-CS submatrix normalization
remain in the fitting stage. Small transitions between native spans are
omitted. No separately measured cost is subtracted.

The [archived harness and instrumentation](assets/benchmark/archive/README.txt)
include exact frozen scripts, patches, hashes and boundary notes. They retain
cluster paths and are an execution archive, not a portable cluster installer.
The original release artifacts remain unchanged. Eight representative cases
were compared against the original artifacts before formal timing; final
alpha/PIP arrays were bitwise unchanged, with matching recorded iterations,
convergence and CS counts. Complete-call timing shifts are reported in
[Methods](performance.md#methods); negligible overhead is not assumed.

Protocol v1 initially retained seven calls. At the repeat-count amendment,
24 completed cases were preserved and contribute their first three retained
times. Their saved final alpha/PIP arrays are from retained repeat seven;
other cases save retained repeat three. Thirty-one in-flight cases were
aborted and restarted under v2 solely because of this amendment. Completed
raw outputs and claims were preserved; some unarchived node-local partial
outputs from those aborted cases could not be recovered. The archive records
that limitation and does not treat those cases as outcome-based reruns.

Only operational failures permit a whole-case retry with identical settings.
The 19,221-variant nested window required one retry after R was killed by a
NUMA-local out-of-memory event. The failed attempt is preserved; the whole
case completed on another node with the same protocol. Node-wide memory
admission did not guarantee enough free memory within each NUMA domain.
Every fixed case remains in the summaries. Peak memory comes from the
high-water marks of these same backend processes.

One retained susieR call for the 10,000-variant nested window took 1,576 s,
while its fitting stage took 70 s. The published susieR runtime and peak-memory
values for that window come from a dedicated replay on the same node and core: two fresh R processes
with one warm-up and three, then seven, retained calls; all ten agreed within
0.6%. [window10000_retime.json](assets/benchmark/window10000_retime.json)
records the replay.

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
timing and memory series and requires no BLAS-probe compilation. It reports the
[complete-alpha comparison](model.md#computational-compatibility); PIP and
credible-set differences are reported separately. The
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

The AMD EPYC 7742 runtime environment uses CPython 3.12.4, NumPy 1.26.4 and R
4.4.0. NumPy loads OpenBLAS 0.3.23.dev; R uses OpenBLAS 0.3.20, with the
loaded library identities and thread probes recorded in the archive.

| Runtime artifact | SHA-256 |
| --- | --- |
| Original frozen 0.3.0 wheel, code `6d60d5e5723461c3e1fc5a46fd0a0e7b6378cfa7` | `85285c26c1f6ae3caaed99c4d7766ede7f0083ea98eb2d26b7763d8654840b59` |
| Timing-instrumented wheel derived from that release | `ea54f79bae02240370811345ac34d751b6c58f4affd54477b31d649f90fd0af8` |
| Timing-instrumented native extension | `95b764e5c99c4c1c851c0c8b5d1aa9a4577201a826ad2db6eb436bae635df577` |
| Native timing patch | `f558de65c0eb3a6d7047f30a2c0f9a84e944616a42e921a51a2d5387df8ec8c2` |
| R timing helper | `7e688f6cb146befe3f37ce2bd95a966c2f811e11180cdadd25515420a9158e6a` |
| PLINK binary | `47ce3069a1349128bdc5193aa8af33ec20dc99ece920843c80453c34eea1a062` |

All formal fits use `refine=False`. The release native extension has SHA-256
`a9d377e45583b654ddfbe8323a783e32a61bf042c755cb26c3873f2065ed0b3e`; the
timing-instrumented binary is distinct. See
[numerical accuracy](numerical_accuracy.md) for the release qualification and
the full-panel comparison.

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
