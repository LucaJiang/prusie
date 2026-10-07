# Reproducibility

There are two complementary entrances: reproduce the published summaries from
retained records, or rerun fitting from numerical inputs. The independent
synthetic example supports both entirely offline after software installation.
The real-data panel has public result records and separately obtained inputs.

## Regenerate tables and figures

From the source checkout:

```sh
python -m pip install -r tools/results-requirements.txt -r tools/docs-requirements.txt
python tools/generate_results.py
python tools/generate_results.py --check
python tools/build_docs.py
python tools/build_docs.py --check
python tools/check_docs.py
```

The normalized source is `docs/assets/benchmark_records.json`. It contains only
complete fine-mapping stage C, with explicit study groups, case/backend statuses,
retained repeats, numerical diagnostics and measured identities. The generator
recomputes case medians before paired group statistics, and creates README
highlights, document tables, SVG/PDF figures and downloadable records. Fixed
plot seeds make jitter reproducible. The accompanying new teaching-result record
has a separately identified execution and input/reference checksums.

Anonymized real-panel records include source hashes and positional error traces;
they do not contain association vectors or LD matrices. Regenerating these
figures is result-level reproducibility, not a rerun of the private input archive.
The old measured native SHA256 remains identified in the protocol; new local
builds do not acquire the old timing identity.

## Rerun the independent example

The wheel includes `prusie/data/teaching`: input arrays, generation metadata,
checksums, the R reference arrays and reference metadata. `load_example()` uses
`importlib.resources`, so no repository path is required. To inspect the
resources:

```python
from importlib.resources import files
import json

resources = files("prusie").joinpath("data/teaching")
reference = json.loads(resources.joinpath("reference.json").read_text())
print(reference["susieR_version"], reference["susieR_commit"])
```

From a checkout with the package installed:

```sh
python examples/quickstart.py
python -m pytest tests/test_packaged_example.py
python tools/generate_example.py --check
```

The frozen-reference check needs no R. To generate inputs again, use
`python tools/generate_example.py --output-dir NEW_DIRECTORY` with NumPy 2.2.6.
This is one deterministic teaching example, rather than a simulation study.

### Teaching data regeneration

`--check` independently verifies the original SHA256 manifest and the exact
frozen input, metadata and license bytes. It then regenerates the data with
NumPy 2.2.6 and compares array names, shapes, dtypes and finite values. Metadata,
true effects and coordinates must match exactly. The recorded NumPy version
describes the original generation environment and is never replaced during a check.

Single-threaded OpenBLAS 0.3.29 kernels (SkylakeX, Haswell and Sandybridge)
produced maximum absolute differences of 1.60e-14 for z, 1.12e-15 for R,
5.56e-16 for marginal effects and 2.09e-17 for their standard errors.
The respective regeneration bounds are 8e-14, 5e-15, 3e-15 and 1e-16,
with zero relative tolerance. ZIP encoding is not compared for regenerated
arrays. Re-encoding or editing the committed frozen files still fails their
exact integrity checks. These generation checks do not change the independent
R-reference PIP criterion (absolute error at most 1e-5, zero relative tolerance),
credible-set checks or fitting options.

For a live comparison, install susieR 0.16.6 at the recorded commit and jsonlite
in your chosen R library. Check the R installation:

```r
library(susieR)
packageVersion("susieR")
```

Then use:

```sh
python tools/benchmark/compare_r.py --rscript Rscript \
  --validation-only --output-dir new-reference-comparison
```

The [benchmark guide](https://github.com/LucaJiang/prusie/blob/main/tools/benchmark/README.md)
describes the high-resolution clock, seven-repeat timing, fresh-process memory
measurement and portable case format. Generated references go to a new output
directory; the shipped reference is unchanged.

## Real GWAS and eQTL inputs

The [real-data recipes](real_data.md) identify distinct GWAS and eQTL sources,
access conditions and preprocessing requirements. The evaluated panel consists
of one FinnGen R10 study and one BLUEPRINT monocyte eQTL dataset across 100
windows (83 distinct genes). Its manifest was fixed before the original
comparison. It uses external FIN/EUR reference LD and shares participants across
analyses.

An exact input-level rerun needs the original aligned summaries, LD, variant
order and frozen options. A newly downloaded dataset or a newly constructed LD
panel is a separately identified analysis, even when it concerns the same locus.
Use the portable case drivers to record the new hashes and full parameters.

## Software and reference identities

The retained real-panel records identify prusie 0.2.3rc4, its measured native
hash, Python/NumPy versions and loaded numerical libraries, plus susieR 0.16.6
commit `8e56a8e038e989856d106d9ca5175cc664fea9d2`, R and BLAS versions.
The new teaching reference uses the same pinned R source and records its own
raw-record hash. The original attributed regression inputs and R outputs remain
separate in `examples/data` with their existing license and checksums.

See [numerical accuracy](numerical_accuracy.md),
[performance](performance.md) and [contributing](contributing.md).
