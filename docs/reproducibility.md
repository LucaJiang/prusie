# Reproducibility

The repository includes measurement records for regenerating the published
tables and figures, and toy inputs for running the examples offline.
Real association inputs are obtained separately from their source studies.

## Tables and figures

From a checkout with Node.js and the [documentation tools](contributing.md#editing-and-building-documentation):

```sh
python -m pip install -r tools/docs-requirements.txt
python tools/model_examples.py
python tools/generate_results.py
python tools/build_docs.py
```

`docs/assets/benchmark/` contains the prusie 0.2.4.dev2 measurements.
`docs/assets/benchmark_records.json` contains the earlier 0.2.3rc4 records.
The generator reads these records to produce displayed tables and the earlier
comparison figures. It leaves the dev2 source records unchanged. To redraw the
dev2 figures:

```sh
python tools/benchmark/reproduce.py --render-records \
  --output-dir output/benchmark-figures
```

Records contain case/backend statuses, repeats, numerical comparisons and
software identities. Anonymous locus labels and aggregate results are public;
association vectors and LD matrices are not included. The
[performance page](performance.md#reproducibility) links the measured binary hashes.

## Toy example

The wheel includes `prusie/data/toy`: arrays, metadata, checksums and the R
reference. `load_example()` reads these installed resources:

```python
from importlib.resources import files
import json

resources = files("prusie").joinpath("data/toy")
reference = json.loads(resources.joinpath("reference.json").read_text())
print(reference["susieR_version"], reference["susieR_commit"])
```

From a checkout with prusie installed:

```sh
python examples/quickstart.py
python -m pytest tests/test_packaged_example.py
python tools/generate_example.py --check
```

The check needs no R installation. To regenerate the inputs in a new directory,
use `python tools/generate_example.py --output-dir NEW_DIRECTORY` with NumPy 2.2.6.

### Regeneration checks

`--check` compares regenerated arrays by name, shape, dtype and value. True effects and coordinates must match exactly. Metadata retains the
original NumPy version.

With one thread, OpenBLAS 0.3.29 kernels SkylakeX, Haswell and Sandybridge produce
maximum differences of 1.60e-14 for z, 1.12e-15 for R, 5.56e-16 for marginal
effects and 2.09e-17 for their standard errors. Regeneration tolerances are
8e-14, 5e-15, 3e-15 and 1e-16, respectively, with zero relative tolerance.
Compressed archive bytes may differ after regeneration; the original files
and provenance remain frozen. Current comparisons use complete final alpha
matrices in original component and variant order, with absolute tolerance 1e-5
and zero relative tolerance. PIP differences are retained as diagnostics.

### R comparison

Install susieR 0.16.6 at the recorded commit and jsonlite in your R library.
Verify the package version, then run:

```sh
python tools/benchmark/reproduce.py --rscript Rscript \
  --validation-only --output-dir new-reference-comparison
```

The [comparison guide](https://github.com/LucaJiang/prusie/blob/main/tools/benchmark/README.md)
describes the timing protocol, process memory measurements and portable input
format. Write new references to a new output directory.

## Real GWAS and eQTL inputs

The measurements use one FinnGen R10
study and one BLUEPRINT monocyte eQTL dataset across 100 windows, representing
83 genes. They use external FIN/EUR reference LD and share study participants
across loci.

Reproducing a fit requires the same aligned summaries, LD, variant order and
settings. Downloaded revisions or newly constructed LD panels define new
inputs; record their hashes and parameters with the portable case drivers.

## Software and references

Measurement records identify prusie 0.2.4.dev2 or 0.2.3rc4, the native extension,
Python/NumPy versions and loaded numerical libraries. The statistical reference
is susieR 0.16.6 at commit `8e56a8e038e989856d106d9ca5175cc664fea9d2`.
The toy reference records the same R source and its own output hash.
The attributed regression inputs in `tests/fixtures/susier_regression` retain their source,
reference outputs and licenses. [Toy example](worked_examples.md#sources-and-reproducibility)
provide saved posterior states and report rendering commands.
