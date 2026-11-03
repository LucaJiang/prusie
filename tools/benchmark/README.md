# Toy R comparison

Run the independent toy comparison against pinned susieR 0.16.6:

```bash
python tools/benchmark/reproduce.py --rscript Rscript --validation-only --output-dir toy-comparison
```

The output directory must be new. Use the R executable and library containing
susieR commit `8e56a8e038e989856d106d9ca5175cc664fea9d2` and jsonlite.
Omit `--validation-only` for local repeated timing; `--r-build` selects the R
executable used to compile the BLAS probe. These local timings are separate
from the manuscript measurements.

Manuscript input preparation, campaign scripts, recorded observations and figure
regeneration live in [prusie-analysis](https://github.com/LucaJiang/prusie-analysis).
The package retains displayed figures and a compact documentation summary.
