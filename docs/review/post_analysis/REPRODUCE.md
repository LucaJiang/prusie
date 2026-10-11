# Inspect or reproduce this snapshot

This public guide is adapted from the original implementation run's `reports/REPRODUCE.md`. Server-specific environment paths and resource-control commands are omitted. Historical numerical files were **copied, not refitted**, for this review delivery; see [artifact_manifest.json](artifact_manifest.json).

## Inspect without fitting

Download this branch, then open `docs/review/post_analysis/examples/stable/index.html` or `examples/problem/index.html` in a local browser. These are the original self-contained HTML reports; GitHub's file viewer does not execute them. Their PNGs, CSVs and the two `interpretation.md` files can be read directly on GitHub. No R, network, model account, or Python installation is needed to read them.

For each example:

- `evidence.json` preserves the entire original evidence bundle, including IDs and its canonical data hash; only whitespace was compacted.
- `convergence.json` contains decoded baseline/refined traces, final convergence summaries, variances, logged ELBOs and CSs.
- `refinement.json` retains all candidate/stage records and before/after summaries. `comparison.json` retains all variants and many-to-many relationships.
- `ld_diagnostics.json`, `posterior_summary.json`, `tables/*.csv` and `review_values.json` support the interpretation. `selected_ld.json` contains only explicitly selected entries extracted from the frozen input matrix during review, with input checksum and formula; it is not a new fit.
- `fits/analysis.json` and `fits/arrays.npz` preserve the baseline/refined public result states exactly, including binary64 encodings. Private prepared matrices are intentionally not persisted. The redundant report-level typed `analysis.json`/NPZ was omitted; the full report, evidence and exact fits remain available.

After installing the implementation from this branch, this only reads saved state:

```python
from pathlib import Path
from prusie.report import load_analysis

root = Path('docs/review/post_analysis/examples')
for case in ('stable', 'problem'):
    fits = load_analysis(root / case / 'fits')
    for name, fit in fits.items():
        print(case, name, fit.niter, fit.convergence['stop_reason'])
    print(fits['refined'].refinement_status)
```

## Optional reproduction of the existing examples

The source script is [examples/post_analysis.py](../../../examples/post_analysis.py). With the documented Rust toolchain available for a source install, install `.[analysis]` from this checkout, then run:

```bash
python -m pip install '.[analysis]'
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PRUSIE_NUM_THREADS=1 \
  python examples/post_analysis.py --case both --output-dir output/post-analysis
```

Use a new output directory. This command **does** fit and export examples; it was not rerun for the review-material delivery. Respect local CPU/memory limits. No R or AI call is used by default. The retained CLI adapter's real connectivity has not been verified; the interpretations included here were written in the current Codex session, not through `--interpret`.

The [synthetic fixture metadata](../../../src/prusie/data/teaching/metadata.json) and [official convergence fixture metadata](../../../src/prusie/data/convergence/metadata.json) identify inputs, licensing and reference commits. Frozen numerical evidence represents source captured in `e45914e512e04feef37cf7310bc0b309582d6058`, not the later documentation commit. The package version string alone (`0.2.3rc6`) is insufficient to distinguish this branch.

## Validation already executed

The original full Python suite recorded 811 passes and two missing-external-oracle skips; the final targeted 24-test suite supplied the two oracle paths and passed. Across these suites, 813 distinct cases were exercised successfully. Rust vector/scalar suites recorded 56/52 passes. These historical local outcomes do not imply GitHub CI success. See [verification_summary.json](numerical/verification_summary.json) for outcomes and failures retained from development, and the review README for the observed remote CI failure.

Original LD tests can use the **`results` contents**, not the public wrapper objects, of [ld_oracles.json](numerical/ld_oracles.json) and [kriging_certified_oracle.json](numerical/kriging_certified_oracle.json) as `PRUSIE_LD_ORACLE` and `PRUSIE_LD_CERTIFIED_ORACLE`. No R re-estimation or tolerance change is needed to inspect the frozen reference values. The review delivery checks file integrity, JSON readability, all evidence references, offline-image correspondence, lossless-fit round trips and relative links; it does not rerun the full suite or benchmark.
