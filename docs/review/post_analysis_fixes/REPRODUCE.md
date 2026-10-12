# Reproduce the review views

These commands start in the repository root with an installed analysis-capable
development build. NumPy remains the only core runtime dependency; SciPy and
Matplotlib are optional analysis dependencies. No model service is needed.

```sh
python -m venv .venv-review
. .venv-review/bin/activate
python -m pip install -c requirements-dev.lock maturin==1.9.6 '.[test,analysis]'
```

The normal source build also needs the pinned Rust toolchain and a C linker.
See [installation](../../install.md). The source/render identity of the supplied
figures is `997aba947d6246c00b52f7083f2d7bdce49a656e`. A later documentation commit
does not change the numerical producer recorded in `provenance.json`.

## Read or render without fitting

Open either supplied `examples/stable/index.html` or `examples/problem/index.html`
locally after downloading its directory. Images are embedded and machine exports
are relative files. GitHub displays the PNGs but does not execute the HTML page.

Copy the saved alternative fits first; then rendering reuses all fits and LD
records. This command does not run refinement, an LD optimizer or new fits:

```sh
mkdir -p review-output
cp -R docs/review/post_analysis_fixes/examples/stable review-output/stable
cp -R docs/review/post_analysis_fixes/examples/problem review-output/problem
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
  python examples/render_review.py --output-dir review-output \
  --render-source-sha 997aba947d6246c00b52f7083f2d7bdce49a656e
```

`examples/render_review.py` reads the baseline/refined lossless objects and LD
records in the unchanged `docs/review/post_analysis/examples/` snapshot. It
reconstructs summaries, the two comparison layers, nine PNG/SVG figures per case,
the compact summary, full evidence and offline report. Expected figures/results
are in this directory. Byte-identical graphics across different Matplotlib/font
environments are not promised. Inspect `figure_index.json`, `plot_data.json`,
`analysis_summary.json` and `sensitivity_validation.json` for numerical agreement.

The initial bounded export used `--refresh-sensitivity` at source
`1f2b246ebd7970929711b2a51bb08f6a76ddad56`: stable L=6, stable fixed prior variance
0.05 with variance estimation disabled, and problem L=12. Baseline requested ELBO
controls were recovered from the saved diagnostics. Stable coverage=0.90 was a
report-only operation. Exactly three alternatives were fitted; their PIPs match
the original saved comparison exactly. Later passes reuse them. To deliberately
recompute these alternatives, add `--refresh-sensitivity` and supply the actual
source SHA you built; do not label new computations as old results.

## Regression and measurement scope

Run focused regressions or the ordinary suite as needed:

```sh
python -m pytest -q tests/test_convergence_identity_review.py \
  tests/test_comparison.py tests/test_ld_diagnostics.py \
  tests/test_plot_review_semantics.py tests/test_report_review.py
python -m pytest -q
PRUSIE_SER_MATH=scalar python -m pytest -q
```

The static documentation checks additionally require Node.js 24.18.0 (as in CI)
and the pinned documentation dependencies:

```sh
python -m pip install -r tools/docs-requirements.txt
python tools/build_docs.py --check
python tools/check_docs.py
```

The two external raw-R oracle tests require the original external oracle paths
documented by their tests; they skip without them. They were additionally run
against preserved local oracle files (2 passed). No SciPy-dependent test is
skipped in the full analysis environment. [Verification](numerical/verification.json)
records actual local checks, observed failures and the successful corrections.
GitHub CI independently builds/tests the final commit; consult its exact SHA/run.

[Default numerical comparison](numerical/default_numerical_validation.json) is a
public copy of the local `reports/default_numerical_validation.json`: 17 frozen
`tests/fixtures/pinned_susieR_gaussian.json` cases plus the installed 500/200-variant
examples, each with tracking off/compact/dense. Saved posterior fields from
source `e45914e512e04feef37cf7310bc0b309582d6058` were compared to the revised
implementation. All 60,576 array elements were byte-identical; scalar settings,
CS and signatures also matched. This does not test every regional dataset or
every explicitly overridden convergence policy.

[Hash measurements](numerical/hash_memory.json) are a public copy of local
`reports/hash_memory.json`. Each mode ran once in a fresh process using a
preallocated F-order float64 matrix of ones (p=500 or 4162), one numerical thread,
`tracemalloc` started after allocation, `time.perf_counter` around the hash, and
Linux `resource.getrusage(RUSAGE_SELF).ru_maxrss` before/after (KiB converted to
bytes). Legacy hashing updated SHA256 with shape plus `matrix.tobytes(order='C')`;
the revised path called `refinement._hash_float64_c_order`. Digests matched.
The matrix is an isolated allocation probe, not a simulated scientific dataset.
The 262,144-byte buffer limit does not bound total process memory. These single
hash timings do not establish a whole-fit speedup.

The public verification summary is curated from named local report/log basenames;
it is not a verbatim complete server log. Complete local records are retained.
The original [numerical evidence](../post_analysis/README.md) remains unchanged,
including actual R differences and certified-reference results. No historical
benchmark or old inf/ash experiment was resumed.
