# Changelog

## 0.2.4.dev1 — post-analysis (unreleased)

- Preserve requested convergence controls across continuation, refinement and
  sensitivity fits; record inner solver settings and explicit overrides.
- Hash matrix identity in bounded buffers without changing signatures or default
  posterior calculations. Distinguish this development build in sensitivity caches.
- Separate CS-supported signal comparisons from complete component evidence;
  distinguish incomplete LD assessment from evaluated, unflagged variants.
- Show before/after fitting and all retained CS in offline reports, with compact
  deterministic summaries and complete external machine evidence.
- Install analysis dependencies explicitly in full CI while testing NumPy-only
  wheel installations separately.
- Align sensitivity priors and metadata by variant ID, including subsets and
  null augmentation; use one global intersection for common-mode comparisons.
- Share local figure/report styling, add compact and variance-detail layouts,
  and provide `summary_text` without implicit fitting or plotting.
- Organize user documentation around result interpretation, convergence,
  sensitivity and two saved worked examples; retain the original version and
  date attribution of performance measurements.
- Use separate linear PIP and LD heatmap palettes, compact cells and
  contrast-aware annotations while retaining full-precision plotted values.
  Improve mobile navigation, figure viewing and the fit-to-report tutorials.

## 0.2.3rc6 — portable toy example validation

- Verify frozen toy example file hashes separately from numerical regeneration,
  allowing only measured BLAS roundoff instead of requiring compressed bytes
  to match across CPU kernels.
- Preserve all frozen inputs, reference outputs and statistical tolerances;
  test corruption detection, metadata changes and alternate BLAS kernels.

## 0.2.3rc5 — documentation and packaged example

- Add an independent, deterministic 500-variant synthetic toy example,
  installed-resource loader and separately executed susieR reference.
- Organize the documentation around the statistical model, numerical accuracy,
  fine-mapping performance, implementation and reproducibility.
- Separate retained GWAS and eQTL evidence and include completed iteration-limit
  and empty-credible-set analyses in the paired timing summaries.
- Generate tables and vector figures from normalized records; add static syntax
  highlighting and package-data installation checks.

These changes preserve the public fitting functions and statistical behavior.

## Gaussian inference implementation

- Implement susieR 0.16.6 convergence and final low-variance trimming semantics.
- Return actual credible-set posterior mass with original component identities.
- Reuse fit-local SER workspaces and component matrix products.
- Use guarded matrix scaling, verified exact signed redundancy, optional vector
  math and complete retained-set purity calculations.

See [model support](docs/compatibility.md),
[implementation](docs/implementation.md) and [performance](docs/performance.md).
