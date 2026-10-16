# Changelog

## 0.2.4.dev2 — stopping and prior validation (unreleased)

- Continue finite-ELBO fits through repeated alpha/PIP diagnostics until the
  ELBO criterion or iteration limit is reached. Respect minimum iterations
  and distinguish numerical failure from the initial iteration sentinel.
- Check prior variance against the residual variance at the current fitting
  step, including when continuing a saved fit.
- Add a compact text representation of fitted results.
- Removed the optional `prusie.ai` interpretation adapter.
- Clarify installation, result interpretation and the worked examples;
  simplify documentation navigation and the performance report.
- Include license texts and third-party notices in distributions and offline
  documentation.

## 0.2.4.dev1 — post-analysis (unreleased)

- Preserve requested convergence controls across continuation, refinement and
  sensitivity fits; report inner solver settings and explicit overrides.
- Bound temporary memory when hashing input matrices for fit identity checks.
- Separate credible-set signal comparisons from component comparisons, and
  distinguish missing LD assessment from evaluated, unflagged variants.
- Add before/after fitting views and all retained credible sets to offline
  reports, with text summaries and downloadable numerical results.
- Align sensitivity priors and metadata by variant ID, including subsets and
  null columns; compare all fits on one common variant intersection when requested.
- Add compact and variance-detail plot layouts, local styling and `summary_text`.
- Use distinct PIP and LD heatmap palettes with legible cell annotations.
- Improve mobile navigation and figure viewing.

## 0.2.3rc6 — portable toy example validation

- Validate the shipped toy example hashes separately from numerical regeneration,
  allowing small BLAS roundoff differences between CPU kernels while detecting
  altered inputs and metadata.

## 0.2.3rc5 — documentation and packaged example

- Add an offline 500-variant synthetic toy example, installed-data loader and
  separately executed susieR reference.
- Document the statistical model, numerical accuracy, performance and implementation.
- Report GWAS and eQTL timing results separately, including iteration-limit
  fits and fits without credible sets.
- Add static syntax highlighting and figures generated from measurement records.

## Gaussian inference implementation

- Match susieR 0.16.6 convergence and final low-variance trimming semantics.
- Return credible-set posterior mass with original component identities.
- Reuse single-effect regression workspaces and component matrix products within a fit.
- Support matrix scaling checks, exact signed redundancy, optional vector math
  and purity calculations using every pair of variants in a credible set.

See [model support](docs/inputs.md#supported-model-and-input-routes),
[implementation](docs/implementation.md) and [performance](docs/performance.md).
