# Changelog

- Unreleased: replace `prusie.api` with `suff_stat`, `rss`, `workhorse`,
  `_request` and `_validation`; `SusieInputs` and `susie_workhorse` are now
  defined in `prusie.workhorse`. Public `prusie.*` names are unchanged.
  `SusieInputs` objects pickled by earlier versions do not load.

## 0.2.4.dev2 — Gaussian model and API expansion (unreleased)

- Warn when `susie_rss` replaces missing z-scores with zero. The multi-panel
  joint fit now also uses zero for them; previously the missing values entered
  its panel-weight updates.
- Reject `ConvergenceConfig` settings that the `R_finite`, `R_mismatch` and
  multi-panel sweep loop does not implement, instead of silently applying the
  ELBO rule.
- Check the type of `refine` before fitting.
- `run_sensitivity` passes convergence controls to the fitting function only
  when they were requested. Previously it always passed the default
  `ConvergenceConfig`, so `R_finite` and `R_mismatch` fits used the ELBO rule
  instead of susieR's PIP rule, and other fits lost susieR's NaN-ELBO handling.
- Separate the pure Rust solver from the PyO3 binding in a Cargo workspace.
- Use mathematically equivalent float64 calculations without R-rounding emulation.
- Use variance grouping, vector exponentials and implicit matrix scaling
  without mathematical-path switches.
- Add individual-level dense/sparse fitting, reference-factor RSS, regularized
  RSS, automatic effect-count fitting, implicit trend filtering and standalone SER.
- Add fixed mixture priors, NIG and MLE residual updates, residual bounds, slot
  activity priors, greedy effects and posterior convergence controls.
- Add credible-set purity/extension options, posterior accessors, prediction,
  model summaries and posterior outcome configurations.
- Validate complete final component alpha matrices against susieR 0.16.6;
  retain other model quantities as numerical diagnostics.
- Keep post-analysis, convergence diagnostics, plotting and report interfaces.
- Support `refine=True` for matrix-free, dynamic RSS, mixture, NIG, slot, greedy,
  PIP-stopping and nondefault credible-set models by susieR's refitting search.
- Keep the PIP history of earlier sweeps when checking PIP-based convergence, as
  susieR does; previously all history was recomputed with the current prior
  variances, which could stop one sweep early.
- Check sweeps whose ELBO is NaN with susieR's PIP rule (tolerance `tol`, cycle
  averaging) unless a `ConvergenceConfig` is supplied, and skip the first sweep
  and seed the PIP history as susieR does in dynamic RSS fits.

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
