# Changelog

## 0.3.0 — 2026-10-26

- Add per-call `n_threads` to the main fitting functions, defaulting to 1.
  Counts from 1 to 32 use the integer-index protocol, including NumPy integers;
  booleans and floats are rejected. `fit.params['n_threads']` records the
  normalized count. Try 2–4 threads for larger problems.
  Matrix products keep their ordered arithmetic and IBSS effects remain sequential.
- Reduce matrix-validation passes and fuse owned RSS validation and repair.
  Eligible GNU/Linux x86-64 builds without `portable-math` also parallelize
  preparation; other builds prepare serially and can still parallelize fitting.
  stderr replacement, tracing, profiling and logging warning capture do not
  change preparation eligibility. Warnings are emitted on the calling thread
  after preparation workers join, with their original text and order.
- Add opt-in `overwrite_ld=True` for the restricted ordinary z/R/n route.
  It may consume LD and its aliases as an XtX workspace, including when a later
  error occurs. The default preserves inputs. `fit.params['overwrite_ld']` and
  `fit.qc['ld_overwrite']` record the request and actual action.
- Organize entry points into `rss`, `suff_stat`, `individual` and `rss_lambda`,
  with shared request construction and fitting in `_request` and `workhorse`.
  Remove `prusie.api` without an alias; public `prusie.*` names stay unchanged.
  `SusieInputs` is now defined in `prusie.workhorse`, so older pickles do not load.
  Warning filename/module attribution moves with the code; filters matching
  `module='prusie.api'` must be updated.
- Add the `prusie.viz` package while retaining `prusie.plots`, and export typed
  result schemas with `py.typed`. Analysis imports are lazy: importing `prusie`
  does not load SciPy or Matplotlib.
- Add the `portable-math` build feature for the existing scalar fallback.
- Default `susie_rss_lambda` to `L=10`, validate explicit L before capping it
  at the retained variant count, and exclude an added null column from that cap.
  Explicit `L=None` remains an alias for 10.
- Warn when missing RSS z-scores are replaced by zero, and use those zeros in
  joint multi-panel fits. Reject unsupported dynamic-model convergence controls
  and invalid `refine` values before fitting. Sensitivity fits forward convergence
  controls only when requested, preserving the fitting route's PIP convergence
  and NaN-ELBO handling defaults.
- Input mutation by callbacks during a fit is unsupported and no longer changes
  the completed validation/repair result. Sensitivity refits do not inherit
  destructive LD workspace policy from a supplied baseline.
- Reorganize benchmark helpers under `tools/benchmark/_internal` and update
  documentation, plotting and typing contracts.

## 0.2.4.dev2 — Gaussian model and API expansion (unreleased)

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
