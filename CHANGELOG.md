# Changelog

## 0.2.3rc5 — documentation and packaged example

- Add an independent, deterministic 500-variant synthetic teaching example,
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
