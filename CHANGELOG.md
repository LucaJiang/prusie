# Changelog

## 0.2.3rc4

- Reuse fit-local single-effect regression storage while retaining sequential
  float64 component updates.
- Exclude the known-phenotype-variance allocation experiment, which lacked runtime
  evidence for its applicable input path.
- Add the offline 500-SNP synthetic teaching example, executed
  [R agreement report](docs/r_agreement.md) and static documentation.

## Earlier changes

- Adopt Gaussian susieR 0.16.6 convergence and final low-variance trimming semantics.
- Return actual credible-set posterior mass using original component IDs.
- Add guarded matrix operators, exact signed redundancy handling, optional vector
  math and complete retained-set purity checks.

The [compatibility guide](docs/compatibility.md) defines the supported model.
[Performance evidence](docs/performance.md) identifies the historical binaries
actually timed; it does not represent a new benchmark of this checkout.
