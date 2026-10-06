# prusie 0.2.3rc4: public package rename (2026-10-06)

- Rename distribution, Python package, native/build identifiers and environment settings to prusie.
- Preserve numerical implementation, frozen evidence and existing licenses; no compatibility aliases.
- See [migration notes](docs/migration.md), including old pickle module paths.
- Entries below describe the original pre-rename binaries.

# 0.2.3rc4

- Remove the unmeasured B01 known-variance RSS allocation change; retain its correctness tests.
- Keep A02 as a candidate subject to the predeclared complete-call majority gate against rc17.
- Add an offline upstream teaching fixture check and static Pages guides with executed reference evidence.

# Changelog

## 0.2.3rc3

- Reuse fit-local SER search and result storage while retaining sequential float64 updates.
- Reuse owned RSS matrix temporaries for known phenotype variance, preserving operation order and validation.
- Add independent repository packaging, pinned analytical reference fixtures, API/schema documentation and public CI.

This local export has no assigned remote, DOI or package-index release. No publication or default-environment replacement was performed.

The development starting point was experimental pyrsusie0.2.2rc17; the separately accepted0.2.0 installation remains unchanged. Python/Cargo/CITATION versions identify this cleaned source.

The validation protocol uses all100 frozen regions/200 trait cases, L=5, max_iter=100 and tol=.001 against pinned susieR0.16.6. PIP absolute error ≤1e-5 (rtol0), probability validity and unchanged input/model semantics define numerical acceptance. Intermediate/credible-set and downstream colocalisation changes remain diagnostics. One known GWAS case reaches100 iterations without convergence and must remain visible in the accompanying validation report.

Local checks use CPython3.12/Linux x86-64. Public tests and examples are self-contained analytical inputs; private benchmark input data and annotation databases are not redistributable artifacts here. The source includes a future GitHub CI workflow; no remote CI success is claimed. Formal measurements and local package/check results accompany the delivery separately. No universal speed or hardware bottleneck claim is made.

## Earlier semantic changes

The0.2.0 Gaussian migration adopts pinned susieR0.16.6 convergence, final low-V trimming and actual CS coverage. The0.2.2rc17 baseline adds exact signed redundancy representation, guarded matrix operators and optional SIMD, complete public/layout validation and retained-CS purity. See docs/compatibility.md and docs/upstream_differences.md.
