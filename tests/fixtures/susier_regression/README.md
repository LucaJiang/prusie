# Frozen susieR regression cases

These seven cases test fine-mapping against frozen official susieR 0.16.6
outputs, including partial-zero prior weights, no retained credible sets and
one-iteration nonconvergence. They are test fixtures, separate from the
independent packaged example used by `examples/quickstart.py`.

The inputs are **synthetic teaching data from coloc 6.0.3**, commit
`8f20f0bc5e60ffc99e4c2f787bd55fd30cfe7c45`, credited to Chris Wallace and
the coloc contributors in `upstream/coloc_DESCRIPTION`. The upstream package
declares unversioned GPL; `LICENSE.GPL-3` supplies the retained license text.
`reference.lock.json` records the original archive, data and Git blob identities.
No new data were generated for these checks.

All four stored datasets D1–D4 retain their full 500 variables, labelled s1–s500,
with N=1000. `inputs.npz` contains the original float64 beta, varbeta, MAF and
signed correlation matrices; `metadata.json` retains N, sdY and synthetic
positions. `variants.tsv` maps variant order to both LD axes. Alleles, genome
build, ancestry and physical coordinates are unavailable. The LD is an upstream
synthetic reference sample, not established in-sample LD; residual variance is
fixed. Do not interpret these labels or positions as biological variants.

`parameters.json` fixes all seven calls. The four primary calls use
z=beta/√varbeta, L=10, max_iter=100 and tol=0.001. The diagnostic cases use
partial-zero weights, min_abs_corr=1 or max_iter=1. Zero supplied weights in
pinned susieR are not hard exclusions. The one-iteration case does not converge;
the strict-purity case has no credible sets and no available coverage.

`r_fits.npz` and `r_fits.json` retain the original susieR 0.16.6 reference bytes
from commit `8e56a8e038e989856d106d9ca5175cc664fea9d2`. JSON records retain
original component IDs, CS members, returned coverage, purity and warnings.
`input_qc.json`, `reference_environment.json` and `reference.lock.json` are
unchanged historical scientific records. The lock and environment still identify
coloc because the inputs and original shared export came from that package;
their colocalization-specific fields are not used by these fine-mapping checks.
There is no coloc implementation or R runtime dependency here.

`expected_native.*` and `native_checksums.json` retain a historical native
snapshot as diagnostics, including its original package/version metadata. It
does not supply the R expectation. `manifest.json` describes this retained
subset; `checksums.json` covers its reference files. Load NPZ with
`allow_pickle=False`.

From the repository root, with prusie installed:

```sh
python tests/reference_susier.py --output-dir regression-results
python -m pytest tests/test_offline_example.py
```

The checker requires only prusie and NumPy, validates hashes and variant order,
and checks PIP absolute error ≤1e-5 with rtol=0 plus input/model/probability
validity. Intermediate fields and CS differences remain diagnostics. It never
calls R or regenerates references. The default pytest suite also retains
checksum-corruption and numerical-failure checks.
