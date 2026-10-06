# Frozen official coloc teaching data

This is **synthetic upstream teaching data**, not human study data or new biological
validation. The entire stored D1, D2, D3 and D4 datasets each contain 500 distinct
variables, labelled s1 through s500, and N=1000. Nothing was simulated, subsetted,
padded, duplicated or selected using Python/R agreement in this export.

The source is `coloc_test_data.rda` from coloc 6.0.3, official commit
`8f20f0bc5e60ffc99e4c2f787bd55fd30cfe7c45`. See `reference.lock.json` for the
archive SHA256, exact data SHA256 and Git blob identity. The source package's
`R/data.R` records the generation recipe under `if(FALSE)`; that code was inspected,
not run. It is retained as `upstream/coloc_data.R`. Its prose about 50/100 variants
is stale. The stored object and its 500-SNP generation calls establish the size.
Its stored causal indices are D1/D2/D4: 105; D3: 105 and 89. Old comments naming
other causal indices do not override the stored object.

## Terms and attribution

Attribution: Chris Wallace and the coloc contributors, as listed in the included
`upstream/coloc_DESCRIPTION`. The source package declares `License: GPL`; no
separate data-specific restriction appears in the pinned source. These converted
teaching files retain that GPL declaration, rather than being relicensed as the
Python package's own data. `LICENSE.GPL-3` supplies a GPL version for this unversioned
upstream declaration. Conversion into portable arrays is the only data modification.
The original source and its generator remain linked in `reference.lock.json`.

## Model and ordering

The four quantitative datasets retain their actual stored beta, varbeta, MAF, N
and sdY. `metadata.json` contains N and sdY; `inputs.npz` contains numeric float64
arrays named `{D1,D2,D3,D4}_{beta,varbeta,MAF,LD}`. Always load NPZ with
`allow_pickle=False`. The input arrays are bit-preserving conversions from the
original R doubles through little-endian `writeBin`, not numbers recovered from
printed output. The optional TSV columns are readable decimal representations.

`variants.tsv` maps each dataset's zero-based array index to its synthetic variable
ID and both LD axes. All matrices are **signed correlation r**, not r-squared.
Both rows and columns use exactly the matching metadata SNP list. Upstream constructs
LD with `cor(X3)` in the same synthetic variable convention as the effects. The
stored `position` is only a synthetic index. Genome build, ancestry, physical
coordinates, effect/counting alleles and other alleles are unavailable and are
explicitly null. Do not interpret these as real biological variant identities or
use the MAF column to infer alleles. LD is from the upstream synthetic reference
sample, not established in-sample LD; residual variance remains fixed.

## Independent reference files

`parameters.json` predeclares every SuSiE call. The primary call is known-n
`susie_rss(z=beta/sqrt(varbeta), R=LD, n=1000)`, L=10, max_iter=100, tol=0.001,
coverage=0.95, min_abs_corr=0.5, full 500-variable purity, sequential IBSS,
standardization enabled, scaled prior variance 0.2 estimated using optim,
fixed residual variance, prior_tol=1e-9, no null column and no refinement.
The D1_partial_zero weights are 0 when the one-based index is divisible by 7,
1 otherwise; they are a software check, not biological priors. In pinned susieR
active SER uses log(prior + sqrt(machine epsilon)), so a zero supplied weight
is **not a hard exclusion**. This differs from coloc's shared-prior zero semantics.
D3_strict_purity only changes min_abs_corr to 1 and actually returns no credible
sets. D3_one_iteration changes max_iter to 1 and actually does not converge; it is
an explicitly labelled diagnostic, not a primary fit or accepted scientific result.

`r_fits.npz` has `{case}_{field}` arrays from official susieR 0.16.6 commit
`8e56a8e038e989856d106d9ca5175cc664fea9d2`. `r_fits.json` holds convergence,
iterations, warnings and credible sets. Members and original component IDs are
provided in both zero- and one-based forms. Coverage is the actual R returned
coverage and was independently checked against the component's alpha sum; the
requested 0.95 is separate. A no-CS result has an empty list and unavailable
coverage, not zero coverage. Arrays derived from R JSON retain about 15 significant
digits; this rounding is far below the declared numerical tolerances.

`r_coloc.json` contains raw official coloc ABF and identical-fit pairing outputs,
including R's no-CS unavailable result and its partial-zero BF error. R ABF summary
and priors named vectors were mapped to explicit JSON objects using the documented
field order. Pairing indices in R tables remain **one-based**; map to original
component IDs before comparing. `independent_oracles.json` separately enumerates
two-variant hypotheses in 70-digit Decimal arithmetic. R's equal-logBF weighted
SNP posterior is [0.5,0.5]; the shared-prior conditional posterior is [0.9,0.1].
No official result is silently corrected, and no Python-package output supplies an
independent expectation.

`input_qc.json`, `reference_environment.json` and `reference.lock.json` preserve
input checks, actual R/dependency versions and source identities. Package-specific
native expectations, if included by a repository, must remain separately labelled.
`checksums.json` hashes every immutable shared-bundle file except itself.

## Regeneration (optional, never needed by the offline check)

Obtain the two **pinned** upstream source archives named in `reference.lock.json`;
verify their SHA256 values before extracting. Use R 4.4.0 with coloc 6.0.3,
susieR 0.16.6 and the exact dependency versions listed in
`reference_environment.json`, installed into an isolated R library. Do not substitute
floating CRAN/GitHub versions. This reference used Netlib BLAS and one numerical
thread. Provide the pinned coloc tree's original `data/coloc_test_data.rda`.

From this directory, with `R_LIBS` and `R_LIBS_USER` pointing first to that isolated
library and an existing local temporary directory:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python verify_reference_input.py /path/to/pinned/coloc_test_data.rda
Rscript --vanilla export_reference.R /path/to/pinned/coloc_test_data.rda raw-reference
python convert_exports.py raw-reference regenerated-portable
```

`verify_reference_input.py` checks the original data hash. The R driver refuses
wrong coloc/susieR versions and exports existing data and all seven fixed calls.
The Python converter needs NumPy. Regeneration writes a new directory; it does not
update or overwrite the shipped expected outputs or checksum manifest. Raw RDS
outputs belong to the regeneration directory only and are never deserialized by
runtime examples. The private validation run additionally retains execution paths,
raw R objects and failed setup logs; those are not public runtime dependencies.
