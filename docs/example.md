# Analyze the 500-SNP example

Fit and interpret the complete **500-SNP synthetic teaching datasets** D1–D4
bundled from official coloc. These fixed examples preserve the stored variable
order and include single- and multiple-signal cases.

## Fit a region

After [installation](install.md), run this from the repository root. All 500
variables are fitted in their stored order.

```python
import json
from pathlib import Path
import numpy as np
import prusie

data = Path("examples/data")
metadata = json.loads((data / "metadata.json").read_text())["D3"]
with np.load(data / "inputs.npz", allow_pickle=False) as arrays:
    z = arrays["D3_beta"] / np.sqrt(arrays["D3_varbeta"])
    R = arrays["D3_LD"]

fit = prusie.susie_rss(
    z=z, R=R, n=metadata["N"], variant_ids=metadata["snp"],
    L=10, max_iter=100, tol=0.001, estimate_residual_variance=False,
    coverage=0.95, min_abs_corr=0.5, n_purity=500,
)
lead = int(np.argmax(fit.pip))
print(f"{fit.variant_ids[lead]}: PIP={fit.pip[lead]:.3f}")
print("Converged:", fit.converged, "Iterations:", fit.niter)
for k, component in enumerate(fit.sets["cs_index"]):
    members = fit.variant_ids[fit.sets["cs"][k]].tolist()
    print("Component:", int(component), "SNPs:", members,
          "Posterior mass:", round(float(fit.sets["coverage"][k]), 3))
```

The D3 fit returns two credible sets and converges in eight iterations in the
recorded environment. The leading PIP describes marginal variant inclusion;
`fit.sets["coverage"]` is the posterior mass of each set within its original
component. Read `fit.sets["purity"]` alongside that mass and the convergence flag.

## Reuse fits for colocalisation

With [pycoloc](https://github.com/LucaJiang/pycoloc) installed, the following
continues the example with the stored D4 dataset. Each trait is fitted once;
`coloc_susie` reuses component Bayes factors and retained set identities.

```python
from pycoloc import coloc_susie

metadata4 = json.loads((data / "metadata.json").read_text())["D4"]
with np.load(data / "inputs.npz", allow_pickle=False) as arrays:
    z4 = arrays["D4_beta"] / np.sqrt(arrays["D4_varbeta"])
    R4 = arrays["D4_LD"]
fit4 = prusie.susie_rss(
    z=z4, R=R4, n=metadata4["N"], variant_ids=metadata4["snp"],
    L=10, max_iter=100, tol=0.001, estimate_residual_variance=False,
    coverage=0.95, min_abs_corr=0.5, n_purity=500,
)
paired = coloc_susie(fit, fit4)
print(paired.status)
print(paired.summary[["idx1", "idx2", "PP.H4.abf"]])
```

`idx1` and `idx2` retain original zero-based component IDs. Each PP.H4 is a
signal-pair posterior, not a region-wide posterior. The associated `SNP.PP.H4`
values are conditional on H4 and differ in meaning from fine-mapping PIP.
No-CS and inadequate-overlap statuses need inspection before interpretation.
Keep the original SNP/effect-allele alignment and full Bayes-factor arrays when
saving fits for reuse. See [results](results.md) for portable storage.

## Validate the installation

From the checkout, run:

```sh
python examples/check_example.py --output-dir example-results
```

The check runs offline after installation and needs only prusie and NumPy.
It uses all seven fixed cases and checks source/reference hashes, SNP order,
model parameters, PIP validity, actual CS mass and complete-pair purity.
`example-results/report.json` records the installed version, warnings and
per-field differences; `actual_native.npz` contains the computed arrays.
The input and reference files stay unchanged.

PIP acceptance is absolute error ≤1e-5, rtol=0. Posterior and coverage
identities have a separate fixed 1e-10 arithmetic tolerance. A changed
version string is recorded; an invalid input/probability structure or failed
PIP comparison produces FAIL and a nonzero exit. Component BFs, iterations
and credible sets retain their own diagnostics in the [agreement report](r_agreement.md).

| Case | Purpose | Change from primary settings |
| --- | --- | --- |
| D1 | Single-signal teaching data | None |
| D2 | Second single-signal dataset | None |
| D3 | Two stored causal variables | None |
| D4 | Further single-signal dataset | None |
| D1_partial_zero | Prior-weight diagnostic | Every seventh one-based SNP weight is zero |
| D3_strict_purity | No returned credible set | `min_abs_corr=1` |
| D3_one_iteration | Nonconvergence diagnostic | `max_iter=1` |

All primary cases use signed z = beta / √varbeta, stored signed LD r,
n=1000, L=10, max_iter=100, tol=0.001, coverage=0.95 and min_abs_corr=0.5.
`examples/data/parameters.json` records all arguments, including complete
matrix purity, fixed residual variance and estimated prior variance. The
RSS call uses the known-n adjustment and no supplied phenotype variance.

## Interpret the diagnostics

A credible set's coverage is its returned posterior mass in the original
component; it is distinct from frequentist repeated-sampling coverage.
A no-CS fit retains PIPs but has empty member/coverage/purity collections.
The one-iteration case remains nonconverged and is excluded from primary
scientific-fit summaries.

The pinned reference adds √ε before logging active-component prior weights.
A supplied zero weight therefore permits a small posterior contribution; final
low-variance trimming restores the exact normalized prior.

## Data and reference provenance

The fixture preserves all 500 variables in stored upstream D1–D4. Its synthetic
IDs and indices carry no biological build, ancestry, allele or coordinate
annotations. Full metadata, data licensing and regeneration commands are in
the [data README](https://github.com/LucaJiang/prusie/blob/main/examples/data/README.md).

`r_fits.npz` and `r_fits.json` are official susieR 0.16.6 reference outputs.
`expected_native.*` separately records an earlier native execution; it is an
additional reproducibility check. Independent probability and CS identities
use the returned alpha and signed LD, without calling production summary
helpers. Source/archive hashes and R dependencies are in `reference.lock.json`
and `reference_environment.json`. Regeneration uses the supplied export and
conversion scripts in a new directory, preserving shipped expectations.

Read [inputs](inputs.md) and [supported scope](compatibility.md) before
substituting your own aligned data.
