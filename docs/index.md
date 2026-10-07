# prusie

SuSiE fine-mapping in Python, powered by Rust.

Fit the established SuSiE-RSS model to association summary statistics and signed
linkage disequilibrium (LD). Obtain variant posterior inclusion probabilities
(PIPs), credible sets and component Bayes factors through a Python API.

- **Numerical agreement:** all 200 real trait cases meet the fixed PIP criterion against susieR 0.16.6.
- **Measured runtime:** 199/199 comparable real cases run faster than optimized R; paired geometric-mean speedup is 35.2×.
- **Lower peak process RSS:** median 37.7 MiB versus 266 MiB for R on the same comparable cases.

## Numerical agreement

Current prusie 0.2.3rc4 was compared with freshly executed pinned susieR 0.16.6 using matched float64 inputs.

| Comparison with susieR 0.16.6 | Passed cases | Maximum absolute PIP error |
| --- | ---: | ---: |
| Real regions, 103–4162 SNPs | 200/200 | 1.22e-08 |
| 500-SNP teaching cases, converged | 6/6 | 8.81e-10 |

The PIP criterion is absolute error ≤1e-5, rtol=0, with input/model/probability checks. 199/200 real fits converged on both sides; the nonconverged case and separate one-iteration teaching diagnostic remain reported. Component BFs and downstream colocalisation have distinct checks in [R agreement](r_agreement.md).

## Performance

Complete calls with one numerical thread against susieR 0.16.6 in R 4.4.0 / OpenBLAS 0.3.20. The fixed panel contains 100 real regions, 103–4162 SNPs each. Tables summarize the 199/200 cases in which both fits converged.

| Task | Comparable cases | Median time, R → Python (ms) | Paired speedup¹ | Faster / tied / slower |
| --- | ---: | ---: | ---: | ---: |
| SuSiE-RSS fine-mapping | 199/200 | 86.9 → 2.52 | 35.2× | 199 / 0 / 0 |

¹ Geometric mean of per-case speedups; seven retained repetitions after warmup. Speedup pairs each case; the displayed time columns are separate across-case medians.

| Task | Median peak process RSS, R → Python (MiB) | Lower / tied / higher in Python |
| --- | ---: | ---: |
| SuSiE-RSS fine-mapping | 266 → 37.7 | 199 / 0 / 0 |

![Runtime and peak process RSS ratios for every measured real case, including slower and higher-memory cases.](assets/r_comparison.svg)

Peak process RSS includes imports, inputs and full results, measured in a separate fresh process. [Methods, all cases, repeat variability and public-fixture results](performance.md) give the comparison scope.

## Installation

A locally supplied CPython 3.12 Linux x86-64 wheel can be installed directly,
without Rust on the target system:

```sh
python -m pip install ./prusie-0.2.3rc4-cp312-cp312-manylinux_2_34_x86_64.whl
```

Public wheel downloads are not yet available. To install from the source
repository, use Python ≥3.10, the pinned Rust toolchain, a C linker and Python
development headers:

```sh
git clone https://github.com/LucaJiang/prusie.git
cd prusie
python -m venv .venv
. .venv/bin/activate
python -m pip install .
```

See [installation](install.md) for tested environments and build details.

## Quick start

Run from the checkout. This uses the complete **500-SNP synthetic teaching
dataset D3** bundled from official coloc; its IDs are teaching variables.

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

PIP summarizes a variant's inclusion across effects. Each credible set belongs
to an original SuSiE component; its returned coverage is actual posterior mass.
Inspect convergence and set purity before interpreting the result. The
[worked example](example.md) explains these outputs and the offline check.

## Methods and scope

`susie_rss` accepts signed z or beta/SE with signed LD **r**, in exactly the same
SNP and effect-allele order. `susie_suff_stat` supports centered Gaussian
sufficient statistics. The Rust core performs sequential iterative Bayesian stepwise selection (IBSS)
updates in float64, following the supported Gaussian semantics of susieR 0.16.6. Dense LD storage requires 8m² bytes for m
variants, before fitting workspace and outputs.

Supply sample-size and allele/build/population provenance. Allele harmonization
and LD quality control precede fitting; residual variance stays fixed by default
for external reference LD. Read [inputs](inputs.md) and
[supported models](compatibility.md) for assumptions and options.

Fit each trait or gene once, then reuse its results. With pycoloc installed,
`pycoloc.coloc_susie(fit1, fit2)` pairs existing component Bayes factors without
refitting. See [the integration example](example.md#reuse-fits-for-colocalisation).
Each PP.H4 row describes one signal pair, rather than a region-wide posterior.

## Documentation and reproducibility

Browse the [documentation site](https://lucajiang.github.io/prusie/),
[API](api.md), [result schema](results.md) and
[manual R comparison scripts](https://github.com/LucaJiang/prusie/tree/main/tools/benchmark/README.md).
To validate an installation against all seven frozen cases, run
`python examples/check_example.py --output-dir example-results`.
The R agreement criterion applies to PIPs; component and downstream posterior
agreement are checked separately.

## Citation and license

Cite the SuSiE method of Wang et al. (2020) and SuSiE-RSS of Zou et al. (2022),
and identify the prusie version using [CITATION.cff](https://github.com/LucaJiang/prusie/blob/main/CITATION.cff).
Maintained by Wenxin Jiang.

GPL-3.0-or-later. See [LICENSE](https://github.com/LucaJiang/prusie/blob/main/LICENSE) and
[third-party notices](https://github.com/LucaJiang/prusie/blob/main/THIRD_PARTY_NOTICES.md) for retained upstream attribution
and the bundled teaching-data terms.
