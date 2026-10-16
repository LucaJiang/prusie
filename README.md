# prusie: SuSiE fine-mapping from summary statistics

prusie is a fast Rust implementation of SuSiE and SuSiE-RSS fine-mapping for
Python. It fits association summary statistics and linkage disequilibrium (LD),
or sufficient statistics, and returns posterior inclusion probabilities (PIPs)
and credible sets. Numerical comparisons with susieR 0.16.6 show agreement to
within 1e-5 in PIP across the 200 GWAS and eQTL analyses described below.

## Installation

This checkout contains **0.2.4.dev2** (development version). Building from source requires CPython 3.10–3.13,
the Rust toolchain pinned in `rust-toolchain.toml`, a C linker and the Python
development headers:

```sh
git clone --branch main https://github.com/LucaJiang/prusie.git
cd prusie
python -m venv .venv
. .venv/bin/activate
python -m pip install '.[analysis]'
```

Fitting requires only NumPy at run time; the optional `analysis` extra adds
Matplotlib and SciPy for figures and LD diagnostics. An installed package
requires neither R nor Rust. The [installation guide](docs/install.md) also
describes the earlier prerelease wheel for CPython 3.12 on Linux x86-64.

## Example

The packaged synthetic toy example with 500 variants can be fitted offline:

```python
import prusie

example = prusie.load_example()
fit = prusie.susie_rss(
    **example["inputs"], **example["parameters"], track_fit="compact", retain_data=True
)
summary = prusie.summarize(fit, example["inputs"]["R"], z=example["inputs"]["z"])
print(prusie.summary_text(summary).splitlines()[0])
print(f"IBSS: {fit.convergence['stop_reason']} ({fit.niter} sweeps)")
for cs in summary["credible_sets"]:
    print(f"CS {cs['component_id']}: {', '.join(cs['members'])}")
```

Expected output:

```text
500 variants; 2 retained credible sets; 2 active components.
IBSS: criterion_satisfied (3 sweeps)
CS 0: syn0100
CS 1: syn0350
```

The [quick-start tutorial](docs/example.md) explains these PIPs and credible
sets (CS) and then produces figures and an offline report from the same fit.

## Accuracy and performance

<!-- current:highlights:start -->
Measured **0.2.4.dev2** against susieR 0.16.6.

| Analysis | Loci | Max. absolute PIP difference vs susieR | Speed-up¹ | Memory ratio² |
| --- | --- | --- | --- | --- |
| GWAS | 100 | 9.14e-09 | 26× | 6.3× |
| eQTL | 100 | 1.22e-08 | 22× | 6.2× |

¹ Geometric mean of per-locus susieR/prusie runtime ratios. ² Median per-locus susieR/prusie ratio of peak resident memory (RSS), including the interpreter and inputs.
<!-- current:highlights:end -->

AMD EPYC 9754, Linux x86-64; one numerical thread per fit.
See [performance](docs/performance.md) for the data, software and measurement
protocol, and [numerical accuracy](docs/numerical_accuracy.md) for validation.

## Documentation

[Quick start](docs/example.md) · [Reading results](docs/results.md) ·
[Convergence and refinement](docs/post_analysis.md) ·
[Sensitivity and LD](docs/sensitivity.md) · [API](docs/api.md)

The [worked examples](docs/worked_examples.md) show how credible sets and PIPs
change with prior assumptions and refinement. Browse the
[documentation site](https://lucajiang.github.io/prusie/) for the deployed version.

## Citation and license

Please cite [SuSiE, Wang et al. (2020)](https://doi.org/10.1111/rssb.12388) and
[SuSiE-RSS, Zou et al. (2022)](https://doi.org/10.1371/journal.pgen.1010299),
and identify the prusie version using [CITATION.cff](CITATION.cff).
Author and maintainer: Wenxin Jiang. No DOI is assigned to the software.

Copyright © 2026 Wenxin Jiang. Licensed under [GPL-3.0-or-later](LICENSE);
see [third-party notices](THIRD_PARTY_NOTICES.md) for upstream attribution.
