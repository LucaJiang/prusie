# prusie: SuSiE fine-mapping in Python

[![tests](https://github.com/LucaJiang/prusie/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/LucaJiang/prusie/actions/workflows/ci.yml)
[![coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/LucaJiang/prusie/badges/coverage.json)](https://github.com/LucaJiang/prusie/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://github.com/LucaJiang/prusie/blob/main/pyproject.toml)
[![license: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)](https://github.com/LucaJiang/prusie/blob/main/LICENSE)

<p align="center">
  <a href="https://raw.githubusercontent.com/LucaJiang/prusie/main/docs/assets/logos/logo_title.jpeg"><img src="https://raw.githubusercontent.com/LucaJiang/prusie/main/docs/assets/logos/prusie-banner-1792.jpg" alt="prusie — Python and Rust software for SuSiE fine-mapping" width="960"></a>
</p>

prusie fits SuSiE fine-mapping models in Python with a Rust solver. It returns
component inclusion probabilities (`alpha`), posterior inclusion probabilities (PIPs)
and credible sets from association summaries, sufficient statistics or
individual-level data. See [numerical accuracy](https://lucajiang.github.io/prusie/numerical_accuracy.html) for
validation against susieR 0.16.6 and [the model](https://lucajiang.github.io/prusie/model.html) for assumptions.

- Dense and sparse individual data, summary statistics, sufficient statistics,
  regularized RSS, single-effect regression and trend filtering.
- Gaussian or fixed mixture priors, conjugate NIG residual inference, slot
  activity models and automatic or greedy effect counts.
- Posterior accessors, credible-set controls, prediction and model summaries.

SuSiE-inf, SuSiE-ash and Mr.ASH are outside the supported model set.

## Installation

**prusie 0.3.1** supports CPython 3.10 and newer (GIL builds). Install from PyPI:

```sh
pip install prusie
```

Prebuilt stable-ABI (`abi3`) wheels cover Linux x86-64/ARM64, macOS Intel/Apple
silicon and Windows x86-64. The same wheel is tested on CPython 3.10–3.15;
free-threaded builds are not supported. Wheels need neither Rust nor a compiler.

NumPy is the only required Python dependency for core fitting. Use
`pip install 'prusie[analysis]'` to add SciPy for sparse input, advanced RSS models
and posterior outcome configurations, and Matplotlib for plotting. Fitting does
not require R. See the [installation guide](https://lucajiang.github.io/prusie/install.html) for platform requirements,
local wheels and [building from source](https://lucajiang.github.io/prusie/install.html#build-from-source).

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

The [quick-start tutorial](https://lucajiang.github.io/prusie/example.html) explains these PIPs and credible
sets (CS) and then produces figures and an offline report from the same fit.

## Accuracy and performance

<!-- current:highlights:start -->
Measured **prusie 0.3.0** against **susieR 0.16.6** on the same inputs.

| Evaluated cases | Cases | Cumulative runtime ratio | Cumulative fitting-stage ratio | Median peak-memory ratio |
| --- | --- | --- | --- | --- |
| GWAS | 200 | 49.2× | 55.4× | 5.11× |
| eQTL | 14,157 | 26.6× | 29.8× | 4.60× |

The cumulative ratios divide the summed susieR time by the summed prusie time over three retained complete calls of every evaluated case, with one numerical thread; the fitting stage excludes input and LD preparation. The memory ratio is the median of paired susieR/prusie ratios of peak backend-process memory. All measurements are from exclusive AMD EPYC 7742 nodes. Ratios for other traits, tissues, region sizes and LD references may differ; see [performance](https://lucajiang.github.io/prusie/performance.html).
<!-- current:highlights:end -->

The performance report identifies the measured software, hardware, data and protocol.

## Documentation

[Quick start](https://lucajiang.github.io/prusie/example.html) · [Reading results](https://lucajiang.github.io/prusie/results.html) ·
[Convergence and refinement](https://lucajiang.github.io/prusie/post_analysis.html) ·
[Sensitivity and LD](https://lucajiang.github.io/prusie/sensitivity.html) · [API](https://lucajiang.github.io/prusie/api.html)

The official susieR [Toy example](https://lucajiang.github.io/prusie/worked_examples.html) shows how credible sets
and PIPs change under refinement. Browse the
[documentation site](https://lucajiang.github.io/prusie/) for the deployed version.

## Citation and license

Please cite [SuSiE, Wang et al. (2020)](https://doi.org/10.1111/rssb.12388) and
[SuSiE-RSS, Zou et al. (2022)](https://doi.org/10.1371/journal.pgen.1010299),
and identify the prusie version using [CITATION.cff](https://github.com/LucaJiang/prusie/blob/main/CITATION.cff).
Copyright © 2026 Wenxin Jiang; licensed under [GPL-3.0-or-later](https://github.com/LucaJiang/prusie/blob/main/LICENSE);
see [third-party notices](https://github.com/LucaJiang/prusie/blob/main/THIRD_PARTY_NOTICES.md) for upstream attribution.
