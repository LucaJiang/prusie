# prusie: SuSiE fine-mapping in Python

<div class="logo-banner">
  <a href="assets/logos/logo_title.jpeg"><img src="assets/logos/prusie-banner-1792.jpg" alt="prusie — Python and Rust software for SuSiE fine-mapping" width="1792" height="592"></a>
</div>

prusie fits SuSiE fine-mapping models in Python with a Rust solver. It returns
component-level posterior probabilities, marginal inclusion probabilities (PIPs)
and credible sets from association summaries, sufficient statistics or
individual-level data. See [numerical accuracy](numerical_accuracy.md) for
validation against susieR 0.16.6 and [the model](model.md) for assumptions.

## Installation

This checkout contains **0.3.0**. Building from source requires CPython 3.10–3.13,
the Rust toolchain pinned in `rust-toolchain.toml`, a C linker and the Python
development headers:

```sh
cd prusie  # source checkout for the version documented here
python -m venv .venv
. .venv/bin/activate
python -m pip install '.[analysis]'
```

Ordinary dense fitting requires only NumPy at run time. The optional `analysis`
extra adds SciPy for sparse designs and advanced RSS models, and Matplotlib
for figures. An installed package
requires neither R nor Rust. The [installation guide](install.md) also
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

The [quick-start tutorial](example.md) explains these PIPs and credible
sets (CS) and then produces figures and an offline report from the same fit.

## Accuracy and performance

<!-- current:highlights:start -->
Measured **prusie 0.3.0** against fresh **susieR 0.16.6** fits.

| Analysis | Cases | Timing pairs | Total-time speed-up¹ | Per-case geometric mean² | Peak RSS ratio³ |
| --- | --- | --- | --- | --- | --- |
| GWAS | 200 | 200 | 27.7× | 35× | 3.66× |
| eQTL | 14157 | 14157 | 19.1× | 22.7× | 3.77× |

¹ Sum of all seven retained R fit times divided by the corresponding prusie total. ² Geometric mean of within-case median-time ratios. ³ Median paired R/prusie fresh-process peak RSS, including interpreter and inputs. The two speed-up summaries use different weighting.
<!-- current:highlights:end -->

AMD EPYC 9754, Linux x86-64; one numerical thread per fit.
See [performance](performance.md) for the data, software and measurement
protocol, and [numerical accuracy](numerical_accuracy.md) for validation.

## Sensitivity and refinement

![PIPs before refinement, after refinement and with a larger component limit in the official susieR example.](examples/post_analysis/official/figure_5.png)

The official susieR [Toy example](worked_examples.md) shows a fit that meets its
stopping rule and then improves under refinement. Posterior support moves from
two singleton credible sets to a different singleton, while a pair of variants
in strong LD remains unresolved. The saved report presents every retained
credible set and the LD to each set's top variant.

## Documentation

<div class="start-links" markdown="1">

[Quick start](example.md) [Reading results](results.md) [Toy example](worked_examples.md)

</div>

- [Convergence and refinement](post_analysis.md): stopping rules and alternative starting solutions.
- [Sensitivity and LD](sensitivity.md): model choices, variant alignment and input diagnostics.
- [API](api.md), [inputs](inputs.md) and [statistical model](model.md): definitions and interfaces.
- [Reproducibility](reproducibility.md): example data, reference results and measurement scripts.

## Citation and license

Please cite [SuSiE, Wang et al. (2020)](https://doi.org/10.1111/rssb.12388) and
[SuSiE-RSS, Zou et al. (2022)](https://doi.org/10.1371/journal.pgen.1010299),
and identify the prusie version using [CITATION.cff](https://github.com/LucaJiang/prusie/blob/main/CITATION.cff).
Author and maintainer: Wenxin Jiang. No DOI is assigned to the software.

Copyright © 2026 Wenxin Jiang. Licensed under [GPL-3.0-or-later](https://github.com/LucaJiang/prusie/blob/main/LICENSE);
see [third-party notices](https://github.com/LucaJiang/prusie/blob/main/THIRD_PARTY_NOTICES.md) for upstream attribution.
