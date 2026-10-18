# prusie: SuSiE fine-mapping from summary statistics

<div class="logo-banner">
  <a href="assets/logos/logo_title.jpeg"><img src="assets/logos/prusie-banner-1792.jpg" alt="prusie — Python and Rust software for SuSiE fine-mapping" width="1792" height="592"></a>
</div>

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
Measured **0.2.4.dev2** against susieR 0.16.6.

| Analysis | Loci | Max. absolute PIP difference vs susieR | Speed-up¹ | Memory ratio² |
| --- | --- | --- | --- | --- |
| GWAS | 100 | 9.14e-09 | 26× | 6.3× |
| eQTL | 100 | 1.22e-08 | 22× | 6.2× |

¹ Geometric mean of per-locus susieR/prusie runtime ratios. ² Median per-locus susieR/prusie ratio of peak resident memory (RSS), including the interpreter and inputs.
<!-- current:highlights:end -->

AMD EPYC 9754, Linux x86-64; one numerical thread per fit.
See [performance](performance.md) for the data, software and measurement
protocol, and [numerical accuracy](numerical_accuracy.md) for validation.

## Sensitivity and refinement

![Candidate PIPs under four settings: leading credible sets persist while a fixed prior variance changes PIPs elsewhere.](examples/post_analysis/stable/figure_5.png)

Fixing the scaled prior variance at a positive value preserves the two leading
credible sets but raises PIP at other variants: `syn0143` changes from about
5.3e-11 to 0.2544. All five components retain positive prior variance, including
three without a credible set. The [worked examples](worked_examples.md) explain
this sensitivity and a second example in which refinement finds a higher ELBO.

## Documentation

<div class="start-links" markdown="1">

[Quick start](example.md) [Reading results](results.md) [Worked examples](worked_examples.md)

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

<div class="logo-signature">
  <a href="assets/logos/logo.jpeg"><img src="assets/logos/prusie-logo-384.jpg" alt="prusie logo: a Python snake, DNA helix, association dots and a gear" width="384" height="384" loading="lazy"></a>
</div>
