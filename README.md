# prusie — SuSiE fine-mapping in Python

prusie fits the Sum of Single Effects (SuSiE) regression model to association
summary statistics and LD (SuSiE-RSS), or to sufficient statistics, and returns
posterior inclusion probabilities (PIPs) and credible sets. The iterative
Bayesian stepwise selection (IBSS) algorithm is implemented in Rust and follows
susieR 0.16.6. Further functions report convergence, refinement, LD
diagnostics and sensitivity to model settings.

## Recorded numerical accuracy and performance

<!-- current:highlights:start -->
Measured **0.2.4.dev2** against susieR 0.16.6.

| Analysis | Cases | Maximum PIP error | Paired runtime speedup | Median full-process memory ratio |
| --- | --- | --- | --- | --- |
| GWAS | 100 | 9.14e-09 | 26.23× | 6.345× |
| eQTL | 100 | 1.22e-08 | 21.62× | 6.24× |

Runtime is the geometric mean of per-case susieR/prusie ratios of seven-repeat medians. Memory is the median ratio of fresh-process peaks, including interpreters and inputs. These are complete-call observations on a fixed panel, not measurements of a particular optimization.
<!-- current:highlights:end -->

<details markdown="1">
<summary>Historical 0.2.3rc4 comparison</summary>


<!-- generated:highlights:start -->

| Analysis | Maximum absolute PIP difference | Paired runtime speedup¹ | Median peak-memory ratio² |
| --- | --- | --- | --- |
| GWAS | 9.14e-09 | 37.4× | 6.94× |
| eQTL | 1.22e-08 | 33.6× | 6.8× |

100 GWAS locus analyses and 100 eQTL analyses, each with 103–4162 variants, compared with susieR 0.16.6. ¹ Geometric mean of per-case ratios of median runtimes (susieR/prusie). ² Median per-case ratio of full-process peak memory (susieR/prusie). Results reanalyze measurements of prusie 0.2.3rc4 made on 6 October 2026; these values identify that measured version.
<!-- generated:highlights:end -->

</details>

See [numerical accuracy](docs/numerical_accuracy.md) and
[performance](docs/performance.md) for the methods and complete results.
Agreement with susieR is a check of the implementation; it does not by itself
establish that the model assumptions hold for a given dataset.

[Install](docs/install.md) · [Quick start](docs/example.md) · [Worked examples](docs/worked_examples.md)

## Install the development version

The `main` branch contains the **0.2.4.dev2 development version** described
here; it is not a release. Building from source requires CPython 3.10–3.13,
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

## One fit, a readable result

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

## Sensitivity to model settings

![Candidate PIPs of the toy example under four settings: the two leading credible sets persist, while a fixed scaled prior variance raises the PIPs of other variants.](docs/examples/post_analysis/stable/figure_5.png)

The two leading credible sets persist when the scaled prior variance is fixed
at a positive value, whereas the PIP of `syn0143` increases from about
5.3×10⁻¹¹ to 0.2544. Without prior-variance estimation, all five single effects
keep a positive prior variance, including three without a credible set, and
these effects place posterior probability on further variants. The
[worked examples](docs/worked_examples.md) describe this separate sensitivity
analysis; the code above fits and summarizes only the baseline.

See [reading results](docs/results.md), [convergence and refinement](docs/post_analysis.md)
and [sensitivity and LD](docs/sensitivity.md), and for reference the [API](docs/api.md)
and the [statistical model](docs/model.md). The [documentation site](https://lucajiang.github.io/prusie/)
shows the most recently deployed version, which can lag behind this branch.

## Citation and license

Please cite [SuSiE, Wang et al. (2020)](https://doi.org/10.1111/rssb.12388) and
[SuSiE-RSS, Zou et al. (2022)](https://doi.org/10.1371/journal.pgen.1010299),
and identify the prusie version (here 0.2.4.dev2, a development version) as
given in [CITATION.cff](CITATION.cff). Author and maintainer: Wenxin Jiang.

prusie is distributed under GPL-3.0-or-later. [LICENSE](LICENSE) and the
[third-party notices](THIRD_PARTY_NOTICES.md) retain the upstream attributions.
