# prusie — SuSiE fine-mapping in Python

Fit association summary statistics with a Rust-powered SuSiE implementation,
then inspect credible sets, convergence and sensitivity in one workflow.

## Numerical accuracy and performance

<!-- generated:highlights:start -->

| Analysis | Maximum absolute PIP difference | Paired runtime speedup¹ | Median peak-memory ratio² |
| --- | --- | --- | --- |
| GWAS | 9.14e-09 | 37.4× | 6.94× |
| eQTL | 1.22e-08 | 33.6× | 6.8× |

100 GWAS locus analyses and 100 eQTL analyses, each with 103–4162 variants, compared with susieR 0.16.6. ¹ Geometric mean of per-case ratios of median runtimes (susieR/prusie). ² Median per-case ratio of full-process peak memory (susieR/prusie). Results reanalyze measurements of prusie 0.2.3rc4 made on 6 October 2026; these values identify that measured version.
<!-- generated:highlights:end -->

See [numerical accuracy](docs/numerical_accuracy.md) and
[performance](docs/performance.md) for methods and complete results. The Python
workflow combines PIPs and credible sets with explicit continuation, refinement,
LD diagnostics, sensitivity comparisons and offline reports.

[Install](docs/install.md) · [Quick start](docs/example.md) · [Worked examples](docs/worked_examples.md)

## Install the development preview

This is the **0.2.4.dev1 preview** on `post_analysis`. Source installation needs
Python ≥3.10, the pinned Rust toolchain, a C linker and Python headers:

```sh
git clone --branch post_analysis https://github.com/LucaJiang/prusie.git
cd prusie
python -m venv .venv
. .venv/bin/activate
python -m pip install '.[analysis]'
```

Core fitting needs NumPy at runtime; the optional `analysis` extra adds Matplotlib
and SciPy. Neither R nor Rust is required after installation.

## One fit, a readable result

The installed 500-variant synthetic toy example works offline:

```python
import prusie

example = prusie.load_example()
fit = prusie.susie_rss(**example["inputs"], **example["parameters"],
                       track_fit="compact", retain_data=True)
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

The [quick-start tutorial](docs/example.md) explains the candidate PIPs and CS,
then adds figures and an offline report from the fitted posterior.

## Inspect what changes across settings

![Stable-example candidate PIPs: the leading credible sets persist while a fixed scaled prior variance raises background probabilities.](docs/examples/post_analysis/stable/figure_5.png)

The same two leading CS persist under a fixed positive scaled prior variance,
while `syn0143` rises from PIP about 5.3×10⁻¹¹ to 0.2544. With variance estimation
disabled, all five components retain positive variance, including three without
retained CS. Those components can spread probability beyond the leading candidates.
This separate sensitivity analysis is explained in the
[worked examples](docs/worked_examples.md); the minimal code above only fits and
summarizes its baseline.

Read [results](docs/results.md), [convergence and refinement](docs/post_analysis.md),
and [sensitivity and LD](docs/sensitivity.md), then consult the [API](docs/api.md)
and [model](docs/model.md). The [documentation site](https://lucajiang.github.io/prusie/)
shows its latest deployment; source changes on this branch require a new deployment
to appear there.

## Citation and license

Cite [SuSiE, Wang et al. (2020)](https://doi.org/10.1111/rssb.12388) and
[SuSiE-RSS, Zou et al. (2022)](https://doi.org/10.1371/journal.pgen.1010299),
and identify this prusie 0.2.4.dev1 development snapshot with [CITATION.cff](CITATION.cff).
Author and maintainer: Wenxin Jiang.

GPL-3.0-or-later. [LICENSE](LICENSE) and
[third-party notices](THIRD_PARTY_NOTICES.md) retain upstream attribution.

For historical development evidence, the [original review snapshot](docs/review/post_analysis/README.md),
[review-fixes snapshot](docs/review/post_analysis_fixes/README.md) and
[sensitivity follow-up](docs/review/sensitivity_followup.md) remain preserved.
