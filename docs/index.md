# prusie: SuSiE fine-mapping in Python

<div class="logo-banner">
  <a href="assets/logos/logo_title.jpeg"><img src="assets/logos/prusie-banner-1792.jpg" alt="prusie — Python and Rust software for SuSiE fine-mapping" width="1792" height="592"></a>
</div>

Fine-mapping seeks to identify variants that affect a trait despite uncertainty
from linkage disequilibrium (LD). prusie fits the sum of single effects (SuSiE)
model in Python with a Rust solver, allowing multiple effects within a region.

Fits accept association summary statistics with signed LD, centered sufficient
statistics, or individual-level data. Results include component inclusion
probabilities (`alpha`), marginal posterior inclusion probabilities (PIPs) and
credible sets. See [the statistical model](model.md) for assumptions and
[numerical accuracy](numerical_accuracy.md) for comparisons with susieR 0.16.6.

## Installation

From the prusie 0.3.0 source checkout, use CPython 3.10–3.13 and install the
[build requirements](install.md#build-from-source), including the pinned Rust
toolchain:

```sh
cd prusie
python -m venv .venv
. .venv/bin/activate
python -m pip install '.[analysis]' --config-settings=build-args="--locked"
```

NumPy is the only required Python dependency for core fitting. The optional
`analysis` extra installed above adds SciPy for sparse inputs, LD diagnostics
and advanced RSS models, and Matplotlib for plotting. Fitting does not require
R. To install without Rust or a compiler, use a compatible
[prebuilt wheel](install.md#use-a-prebuilt-wheel).

## Example

Fit the packaged 500-variant synthetic toy example offline and list its retained
credible sets (CS):

```python
import prusie

example = prusie.load_example()
fit = prusie.susie_rss(**example["inputs"], **example["parameters"])
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

Component IDs are zero-based. The [quick-start tutorial](example.md) explains
PIPs and credible sets, then adds figures and an offline report.

## Accuracy and performance

The comparison uses SuSiE-RSS fits for FinnGen R13 type 2 diabetes GWAS loci
and GTEx v11 Whole_Blood eQTL analyses. Time and memory ratios below are
susieR divided by prusie.

<!-- current:highlights:start -->
Measured **prusie 0.3.0** against **susieR 0.16.6** on the same inputs.

| Evaluated cases | Cases | Cumulative runtime ratio | Cumulative fitting-stage ratio | Median peak-memory ratio |
| --- | --- | --- | --- | --- |
| GWAS | 200 | 49.2× | 55.4× | 5.11× |
| eQTL | 14,157 | 26.6× | 29.8× | 4.60× |

The cumulative ratios divide the summed susieR time by the summed prusie time over three retained complete calls of every evaluated case, with one numerical thread; the fitting stage excludes input and LD preparation. The memory ratio is the median of paired susieR/prusie ratios of peak backend-process memory. All measurements are from exclusive AMD EPYC 7742 nodes. Ratios for other traits, tissues, region sizes and LD references may differ; see [performance](performance.md).
<!-- current:highlights:end -->

See [performance](performance.md) for methods and software identities, and
[numerical accuracy](numerical_accuracy.md) for the complete-alpha comparison
and convergence diagnostics.

## Sensitivity and refinement

The official susieR [toy example](worked_examples.md) shows a fit that meets its
stopping rule before refinement finds a higher evidence lower bound (ELBO).
Refinement changes the singleton credible sets, while a set of strongly
correlated variants remains unresolved.

![PIPs for the baseline fit, refined fit and a fit with a larger maximum number of single-effect components.](examples/post_analysis/official/figure_5.png)

Increasing the maximum number of single-effect components, `L`, retains the
baseline's credible sets in this example. The
[saved report](examples/post_analysis/official/index.html) shows every retained
credible set and separate LD panels for each set's top variant.

## Documentation

<div class="start-links" markdown="1">

[Quick start](example.md) [Reading results](results.md) [Toy example](worked_examples.md)

</div>

- [Inputs, alignment and priors](inputs.md): input routes, variant order, effect alleles and prior weights.
- [Statistical model](model.md): assumptions, single-effect components, PIPs and credible sets.
- [Python API](api.md): fitting functions, options and result fields.
- [Convergence and refinement](post_analysis.md): stopping diagnostics, continuation and alternative starting solutions.
- [Sensitivity and LD diagnostics](sensitivity.md): model comparisons and consistency between z-scores and LD.
- [Reproducibility](reproducibility.md): data sources, reference fits and regeneration of benchmark tables and figures.

## Citation and license

Please cite [SuSiE, Wang et al. (2020)](https://doi.org/10.1111/rssb.12388) and
[SuSiE-RSS, Zou et al. (2022)](https://doi.org/10.1371/journal.pgen.1010299)
for the statistical methods. Include the installed prusie version
(`prusie.__version__`); software citation metadata are in
[CITATION.cff](https://github.com/LucaJiang/prusie/blob/main/CITATION.cff).

Wenxin Jiang is the author and maintainer. No DOI is assigned to the software.

Copyright © 2026 Wenxin Jiang. Licensed under [GPL-3.0-or-later](https://github.com/LucaJiang/prusie/blob/main/LICENSE);
see [third-party notices](https://github.com/LucaJiang/prusie/blob/main/THIRD_PARTY_NOTICES.md) for upstream attribution.
