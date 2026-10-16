# SuSiE fine-mapping in Python

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

See [numerical accuracy](numerical_accuracy.md) and [performance](performance.md)
for the methods and complete results. Agreement with susieR is a check of the
implementation; it does not by itself establish that the model assumptions hold
for a given dataset.

<div class="start-links" markdown="1">

[Install](install.md) [Quick start](example.md) [Worked examples](worked_examples.md)

</div>

## One fit, a readable result

The packaged synthetic toy example with 500 variants can be fitted offline
after installation. This code fits the model once and summarizes the posterior:

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

Each of the two credible sets contains a single variant in this toy example.
The [quick start](example.md) explains their PIPs and posterior mass and then
produces figures and an offline report. Refinement, LD diagnostics and
sensitivity analyses are separate steps that the code above does not run.

## Sensitivity and refinement

![Candidate PIPs of the toy example under four settings: the two leading credible sets persist, while a fixed scaled prior variance changes the PIPs of other variants.](examples/post_analysis/stable/figure_5.png)

In the toy example the two leading credible sets persist across these
settings, whereas a fixed positive scaled prior variance increases the PIPs of
other variants. The [worked examples](worked_examples.md) discuss this
distinction and a second example in which refinement finds a solution with a
higher objective, while a pair of variants in strong LD remains unresolved.

## Reading a result

- [Reading results](results.md): which variants have posterior support, and how concentrated is each single effect?
- [Convergence and refinement](post_analysis.md): did IBSS satisfy its stopping rule, and does refinement find a solution with a higher objective?
- [Sensitivity and LD](sensitivity.md): which conclusions change with model settings or input assumptions?
- [Worked examples](worked_examples.md): how do these results combine in an interpretation?

The [API](api.md), [input guide](inputs.md) and [statistical model](model.md)
give the precise inputs, outputs and definitions. [Reproducibility](reproducibility.md)
distinguishes regenerating figures from refitting models, and
[citation and license](citation.md) identifies the methods and the software version.
