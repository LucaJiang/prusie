# Fine-mapping, from candidates to interpretation

prusie fits the SuSiE model to signed association statistics and LD. The Python
API returns posterior inclusion probabilities (PIPs), credible sets and component
posteriors, with optional convergence tracking, refinement, sensitivity comparisons
and offline reports.

Start with the [quick start](example.md) to fit the packaged synthetic example.
Then use the result to answer four questions:

| Question | Read next |
| --- | --- |
| Which candidates carry support, and how concentrated is each signal? | [Reading results](results.md) |
| Did IBSS stop, and does a new search improve the solution? | [Convergence and refinement](post_analysis.md) |
| Which conclusions change with settings or LD assumptions? | [Sensitivity and LD diagnostics](sensitivity.md) |
| What does a complete interpretation look like? | [Worked examples](worked_examples.md) |

## One fit and a short summary

```python
import prusie

example = prusie.load_example()
fit = prusie.susie_rss(**example["inputs"], **example["parameters"],
                       track_fit="compact", retain_data=True)
summary = prusie.summarize(fit, example["inputs"]["R"], z=example["inputs"]["z"])
print(prusie.summary_text(summary))
```

The example returns two singleton credible sets after three sweeps. The
[quick start](example.md#write-an-offline-report) adds figures and an offline report
without fitting again. [Installation](install.md) describes this 0.2.4.dev1
branch preview and optional analysis dependencies.

## Two different diagnostic questions

![Candidate PIPs in the stable example show persistent leading sets and a prior-sensitive background.](examples/post_analysis/stable/figure_5.png)

In the stable example, fixing a positive component variance preserves the two
leading CS but raises background PIPs. In the problem example, refinement improves
the original-model objective by 17.46 after the initial fit has already met its
stopping condition. The remaining `{86,90}` pair has limited resolution.
[Read both examples and open their complete reports](worked_examples.md).

## Measured numerical accuracy and performance

<!-- generated:highlights:start -->

| Analysis | Maximum absolute PIP difference | Paired runtime speedup¹ | Median peak-memory ratio² |
| --- | --- | --- | --- |
| GWAS | 9.14e-09 | 37.4× | 6.94× |
| eQTL | 1.22e-08 | 33.6× | 6.8× |

100 GWAS locus analyses and 100 eQTL analyses, each with 103–4162 variants, compared with susieR 0.16.6. ¹ Geometric mean of per-case ratios of median runtimes (susieR/prusie). ² Median per-case ratio of full-process peak memory (susieR/prusie). Results reanalyze measurements of prusie 0.2.3rc4 made on 6 October 2026; these values identify that measured version.
<!-- generated:highlights:end -->

![Runtime and full-process peak-memory ratios for GWAS and eQTL, with ratios above one favoring prusie.](assets/r_comparison.svg)

[Numerical accuracy](numerical_accuracy.md) and [performance](performance.md)
explain the data, methods and original measurement environment. These measurements
are attributed to prusie 0.2.3rc4, not to the development preview.

For precise inputs and return values, use the [API](api.md), [input guide](inputs.md)
and [statistical model](model.md). [Reproducibility](reproducibility.md) separates
regenerating figures from rerunning fits. Cite the original statistical methods
and the software revision as described in [citation and license](citation.md).
