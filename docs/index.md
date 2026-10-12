# SuSiE fine-mapping in Python

Fit association summary statistics with a Rust-powered SuSiE implementation,
then inspect credible sets, convergence and sensitivity in one workflow.

## Recorded numerical accuracy and performance

The comparison below measures **0.2.3rc4**. It is historical evidence for that
version; the current development snapshot is **0.2.4.dev2**.

<!-- generated:highlights:start -->

| Analysis | Maximum absolute PIP difference | Paired runtime speedup¹ | Median peak-memory ratio² |
| --- | --- | --- | --- |
| GWAS | 9.14e-09 | 37.4× | 6.94× |
| eQTL | 1.22e-08 | 33.6× | 6.8× |

100 GWAS locus analyses and 100 eQTL analyses, each with 103–4162 variants, compared with susieR 0.16.6. ¹ Geometric mean of per-case ratios of median runtimes (susieR/prusie). ² Median per-case ratio of full-process peak memory (susieR/prusie). Results reanalyze measurements of prusie 0.2.3rc4 made on 6 October 2026; these values identify that measured version.
<!-- generated:highlights:end -->

Use [numerical accuracy](numerical_accuracy.md) and [performance](performance.md)
for the full comparison and measurement methods. The Python workflow adds
explicit continuation and refinement, LD diagnostics, comparisons across settings,
and offline reports alongside variant PIPs and credible sets.

<div class="start-links" markdown="1">

[Install](install.md) [Quick start](example.md) [Worked examples](worked_examples.md)

</div>

## One fit, a readable result

The packaged 500-variant synthetic toy example runs offline after installation.
This code fits once and summarizes that posterior:

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

The two singleton credible sets identify concentrated candidates in this toy fit.
The [quick start](example.md) explains their PIPs and posterior mass, then adds
plots and an offline report. Refinement, LD and sensitivity are separate explicit
steps; they have not been run by the code above.

## From candidates to interpretation

![Candidate PIPs across settings: leading credible sets persist while fixing scaled prior variance changes the background.](examples/post_analysis/stable/figure_5.png)

The stable example retains its two main CS across these settings, while a fixed
positive scaled prior variance raises background probabilities. The
[worked examples](worked_examples.md) explain this distinction and a second fit
where refinement improves the solution but an LD-correlated pair remains unresolved.

## Read the result in context

- [Reading results](results.md): which candidates carry support, and how concentrated is each signal?
- [Convergence and refinement](post_analysis.md): did IBSS stop, and can search improve the solution?
- [Sensitivity and LD](sensitivity.md): which conclusions change with settings or input assumptions?
- [Worked examples](worked_examples.md): how do these observations fit into a complete interpretation?

For precise inputs and return values, use the [API](api.md), [input guide](inputs.md)
and [statistical model](model.md). [Reproducibility](reproducibility.md) separates
regenerating figures from rerunning fits. [Citation and license](citation.md)
identifies the statistical methods and software revision.
