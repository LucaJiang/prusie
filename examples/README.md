# Examples

Run `python examples/quickstart.py` after installation to analyze the independent
packaged 500-variant synthetic toy example and inspect PIPs, credible sets and
convergence. Its inputs and separate susieR reference are installed with prusie;
the [quick-start tutorial](../docs/example.md) gives a complete Python workflow.

With the analysis extra installed, execute both posterior-analysis examples:

```sh
python examples/post_analysis.py --output-dir output/post-analysis --case both
```

Each example saves summaries, fits, figures and an offline HTML report.

[Manual comparison scripts](../tools/benchmark/README.md) execute fresh R
validation, timing and separate peak process RSS measurements.
