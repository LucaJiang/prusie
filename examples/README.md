# Examples

Run `python examples/quickstart.py` after installation to analyze the independent
packaged 500-variant synthetic toy example and inspect PIPs, credible sets and
convergence. Its inputs and separate susieR reference are installed with prusie;
the [quick-start tutorial](../docs/example.md) gives a complete Python workflow.

Run `python examples/check_example.py --output-dir example-results` to validate
all seven predefined cases against frozen official susieR 0.16.6 outputs.
The one-iteration case is a separate nonconvergence diagnostic. The checker
runs offline with the installed package and NumPy; it preserves reference bytes.
These older regression cases use the fixed, licensed synthetic toy fixture from
official coloc in `data/`; see [its provenance](data/README.md).

With the analysis extra installed, execute both posterior-analysis examples:

```sh
python examples/post_analysis.py --output-dir output/post-analysis --case both
```

Each example saves summaries, fits, figures and an offline HTML report.

[Manual comparison scripts](../tools/benchmark/README.md) execute fresh R
validation, timing and separate peak process RSS measurements.
