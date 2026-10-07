# Examples

Run `python examples/quickstart.py` after installation to analyze the complete
500-SNP D3 teaching dataset and inspect PIPs, credible sets and convergence.
The files in `data/` are the fixed, licensed synthetic teaching fixture from
official coloc. See `data/README.md` for provenance.

Run `python examples/check_example.py --output-dir example-results` to validate
all seven predefined cases against frozen official susieR 0.16.6 outputs.
The one-iteration case is a separate nonconvergence diagnostic. The checker
runs offline with the installed package and NumPy; it preserves reference bytes.

The [worked example](../docs/example.md) includes optional pycoloc pairing of
existing fits. [Manual comparison scripts](../tools/benchmark/README.md) execute
fresh R validation, timing and separate peak process RSS measurements.
