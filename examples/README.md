# Offline reference example

From this repository root, after installing prusie and NumPy:

```sh
python examples/check_example.py --help
python examples/check_example.py --output-dir example-results
```

All seven cases call the installed public API with the full fixed 500-SNP
upstream synthetic teaching fixture. No R or network is used. PASS/FAIL lines
and the JSON report give PIP errors, posterior validity, convergence and
credible-set diagnostics. A substantive failure exits nonzero. Version strings
are recorded transparently and are not numerical acceptance criteria.

See [the example guide](../docs/example.md),
[executed R agreement](../docs/r_agreement.md) and
[the data provenance and pinned regeneration recipe](data/README.md).
`--output-dir` must be outside `data/`; expected outputs are never regenerated
by the check. The smaller `quickstart.py` is a handwritten analytical API
illustration and is not biological evidence.
