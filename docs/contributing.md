# Contributing

Use the build requirements in [installation](install.md). Changes to inference
should retain sequential component updates and verify every affected model path
against independently identified references. Keep numerical tolerances and
reference changes explicit.

## Tests and packages

```sh
python -m pip install -c requirements-dev.lock '.[test]' maturin==1.9.6
python -m pip install -r tools/docs-requirements.txt -r tools/results-requirements.txt
python -m pytest tests
PRUSIE_SER_MATH=scalar python -m pytest tests
cargo test --locked --lib --no-default-features -- --test-threads=1
cargo test --locked --lib --no-default-features --features vector-math -- --test-threads=1
python examples/quickstart.py
python examples/check_example.py --output-dir regression-results
python tools/check_checkout.py --output-dir checkout-check
maturin build --release --locked --out dist
maturin sdist --out dist
```

The installed-resource test fits the independent toy example and compares
it with its packaged R reference. The older attributed fixture supplies seven
unchanged regression cases, including partial-zero priors, empty credible sets
and an iteration-limit diagnostic. Native tests exercise exact-column discovery,
SER workspaces, array boundaries, scalar/vector math and supported variance paths.

`tools/check_artifacts.py --wheel FILE --sdist FILE --work-dir NEW_DIRECTORY`
inspects license/data payloads, installs binary wheels in clean environments
without Rust on PATH, runs examples from an independent directory, rebuilds the
sdist and checks the rebuilt wheel. On Linux it also records auditwheel output.
The CI workflow runs these checks; actual workflow status is separate from local
verification.

## Documentation and figures

```sh
python tools/generate_example.py --check
python tools/generate_results.py
python tools/generate_results.py --check
python tools/build_docs.py
python tools/build_docs.py --check
python tools/check_docs.py
python -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Markdown is the source for generated HTML. Pinned Markdown and Pygments produce
local token-level highlighting with language guessing disabled. Code copying
returns the original code, without line numbers. Mathematical notation is static
Unicode and remains readable offline. Review desktop and mobile rendering after
layout changes. Check the result generator before the HTML generator so tables,
figures and rendered pages stay synchronized.

`docs/assets/benchmark_records.json` is the normalized retained-measurement
source. The generator produces both README and Overview highlights, detailed
tables, machine summaries and SVG/PDF figures. `--check` detects stale outputs.
[Reproducibility](reproducibility.md) describes what can be regenerated from
public records and which fits require separately obtained inputs.

## Scientific references and attribution

Use [source mapping](reference_mapping.md) and
[implementation](implementation.md) when changing a computational path. Preserve
independent frozen references; regenerate a new reference into a separate
location and explain its source. Whole-call comparisons and isolated mechanism
measurements answer different questions.

Keep the GPL license, BSD susieR notices, R optimizer attribution and dataset
licenses in both source and binary distributions. Cite the statistical methods
and identify the software version as described in [citation](citation.md).
