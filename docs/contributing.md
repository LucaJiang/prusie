# Contributing

## Development environment

Use Python 3.12 for the pinned development environment, the Rust toolchain
in `rust-toolchain.toml`, and a C compiler. The package supports Python
3.10–3.13. From a checkout:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.lock
python -m pip install '.[test,analysis]'
```

After source changes, reinstall the package or use
`maturin develop --release --locked` to update the development build.

## Lint and tests

```sh
ruff check
ruff format --check
cargo fmt --check
cargo clippy --locked --all-targets --no-default-features -- -D warnings
cargo clippy --locked --all-targets --no-default-features --features vector-math -- -D warnings
python -m pytest tests
cargo test --locked --lib --no-default-features -- --test-threads=1
cargo test --locked --lib --no-default-features --features vector-math -- --test-threads=1
```

Tests cover frozen susieR outputs, an independent NumPy reference, analytical
cases, input validation, plotting and reports. Preserve frozen inputs and
reference outputs; add regression tests for numerical fixes.

## Editing and building documentation

Install Node.js (tested with 24.18.0), then:

```sh
python -m pip install -r tools/docs-requirements.txt
python tools/build_docs.py
python -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Edit Markdown and commit the rebuilt HTML alongside it; GitHub Pages serves
`docs/` with `.nojekyll`. The build renders math locally with vendored KaTeX and
checks local link targets. Inspect changed pages in a browser.
`tools/model_examples.py` updates explanatory calculations;
`tools/generate_results.py` regenerates tables and figures from saved measurements.
See [reproducibility](reproducibility.md) before updating either.

## Building packages

```sh
maturin build --release --locked --out dist
maturin sdist --out dist
python -m pip wheel --no-deps dist/*.tar.gz -w sdist-wheel
```

Try the wheel in a fresh environment and run `examples/quickstart.py`.
The sdist contains source, tests and their data; website and benchmark tools
remain in the Git checkout. Retain `LICENSE`, `THIRD_PARTY_NOTICES.md` and the
notices in adapted source and frozen data when distributing packages.

## Benchmark scripts

[`tools/benchmark`](https://github.com/LucaJiang/prusie/tree/main/tools/benchmark)
contains the R comparison and measurement scripts. Keep numerical tolerances fixed.
Measure each proposed optimization against its parent on cases chosen before
inspecting outcomes, and report wins, ties, regressions and repeat variability.
Merge it only if more than half of those cases have lower median complete-call
runtime; otherwise retain it as experimental.
