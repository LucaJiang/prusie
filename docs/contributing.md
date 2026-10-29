# Contributing

This guide covers the development setup, tests, documentation builds, packaging
and benchmark scripts for prusie. Before changing the numerical implementation,
read the [statistical model](model.md) and
[numerical validation criteria](numerical_accuracy.md).

## Development environment

The pinned development dependencies in `requirements-dev.lock` target
CPython 3.12 on Linux. prusie supports Python 3.10–3.13. Install the Rust
toolchain in `rust-toolchain.toml`, a C compiler and an archiver. Run the
following from the repository root in a POSIX shell:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.lock
python -m pip install '.[test,analysis]'
```

After source changes, rerun `python -m pip install '.[test,analysis]'` to
update the installed package. Alternatively, use
`maturin develop --release --locked` in the active virtual environment for a
development build.

## Lint and tests

Run these checks from the repository root in the development environment:

```sh
ruff check
ruff format --check
cargo fmt --all --check
cargo clippy --locked --workspace --all-targets -- -D warnings
python -m pytest tests
python -m pytest tools/benchmark/_internal
cargo test --locked -p prusie-core -- --test-threads=1
```

Tests compare prusie with frozen susieR outputs, a restricted independent
NumPy reference and analytical cases, and check input validation, plotting
and reports. Preserve frozen inputs and reference outputs, and add
regression tests for numerical fixes.

The CI workflow defines builds and tests on Linux, macOS and Windows, with
an additional Linux `portable-math` build. For an optional live R check, manually run the
`tests-and-packages` workflow with `r_reference=true`. It installs the
[pinned susieR 0.16.6 reference](model.md#computational-compatibility) and
compares complete `alpha` matrices from individual-level, sufficient-statistic
and RSS fits with the frozen R outputs, without regenerating the references.

To run the R check locally, install R, `jsonlite` and the pinned susieR commit
through `remotes::install_github`, then run:

```sh
Rscript --vanilla tools/check_r_reference.R
```

The script verifies the installed susieR version and its `RemoteSha` commit.

## Editing and building documentation

Edit Markdown in `docs/` and commit the rebuilt HTML alongside it. Install
Node.js (CI uses 24.18.0), then run these commands from the repository root:

```sh
python -m pip install -r tools/docs-requirements.txt
python tools/build_docs.py
python -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Open `http://127.0.0.1:8000/` and inspect the changed pages in a browser.
The build renders math locally with vendored KaTeX, verifies its assets,
and checks local files and HTML section links. It writes `docs/.nojekyll`
for static GitHub Pages hosting.

Use the generators to update marked numerical blocks:
`tools/model_examples.py` updates explanatory calculations, and
`tools/generate_results.py` updates measurement tables and text from saved
records. See [reproducibility](reproducibility.md) before regenerating
numerical content.

## Building packages

From the repository root, build a release wheel and source distribution
(sdist), then verify that the sdist builds into a wheel:

```sh
maturin build --release --locked --out dist
maturin sdist --out dist
python -m pip wheel --no-deps dist/*.tar.gz -w sdist-wheel
```

Install the built wheel in a fresh virtual environment and run
`python examples/quickstart.py` from the checkout. The sdist includes Python
and Rust sources, tests and frozen test data; documentation, examples and
benchmark tools remain in the checkout.

Retain `LICENSE`, `THIRD_PARTY_NOTICES.md` and the
notices in adapted source and frozen data when distributing packages.

Pushing a tag `vX.Y.Z` runs `.github/workflows/release.yml`. It builds wheels
for CPython 3.10–3.13 on Linux (x86-64 and ARM64, manylinux 2.28), macOS
(Intel and Apple silicon) and Windows x86-64, plus the sdist. It installs
each wheel without a source build and runs the test suite on both the oldest
and newest supported Python, then attaches the packages and `SHA256SUMS.txt`
to the GitHub Release for that tag. Tags ending in `aN`, `bN` or `rcN` are
marked as prereleases. Running the workflow by hand builds and tests the
packages without publishing them.

## Benchmark scripts

`tools/benchmark/reproduce.py` runs new comparisons with pinned susieR;
helper drivers live in `tools/benchmark/_internal/`. The
[comparison guide](https://github.com/LucaJiang/prusie/blob/main/tools/benchmark/README.md)
describes inputs and options. Use `python tools/benchmark/v030.py figures`
to redraw the 0.3.0 figures from retained records without fitting; see
[reproducibility](reproducibility.md).

Before measuring an optimization against its immediate parent, freeze its
case panel, applicability rule, fitting settings and hardware. Require
[numerical validation](numerical_accuracy.md) with unchanged tolerances.
Include GWAS and eQTL cases with large variant counts and many IBSS sweeps,
and report applicable case counts, wins, ties, regressions, failures and
repeat variability.

A runtime optimization needs more than half of applicable cases to have a
median paired runtime ratio below one (candidate time ÷ parent time for the
complete fitting call). A memory-only optimization needs lower median
incremental peak resident set size (RSS) in more than half of applicable cases.

Both require the predeclared runtime regression veto to pass across the full
panel, including individual cases and strata defined by data type,
variant-count tertile and parent iteration count. The majority and veto
decisions must each be reproduced in at least 95% of resamples of paired
rounds. Keep failed or unresolved optimizations experimental.
