# Contributing

A useful contribution makes a statistical behavior easier to understand or
maintain, with a focused example that distinguishes the intended result from
plausible mistakes. Start from the [model](model.md) and
[implementation](implementation.md), then add a regression test alongside the
path being changed. Keep reference identities and numerical tolerances explicit.

## Development environment

Source builds use the Rust toolchain in `rust-toolchain.toml`, a C compiler and
Python development headers. The local checks use Python 3.12 and NumPy 2.2.6.
From a checkout, create an environment and install the package and development
tools:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -c requirements-dev.lock '.[test]' maturin==1.9.6
python -m pip install -r tools/docs-requirements.txt -r tools/results-requirements.txt
```

Reinstall the package after editing Python or Rust source so tests exercise the
compiled checkout. For frequent Rust work, `maturin develop --release --locked`
installs a development build in the active environment. Ordinary inference
installs depend only on NumPy; documentation and plotting dependencies stay in
their separate requirements files.

## Source structure and tests

`src/prusie` contains the public API, result object and posterior helpers.
`bindings` validates borrowed arrays, prepares matrix operations and transfers
owned outputs. `rust/core.rs` contains SER, matrix products and the IBSS schedule;
`rust/optimizer.rs` and `rust/vector_math.*` supply optimization and optional
vector exponentials. The [source navigation](implementation.md#source-navigation)
connects functions to their responsibilities.

```sh
python -m pytest tests
PRUSIE_SER_MATH=scalar python -m pytest tests
cargo test --locked --lib --no-default-features -- --test-threads=1
cargo test --locked --lib --no-default-features --features vector-math -- --test-threads=1
cargo fmt --all --check
cargo clippy --locked --lib --tests --no-default-features --features vector-math
python examples/quickstart.py
python examples/check_example.py --output-dir regression-results
python tools/check_checkout.py --output-dir checkout-check
```

The Python reference tests include fixed Gaussian analytical cases and frozen
R outputs. Layout tests cover C/F storage, strided and read-only views,
alignment and conversion callbacks. Rust tests retain scalar mathematical
oracles and directly observe the matrix-discovery branch at its real size
threshold. When adding a branch, test both its execution condition and an
independent expected result. Keep scalar fallbacks useful on CPUs without the
optional vector instructions. Independent regions may run in separate
processes; component updates within one fit remain sequential.

## Editing and building documentation

Markdown is the body source; generated HTML is committed alongside it. The
build uses Python-Markdown 3.7, Pygments 2.19.2, Arithmatex 10.16.1 and vendored
KaTeX 0.16.22. Node.js is needed only at build time (tested with Node.js 24.18.0).
Once these tools are installed, rebuilding math requires no network access.

```sh
python tools/generate_example.py --check
python tools/model_examples.py
python tools/model_examples.py --check
python tools/generate_results.py
python tools/generate_results.py --check
python tools/build_docs.py
python tools/build_docs.py --check
python tools/check_docs.py
python -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Open `http://127.0.0.1:8000/`. Use `$...$` for inline math and a separate
`$$...$$` block for display math, with blank lines around the block. Arithmatex
protects TeX during Markdown parsing; the builder renders only those protected
nodes into HTML plus accessible MathML. Invalid TeX stops the build with its
page and formula. Fenced Python, R and shell blocks retain token highlighting
and their original copyable text. The [math renderer resources and lock](https://github.com/LucaJiang/prusie/tree/main/tools/vendor/katex)
identify the matching renderer, styles, fonts and license.

`model_examples.py` checks and generates the small explanatory calculations.
`generate_results.py` rebuilds the measured summaries and figures from
`docs/assets/benchmark_records.json`; it also updates marked blocks in the
README and Overview. These scripts must precede HTML generation. The
[reproducibility guide](reproducibility.md) distinguishes saved-record
regeneration from fitting the underlying inputs.

After layout changes, inspect wide and narrow browser views, local HTTP,
`/prusie/` subpaths and offline file reading with JavaScript disabled. Long
formulas and code should scroll inside their own region, without widening the
page. Navigation, text, figures and mathematical rendering remain available
without JavaScript; the copy button is optional.

## Packages and scientific attribution

```sh
maturin build --release --locked --out dist
maturin sdist --out dist
python -m pip install 'auditwheel>=6,<7'
python tools/check_artifacts.py --wheel PATH_TO_WHEEL \
  --sdist PATH_TO_SDIST --work-dir artifact-check
```

Substitute the wheel and sdist just produced. The artifact checker inspects
license and data payloads, installs wheels in clean environments without Rust
on PATH, runs examples from an independent directory, rebuilds the sdist and
checks its wheel. Linux checks also report auditwheel compatibility. Use a new
output directory so earlier results are retained. CI runs the test, docs and
package checks; its observed status is separate from a local run.

Preserve frozen numerical inputs, raw reference records and attribution. If a
new reference is needed, generate it separately and identify its source and
parameters. Runtime optimizations should compare complete calls on an
outcome-independent case panel with retained repeats, reporting wins, ties,
regressions and variability. An optimization enters the main implementation
only when more than half of applicable cases have lower median runtime;
correctness or allocation savings alone do not establish that result.

Retain GPL package terms, susieR's BSD notices, the R optimizer's attribution
and data licenses. KaTeX's local documentation resources carry their own MIT
notice. Cite the methods and identify the source revision as described in
[Citation](citation.md). Focused changes and their tests can be proposed through
the repository's issue and pull-request workflow.
