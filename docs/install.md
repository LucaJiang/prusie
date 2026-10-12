# Installation

Install prusie from source with Python ≥3.10. NumPy ≥1.26,<3 is the only Python
runtime dependency; fitting does not invoke R.

The source instructions below use `main`. This documentation describes the
**0.2.4.dev2 development snapshot**; record the checkout revision with every
analysis because development source can change.

## Build from source

Source builds need the Rust toolchain pinned in `rust-toolchain.toml` (1.98.1),
a C linker and Python development headers. The build uses maturin 1.9.6 and
locked Cargo dependencies.

```sh
git clone --branch main https://github.com/LucaJiang/prusie.git
cd prusie
python -m venv .venv
. .venv/bin/activate
python -m pip install .
```

pip and Cargo need network access or cached build dependencies. Install the
pinned Rust toolchain using your usual Rust installation before building.
No BLAS development library is required by the extension. Runtime CPU checks
select optional system vector math where available, with a scalar fallback.

## Use a prebuilt wheel

The [v0.2.3rc6 prerelease](https://github.com/LucaJiang/prusie/releases/tag/v0.2.3rc6)
contains a CPython 3.12 Linux x86-64 wheel tagged `manylinux_2_34` (glibc ≥2.34),
a source distribution and `SHA256SUMS`. These assets belong to that older version;
they do not contain the development snapshot described here. The release asset
list, checked on 12 October 2026, contains no macOS, Windows or other Python-ABI
wheels.

Download the matching asset and verify it against the release's `SHA256SUMS`.
For that specific wheel:

```sh
python -m pip install ./prusie-0.2.3rc6-cp312-cp312-manylinux_2_34_x86_64.whl
```

Installing a compatible wheel requires neither Cargo nor a compiler. A source
distribution still requires the build toolchain above. Choose source installation
for the development API; a wheel's version and platform tags identify the code
and environment it supports.

The published rc6 package checks cover CPython 3.12 on GNU Linux x86-64 with
NumPy 2.2.6. These checks do not establish support for unbuilt platform wheels.
See [contributing](contributing.md) for reproducible builds and payload checks.

## Optional analysis tools

From the same checkout, install plotting and LD-diagnostic dependencies:

```sh
python -m pip install '.[analysis]'
```

The analysis extra adds Matplotlib and SciPy. Core fitting and deterministic
posterior summaries remain available with NumPy alone. Reports work offline;
plots and diagnostic computation run only when requested.

## Run the installed example

This works from any directory after installation:

```python
import prusie

example = prusie.load_example()
fit = prusie.susie_rss(**example["inputs"], **example["parameters"])
print(fit.converged, fit.niter, fit.pip.max())
```

The package contains the inputs, metadata and frozen reference for this example.
No source checkout, R installation or network is needed to load and fit it.
Continue with the [toy example tutorial](example.md).
