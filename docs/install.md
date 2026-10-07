# Installation

Install prusie from source with Python ≥3.10. NumPy ≥1.26,<3 is the only Python
runtime dependency; fitting does not invoke R.

## Build from source

Source builds need the Rust toolchain pinned in `rust-toolchain.toml` (1.98.1),
a C linker and Python development headers. The build uses maturin 1.9.6 and
locked Cargo dependencies.

```sh
git clone https://github.com/LucaJiang/prusie.git
cd prusie
python -m venv .venv
. .venv/bin/activate
python -m pip install .
```

pip and Cargo need network access or cached build dependencies. Install the
pinned Rust toolchain using your usual Rust installation before building.
No BLAS development library is required by the extension. Runtime CPU checks
select optional system vector math where available, with a scalar fallback.

## Use an available wheel

When you have a wheel built for your Python and platform, install its actual
file with `python -m pip install PATH_TO_WHEEL`. A wheel installation requires
neither Cargo nor a compiler. Locally prepared artifacts are distinct from a
published binary release; the source route above is the documented distribution
route.

Executed package checks cover CPython 3.12 on GNU Linux x86-64 with NumPy 2.2.6.
Wheel filenames specify their Python ABI and minimum manylinux platform tag.
See [contributing](contributing.md) for reproducible builds and payload checks.

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
