# Installation

Install from the [GitHub source](https://github.com/LucaJiang/prusie). No PyPI
release or public prebuilt wheel is provided. The package declares Python ≥3.10
and NumPy ≥1.26,<3; executed installation checks cover CPython 3.12 on Linux
x86-64 with NumPy 2.2.6. Other declared Python/platform combinations need their
own build and validation.

## Build from a checkout

Install the Rust toolchain pinned in `rust-toolchain.toml` (currently 1.98.1),
a working C linker and Python development headers first. Rustup can manage the
pinned toolchain. The build uses maturin 1.9.6 and the native dependencies in
`Cargo.lock`; pip and Cargo need either cached dependencies or network access.

```sh
git clone https://github.com/LucaJiang/prusie.git
cd prusie
python -m venv .venv
. .venv/bin/activate
python -m pip install .
python examples/check_example.py --output-dir example-results
```

NumPy is the only required Python runtime dependency. R, Rust and a compiler
are not needed to run an already installed package. There is no mandatory BLAS
development library: optional glibc vector math is discovered at runtime, with
a scalar fallback and runtime-guarded AVX2.

The [offline example](example.md) uses repository files, so keep the checkout
when running it. Once installation is complete it needs no network or R.
For building distributable artifacts and running development checks, see
[contributing](contributing.md).
