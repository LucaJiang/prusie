# Installation

prusie is installed from source for CPython 3.10–3.13. NumPy (≥1.26, <3)
supports ordinary dense fitting. Install the analysis extra below for sparse
designs, advanced RSS models and auxiliary functions that use SciPy.
Fitting does not call R.

This checkout contains **0.2.4.dev2** (development version). Record the package
version and source revision with each analysis.

## Build from source

Building requires the Rust toolchain pinned in `rust-toolchain.toml` (1.98.1),
a C compiler and archiver and the Python development headers. The build uses maturin 1.10.2
and the locked Cargo dependencies.

The pinned PyO3 0.23.5 supports CPython up to 3.13, and the package metadata
therefore requires Python ≥3.10, <3.14. Support for newer interpreters would
require an updated binding and its validation.

```sh
cd prusie  # source checkout for the version documented here
python -m venv .venv
. .venv/bin/activate
python -m pip install .
```

pip and Cargo need network access or locally cached build dependencies. Install
the pinned Rust toolchain with your usual Rust installer before building. The
extension does not require a BLAS development library. The portable scalar
solver requires no system vector-math library.

## Use a prebuilt wheel

The [v0.2.3rc6 prerelease](https://github.com/LucaJiang/prusie/releases/tag/v0.2.3rc6)
provides a CPython 3.12 wheel for Linux x86-64, tagged `manylinux_2_34`
(glibc ≥2.34), a source distribution and `SHA256SUMS`. These files belong to
that earlier version and do not contain the development version described here.

Download the matching file and verify it against `SHA256SUMS`. For that wheel:

```sh
python -m pip install ./prusie-0.2.3rc6-cp312-cp312-manylinux_2_34_x86_64.whl
```

Installing a compatible wheel requires neither Cargo nor a compiler; a source
distribution requires the build toolchain above. The API documented here
requires a build of this source revision. The version and platform tags of a wheel identify
its code and supported environment.

Earlier snapshots had local installation checks with Python 3.10–3.13.
The current refactor is validated on Linux x86-64 with CPython 3.12;
other interpreter/platform combinations have not been retested for this revision.
See [contributing](contributing.md) for package validation commands.

## Optional analysis tools

To install the plotting and LD-diagnostic dependencies from the same checkout:

```sh
python -m pip install '.[analysis]'
```

The analysis extra adds Matplotlib and SciPy. Ordinary dense Gaussian fitting
needs only NumPy; sparse input, dynamic RSS and posterior outcome configurations
also use SciPy. Reports are written for offline reading, and figures and diagnostics
are computed only when requested.

## Run the installed example

The following works from any directory after installation:

```python
import prusie

example = prusie.load_example()
fit = prusie.susie_rss(**example["inputs"], **example["parameters"])
print(fit.converged, fit.niter, fit.pip.max())
```

The package contains the inputs, metadata and frozen reference fit of this
example, so neither a source checkout, R nor network access is needed to load
and fit it. Continue with the [toy example tutorial](example.md).
