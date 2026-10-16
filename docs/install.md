# Installation

prusie is installed from source for CPython 3.10–3.13. Its only Python run-time
dependency is NumPy (≥1.26, <3); fitting does not call R.

The `main` branch contains the **0.2.4.dev2 development version** described
here; it is not a release. Record the installed version and the source
revision with every analysis. The earlier prerelease files described below
contain different code.

## Build from source

Building requires the Rust toolchain pinned in `rust-toolchain.toml` (1.98.1),
a C linker and the Python development headers. The build uses maturin 1.9.6
and the locked Cargo dependencies.

The pinned PyO3 0.23.5 supports CPython up to 3.13, and the package metadata
therefore requires Python ≥3.10, <3.14. Support for newer interpreters would
require an updated binding and its validation.

```sh
git clone --branch main https://github.com/LucaJiang/prusie.git
cd prusie
python -m venv .venv
. .venv/bin/activate
python -m pip install .
```

pip and Cargo need network access or locally cached build dependencies. Install
the pinned Rust toolchain with your usual Rust installer before building. The
extension does not require a BLAS development library. At run time, the CPU is
checked and the system vector math library is used where available, otherwise
scalar code.

## Use a prebuilt wheel

The [v0.2.3rc6 prerelease](https://github.com/LucaJiang/prusie/releases/tag/v0.2.3rc6)
provides a CPython 3.12 wheel for Linux x86-64, tagged `manylinux_2_34`
(glibc ≥2.34), a source distribution and `SHA256SUMS`. These files belong to
that earlier version and do not contain the development version described here.
On 12 October 2026 the release contained no wheels for macOS, Windows or other
Python ABIs.

Download the matching file and verify it against `SHA256SUMS`. For that wheel:

```sh
python -m pip install ./prusie-0.2.3rc6-cp312-cp312-manylinux_2_34_x86_64.whl
```

Installing a compatible wheel requires neither Cargo nor a compiler; a source
distribution requires the build toolchain above. The API documented here
requires a build from `main`. The version and platform tags of a wheel identify
its code and supported environment; wheels built for local validation are not
published.

The package checks of the published rc6 release covered CPython 3.12 on GNU
Linux x86-64 with NumPy 2.2.6; they do not establish support on platforms for
which no wheel was built. See [contributing](contributing.md) for reproducible
builds and checks of the package contents.

For 0.2.4.dev2, installed Linux x86-64 wheels were also tested locally with
Python 3.10.12, 3.12.2 and 3.13.5 and NumPy 1.26.4, 2.2.6 and 2.5.3 in
supported combinations. These local tests do not imply that development wheels
have been published, nor that other operating systems were tested.

## Optional analysis tools

To install the plotting and LD-diagnostic dependencies from the same checkout:

```sh
python -m pip install '.[analysis]'
```

The analysis extra adds Matplotlib and SciPy. Fitting and posterior summaries
require only NumPy. Reports are written for offline reading, and figures and
diagnostics are computed only when requested.

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
