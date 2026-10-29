# Installation

Build prusie 0.3.0 [from source](#build-from-source) using CPython 3.10–3.13.
NumPy (≥1.26, <3) is its only required Python dependency. Fitting uses a compiled
Rust extension and does not require R.

Install the [analysis extra](#optional-analysis-tools) for sparse input, LD
diagnostics and plotting. Record the package version and source revision with
each analysis.

## Build from source

Install the Rust toolchain pinned in `rust-toolchain.toml` (1.98.1) and your
platform's linker tools. The default native GNU/Linux x86-64 build also uses a
C compiler and `ar` for the vector-math helper. pip uses maturin 1.10.2, pinned in
`pyproject.toml`, as the build backend.

From the source checkout, run these commands in a POSIX shell:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install . --config-settings=build-args="--locked"
```

`--locked` requires the dependency versions recorded in `Cargo.lock`. pip and
Cargo need network access or locally available dependencies. The extension does
not require BLAS development libraries.

To disable the optional vector-math helper, add `portable-math`:

```sh
python -m pip install . --config-settings=build-args="--locked --features portable-math"
```

This uses scalar exponentials and keeps RSS preparation sequential; see
[native threading](implementation.md#native-threading). It does not guarantee
bitwise identical results across platforms; see
[numerical reproducibility](implementation.md#numerical-reproducibility).

## Use a prebuilt wheel

Each tagged [GitHub Release](https://github.com/LucaJiang/prusie/releases)
provides wheels for CPython 3.10–3.13 on these platforms, together with a
source distribution and `SHA256SUMS.txt`:

| Platform | Wheel tag |
| --- | --- |
| Linux x86-64, glibc ≥2.28 | `manylinux_2_28_x86_64` |
| Linux ARM64, glibc ≥2.28 | `manylinux_2_28_aarch64` |
| macOS Intel | `macosx_*_x86_64` |
| macOS Apple silicon | `macosx_*_arm64` |
| Windows x86-64 | `win_amd64` |

Download the wheel for your interpreter and platform, check it against
`SHA256SUMS.txt`, and install it from its directory, for example:

```sh
python -m pip install ./prusie-0.3.0-cp312-cp312-manylinux_2_28_x86_64.whl
```

A wheel contains the compiled extension, so it needs neither Rust nor a
compiler. Only the Linux x86-64 wheel includes the optional vector-math helper;
the other wheels use scalar exponentials, as with
[`portable-math`](#build-from-source). A source distribution requires the
[build toolchain](#build-from-source).

See [contributing](contributing.md#building-packages) for package-building and
validation commands.

## Optional analysis tools

The `analysis` extra adds SciPy (≥1.11) and Matplotlib (≥3.7). From the same
source checkout:

```sh
python -m pip install '.[analysis]' --config-settings=build-args="--locked"
```

If using `portable-math`, include `--features portable-math` in these build
arguments.

SciPy is used for sparse input, LD diagnostics, RSS models with reference
uncertainty or mismatch, multi-panel RSS, and
`prusie.susie_post_outcome_configuration`. Plotting functions in `prusie.viz`
use Matplotlib; `prusie.report` writes offline reports. Both modules are loaded
on demand. See the [analysis and plotting API](api.md#summaries-comparisons-and-reports).

## Run the installed example

To check the installation, fit the packaged synthetic toy example:

```python
import prusie

example = prusie.load_example()
fit = prusie.susie_rss(**example["inputs"], **example["parameters"])
print("Version:", prusie.__version__)
print("Converged:", fit.converged, "IBSS sweeps:", fit.niter)
print("Maximum PIP:", fit.pip.max())
```

`load_example()` reads inputs and metadata from the installed package, so this
example can run from any directory without a source checkout, R or network
access. It needs no optional analysis dependencies. Continue with the
[toy example tutorial](example.md) to interpret PIPs and credible sets and add
figures and an offline report.
