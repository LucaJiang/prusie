# Installation

prusie 0.3.1 supports CPython 3.10 and newer (GIL builds). NumPy (≥1.26, <3)
is its only required Python dependency. Fitting uses a compiled Rust extension
and does not require R. Record the package version and source revision with
each analysis.

## PyPI

```sh
pip install prusie
```

For sparse input, LD diagnostics and plotting, add the [analysis extra](#optional-analysis-tools):

```sh
pip install 'prusie[analysis]'
```

## Use a prebuilt wheel

The same `cp310-abi3` wheel is tested on CPython 3.10–3.15. Free-threaded builds
(3.13t, 3.14t and 3.15t) are not supported by these wheels.

Download the wheel for your platform from [GitHub Release v0.3.1](https://github.com/LucaJiang/prusie/releases/tag/v0.3.1):

| Platform | Wheel filename |
| --- | --- |
| Linux x86-64, glibc ≥2.28 | [`prusie-0.3.1-cp310-abi3-manylinux_2_28_x86_64.whl`](https://github.com/LucaJiang/prusie/releases/download/v0.3.1/prusie-0.3.1-cp310-abi3-manylinux_2_28_x86_64.whl) |
| Linux ARM64, glibc ≥2.28 | [`prusie-0.3.1-cp310-abi3-manylinux_2_28_aarch64.whl`](https://github.com/LucaJiang/prusie/releases/download/v0.3.1/prusie-0.3.1-cp310-abi3-manylinux_2_28_aarch64.whl) |
| macOS Intel | [`prusie-0.3.1-cp310-abi3-macosx_10_12_x86_64.whl`](https://github.com/LucaJiang/prusie/releases/download/v0.3.1/prusie-0.3.1-cp310-abi3-macosx_10_12_x86_64.whl) |
| macOS Apple silicon | [`prusie-0.3.1-cp310-abi3-macosx_11_0_arm64.whl`](https://github.com/LucaJiang/prusie/releases/download/v0.3.1/prusie-0.3.1-cp310-abi3-macosx_11_0_arm64.whl) |
| Windows x86-64 | [`prusie-0.3.1-cp310-abi3-win_amd64.whl`](https://github.com/LucaJiang/prusie/releases/download/v0.3.1/prusie-0.3.1-cp310-abi3-win_amd64.whl) |

Download the attached [SHA256SUMS.txt](https://github.com/LucaJiang/prusie/releases/download/v0.3.1/SHA256SUMS.txt) alongside the wheel
and check its SHA256 before installing. For example, on Linux x86-64 with GNU
`sha256sum`:

```sh
curl -fLO https://github.com/LucaJiang/prusie/releases/download/v0.3.1/prusie-0.3.1-cp310-abi3-manylinux_2_28_x86_64.whl
curl -fLO https://github.com/LucaJiang/prusie/releases/download/v0.3.1/SHA256SUMS.txt
sha256sum --check --ignore-missing SHA256SUMS.txt
pip install ./prusie-0.3.1-cp310-abi3-manylinux_2_28_x86_64.whl
```

The checksum command must report `OK` for the downloaded wheel. On macOS use
`shasum -a 256 <wheel>`; in Windows PowerShell use `Get-FileHash <wheel> -Algorithm SHA256`,
and compare the value with the matching entry in `SHA256SUMS.txt`.

A direct-URL install is also available (Linux x86-64 example):

```sh
pip install https://github.com/LucaJiang/prusie/releases/download/v0.3.1/prusie-0.3.1-cp310-abi3-manylinux_2_28_x86_64.whl
```

A wheel contains the compiled extension, so it needs neither Rust nor a
compiler. Only the Linux x86-64 wheel includes the optional vector-math helper;
the other wheels use scalar exponentials, as with
[`portable-math`](#build-from-source). A source distribution requires the
[build toolchain](#build-from-source).

## Build from source

Source installation needs the Rust toolchain pinned in `rust-toolchain.toml`
(currently 1.98.1) and your platform's linker tools. With rustup installed and
`cargo` on `PATH`, rustup selects and downloads the pinned toolchain automatically
if it is missing. The default GNU/Linux x86-64 build also needs a C compiler and
`ar` for the vector-math helper. pip installs the pinned maturin build backend.
The maturin version is 1.15.0, pinned in `pyproject.toml`.

Clone the repository and install into your Python environment:

```sh
git clone https://github.com/LucaJiang/prusie.git
cd prusie
git checkout v0.3.1  # optional: pin the released version
pip install .
```

To require the exact Cargo dependency versions, install with:

```sh
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

See [contributing](contributing.md#building-packages) for package-building and
validation commands.

## Optional analysis tools

The `analysis` extra adds SciPy (≥1.11) and Matplotlib (≥3.7):

```sh
python -m pip install 'prusie[analysis]'
```

For a source build, use `'.[analysis]'` in place of `.` in the build command above.

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
