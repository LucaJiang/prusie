# Installation

## Install a prepared wheel

For a locally supplied Linux x86-64, CPython 3.12 wheel:

```sh
python -m pip install ./prusie-0.2.3rc4-cp312-cp312-manylinux_2_34_x86_64.whl
```

The wheel contains the Python interface and compiled Rust extension. This
installation needs Python and NumPy, without Rust, Cargo or a compiler.
The platform tag specifies glibc 2.34 or later; use the asset matching your
Python and platform. Public wheel downloads are not yet available.
Source is available from the repository below.

## Install from source

The package declares Python ≥3.10 and NumPy ≥1.26,<3. Executed installation
checks cover CPython 3.12 on Linux x86-64 with NumPy 2.2.6. Build checks for
other Python versions and platforms are separate from this evidence.

Source builds need the Rust toolchain pinned in `rust-toolchain.toml`
(currently 1.98.1), a C linker and Python development headers. The build uses
maturin 1.9.6 and the native dependency versions in `Cargo.lock`.

```sh
git clone https://github.com/LucaJiang/prusie.git
cd prusie
python -m venv .venv
. .venv/bin/activate
python -m pip install .
python examples/quickstart.py
```

pip and Cargo need cached dependencies or network access during installation.
NumPy is the only required Python runtime dependency. An installed package
runs without R; optional glibc vector math uses runtime CPU checks and a scalar
fallback. No BLAS development library is required for the native extension.

## Check your installation

Keep the repository checkout for the bundled examples. After installation,
the following check runs offline using all seven fixed examples:

```sh
python examples/check_example.py --output-dir example-results
```

Read the [analysis example](example.md) and [input guide](inputs.md) before
fitting your own data. The [contributor guide](contributing.md) covers source
builds, tests and locally prepared distribution artifacts.
