# Installation and development

The package metadata requires Python ≥3.10 and NumPy ≥1.26,<3. The release
checks use CPython 3.12 on Linux x86-64 and NumPy 2.2.6. Other Python versions and
operating systems are metadata targets, not locally verified binary platforms.
The supplied CPython 3.12 Linux wheel must match your interpreter, architecture
and manylinux tag; pip rejects incompatible tags. No package-index availability
is implied.

From the repository root:

```sh
python -m pip install .
python -m pip install -c requirements-dev.lock '.[test]'
python -m pytest tests
python examples/quickstart.py
python examples/check_example.py --output-dir example-results
```

Source builds use maturin 1.9.6 and the Rust toolchain pinned in rust-toolchain.toml.
Install that toolchain with rustup in your own environment; a C linker and Python
headers are required. Cargo.lock fixes native dependency versions. Dependencies
must be available locally or downloadable by pip/Cargo.

```sh
python -m pip install maturin==1.9.6
maturin build --release --locked --out dist
maturin sdist --out dist
python -m pip install --no-deps dist/prusie-*.whl
```

Alternatively install the supplied source archive with
`python -m pip install path/to/prusie-VERSION.tar.gz`, using its real filename.
The wheel carries the native extension; it needs neither Rust nor R at runtime.
NumPy is the only required Python runtime dependency. pytest is the test extra.
There is no mandatory BLAS development library: optional glibc vector math is
discovered at runtime, with a scalar fallback. AVX2 is runtime guarded; no
unconditional `target-cpu=native` build is required.

The source archive includes the [500-SNP offline check](example.md), its complete
input/reference bundle and the static site. Execute the runner from the checkout
or extracted source archive using the installed package; a wheel installs the
library, while the source archive supplies these repository-level examples.
After installing the library and NumPy, example execution needs no network or R.

The local CI-equivalent checks and artifact member audits are documented in
[testing](testing.md). CI configuration is supplied for later remote execution;
local success does not imply GitHub Actions has run.

To obtain the source and its frozen offline example:

```sh
git clone https://github.com/LucaJiang/prusie.git
cd prusie
python -m pip install .
```

This installs from the checkout; it does not require or imply a PyPI release.
