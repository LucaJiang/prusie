# Tests and reproducibility

From a clean checkout, install `.[test]`, then run `python -m pytest tests`.
Public tests use analytical fixtures and the fixed official synthetic teaching
datasets, including frozen outputs from pinned official susieR0.16.6. Fixture
provenance states the upstream commit and settings. No private genotype/LD/summary data, R executable or
original development workspace is needed.

```sh
python -m pip install -c requirements-dev.lock '.[test]'
python -m pytest tests
PRUSIE_SER_MATH=scalar python -m pytest tests
cargo test --locked --lib --no-default-features -- --test-threads=1
cargo test --locked --lib --no-default-features --features vector-math -- --test-threads=1
python examples/quickstart.py
python examples/check_example.py --output-dir example-results
python -m pip install -r tools/docs-requirements.txt
python tools/build_docs.py --check
python tools/check_docs.py
python tools/check_checkout.py --output-dir .ci-check
```

Native tests may need the interpreter library on the system linker search path;
use the same Python toolchain that builds the extension. `PRUSIE_SER_MATH=scalar`
forces the optional SER vector-math fallback. A build without vector-math also
checks the portable path. Runtime CPU-feature-specific tests may report an
explicit skip when that feature is absent; ordinary mathematical tests remain.

The private release validation separately accounts for100 regions/200 trait
cases, uses max_iter100 and PIP atol1e-5/rtol0, and records intermediate/CS and
downstream colocalisation diagnostics. Public unit success is not a substitute
for that evidence, and unavailable private datasets are not silently counted
as passing tests. Public CI uses only redistributable fixtures.

`test_offline_example.py` executes all seven public API cases with R absent from
PATH and from a working directory outside the repository. It also alters an
input value and a reference value, requiring useful checksum failures; corrupts
and rehashes a reference PIP, requiring a numerical failure; and proves a changed
snapshot version alone does not cause a false numerical mismatch.

`check_checkout.py` copies source into a new independent directory, executes the
installed-package example with an empty executable PATH, and checks generated
HTML and links there. It writes preserved logs and a JSON record. This is a
local source-directory check, not proof of a remote GitHub Actions run or browser
visual review. The output directory must be new.

The checked-in workflow builds wheel/sdist, runs tests and examples and audits
docs from this repository. It has no publication step or secrets. Local job
results are recorded in release notes; GitHub status requires an actual remote run.
