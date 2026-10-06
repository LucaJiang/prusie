# Contributing

Start with the checkout, environment and compiler requirements in
[installation](install.md). Keep inference changes separate from documentation
changes, preserve sequential component updates, and validate affected model paths.

## Tests and packages

```sh
python -m pip install -c requirements-dev.lock '.[test]' maturin==1.9.6
python -m pytest tests
PRUSIE_SER_MATH=scalar python -m pytest tests
cargo test --locked --lib --no-default-features -- --test-threads=1
cargo test --locked --lib --no-default-features --features vector-math -- --test-threads=1
python examples/quickstart.py
python examples/check_example.py --output-dir example-results
maturin build --release --locked --out dist
maturin sdist --out dist
```

The build writes local artifacts into `dist/`; install your wheel by its actual
filename. A wheel contains the library, while the source distribution includes
repository examples. These commands do not publish artifacts. Native tests may
need the Python interpreter library on the linker search path. Runtime CPU
feature checks may explicitly skip a path when the feature is absent.

`tests/test_offline_example.py` covers all seven fixed cases without R, corrupted
input/reference checksums, a rehashed numerical mismatch and a version-only
change. A changed label alone must not cause numerical failure.
`python tools/check_checkout.py --output-dir .ci-check` copies source into a new
directory and checks installed-package examples and docs there with R absent
from the executable path; the output directory must be new. This reuses the
installed environment rather than testing a fresh installation.

Each proposed optimization must improve median complete-call time for more than
half of a fixed, outcome-independent applicable panel against its appropriate
parent. Publish wins, ties, regressions and repeat variability. Correctness,
memory savings and aggregate time alone do not establish that runtime gate.
See [performance](performance.md) for the existing measurements and limits.

## Documentation

Markdown in `docs/` is the source for the committed static HTML. From the
repository root:

```sh
python -m pip install -r tools/docs-requirements.txt
python tools/build_docs.py
python tools/build_docs.py --check
python tools/check_docs.py
python -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Open `http://127.0.0.1:8000/`, or open `docs/index.html` directly. Navigation,
code and tables work without JavaScript; Tab reaches the skip link and scrollable
content. Links are relative so the site also works below a repository subpath.
`--check` detects stale HTML; the link checker validates local pages, anchors,
assets and metadata. Review desktop and mobile rendering after layout changes.
When merging a page, remove its obsolete HTML and update inbound links too.

The published site is [prusie documentation](https://lucajiang.github.io/prusie/).
An owner can configure it under **Settings → Pages → Build and deployment**:
**Deploy from a branch → main → /docs → Save**. `docs/.nojekyll` serves the
prebuilt files. Committing source and deploying Pages are separate operations;
check deployment status after publishing changes.

## Scientific checks and provenance

The [R agreement report](r_agreement.md) identifies the measured versions,
parameters, tolerances, corrected behavior and limits. Keep official-R outputs,
mathematical expectations and native regression snapshots distinct. Never
refresh frozen outputs merely to make a failed check pass. Reference regeneration
uses a separate directory and the pinned recipe in
[the data README](https://github.com/LucaJiang/prusie/blob/main/examples/data/README.md).
See [source mapping](reference_mapping.md) for implementation responsibilities.

Retain GPL and third-party notices with code and data. Cite the statistical
methods as well as the software version. Tests use redistributable analytical
inputs and the fixed synthetic teaching fixture; they do not need private study
data or an R installation. The workflow in `.github/workflows/ci.yml` runs local
build/test/doc commands on GitHub; consult the actual workflow run for CI status.
