# prusie

Python summary-statistics fine-mapping with a Rust SuSiE/IBSS core. The repository, distribution and Python import are all **prusie**. Software author
and maintainer: Wenxin Jiang. Release candidate 0.2.3rc4. No package-index publication is claimed.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install .
python examples/check_example.py --output-dir example-results
```

Source builds need Rust, a C linker and Python development headers. A matching
prebuilt wheel needs only Python and NumPy. See [installation](docs/install.md)
for tested versions, platform limits and wheel/source commands.

The [500-SNP offline check](docs/example.md) calls the installed public API on
the fixed official synthetic teaching data and compares its PIPs to frozen R
outputs. R is not a runtime dependency. Read the [executed R agreement report](docs/r_agreement.md)
and open [the static documentation site](docs/index.html) for results and
limitations. The example exits nonzero on a substantive mismatch.

```python
import numpy as np
from prusie import susie_rss

# Hand-written analytical example, not biological evidence or a simulated study.
fit = susie_rss(z=[4., .5, -.2],
    R=[[1., .2, 0.], [.2, 1., .1], [0., .1, 1.]], n=191,
    L=2, max_iter=100, tol=.001, estimate_residual_variance=False,
    variant_ids=['a', 'b', 'c'])
print(fit.pip, fit.converged, fit.sets['cs_index'])
```

Use signed correlation **r**, with exactly the same SNP and counted-allele order
as signed z=beta/SE. Do not supply r². The caller supplies sample size, ancestry,
build and allele provenance. This package does not align variants or repair LD.

The supported statistical reference is official susieR 0.16.6, commit
`8e56a8e038e989856d106d9ca5175cc664fea9d2`. The numerical acceptance contract is
PIP absolute error ≤1e-5, rtol=0, plus valid probabilities and unchanged inputs
and model semantics. It is an empirical validation contract, not a bound for
all possible inputs. Intermediate arrays and credible sets may differ.

Read [inputs and priors](docs/inputs.md), [API](docs/api.md),
[result schema](docs/results.md), [reference compatibility](docs/compatibility.md),
[performance limits](docs/performance.md), and [testing](docs/testing.md).
See [release notes](docs/release_notes.md), [changelog](CHANGELOG.md),
[license](LICENSE), [attribution](THIRD_PARTY_NOTICES.md), and [citation](CITATION.cff).
The separate pycoloc package consumes these fits for colocalisation; it is not
needed for fine-mapping.

See the [name migration and license status](docs/migration.md) for the public rename, saved-result compatibility and retained licensing constraints.

Source repository: [prusie](https://github.com/LucaJiang/prusie). For Pages setup, see [the guide](docs/pages.md).
Related package: [pycoloc](https://github.com/LucaJiang/pycoloc).
