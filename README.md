# prusie

SuSiE summary-statistics fine-mapping in Python with a Rust inference core.
Fit signed association statistics and linkage disequilibrium (LD) to estimate
variant inclusion probabilities and credible sets. R is not needed at runtime.

## Install and run

Source installation requires Python ≥3.10, the Rust toolchain specified in
`rust-toolchain.toml`, a C linker and Python development headers. See
[installation](docs/install.md) for dependencies and tested platforms.

```sh
git clone https://github.com/LucaJiang/prusie.git
cd prusie
python -m venv .venv
. .venv/bin/activate
python -m pip install .
python examples/check_example.py --output-dir example-results
```

The included check runs offline after installation. It fits seven cases from a
frozen, explicitly **synthetic 500-SNP teaching fixture** and compares PIPs with
official susieR outputs. It prints PASS/FAIL and writes a detailed JSON report.
Source installation is the supported download route; no PyPI release or public
wheel download is provided.

## Fit a region

```python
import prusie

# Small analytical illustration; these are not biological data.
fit = prusie.susie_rss(
    z=[4., .5, -.2],
    R=[[1., .2, 0.], [.2, 1., .1], [0., .1, 1.]],
    n=191, L=2, max_iter=100, tol=.001,
    estimate_residual_variance=False, variant_ids=['a', 'b', 'c'],
)
print(fit.pip)
print(fit.converged, fit.niter, fit.sets['cs_index'])
```

Use signed LD correlation **r**, not r², in exactly the same SNP and counted-allele
order as signed z = beta/SE. Supply sample-size and allele/build/ancestry provenance;
the package does not harmonize variants or repair LD. Alternatives accept
beta/SE or centered sufficient statistics. Read [inputs](docs/inputs.md) before
using your own data.

`pip` is the marginal inclusion probability for each variant. Credible sets use
original component IDs and report actual posterior mass and purity. Inspect
convergence and warnings: no credible set means unavailable set evidence, and
a reproducible nonconverged fit is still nonconverged.

## Documentation and validation

Read the [documentation site](https://lucajiang.github.io/prusie/), or the
Markdown [example](docs/example.md), [API](docs/api.md),
[results](docs/results.md) and [supported scope](docs/compatibility.md).
The [executed R agreement report](docs/r_agreement.md) compares with susieR 0.16.6
using PIP absolute error ≤1e-5, rtol=0. This is a tested-case criterion, not a
universal bound or proof of identical Bayes factors, credible sets or downstream
posteriors. [Performance evidence](docs/performance.md) includes slower cases
and identifies its earlier native baseline; it is not an R speedup benchmark.

Maintained by Wenxin Jiang. See [contributing](docs/contributing.md),
[changelog](CHANGELOG.md) and [citation](CITATION.cff).
GPL-3.0-or-later; [license](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md)
retain the upstream code and teaching-data terms.
