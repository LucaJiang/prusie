<p class="eyebrow">Python + Rust / SuSiE summary statistics</p>

# Fine-map with a traceable result

<p class="intro">prusie fits the Gaussian SuSiE summary-statistics model with a native Rust core. Work with signed association statistics and LD, inspect PIPs and credible sets, and check your installation against frozen official-R results.</p>

<div class="cards">
<a class="card" href="install.html"><span>01 / Start</span><strong>Install the package</strong><small>Python and NumPy at runtime. A Rust toolchain for source builds.</small></a>
<a class="card" href="example.html"><span>02 / Verify</span><strong>Run the 500-SNP check</strong><small>A fixed upstream synthetic teaching example. Runs offline, without R.</small></a>
<a class="card" href="r_agreement.html"><span>03 / Understand</span><strong>Read the R agreement report</strong><small>Executed cases, absolute errors, component identities and explicit limitations.</small></a>
</div>

## Quick start

From a source checkout or extracted source archive, after [installation](install.md):

```sh
python examples/check_example.py --output-dir example-results
```

The check fits the bundled data using the installed package, verifies input and
reference checksums, compares PIPs to frozen official-R results, checks posterior
validity and records your runtime. It prints PASS or FAIL and exits nonzero on a
substantive mismatch. A changed version string alone is not a numerical failure.

## What a fit returns

| Output | Interpretation |
| --- | --- |
| `fit.pip` | Marginal variant inclusion probability across retained effects |
| `fit.sets` | Credible sets indexed by original component, with actual posterior mass and purity |
| `fit.converged`, `fit.niter` | The fit's stopping status and iteration count |
| `fit.lbf_variable` | Component-level log Bayes factors for downstream colocalisation |

Use [the result schema](results.md) to interpret these quantities. A missing
credible set is unavailable evidence; it does not mean a zero PIP or zero
colocalisation probability.

## Supply aligned evidence

Use signed LD correlation **r**, with the same SNP and counted-allele order as
signed z = beta / SE. Supply known sample size and retain its provenance.
prusie validates arrays; it does not harmonize alleles, infer ancestry or repair
LD. Read [inputs and priors](inputs.md) before fitting your own region.

The reference scope is Gaussian summary-statistics SuSiE against pinned susieR
0.16.6. The observed numerical gate is PIP absolute error ≤1e-5, rtol=0 plus
input, probability and model validity. It is not a guarantee for every possible
input, and intermediate arrays or credible sets can differ.

<div class="callout"><p><strong>One package name throughout.</strong> The repository, distribution and Python import are all <code>prusie</code>. The separate <code>pycoloc</code> package can consume these results for colocalisation.</p></div>

Maintained by Wenxin Jiang. The source repository is [LucaJiang/prusie](https://github.com/LucaJiang/prusie). No package-index release is claimed. See [release notes](release_notes.md), [supported scope](compatibility.md)
and [performance evidence](performance.md).

Source: [prusie on GitHub](https://github.com/LucaJiang/prusie). Related package: [pycoloc](https://github.com/LucaJiang/pycoloc).
