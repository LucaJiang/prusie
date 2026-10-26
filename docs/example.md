# Quick start: fit, read and report

This toy example uses posterior inclusion probabilities (PIPs) and credible sets
to summarize evidence about which variants carry an effect. It fits a SuSiE
model once and uses the same posterior for figures and an offline report. The
packaged synthetic data contain 500 correlated predictors, two with nonzero
effects, and are available offline after [installation](install.md).

## Load inputs and fit

Load the packaged summary statistics and linkage disequilibrium (LD) matrix,
then fit SuSiE-RSS:

```python
import numpy as np
import prusie

example = prusie.load_example()
inputs = example["inputs"]
print(inputs["R"].shape, inputs["n"])
fit = prusie.susie_rss(
    **inputs,
    **example["parameters"],
    track_fit="compact",
    retain_data=True,
)
print("Converged:", fit.converged, "Iterations:", fit.niter)
```

Expected output:

```text
(500, 500) 1000
Converged: True Iterations: 3
```

`example["parameters"]` allows up to five single-effect components (`L=5`),
with `max_iter=100` and ELBO tolerance `tol=0.001`. Prior variances are
estimated and residual variance is fixed at 1. The credible-set thresholds,
`coverage=0.95` and `min_abs_corr=0.5`, match the
[API defaults](api.md#shared-fitting-options).

`track_fit="compact"` records per-sweep metrics in `fit.trace` for the
convergence plot below. `retain_data=True` keeps prepared inputs for
[continuation or refinement](post_analysis.md).

For your own data, align the signed z-scores and both axes of `R` to the same
variant order and effect alleles. Keep `variant_ids` in that order and supply
the study sample size as `n`. The LD matrix must contain signed correlations
$r$; $r^2$ cannot replace them. Read the
[input requirements](inputs.md#variant-order-and-alleles) before replacing the
example inputs.

## Read PIPs and credible sets

A PIP combines inclusion across active single-effect components; a credible set
summarizes uncertainty within one component. Find the variant with the highest
PIP and list the retained sets:

```python
lead = int(np.argmax(fit.pip))
print(f"Lead: {fit.variant_ids[lead]}, PIP={fit.pip[lead]:.6g}")
for k, component in enumerate(fit.sets["cs_index"]):
    members = fit.variant_ids[fit.sets["cs"][k]].tolist()
    print(
        f"Component {int(component)}: {', '.join(members)}; "
        f"mass={fit.sets['coverage'][k]:.6g}; "
        f"minimum |r|={fit.sets['purity']['min_abs_corr'][k]:.6g}"
    )
```

Expected output:

```text
Lead: syn0350, PIP=0.999937
Component 0: syn0100; mass=0.969674; minimum |r|=1
Component 1: syn0350; mass=0.999937; minimum |r|=1
```

`fit.sets["coverage"]` records each set's attained posterior mass within its
component; it can exceed the requested `coverage=0.95`. The component IDs in
`fit.sets["cs_index"]` refer to original, zero-based rows of `fit.alpha` and
need not be consecutive.

Both sets here contain one variant, so their purity is 1 by convention. This
says nothing about LD with neighboring variants. See
[credible-set fields](results.md#credible-set-fields) for the returned fields
and filtering rules.

## Summarize the fitted posterior

`summarize` organizes the existing posterior into tables of variants,
single-effect components and credible sets without refitting. Supplying `R`
adds LD proxies and relationships between credible sets:

```python
summary = prusie.summarize(fit, inputs["R"], z=inputs["z"])
print(prusie.summary_text(summary).splitlines()[0])
print("Stop:", summary["convergence"]["stop_reason"])
```

Expected output:

```text
500 variants; 2 retained credible sets; 2 active components.
Stop: criterion_satisfied
```

Print `prusie.summary_text(summary)` without `.splitlines()[0]` to read
credible-set details, leading PIPs and the recorded analysis status. For this
fit it reports refinement as not requested and LD diagnostics as not assessed.
Supplying `R` for tables does not run an
[LD consistency diagnostic](sensitivity.md#diagnose-consistency-of-z-and-ld).

## Write an offline report

Save the posterior summaries and figures in an offline report. Install the
[analysis extra](install.md#optional-analysis-tools) to run the plotting calls:

```python
from prusie.report import write_report
from prusie.viz import plot_convergence, plot_locus

report = write_report(
    summary,
    "output/analysis",
    title="Synthetic fine-mapping example",
    figures={
        "Locus overview": plot_locus(summary, R=inputs["R"], layout="compact"),
        "Convergence": plot_convergence(fit, layout="compact"),
    },
)
print(report["html"])
```

Open the printed path in your browser. The report starts with credible sets
and candidate variants, then shows analysis status and the supplied figures.
The plotting and report functions use the existing posterior.

| Export | Use |
| --- | --- |
| `index.html` | Offline HTML report |
| `summary.txt`, `analysis_summary.json` | Text and compact structured summaries |
| `tables/*.csv`, `plot_data.json` | Full tables and the supplied figures' plotted values |
| `figure_index.json`, `*.png`, `*.svg` | Figure labels, file mapping and image exports |
| `evidence.json` | Full-precision scalar facts with stable IDs and JSON pointers |
| `analysis.json`, `arrays.npz` | Reloadable summary, supplied diagnostics and metadata |

[Save the fitted result separately](#save-and-reload-a-result) to retain its
posterior arrays. Copy or download the whole report directory to keep its
export links valid.

## Save and reload a result

Save the posterior arrays, variant identifiers and result metadata in
`analysis.json` and `arrays.npz`, preserving array dtypes. Loading a saved fit
reconstructs a `SusieResult`:

```python
from prusie.report import load_analysis, save_analysis

save_analysis(fit, "output/fit")
restored = load_analysis("output/fit")
print("PIPs identical:", np.array_equal(restored.pip, fit.pip))
```

The expected output is `PIPs identical: True`. Keep both files together.
`load_analysis("output/analysis")` reads the report's saved mapping, whose
`"summary"` entry contains the analysis tables.

Saved fits exclude the retained input snapshot, so `prusie.continue_fit(restored)`
cannot run directly. To warm-start a new `susie_rss` fit, supply the original
inputs and fitting options with `s_init=restored`; see
[warm-start validation](post_analysis.md#technical-details) for the alignment
and scale checks.

Continue with [reading results](results.md), [convergence and refinement](post_analysis.md),
[sensitivity and LD](sensitivity.md) or the [Toy example](worked_examples.md).

## Advanced export and example provenance

<details markdown="1">
<summary>Manual arrays and metadata export</summary>

Use [Save and reload a result](#save-and-reload-a-result) for a reloadable
`SusieResult`. For exchange with other tools, export selected arrays and
metadata explicitly:

```python
import json
from pathlib import Path

out = Path("fine-mapping-result")
out.mkdir(exist_ok=True)
np.savez_compressed(
    out / "posterior.npz",
    pip=fit.pip, alpha=fit.alpha, mu=fit.mu, mu2=fit.mu2,
    lbf_variable=fit.lbf_variable, V=fit.V, elbo=fit.elbo,
    variant_ids=fit.variant_ids,
)
sets = [
    {
        "component": int(component),
        "members": fit.variant_ids[fit.sets["cs"][k]].tolist(),
        "posterior_mass": float(fit.sets["coverage"][k]),
    }
    for k, component in enumerate(fit.sets["cs_index"])
]
metadata = {
    "prusie_version": prusie.__version__,
    "parameters": fit.params,
    "converged": fit.converged,
    "niter": fit.niter,
    "sets": sets,
    "generation": example["metadata"],
}
(out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
```

</details>

<details markdown="1">
<summary>Generation, frozen reference and numerical agreement</summary>

The packaged inputs come from 1000 simulated observations of 500 Gaussian
predictors. Signed z-scores are marginal regression estimates divided by their
standard errors, and `R` is the sample Pearson correlation matrix of the same
centered, sample-standardized predictors. The data are synthetic; no human
study or genome build is implied.

<!-- generated:toy:start -->
The independent 500-variant toy example fit converged in 3 iterations and returned 2 credible sets. Its maximum absolute PIP difference from the separately generated susieR 0.16.6 reference was 1.66e-10.
<!-- generated:toy:end -->

<!-- generated:toy-cs:start -->

| Original component | Synthetic variants | Returned posterior mass | Minimum &#124;r&#124; |
| --- | --- | --- | --- |
| 0 | syn0100 | 0.969674 | 1.000000 |
| 1 | syn0350 | 0.999937 | 1.000000 |
<!-- generated:toy-cs:end -->

![Signed z and fitted PIPs, with the two synthetic true effects marked.](assets/toy_fit.svg){.figure-complex}

`tools/generate_example.py` uses NumPy PCG64 with seed 20261007. It generates
Gaussian AR(1) predictors with adjacent population correlation 0.85, then
centers and sample-standardizes them. Nonzero coefficients are 0.35 and −0.30
at zero-based indices 99 and 349; the response also includes independent
standard Gaussian noise. Stored positions are arbitrary synthetic coordinates
spaced 1000 units apart.

`tools/generate_example.py --check` compares regenerated arrays and generation
metadata with the packaged data, allowing documented bounds on floating-point
rounding error. `load_example()` separately verifies SHA-256 hashes of the
packaged inputs and metadata. See the
[regeneration checks](reproducibility.md#regeneration-checks).

The packaged reference was fitted separately with susieR 0.16.6 using matching
inputs and fitting options. Input and reference-array hashes, parameters,
convergence status and credible sets are distributed with the data. The
dataset and generation script are licensed under GPL-3.0-or-later.
[Reproducibility](reproducibility.md) gives commands for offline checks and for
refitting with R; [numerical accuracy](numerical_accuracy.md) reports the
measured differences.

</details>
