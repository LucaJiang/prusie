# Quick start: fit, read and report

This tutorial loads the packaged synthetic toy example, fits the model once and
uses that posterior for PIPs, credible sets, figures and an offline report. The
example has 500 correlated predictors, two of which have nonzero effects, and
can be used offline after [installation](install.md).

## Load inputs and fit

```python
import numpy as np
import prusie

example = prusie.load_example()
inputs = example["inputs"]
print(inputs["R"].shape, inputs["n"])
fit = prusie.susie_rss(**inputs, **example["parameters"],
                       track_fit="compact", retain_data=True)
print("Converged:", fit.converged, "Iterations:", fit.niter)
```

Expected output:

```text
(500, 500) 1000
Converged: True Iterations: 3
```

This call fits the model. `track_fit` records the per-sweep convergence
metrics, and `retain_data` keeps the validated inputs needed for continuation or
refinement. The example uses L = 5 single effects, ELBO tolerance 0.001, requested
credible-set coverage 0.95, minimum absolute correlation 0.5, an estimated prior
variance and a fixed residual variance of 1. These settings differ from the
[API defaults](api.md).

For your own data, supply signed z-scores aligned to the effect allele, the
signed LD correlation matrix R and the sample size, in the same variant order.
Read [inputs](inputs.md) before replacing these arrays; r² is a display
quantity and cannot replace signed r in the model.

## Read PIPs and credible sets

```python
lead = int(np.argmax(fit.pip))
print(f"Lead: {fit.variant_ids[lead]}, PIP={fit.pip[lead]:.6g}")
for k, component in enumerate(fit.sets["cs_index"]):
    members = fit.variant_ids[fit.sets["cs"][k]].tolist()
    print(f"Component {int(component)}: {', '.join(members)}; "
          f"mass={fit.sets['coverage'][k]:.6g}; "
          f"minimum |r|={fit.sets['purity']['min_abs_corr'][k]:.6g}")
```

Expected output:

```text
Lead: syn0350, PIP=0.999937
Component 0: syn0100; mass=0.969674; minimum |r|=1
Component 1: syn0350; mass=0.999937; minimum |r|=1
```

Here each credible set contains a single variant. A PIP combines inclusion over
all active single effects, whereas the mass of a credible set refers to one
single effect. The purity of a single-variant set is 1 by definition and says
nothing about LD with neighboring variants. [Reading results](results.md)
explains these quantities and the effect indices.

## Summarize the fitted posterior

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

`summarize` builds variant, single-effect, credible-set and resolution tables
from the fit. Print the complete `prusie.summary_text(summary)` to include the
leading PIPs and the status of the analysis. This fit has no refinement search
or LD diagnostic, so these are reported as not requested and not assessed.

## Write an offline report

With the [analysis extra](install.md#optional-analysis-tools) installed, figures
and a report can be produced from the same `fit` and `summary`:

```python
from prusie.plots import plot_locus, plot_convergence
from prusie.report import write_report

report = write_report(
    summary, "output/analysis", title="Synthetic fine-mapping example",
    figures={"Locus overview": plot_locus(summary, R=inputs["R"], layout="compact"),
             "Convergence": plot_convergence(fit, layout="compact")},
)
print("Open output/analysis/index.html")
```

Open `output/analysis/index.html` locally. The report lists the candidate
variants and their resolution first, followed by the stopping status and the
supplied figures. These calls use the existing posterior;
[continuation, refinement](post_analysis.md) and
[LD diagnostics](sensitivity.md#diagnose-consistency-of-z-and-ld) are separate
computations.

| Export | Use |
| --- | --- |
| `summary.txt`, `analysis_summary.json` | Short text and structured summaries |
| `index.html`, `figure_index.json`, PNG/SVG | Offline report and named figure mapping |
| `tables/*.csv`, `plot_data.json` | Full tables and the supplied figures' plotted data |
| `evidence.json` | Full-precision facts with stable IDs and JSON paths |
| `analysis.json`, `arrays.npz` | Lossless saved analysis structure and arrays |

Copy the whole report directory to keep its links valid. The GitHub file viewer
shows the HTML source; download the directory to open the report.

## Save and reload a result

The lossless storage functions keep array dtypes, identifiers and metadata:

```python
from prusie.report import save_analysis, load_analysis

saved = save_analysis(fit, "output/fit")
restored = load_analysis("output/fit")
print("PIPs identical:", np.array_equal(restored.pip, fit.pip))
```

The expected output is `PIPs identical: True`. The same function reads a report
directory into its saved analysis mapping. Saved fits do not include the
retained inputs; to start a new fit from a reloaded result, supply the original
inputs together with `s_init=restored`. The [warm-start details](post_analysis.md#technical-details)
describe the checks applied.

Continue with [reading results](results.md), [convergence and refinement](post_analysis.md),
[sensitivity and LD](sensitivity.md) or the [two worked examples](worked_examples.md).

## Advanced export and example provenance

<details markdown="1">
<summary>Manual arrays and metadata export</summary>

The storage functions above suffice for ordinary use. For exchange with other
tools, selected arrays and metadata can be written explicitly:

```python
import json
from pathlib import Path

out = Path("fine-mapping-result")
out.mkdir(exist_ok=True)
np.savez_compressed(out / "posterior.npz", pip=fit.pip, alpha=fit.alpha,
                    mu=fit.mu, mu2=fit.mu2, lbf_variable=fit.lbf_variable,
                    V=fit.V, elbo=fit.elbo, variant_ids=fit.variant_ids)
sets = [{"component": int(c), "members": fit.sets["cs"][k].tolist(),
         "posterior_mass": float(fit.sets["coverage"][k])}
        for k, c in enumerate(fit.sets["cs_index"])]
metadata = {"prusie_version": prusie.__version__, "parameters": fit.params,
            "converged": fit.converged, "niter": fit.niter, "sets": sets,
            "generation": example["metadata"]}
(out / "metadata.json").write_text(json.dumps(metadata, indent=2))
```

</details>

<details markdown="1">
<summary>Generation, frozen reference and numerical agreement</summary>

The example has 1000 simulated observations of 500 Gaussian predictors. The
z-scores are the marginal regression estimates divided by their standard
errors, and R is the sample Pearson correlation of the same standardized
predictors. The data are synthetic; no human study or genome build is implied.

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

`tools/generate_example.py` uses NumPy PCG64 with seed 20261007, a stationary
AR(1) correlation of 0.85 between adjacent predictors, sample-standardized
predictors, nonzero coefficients 0.35 and −0.30 at zero-based indices 99 and
349, and independent standard Gaussian noise. The coordinates are arbitrary
synthetic positions at intervals of 1000 units. `tools/generate_example.py --check`
verifies the exact SHA-256 hashes of the frozen files and compares regenerated
values within bounds on floating-point rounding error; see the
[regeneration criteria](reproducibility.md#regeneration-checks).

The packaged reference was fitted separately with susieR 0.16.6 using the same
inputs and options. The hashes of inputs and reference, the parameters, the
convergence status and the credible sets are distributed with the data. The
dataset and generation script are licensed under GPL-3.0-or-later.
[Reproducibility](reproducibility.md) gives commands for offline checks and for
refitting with R; [numerical accuracy](numerical_accuracy.md) reports the
measured differences.

</details>
