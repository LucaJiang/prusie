# Quick start: fit, read and report

Load the packaged synthetic example, fit once, then reuse that posterior for
PIPs, credible sets, figures and an offline report. The example contains 500
correlated predictors and two nonzero effects; its inputs work offline after
[installation](install.md).

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

This call performs the fit. `track_fit` retains iteration history, and
`retain_data` keeps the validated inputs needed for explicit continuation or
refinement. The example uses L=5, ELBO tolerance 0.001, requested CS mass 0.95,
minimum absolute correlation 0.5, estimated prior variance and fixed residual
variance 1. These example settings are distinct from the [API defaults](api.md).

For your own data, provide effect-aligned signed z, signed LD correlation R and
sample size in the same variant order. Read [inputs](inputs.md) before replacing
these arrays; r² is a display quantity and cannot replace signed r in the model.

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

Each retained CS is a singleton here. PIP combines inclusion across active
components; CS mass describes one component. Singleton purity is 1 by definition
and says nothing about LD to surrounding variants. [Reading results](results.md)
explains these quantities and the original component IDs.

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

`summarize` reuses the fit to build variant, component, CS and resolution tables.
Print the complete `prusie.summary_text(summary)` to include candidate PIPs and
analysis status. This fit has no refinement search or LD diagnostic yet, so those
operations are shown as not requested or not assessed.

## Write an offline report

With the [analysis extra](install.md#optional-analysis-tools) installed, make
figures and a report from the same `fit` and `summary`:

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

Open `output/analysis/index.html` locally. Candidates and resolution come first,
followed by stopping status and the supplied figures. These calls reuse the
posterior; [continuation, refinement](post_analysis.md) and
[LD diagnostics](sensitivity.md#diagnose-consistency-of-z-and-ld) are separate,
explicit calculations.

| Export | Use |
| --- | --- |
| `summary.txt`, `analysis_summary.json` | Short text and structured summaries |
| `index.html`, `figure_index.json`, PNG/SVG | Offline report and named figure mapping |
| `tables/*.csv`, `plot_data.json` | Full tables and the supplied figures' plotted data |
| `evidence.json` | Full-precision facts with stable IDs and JSON paths |
| `analysis.json`, `arrays.npz` | Lossless saved analysis structure and arrays |

Copy the whole report directory to retain its local links. GitHub's file viewer
shows the HTML source; download the directory to open the offline report.

## Save and reload a result

Use the lossless result interface to retain array dtypes, identifiers and metadata:

```python
from prusie.report import save_analysis, load_analysis

saved = save_analysis(fit, "output/fit")
restored = load_analysis("output/fit")
print("PIPs identical:", np.array_equal(restored.pip, fit.pip))
```

Expected output is `PIPs identical: True`. The same interface reads a report
directory into its saved analysis mapping. Saved fits omit the optimization
context; to warm-start a new call, supply the original inputs with
`s_init=restored`. [Warm-start details](post_analysis.md#technical-details)
explain the identity checks.

Continue with [reading results](results.md), [convergence and refinement](post_analysis.md),
[sensitivity and LD](sensitivity.md), or the [two worked examples](worked_examples.md).

## Advanced export and example provenance

<details markdown="1">
<summary>Manual arrays and metadata export</summary>

The high-level persistence interface above is sufficient for ordinary use. For
interchange with other tools, write selected arrays and metadata explicitly:

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

The example has 1000 simulated observations and 500 Gaussian predictors.
Signed z is beta/SE from marginal regression; R is the sample Pearson
correlation of the same standardized predictors. It is synthetic, with no human
study or genome build implied.

<!-- generated:teaching:start -->
The independent 500-variant toy example fit converged in 3 iterations and returned 2 credible sets. Its maximum absolute PIP difference from the separately generated susieR 0.16.6 reference was 1.66e-10.
<!-- generated:teaching:end -->

<!-- generated:teaching-cs:start -->

| Original component | Synthetic variants | Returned posterior mass | Minimum &#124;r&#124; |
| --- | --- | --- | --- |
| 0 | syn0100 | 0.969674 | 1.000000 |
| 1 | syn0350 | 0.999937 | 1.000000 |
<!-- generated:teaching-cs:end -->

![Signed z and fitted PIPs, with the two synthetic true effects marked.](assets/teaching_fit.svg){.figure-complex}

`tools/generate_example.py` uses NumPy PCG64 seed 20261007, stationary AR(1)
correlation 0.85, sample-standardized predictors, nonzero coefficients 0.35 and
−0.30 at zero-based indices 99 and 349, and independent unit-variance Gaussian
noise. Coordinates in the generator are arbitrary synthetic positions at
1000-unit intervals. `tools/generate_example.py --check` verifies the exact
frozen-file SHA256 hashes and checks regenerated values within tight
floating-point roundoff bounds; see the
[regeneration criteria](reproducibility.md#toy-data-regeneration).

The packaged reference was separately fitted with susieR 0.16.6 using the same
inputs and options. Input and reference hashes, parameters, convergence and CS
accompany the resources. The dataset and generation script are GPL-3.0-or-later.
[Reproducibility](reproducibility.md) provides offline and live-R commands;
[numerical accuracy](numerical_accuracy.md) gives measured errors.

</details>
