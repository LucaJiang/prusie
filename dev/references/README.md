# Independent Gaussian SuSiE checks

These developer tools compare standard fits with pinned susieR 0.16.6 and a
small, independent NumPy implementation. They are not runtime dependencies.
No input or reference is regenerated to obtain agreement. The panel uses the
bundled 500-variable synthetic teaching example, the frozen complete seed-312
regression, small deterministic software fixtures, and the official packaged
convergence example. It is a software validation panel, not a simulation study.

`validation_panel.py` fixes 56 standard/error cases before execution;
`run_validation.py` adds the official baseline and refinement comparison.
Settings cover RSS and sufficient statistics, fixed and estimated residual
variance, optim/EM/simple/fixed prior variance, nonuniform and boundary-zero
weights, both standardization modes, and zero-information variables. All raw
component arrays remain in their original order. Credible sets compare original
component IDs, members, coverage, full-pair purity and deduplication. The R driver
asserts the actual loaded package version, records its library and session,
and receives separate pinned-package and dependency-library paths.

Example invocation, within the project's required resource wrapper:

```bash
python dev/references/run_validation.py \
  --output /absolute/new/output-directory \
  --rscript /absolute/verified/Rscript \
  --rlib /absolute/pinned/library \
  --rdeps /absolute/dependency/library
```

The output directory must be new. `--reuse-r /absolute/previous/output` reuses
actual R outputs only if the serialized panel is byte-identical; this supports
verification of a new native artifact without repeating unchanged R work.
`--summary-only` reads completed raw outputs after a summary-writing failure;
use the same installed artifact so its identity remains accurate.

## Restricted NumPy reference

[`tests/reference_numpy.py`](../../tests/reference_numpy.py) adapts the supplied independent reference. The original
file remains preserved with SHA256
`d036e2b48046386df848386a5eb292d672dc6aaf4a3378d6bf123d5a17fb26e0`.
Its zero-column `r/d` and `sigma2/d` divisions are undefined, and its likelihood
clips sampling variance at machine epsilon. It also adds `sqrt(eps)` to prior
weights, including zero weights. Those expressions are unsuitable as an
unqualified mathematical oracle at these boundaries.

The restricted reference rejects zero/tiny diagonals, materially indefinite or
asymmetric crossproducts, nonfinite or extreme scales, nonpositive/extreme prior
weights, invalid scalars and variance iterates outside its explicit domain.
Sampling variances at or below machine epsilon are rejected; no floor changes
statistics. It uses exact positive normalized priors. Consequently it deliberately
omits pinned R's `sqrt(eps)` prior-weight smoothing: for p uniform weights the
model log-BF offset is `log(1 + p*sqrt(eps))`, even when posterior probabilities
agree. These raw likelihood/KL differences are reported, not suppressed or fixed
by changing tolerances. Positive-semidefiniteness tolerance only classifies input;
it never alters a matrix.

Supported features are finite n≥3, Gaussian sufficient statistics or z/R RSS,
positive scalar residual variance (fixed or MoM-estimated), optim/EM/simple/fixed
V, standardization, positive weights and the signed finite-ELBO stopping test.
The optimizer is SciPy bounded minimization; optimizer paths need not be identical.
Null columns, zero-information columns, check_prior, warm starts, refinement,
missing n, special convergence strategies and other APIs are outside this oracle.
They are checked against actual R or native analytic/invariant tests separately.

`tests/test_independent_numpy_reference.py` exercises native probability agreement,
closed-form one-effect posterior moments, the zero-V boundary, exact prior scaling,
CS coverage/purity/deduplication, and rejection of unsupported domains. Statistical
acceptance uses PIP absolute error ≤1e-5, rtol=0. Other field comparisons retain
the existing diagnostic atol=1e-6, rtol=1e-6 and report every exceeded diagnostic.
Actual counts and errors belong in the dated developer delivery report.
