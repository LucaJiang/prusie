# Numerical accuracy

## Component posterior validation

Current core validation uses susieR 0.16.6, commit
`8e56a8e038e989856d106d9ca5175cc664fea9d2`, with identical arguments and verified
variant order. The acceptance criterion is maximum absolute difference `1e-5`
(relative tolerance zero) across the **complete final alpha matrix**, including
inactive components and components without credible sets. Rows retain their
original component order; they are never permuted to improve agreement.
Each row must be finite, lie in [0, 1], and sum to one within `1e-10`.

PIPs, credible-set memberships, moments, variances, Bayes factors, objectives,
iteration counts and convergence flags remain diagnostics. Small differences
near activity thresholds can change these summaries even when alpha agrees.
The implementation uses mathematically equivalent float64 arithmetic rather
than reproducing R rounding. No thresholds or stopping tolerances are relaxed
for validation.

The current ordinary build passes all 415 fitted configurations with identical
arguments and verified variant order:

| Panel | Fitted configurations | Maximum absolute alpha difference |
| --- | --- | --- |
| Bundled examples | 7 | 3.55e-09 |
| Parameter cases | 54 | 4.6e-09 |
| Core feature cases | 69 | 7.02e-06 |
| NIG interactions | 2 | 5.5e-13 |
| Real analyses | 200 | 1.22e-08 |
| Real RSS model options | 60 | 1.2e-08 |
| Dense scaling fixtures | 6 | 2.6e-11 |
| Targeted regressions | 17 | 7.78e-09 |

Six expected invalid-model errors also match. Auxiliary-function boundaries
are checked separately. This finite panel does not establish agreement for
every combination of model options or identical stochastic draws across
languages. The historical tables below retain their original source and
acceptance criteria.

## Recorded earlier 0.2.4.dev2 snapshot

Comparisons against susieR 0.16.6 at commit `8e56a8e038e989856d106d9ca5175cc664fea9d2`
use compute commit `ff379ada284a63a3cf1770d43faacc6b929dc923`. All 100 GWAS and 100 eQTL
locus analyses have maximum absolute PIP difference ≤1e-5 with zero relative tolerance,
plus input, model and probability validity checks. Iteration counts, convergence
flags and credible-set component/member identities agree in all 200 cases.
CS order is immaterial; sets are matched by their original component IDs.

| Quantity | GWAS maximum absolute difference | eQTL maximum absolute difference |
| --- | --- | --- |
| PIP | 9.14e-09 | 1.22e-08 |
| alpha | 9.140027e-9 | 1.22e-08 |
| mu | 5.260663e-10 | 2.340635e-8 |
| mu2 | 1.394811e-10 | 3.704008e-9 |
| V | 4.078631e-6 | 1.486289e-7 |
| sigma2 | 0 | 0 |
| ELBO | 1.000124e-5 | 3.917756e-8 |
| KL | 7.010086e-8 | 1.171606e-7 |
| lbf | 2.821602e-5 | 2.743060e-7 |
| lbf_variable | 2.821542e-5 | 3.389438e-7 |

Intermediate fields are retained as diagnostics using absolute and relative
tolerances of 10⁻⁶. Region25/GWAS exceeds that diagnostic rule for V and
lbf_variable; its PIP passes the stated PIP criterion. Maximum credible-set
coverage differences are 2.14e-9 (GWAS) and 2.59e-9 (eQTL); maximum purity
difference is 8.88e-16 in both groups. These results describe standard fits with
the [fixed real-panel settings](performance.md#methods).

A separate parameter panel executes the same pinned R reference for 58 outcomes:
54 returned fits and four expected prior-guard errors. It covers RSS and sufficient
statistics, fixed and estimated residual variance, optim/EM/simple/fixed prior
variance, nonuniform priors, standardization and stopping regressions. All 54
returned fits pass PIP and validity checks; the four error cases match the
expected prior-guard rejection in both implementations. A restricted independent NumPy reference also passes
31 supported comparisons; it rejects unsupported boundary inputs explicitly.
The preserved cycling example converges in 49 sweeps in both implementations,
with zero maximum PIP difference. Refinement is tested separately and retains
the [documented search differences](post_analysis.md#refinement-compared-with-susier).

## Earlier measurements: 0.2.3rc4

The earlier comparison uses prusie 0.2.3rc4 and susieR 0.16.6 at commit
`8e56a8e038e989856d106d9ca5175cc664fea9d2`. It reanalyzes saved full fits from
6 October 2026 using the matched inputs and parameters described in
[performance methods](performance.md#methods). The records identify the measured source and binary.

The panel covers ordinary RSS fits with fixed residual variance and optimized
prior variance, as specified in those methods. It does not test refinement,
alternate convergence criteria or the current development snapshot's stopping
and estimated-residual-variance checks.

## Variant inclusion probabilities

<!-- generated:maxima:start -->

| Analysis | Analyses | Variants per analysis | Maximum absolute PIP difference | Trace: case / variant index (zero-based) |
| --- | --- | --- | --- | --- |
| GWAS | 100 | 103–4162 | 9.14e-09 | region16/gwas / 49 |
| eQTL | 100 | 103–4162 | 1.22e-08 | region31/eqtl / 238 |

Across the evaluated GWAS analyses, the maximum absolute difference in variant PIPs relative to susieR 0.16.6 was 9.14e-09. Across the evaluated eQTL analyses, the maximum absolute difference in variant PIPs relative to susieR 0.16.6 was 1.22e-08.
<!-- generated:maxima:end -->

The numerical criterion is absolute PIP difference ≤10⁻⁵ with zero relative
tolerance, together with input/model/probability validity. The table reports
observed elementwise maxima, rather than that threshold. Each maximum is
traceable to a zero-based variant index, both fitted values and hashes of the
two raw fit records in the [machine summary](assets/r_comparison_summary.json).

![Each analysis's maximum absolute PIP difference against its variant count, separated into GWAS and eQTL.](assets/pip_accuracy.svg)

The vertical axis is symmetric-log with a linear interval below 10⁻¹⁴; exact
zeros are retained at zero. [Vector PDF](assets/pip_accuracy.pdf).

<!-- generated:limits:start -->
region25/gwas: prusie 100 iterations, susieR 100 iterations, both unconverged. Both calls completed and their outputs enter the numerical maxima and timing summaries.
<!-- generated:limits:end -->

## Posterior arrays, Bayes factors and objective

<!-- generated:errors:start -->

| Quantity | Scale / units | GWAS maximum absolute difference | eQTL maximum absolute difference |
| --- | --- | --- | --- |
| alpha | Probability | 9.14e-09 | 1.22e-08 |
| mu | Standardized effect | 5.26e-10 | 2.34e-08 |
| mu2 | Squared standardized effect | 1.39e-10 | 3.7e-09 |
| lbf_variable | Natural-log BF | 2.82e-05 | 3.39e-07 |
| lbf | Natural-log BF | 2.82e-05 | 2.74e-07 |
| V | Squared effect | 4.08e-06 | 1.49e-07 |
| sigma2 | Phenotype variance | 0 | 0 |
| elbo | Natural-log objective | 1e-05 | 3.92e-08 |
| KL | Natural-log divergence | 7.01e-08 | 1.17e-07 |

Maxima are computed elementwise across every valid analysis, including iteration-limit outputs. The machine summary includes the case, array index, both values at the maximum and reference magnitude for each field. ELBO traces are compared only with matching shapes; per-case records retain any shape mismatch.
<!-- generated:errors:end -->

PIP and alpha differences have probability units. Conditional effects are on
the fitted standardized predictor/phenotype scale for these z-input analyses;
mu2 is a second moment, not a variance. Component and variant log Bayes factors
use natural logarithms. Their numerical differences are recorded separately
from PIP. ELBO is the iteration objective before final low-variance trimming;
V, moments, Bayes factors and KL follow the finalization semantics described in
[results](results.md).

## Credible sets

<!-- generated:cs:start -->

| Analysis | prusie sets | susieR sets | Component/member differences | Maximum mass difference | Maximum purity difference |
| --- | --- | --- | --- | --- | --- |
| GWAS | 154 | 154 | 0 | 2.14e-09 | 8.88e-16 |
| eQTL | 97 | 97 | 0 | 2.59e-09 | 8.88e-16 |

Sets are matched by original component ID and compared as member sets; returned ordering does not affect the comparison. All actual masses, purity statistics and zero-based members are retained.
<!-- generated:cs:end -->

The requested mass is 0.95. Returned mass is the sum of the selected alpha
entries in the original component; purity uses every selected pair's absolute
correlation. The retained numerical records include independent checks of PIP
identity, alpha normalization, actual set mass and purity, including empty sets.
Convergence and the presence of a credible set are different properties.

## Independent toy example

<!-- generated:toy:start -->
The independent 500-variant toy example fit converged in 3 iterations and returned 2 credible sets. Its maximum absolute PIP difference from the separately generated susieR 0.16.6 reference was 1.66e-10.
<!-- generated:toy:end -->

![Signed association statistics and fitted PIPs for the independent synthetic toy example, with its two true effect positions marked.](assets/toy_fit.svg)

The dashed lines mark the two simulated nonzero effects. This single toy
example has its own newly executed reference, generation script and packaged
inputs. See the [toy example tutorial](example.md) and [vector PDF](assets/toy_fit.pdf).

## Reproduce the comparisons

[Retained records](assets/r_comparison_numerical.json) include the real panel
and a separately identified older synthetic regression panel. The older fixture
and references remain byte-for-byte regression inputs. The installed package's
independent toy example is checked against its separate packaged R reference
by `tests/test_packaged_example.py`.

The [reproducibility guide](reproducibility.md) distinguishes regeneration from
saved records, offline frozen-reference checks, and live-R fitting.
`tools/benchmark/compare_fits.py` compares original component IDs and retains all
field diagnostics. Existing frozen references and tolerances are unchanged.
