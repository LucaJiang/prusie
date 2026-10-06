# Historical 0.14.2 findings (not the current release contract)

The current pinned0.16.6 reference and numerical contract are documented in compatibility.md and release_notes.md. The reduction-order repair below describes earlier
versions, not a requirement imposed on the current native implementation.

## Pinned upstream behavior and explicit differences

The reference is untouched CRAN susieR 0.14.2, archive SHA256
`ba02322eb1f7a7cc024c9278aa7903a34d8ad5d6f3b12c168374bc6214ed2c6e`.
These findings apply to that version. They do not assert complete API parity.

## SUSIER-CS-COVERAGE-INDEX: confirmed implementation bug

Definition: `sets.coverage[k]` is the sum of `alpha[sets.cs_index[k], j]` over
members of the returned set `sets.cs[k]`, using the original component ID.
This fitted posterior mass is distinct from the requested threshold and is not
empirical repeated-sampling coverage.

Minimal input: alpha rows (0.48, 0.48, 0.04) and (0.01, 0.01, 0.98), V=(1,1),
identity correlation, requested coverage 0.95 and min_abs_corr=0.5. Component 0's
set {0,1}, mass 0.96, fails purity. Component 1's singleton {2}, mass 0.98,
remains. The historical independently executed R probe uses (0.51, 0.45, 0.04)
as its first row and establishes the same discrepancy.

| Field | Pinned R | pyrsusie 0.1.0 | pyrsusie 0.1.1 |
| --- | --- | --- | --- |
| Original component (zero-based) | 1 | 1 | 1 |
| CS members (zero-based) | {2} | {2} | {2} |
| coverage | 0.96 | 0.96 | 0.98 |
| actual_coverage (native alias) | unavailable | 0.98 | 0.98 |
| requested_coverage | 0.95 | 0.95 | 0.95 |

In `R/susie_utils.R:susie_get_cs`, claimed_coverage is subset by include_idx but
not is_pure before final ordering. Python 0.1.1 sums from the final returned CS
and original alpha row. Set selection, stable purity sorting and numerical
fitting remain unchanged. The obsolete warning asking users to use the extra
field has been removed. This affected field is correctness PASS with
EXPECTED_DIFFERENCE against the pinned erroneous return when the bug manifests;
unaffected fields must still satisfy their frozen comparisons individually.

Private historical audit outputs remain preserved. Package regression tests include `test_coverage_tracks_original_components`,
`test_empty_coverage`, and the independent
`test_official_cs_coverage_indexing_discrepancy`. Analytic expectations are explicit
posterior masses; independent oracles do not call production coverage helpers.

## Numerical reduction order: resolved implementation discrepancy

The original row-major ndarray dot implementation accumulated eight partial sums.
The locked R environment uses netlib BLAS 3.9.0 forward accumulation. On the first
real GWAS region, early residual differences around floating precision perturbed
an almost-flat optimized V, amplifying into an XtXr difference around 1.05e-5.
Isolated SER calls matched exactly for identical binary input arrays, and all
standardized kernel input values matched exactly, localizing the difference to
matrix-vector reduction order.

The solver stores one exact-valued column-major matrix copy and uses ndarray's
forward reduction. The previously failing XtXr became bit-exact in the historical
repair. All 18 original real cases and 432 field comparisons then passed the
frozen thresholds. Original diagnostics remain in
private historical reports. The correctness
patch leaves this solver and reduction order unchanged. This is not a newly
claimed upstream bug or permission to relax tolerances.

## API-domain and model choices

Full purity replaces upstream random subsampling by default; missing and
asymmetric inputs raise errors; PIP remains available when CS is disabled.
Refinement, custom initialization, trace capture, MAF filtering, random purity
subsampling and nonzero legacy z_ld_weight explicitly error. These are documented
API boundaries, not confirmed statistical bugs. See compatibility.md.
