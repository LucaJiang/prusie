# Sensitivity follow-up

This addendum addresses variant-prior identity in alternative inputs and the
support used by common-mode comparisons. The earlier
[review snapshot](post_analysis/README.md) and
[review-fixes snapshot](post_analysis_fixes/README.md) remain unchanged, including
their stored fits, figures, reports and numerical producers. No existing example
was refitted or rerendered for this follow-up.

## Small installed-package reproduction

The original implementation at `dae00fbef8488b066ec2978e4110555a1b0a62ce`
was exercised with the installed native package before editing. A three-variant
RSS fit used z=(3,2,1), n=200, L=1, fixed scaled prior variance 0.1, tol=1e−8,
prior weights (0.80,0.15,0.05), and LD off-diagonals
r(A,B)=0.2, r(A,C)=0.1, r(B,C)=0.15. Reordering all data from A,B,C to C,A,B
incorrectly retained the original prior order and changed PIPs after aligning
the variants back: maximum absolute difference **0.12783108582586966**.
The corrected prior is (0.05,0.80,0.15), and the same reproduction has maximum
absolute PIP difference **0**. A subset A,B now receives normalized weights
(0.8421052631578947,0.15789473684210525), instead of a vector-length error.

With A={rsA,rsB,rsC}, B={rsA,rsB}, C={rsA,rsC}, the original common-mode comparison
raised `KeyError('rsB')`. The corrected comparison uses {rsA} for every pair,
reports shared count 1, and leaves correlation unavailable with only one point.
It preserves each component's mass outside that support rather than renormalizing
it away. [Usage and exact policies](../post_analysis.md#compare-explicit-settings)
describe null augmentation, explicit new-variant priors, metadata and support.
The [numerical record](sensitivity_followup_checks.json) retains the before/after
values and source commits. Reusing existing stable/problem fits, all existing
comparison fields agree with the parent implementation (maximum numeric
difference 0) in union mode and in common mode with identical variant support;
the additional fields explicitly label support. This is a bounded regression,
not a new regional benchmark or comprehensive scientific validation.

Tests: [prior identity](../../tests/test_sensitivity_prior_identity.py) covers
actual posterior invariance, uniform/nonuniform subsets, unnormalized weights,
nonzero null, new IDs, explicit alternatives, zero support, preflight validation,
metadata and cache behavior. [Common support](../../tests/test_comparison_common_support.py)
covers heterogeneous two/three/four-fit comparisons, empty intersection, union,
component permutation, non-first baseline and missing metadata.

## Stable-example interpretation addendum

The recorded [stable comparison](post_analysis_fixes/examples/stable/comparison.json)
has two active components under the estimated-variance baseline and five under
`fixed_variance` (`layer_summary.settings`). The latter uses
`scaled_prior_variance=0.05` and `estimate_prior_variance=False`
(`fit_settings.fixed_variance`). Fixing a positive component variance disables
the estimation step that could otherwise set unused component variances to zero.
All five components therefore remain available to distribute posterior mass,
including three without retained credible sets. This model difference helps
explain why background PIPs rise even though the two leading CS remain stable.

For example, the existing `variants` records give PIP 0.254364932495293 for
`syn0143` and 0.14297897778579471 for `syn0145` under that fixed-variance setting.
This is an explanation from the recorded model and its posterior components,
not proof of additional causal signals. Read it alongside the preserved
[stable interpretation](post_analysis_fixes/examples/stable/interpretation.md)
and [sensitivity figure](post_analysis_fixes/examples/stable/figure_5.png).
No new joint sensitivity grid is implied or generated.
