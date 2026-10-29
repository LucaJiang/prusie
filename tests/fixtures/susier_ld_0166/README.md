# susieR 0.16.6 LD-consistency fixtures

Source: susieR 0.16.6 (commit 8e56a8e038e989856d106d9ca5175cc664fea9d2) on
R 4.4.0. The upstream authors and BSD-3-Clause terms are retained in
`LICENSE.BSD-3-Clause` and `BSD-3-Clause.txt`.

`ld_oracles.json` stores `estimate_s_rss` (null-MLE, null-partial-MLE and
null-pseudo-MLE) and `kriging_rss` results for three deterministic six-variant
cases (positive-definite LD, singular LD with consistent z-scores and singular
LD with an allele flip), each with n = 200 and without n, and for the estimated
s, s = 0.1 and s = 0. The kriging function is the installed one with its
final expression extended to also return its internal grid and mixture
weights.

`kriging_certified.json` stores, for every kriging fit, the global optimum of
the full-likelihood normal mixture, certified by the full simplex KKT
condition, and the resulting allele-flip log-likelihood ratios. susieR's
default `mixsqp` fit is recorded alongside for comparison.

`generate_reference.R` regenerates both files:
`Rscript generate_reference.R tests/fixtures/susier_ld_0166`.
