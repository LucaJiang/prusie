# susieR 0.16.6 refinement and PIP-convergence fixtures

Source: susieR 0.16.6 (commit 8e56a8e038e989856d106d9ca5175cc664fea9d2) fitted with
`susie_ss` to the sufficient statistics of `prusie.load_convergence_example()`, which
derive from susieR's bundled `FinemappingConvergence` data. The upstream authors and
BSD-3-Clause terms are retained in `LICENSE.BSD-3-Clause` and `BSD-3-Clause.txt`.

`refine_convergence.json` stores the final `alpha` and ELBO of four fits with
`L = 10`, `max_iter = 100` and `tol = 1e-3`: PIP-based stopping with zero prior
weight on the first credible set, and `refine = TRUE` with PIP stopping, a mixture
prior and the NIG residual model. `generate_reference.R` regenerates it.
