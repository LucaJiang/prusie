# Reference mapping

Reference: official susieR 0.16.6 commit
`8e56a8e038e989856d106d9ca5175cc664fea9d2`; source archive SHA256
`c88c6324da061c83ac9fbac23972cd1e501210c916596ea7678c3115b33f6971`.
The old CRAN 0.14.2 lock is immutable historical evidence, not the new target.

| Python/Rust | Pinned R source and responsibility |
| --- | --- |
| api.py RSS | susie_constructors.R summary_stats_constructor and summary_stats_working_quantities: Wald PVE, known/missing n, bhat/shat/var_y |
| api.py sufficient statistics | sufficient_stats_constructor: priors, null, predictor scales; susie.R public defaults |
| core.rs single_effect | single_effect_regression.R gaussian_ser_lbf and gaussian_ser_moments; susie_utils.R posterior-weight helpers |
| core.rs V optimization | scalar optimizer and prior-update hooks; generic_methods.R Gaussian EM |
| optimizer.rs | R stats optimize.c Brent_fmin with optim's default tolerance; original license retained |
| core.rs IBSS | sufficient_stats_methods.R residual/fitted updates; model_methods.R convergence; susie_workhorse.R loop/trim order |
| core.rs ER2/KL | sufficient_stats_methods.R get_ER2.ss and Gaussian expected likelihood |
| posterior.py | susie_get_functions.R CS/PIP; sufficient_stats_methods.R lazy scaled_XtX; susie_utils.R full-matrix purity |
| result.py | Explicit Python result contract, original components, posterior coefficient inverse scales |

Each component sees updated earlier components. Matrix-product reuse is valid
because a component's coefficients stay fixed until its next update. The ER2
quadratic uses the same component products and matrix linearity. No fitted
result survives across public calls. Reductions may be reordered within the
frozen absolute-error contract; R accumulation order is not an API guarantee.

Current finite stopping requires 0 ≤ ELBO increment < tol after the first
iteration. Invalid increments use alpha/PIP fallback and retain diagnostics.
Final trimming runs after the stored trace and before CS/PIP. Its reset is an
intentional semantic migration, not output rounding or a numerical waiver.

The package remains a native Python/Rust implementation. It never invokes R
at runtime. Unsupported SuSiE2 extensions are listed in compatibility.md.
