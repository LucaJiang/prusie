# Supported Gaussian compatibility

The release targets official susieR 0.16.6 at immutable commit
`8e56a8e038e989856d106d9ca5175cc664fea9d2`. Supported entry points are
`susie_rss(z,R,n)`, `susie_rss(bhat,shat,R,n,var_y)`, the explicit missing-n
noncentrality likelihood, and centered `susie_suff_stat` (R `susie_ss`).
Known-n calls use the finite-sample Wald adjustment. Supplied `var_y` sets
the response variance even with z-only input, as in the pinned reference.

Scalar Gaussian priors support fixed V, optim, EM and simple updates; fixed or
estimated residual variance; normalized prior weights and null weight; L,
iteration cap and tolerance; ELBO, CS and PIP. RSS defaults to cap 50,
sufficient statistics to cap 100, and both to Gaussian tolerance 0.0001.
The measured comparison explicitly uses cap 100 and tolerance 0.001 everywhere.

Finite ELBO convergence requires a nonnegative increase below tolerance.
Nonfinite ELBO differences use the current alpha/PIP fallback with warnings.
After iteration, components with V < prior_tol reset alpha to the normalized
prior and zero V, moments, BFs and KL, including at the iteration cap. The
ELBO trace and XtXr cache retain pre-trim iteration values, following upstream.
XtXr is crossproduct times posterior coefficients, not the residual vector.

Degenerate columns have zero Gaussian BF and posterior moments and retain their
prior contribution to posterior odds. As in the reference, the SER adds
sqrt(machine epsilon) before logging prior weights; final null trimming restores
the exact normalized prior, including supplied zeros.

Matrix-input CS purity always uses the full selected submatrix, including when
a finite n_purity is supplied. No random sampling occurs. CS coverage follows
the surviving original component. Low-V PIP and null-column exclusions match
the supported reference path. None for coverage or min_abs_corr disables CS.

Intentional API boundaries:

- Arrays must be finite and symmetric; invalid input errors rather than silent
  missing-value replacement, symmetrization, eigenvalue clipping or LD repair.
- Python keeps named check_input/r_tol/check_prior options; R moves RSS options
  into a control object. Internal matrix products remain CPU float64.
- Result indices are zero-based; original variant IDs exclude the optional null.
  Empty CSs have explicit empty arrays, and the requested threshold remains
  metadata even when R returns NULL for the whole sets object. PIP is retained.
- sets.coverage is actual posterior mass; actual_coverage is an equal alias,
  requested_coverage is the target.
- Nonpositive residual-variance estimates error instead of propagating invalid
  posteriors. Nondefault initialization/refinement, MAF filtering, verbose output
  and trace snapshots remain unsupported.

Unsupported keywords and methods fail explicitly: no NIG, infinitesimal effects,
mixture priors, greedy components, slot priors, LD-mismatch model, low-rank X,
individual-level regression, trend filtering or diagnostic kriging API is claimed.
The [executed R comparison](r_agreement.md) uses PIP absolute error ≤1e-5,
rtol=0, with input/model/probability validity. Passing this criterion does not
prove identical intermediate arrays, credible sets, Bayes factors or downstream
posteriors, and does not bound error on untested inputs.

Returned CS coverage is computed from each surviving original alpha component.
This avoids a filtering/indexing error in the older susieR 0.14.2 reference;
the current 0.16.6 reference includes the coverage correction. The independent
coverage regressions remain in `tests/test_coverage.py`.
