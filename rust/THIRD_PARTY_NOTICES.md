# Native source attribution

`core.rs` implements Gaussian SuSiE updates adapted from susieR. The current
reference is susieR 0.16.6, commit
`8e56a8e038e989856d106d9ca5175cc664fea9d2`; earlier development used 0.14.2.
The corresponding copyright files remain in `LICENSE-susieR-0.16.6` and
`LICENSE-susieR`; the full BSD-3-Clause terms are retained in the distribution's
`THIRD_PARTY_NOTICES.md`.

`optimizer.rs` adapts R 4.4's `Brent_fmin` from
`src/library/stats/src/optimize.c`, retrieved on 2026-10-01 from
https://svn.r-project.org/R/branches/R-4-4-branch/src/library/stats/src/optimize.c.
It implements Richard Brent's *Algorithms for Minimization without Derivatives*
(1973), with R's step selection and stopping rules.

Copyright (C) 1995, 1996 Robert Gentleman and Ross Ihaka.
Copyright (C) 2003–2004 The R Foundation.
Copyright (C) 1998–2023 The R Core Team.

The adapted optimizer's source header specifies GPL-2.0-or-later; the full
license is provided in `LICENSE-R`. prusie is distributed under GPL-3.0-or-later
while retaining upstream file licenses. Source-specific R hashes are retained
in the source distribution's `rust/R-source-sha256.txt`.
