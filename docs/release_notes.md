# prusie 0.2.3rc4 (release candidate)

The current export uses **prusie** for the repository, distribution and import.
See the [rename and migration notes](migration.md). Historical performance and
R-agreement records below retain their original binary identities.

The candidate retains A02 fit-local SER storage and removes B01's known-variance
RSS allocation optimization. The complete known-variance API now matches rc17;
its independent analytical correctness tests are retained. B01 has no applicable
runtime-majority evidence and stays historical experimental work.

A02 previously won 8/8 pilot cases versus rc17 and 5/8 incrementally versus A01.
Those pilots are not an isolated all-200-case A01 ablation. The fresh fixed-panel
comparison selected **A02 only**: **118 faster, 0 tied and 82 slower cases out
of200**, based on seven complete-call repetitions after one warmup, with one
numerical thread and two independent regions. This meets the user's literal
observed-majority gate; it does not imply universal speedup or statistical
significance. B01 remains excluded. All200 PIP/input/model checks passed.

Across all200 cases, the summed medians were1.839772626 seconds for rc17 and
1.707386992 seconds for rc4. For the199 mutually converged cases, they were
1.494473075 and1.461272684 seconds. These totals are diagnostics, not the
selection rule. See [performance evidence](performance.md) for the full panel,
repeat variability and limitations.

The pinned reference is official susieR 0.16.6, commit
`8e56a8e038e989856d106d9ca5175cc664fea9d2`. PIP absolute error ≤1e-5 (rtol=0),
probability validity and unchanged input/model semantics define acceptance.
Intermediate and credible-set differences remain visible diagnostics. The
primary panel uses L=5, max_iter=100 and tol=0.001; one historical GWAS case
reaches the iteration cap without convergence.

This version adds an offline public teaching example and maintained static Pages
documentation. Example execution and release selection are recorded in the
[R agreement report](r_agreement.md). The separate accepted default environment
remains unchanged. No DOI or package-index release is assigned.

Local platform validation targets CPython 3.12 on Linux x86-64. Other platforms
require their own build and validation. The CI workflow is provided for future
use; local checks are not evidence of a remote GitHub Actions run.

## GitHub source delivery

Repository: [prusie](https://github.com/LucaJiang/prusie). The publication changes add repository links and Pages setup instructions; numerical source, frozen inputs and expected outputs are unchanged.

The previously validated 0.2.3rc4 wheels and source distributions are **pre-publication snapshots**. They predate the repository metadata and documentation links in this checkout. Their original hashes and installation evidence are retained; this GitHub delivery does not create replacement distributions with the same filenames or claim a PyPI release. Install this checkout for the maintained metadata and documentation.
