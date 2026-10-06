# Run the 500-SNP offline check

This example uses all 500 synthetic variables in each of the four stored
official coloc teaching datasets, D1–D4. It is an existing fixed software
fixture, not human-study data and not a new simulation. No variables were
selected, padded or generated for agreement with Python.

## Install, then disconnect

From a clean source checkout or extracted source archive:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install .
python examples/check_example.py --help
python examples/check_example.py --output-dir example-results
```

Installation may download build/runtime dependencies; [installation](install.md)
also describes a matching wheel. Once prusie and NumPy are installed, the
check is completely offline. It does not need R, pandas, plotting libraries,
pycoloc or the original development workspace. It imports the installed package
and calls the public `susie_rss` API.

To run just the multi-signal teaching dataset:

```sh
python examples/check_example.py --case D3 --output-dir example-D3
```

The complete check runs all seven declared cases. `--case` is a convenience for
inspection; the executed report and release checks include the complete set.
Use an output directory outside `examples/data/`. The runner never overwrites
input, reference or native expectation files.

## Read PASS and FAIL

Each line gives the maximum absolute PIP error against frozen official R,
the posterior-validity status, credible-set count, iteration count and
convergence flag. `example-results/report.json` stores exact parameters,
input/order/model hashes, runtime/backend/version, warnings, per-field errors
and CS comparisons by original component ID. `actual_native.npz` contains
the newly computed numeric output, with no pickle objects.

A substantive PIP mismatch, failed input/reference checksum or invalid
input/model/probability structure produces **FAIL** and a nonzero process exit.
PIP acceptance is absolute error ≤1e-5, rtol=0. Intermediate arrays, iteration
counts and CS membership differences are diagnostics; they are not silently
used to change this gate. Internal posterior and coverage identities are checked
separately at a fixed 1e-10 absolute arithmetic tolerance.

The caller's version is recorded. A different package version from the frozen
native snapshot does not itself cause FAIL. Read the numerical and structural
results before interpreting a version difference.

## The declared cases

| Case | Purpose | Difference from primary parameters |
| --- | --- | --- |
| D1 | Standard quantitative single-signal teaching data | None |
| D2 | A second single-signal dataset on the same variable convention | None |
| D3 | Two stored causal teaching variables | None |
| D4 | A further single-signal dataset | None |
| D1_partial_zero | Prior-weight edge case | Every seventh one-based SNP weight is zero; the others are one |
| D3_strict_purity | No returned credible set | `min_abs_corr=1` |
| D3_one_iteration | Visible nonconvergence diagnostic | `max_iter=1`; explicitly not a primary fit |

The primary model uses `z=beta/sqrt(varbeta)`, the stored signed LD **r**,
`n=1000`, `L=10`, `max_iter=100`, `tol=0.001`, `coverage=0.95`,
`min_abs_corr=0.5`, `n_purity=500`, standardization, estimated prior variance
using `optim` with initial `scaled_prior_variance=0.2`, fixed residual variance,
`prior_tol=1e-9`, no null column and no refinement. The authoritative complete
arguments are `examples/data/parameters.json`.

The stored sample size is known. This RSS demonstration does not pass `var_y`
or infer phenotype variance. It uses the finite-sample z-RSS path. LD comes
from the upstream synthetic reference construction, so residual variance stays
fixed. The data do not establish ancestry, genome build, effect/counting alleles
or real genomic coordinates. Variable names `s1`–`s500` and positions 1–500 are
synthetic identities and indices.

## Three distinct kinds of expectation

1. `r_fits.npz` and `r_fits.json` are official susieR 0.16.6 outputs and supply
   the PIP comparison. `r_coloc.json` retains separate coloc results for the
   companion example.
2. `expected_native.npz` and `expected_native.json` record an explicitly labelled
   executed prusie snapshot. The check compares its new PIPs to that snapshot
   as an additional diagnostic; this is not an independent R reference.
3. The checker independently sums posterior mass from each original alpha row,
   recomputes marginal PIP across active components, and computes purity from
   the signed LD submatrix. These checks do not call package summary helpers.
   `independent_oracles.json` contains separate coloc-specific mathematical
   counterexamples for the companion package.

`manifest.json`, `checksums.json` and `native_checksums.json` keep identities
explicit. Input NPZ arrays retain the upstream float64 values. The TSV files
are readable views; `variants.tsv` specifies ordering on both matrix axes.

## Interpret the edge cases

In the pinned reference, active SER adds √ε to normalized prior weights before
taking logarithms. A supplied zero prior weight is therefore **not a hard
exclusion**. Final low-variance trimming restores the exact normalized prior.
This differs from zero-weight semantics in a colocalisation prior.

No-CS results have empty member, index, coverage and purity collections, while
PIPs remain available. Returned `sets.coverage` is actual posterior mass for
its original component, not the requested 0.95 and not repeated-sampling
coverage. The one-iteration diagnostic remains `converged=False`; the runner
does not silently retry it or present it as a converged scientific fit.

## Reproduce the reference only if needed

Users never need R to check an installation. Maintainers can separately use
`examples/data/export_reference.R`, `convert_exports.py`,
`verify_reference_input.py`, `reference.lock.json` and
`reference_environment.json`. The [provenance guide](example_provenance.md)
gives the exact pinned versions, archive/data hashes and commands. Regeneration
writes a new directory and does not update the shipped goldens.

Read the [executed agreement report](r_agreement.md) for measured errors and
the [compatibility guide](compatibility.md) before analyzing your own data.
