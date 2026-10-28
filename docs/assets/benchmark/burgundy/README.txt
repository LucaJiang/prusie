# Burgundy measurement archive

These files preserve the executed 2026-10-28 protocol and run-local timing
instrumentation. They are archival source with cluster-specific paths, not a
portable installer. They do not alter the installed package or ship genotype,
association or LD matrices. The `.txt` suffix keeps archived execution code
separate from public tooling. [checksums.json](checksums.json) authenticates
these copies; the [frozen protocol](../burgundy_protocol.json) authenticates
the original runtime scripts and inputs.

- [campaign.py.txt](campaign.py.txt): topology, staging, admission, paired case
  execution, atomic claims and result provenance.
- [m2_native.py.txt](m2_native.py.txt) and [m2_reference.R.txt](m2_reference.R.txt):
  fresh-process complete-call timing and direct fitting-stage observations.
- [reconstruct_ld.py.txt](reconstruct_ld.py.txt): signed PLINK LD, exact frozen
  variant/allele order, sample handling and validation.
- [formal_job.sh.txt](formal_job.sh.txt): exclusive Slurm allocation entry point.
- [stage_timer.py.txt](stage_timer.py.txt), [native_timing.patch](native_timing.patch)
  and [native_boundaries.txt](native_boundaries.txt): native measurement spans.
- [fit_timer.R.txt](fit_timer.R.txt), [R_boundaries.txt](R_boundaries.txt) and
  [blas_probe.c.txt](blas_probe.c.txt): R measurement and monotonic/BLAS probes.
- [environment.json](environment.json) and [qualification.json](qualification.json):
  original/derived identities and the completed eight-case qualification.
- [m2_native.py.diff](m2_native.py.diff) and [m2_reference.R.diff](m2_reference.R.diff):
  initial harness changes from the earlier complete-call timing scripts.
  [v1_to_v2.diff](v1_to_v2.diff) records the later repeat-count amendment.
- [summarize.py.txt](summarize.py.txt): full-panel authentication and aggregation.
- [v1_protocol.json](v1_protocol.json), [v1_adoption.json](v1_adoption.json) and
  [v2_smoke.json](v2_smoke.json): repeat-count amendment and v2 end-to-end smoke.

Protocol v2 uses one warmup and three retained calls per backend. All 24
completed v1 cases keep their original raw results and use their first three
retained times. Their final numerical arrays remain from retained repeat
seven and are labeled in the public records. The 31 in-flight v1 claims were
aborted at the explicit protocol amendment and restarted at 1+3, independently
of timing or numerical outcomes. Claims and completed raw outputs survived;
some unarchived node-local partial outputs from those aborts were unavailable
on recovery. No completed case was discarded.

The R_source directory contains pinned susieR 0.16.6 source files used to
verify the instrumentation boundary, with its original DESCRIPTION and LICENSE.
Preserve the original BSD-3-Clause attribution. The native patch is against
prusie commit 6d60d5e5723461c3e1fc5a46fd0a0e7b6378cfa7 and remains under the
package's applicable GPL terms. Exact source/build hashes remain in the frozen
protocol; a timing-derived binary must not be described as the original wheel.

The nodes directory retains topology and compressed host samples for each allocation.
`extreme_retained_calls.json` identifies all three retained R repeats above twice
the case median. Their shared NUMA placement is consistent with memory pressure
around the observed OOM; they remain in the reported results.
