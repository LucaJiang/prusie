# Performance and its limits

Every public call validates inputs, performs its matrix setup, runs sequential
IBSS and constructs results and complete retained credible sets. No fitted
result cache crosses calls. Fit-local reuse and exactly verified signed matrix
redundancy preserve every original SNP. Dense fallback handles matrices without
useful exact redundancy; gains on redundant LD do not establish speed on all
populations or loci. Float64 and the model/convergence settings are unchanged.

`PRUSIE_NUM_THREADS` explicitly requests 1–32 native threads (default1).
Small matrices stay serial; matrix setup/validation may use scoped workers.
Components remain sequential. Set NumPy/BLAS/OpenMP thread counts separately
before importing numerical libraries. More requested threads can increase CPU
and allocation costs or slow small regions; thread counts alone are not proof
of speedup. Optional runtime vector math has scalar and missing-library fallbacks.

The extension keeps the Python GIL while borrowing input arrays. Use independent
processes for independent regions and budget their total threads and memory.
Do not mutate arrays concurrently during a call. No universal host-utilization
or hard aggregate-memory guarantee is supplied by the package.

An R comparison must identify actual loaded BLAS, R, susieR and dependency
versions. A Netlib baseline does not predict performance against optimized BLAS.
Measure complete calls with input loading/imports/export outside clocks and
validation, setup and result construction inside. Use one warmup, retained
outputs and repeated measurements; report CPU as well as wall time, unfavorable
cases, convergence and iteration counts. Profiling identifies costs but cannot
establish a public-call gain by itself.

The release notes distinguish measured results from historical implementation
changes. Public documentation includes no private cohort identifiers, genomic
windows or machine paths; private benchmark inputs are not distributed.

## Candidate selection rule

Each proposed optimization needs a **strict majority of faster applicable cases**
against its appropriate parent on a fixed, outcome-independent panel. A win
means a lower median complete-call runtime; ties and slower cases do not count.
Report the full denominator, wins/ties/regressions, absolute medians and repeat
variability. Aggregate time, memory savings and correctness tests cannot replace
this rule. There is no additional invented significance or percentage threshold.

The 0.2.3rc4 candidate excludes B01's known-phenotype-variance allocation change:
that branch had analytical memory and correctness evidence but no applicable
runtime-majority evidence. The historical rc3 A02+B01 bundle had135 wins and65
losses out of200; that observed majority did not qualify its unmeasured B01 branch.
Those historical artifacts remain experimental.

The current A02 package is compared as a complete change against rc17, without
B01. Its earlier pilot had8/8 wins versus rc17 and5/8 incrementally versus A01.
Those pilot counts are not an isolated all200 A01 ablation. Final selection used the bounded predeclared200-case fresh campaign described in the
[release notes](release_notes.md): one warmup, seven retained full-call
repetitions, one numerical thread and two independent regions, with each region's
backends run sequentially. There is no new optimization search in this release.

## Fresh measured outcome

A02 only meets the literal majority gate: **118 faster,0 tied,82 slower of200**
fixed trait cases versus rc17. All200 PIP/input/model checks passed, all repeat
outputs were retained and valid, and quiet-window audits passed. The sample is
the predeclared100-region/200-trait panel; no cases were dropped after timing.
This is observed case counting, not a statistical-significance claim.

| Stratum | Cases | Faster / tied / slower | rc17 summed medians (seconds) | rc4 summed medians (seconds) |
| --- | --- | --- | --- | --- |
| All fixed cases |200|118 /0 /82|1.839772626|1.707386992|
| Mutually converged |199|117 /0 /82|1.494473075|1.461272684|
| Same iteration count |200|118 /0 /82|1.839772626|1.707386992|

[All200 anonymized case medians and repeat variability](assets/performance_cases.tsv)
include every regression. They expose runtime measurements, not private study
inputs, genomic locations or frozen biological outputs. Raw per-repeat results
and source/affinity/quiet-window audits are retained in the private release record.

Each case uses one warmup and seven complete calls. Lower median means faster;
exact ties and regressions do not count as wins. Shared-host variability remains
a limitation even with quiet task controls, and another machine or model may
produce different outcomes. No whole-host utilization guarantee is claimed.

This measurement qualifies the A02 package change versus rc17. The only
incremental A02-versus-A01 evidence remains the earlier5/8 pilot; there is no
isolated all200 A01 ablation. B01 and the other excluded layout experiments do
not inherit this package's runtime acceptance.
