# Third-party sources and notices

prusie is Copyright (c) 2026 Wenxin Jiang and is distributed under
**GPL-3.0-or-later**. The complete primary license is [LICENSE](LICENSE).
The components below retain their original terms. The linked texts accompany
the wheel and source distribution; the inventory also covers website assets,
which are not installed as Python runtime code.

## Scientific sources and data

| Source | Use and distribution | Copyright and terms |
| --- | --- | --- |
| [susieR 0.16.6](https://github.com/stephenslab/susieR/tree/8e56a8e038e989856d106d9ca5175cc664fea9d2) | Gaussian SuSiE/SuSiE-RSS algorithms and adapted implementation in source and native wheels | Gao Wang, Peter Carbonetto, Yuxin Zou, Kaiqian Zhang, Matthew Stephens, 2017–2022; [BSD-3-Clause](licenses/susieR-BSD-3-Clause.txt) |
| [R 4.4 Brent_fmin](https://svn.r-project.org/R/branches/R-4-4-branch/src/library/stats/src/optimize.c) | Adaptation in `rust/optimizer.rs`, also compiled into wheels | Robert Gentleman, Ross Ihaka, R Foundation and R Core Team; [original GPL-2.0-or-later notice](licenses/R-optimizer.txt); combined package under GPL-3.0-or-later |
| [coloc 6.0.3 teaching fixture](https://github.com/chr1swallace/coloc/tree/8f20f0bc5e60ffc99e4c2f787bd55fd30cfe7c45) | Frozen format conversions and independent R references in `examples/data`, source distribution only | Chris Wallace and coloc contributors; upstream declares **GPL**, without a version |
| [susieR FinemappingConvergence](https://github.com/stephenslab/susieR/blob/1f9166cf88ed19c1d6acd4ee53a605ff9505a563/data/FinemappingConvergence.RData) | Toy sufficient statistics in `src/prusie/data/convergence`, source and wheel | Same susieR holders; [BSD-3-Clause](licenses/susieR-BSD-3-Clause.txt) |
| prusie teaching data | Independently generated example in `src/prusie/data/teaching`, source and wheel | Wenxin Jiang, 2026; GPL-3.0-or-later |

The susieR source commit is `8e56a8e038e989856d106d9ca5175cc664fea9d2`.
Its source archive SHA256 is
`c88c6324da061c83ac9fbac23972cd1e501210c916596ea7678c3115b33f6971`.
The earlier susieR 0.14.2 reference is retained: its
[CRAN archive](https://cran.r-project.org/src/contrib/Archive/susieR/susieR_0.14.2.tar.gz)
has SHA256 `ba02322eb1f7a7cc024c9278aa7903a34d8ad5d6f3b12c168374bc6214ed2c6e`
and the same copyright holders and BSD terms.

The frozen data directories retain their original provenance and checksum files.
`examples/data/LICENSE.GPL-3` is a supplied GPLv3 text, not evidence that coloc
specified version 3. The unversioned upstream declaration remains unchanged.
The convergence example has a simulated response and real-genotype-derived
sufficient statistics; it contains no individual genotypes.

## Documentation assets

[KaTeX 0.16.22](https://registry.npmjs.org/katex/-/katex-0.16.22.tgz) supplies
`tools/vendor/katex/katex.js` and `docs/assets/katex`. Its JavaScript/CSS use
[MIT](licenses/MIT.txt), with Khan Academy and contributor attribution.
The unchanged font files and SVG glyph material carry **OFL-1.1**, including
Design Science and Khan Academy copyrights and the original reserved font names.
React-derived utilities retain Facebook's Apache-2.0 attribution.
See the [specific notices](licenses/KaTeX.txt), [OFL-1.1](licenses/OFL-1.1.txt)
and [Apache-2.0](licenses/Apache-2.0.txt). Asset hashes remain pinned in
`tools/math-assets.lock.json`.

## Locked Rust dependencies

The table records every registry component in `Cargo.lock`. Official source
archives are available from `https://static.crates.io/crates/NAME/NAME-VERSION.crate`;
their registry checksums are in the lockfile. No dependency source tree is vendored
into the sdist. Runtime crates contribute code to the native extension; build
and procedural-macro crates run during compilation. Their notices are retained
conservatively for generated or embedded code. The project has no direct
development-only Cargo dependencies.

For components offering MIT OR Apache-2.0, this distribution selects MIT.
All original MIT copyrights, including notices that supply no named holder,
are recorded in [Rust-components.txt](licenses/Rust-components.txt), with one
identical common [MIT grant](licenses/MIT.txt). NumPy's **Rust crate** is
[BSD-2-Clause](licenses/numpy-rust-BSD-2-Clause.txt), separately from the Python
NumPy dependency. `unicode-ident` additionally requires
[Unicode-3.0](licenses/Unicode-3.0.txt). `target-lexicon` uses
[Apache-2.0](licenses/Apache-2.0.txt) with the [LLVM exception](licenses/LLVM-exception.txt).

| Component | Version | Scope | Selected terms |
| --- | --- | --- | --- |
| `autocfg` | 1.5.1 | build/code generation | MIT |
| `cfg-if` | 1.0.5 | transitive/runtime | MIT |
| `heck` | 0.5.0 | build/code generation | MIT |
| `indoc` | 2.0.7 | build/code generation | MIT |
| `libc` | 0.2.189 | transitive/runtime | MIT |
| `matrixmultiply` | 0.3.11 | transitive/runtime | MIT |
| `memoffset` | 0.9.1 | transitive/runtime | MIT |
| `ndarray` | 0.16.1 | direct/runtime | MIT |
| `num-complex` | 0.4.6 | transitive/runtime | MIT |
| `num-integer` | 0.1.47 | transitive/runtime | MIT |
| `num-traits` | 0.2.19 | transitive/runtime | MIT |
| `numpy` | 0.23.0 | direct/runtime | BSD-2-Clause |
| `once_cell` | 1.21.4 | transitive/runtime | MIT |
| `portable-atomic` | 1.15.0 | target-dependent/runtime | MIT |
| `portable-atomic-util` | 0.2.8 | target-dependent/runtime | MIT |
| `proc-macro2` | 1.0.107 | build/code generation | MIT |
| `pyo3` | 0.23.5 | direct/runtime | MIT |
| `pyo3-build-config` | 0.23.5 | build/code generation | MIT |
| `pyo3-ffi` | 0.23.5 | transitive/runtime | MIT |
| `pyo3-macros` | 0.23.5 | build/code generation | MIT |
| `pyo3-macros-backend` | 0.23.5 | build/code generation | MIT |
| `quote` | 1.0.47 | build/code generation | MIT |
| `rawpointer` | 0.2.1 | transitive/runtime | MIT |
| `rustc-hash` | 2.1.3 | transitive/runtime | MIT |
| `rustversion` | 1.0.23 | build/code generation | MIT |
| `syn` | 2.0.119 | build/code generation | MIT |
| `target-lexicon` | 0.12.16 | build/code generation | Apache-2.0 WITH LLVM-exception |
| `unicode-ident` | 1.0.26 | build/code generation | MIT AND Unicode-3.0 |
| `unindent` | 0.2.4 | transitive/runtime | MIT |

Reviewed feature configuration: Cargo has no default feature; `extension-module`
enables `pyo3/extension-module`, and `vector-math` adds the project's C helper on
native Linux x86-64 GNU builds. PyO3's default features remain enabled.
`portable-atomic` and `portable-atomic-util` cover target-specific atomic support;
their presence in the lockfile does not imply inclusion in every wheel.

## Rust standard library and system libraries

Rust toolchain: `1.98.1`. The standard library is statically linked into the native
extension. Its [runtime notices](licenses/Rust-runtime.txt) retain Rust, backtrace,
channel, Unicode, compiler-builtins and libm attribution, and the notices for
addr2line 0.25.1, adler2 2.0.1, cfg-if 1.0.4, gimli 0.32.3, hashbrown 0.17.1,
libc 0.2.185, memchr 2.7.6, miniz_oxide 0.8.9, object 0.37.3 and
rustc-demangle 0.1.27. These are separate from versions in the project's lockfile.
Common terms are [MIT](licenses/MIT.txt), [Apache-2.0](licenses/Apache-2.0.txt),
[LLVM exception](licenses/LLVM-exception.txt) and [Unicode-3.0](licenses/Unicode-3.0.txt).
Compiler-builtins requires MIT **and** Apache-2.0 WITH LLVM-exception.
Its [original libm file notices](licenses/Rust-libm.txt) additionally preserve
Sun/fdlibm permission, David Schultz's BSD terms and CORE-MATH attribution.

The compiler executable and its development dependencies are not redistributed.
Python, Python NumPy, and dynamically linked operating-system libraries are
separately supplied dependencies; their full packages are not bundled in prusie.
Building with another toolchain or for another platform requires reviewing its
runtime notices before distributing the resulting wheel.
