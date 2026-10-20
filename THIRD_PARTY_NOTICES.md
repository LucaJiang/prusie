# Third-party notices

prusie is GPL-3.0-or-later; see [LICENSE](LICENSE).

## Adapted source code

### susieR

Gaussian SuSiE and SuSiE-RSS code is adapted from
[susieR 0.16.6](https://github.com/stephenslab/susieR/tree/8e56a8e038e989856d106d9ca5175cc664fea9d2),
commit `8e56a8e038e989856d106d9ca5175cc664fea9d2`, under BSD-3-Clause:

Copyright (c) 2017-2022, Gao Wang, Peter Carbonetto, Yuxin Zou, Kaiqian Zhang,
Matthew Stephens.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of Gao Wang, Peter Carbonetto, Yuxin Zou, Kaiqian
   Zhang, Matthew Stephens, nor the names of its contributors may be used
   to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

### R Brent_fmin

[`rust/prusie-core/src/optimizer.rs`](rust/prusie-core/src/optimizer.rs) adapts
[R 4.4 optimize.c](https://svn.r-project.org/R/branches/R-4-4-branch/src/library/stats/src/optimize.c).
Copyright (C) 1995, 1996 Robert Gentleman and Ross Ihaka;
Copyright (C) 2003–2004 The R Foundation;
Copyright (C) 1998–2023 The R Core Team.
The source retains its GPL-2.0-or-later permission notice and is used under
GPL-3.0 as part of prusie.

## Rust libraries linked into the binary wheel

The Linux x86-64 wheel uses the following libraries. MIT is selected wherever
it is offered; the common MIT permission below applies to these attributions,
except for rust-numpy's separate BSD-2-Clause notice.
Crate sources are available at `https://crates.io/crates/NAME/VERSION`.

| Crate | Version | Copyright / authors |
| --- | --- | --- |
| cfg-if | 1.0.5 | Copyright (c) 2014 Alex Crichton |
| libc | 0.2.189 | Copyright (c) The Rust Project Developers |
| matrixmultiply | 0.3.11 | Copyright (c) 2016–2023, 2026 Ulrik Sverdrup “bluss”; (c) 2018 R. Janis Goldschmidt; (c) 2021 DutchGhost and matrixmultiply authors |
| memoffset | 0.9.1 | Copyright (c) 2017, 2020 Gilad Naaman; (c) 2020 Ralf Jung |
| ndarray | 0.16.1 | Copyright (c) 2014–2024 Ulrik Sverdrup “bluss”, Jim Turner and ndarray developers |
| num-complex | 0.4.6 | Copyright (c) 2013–2014 The Rust Project Developers |
| num-integer | 0.1.47 | Copyright (c) 2013–2014 The Rust Project Developers |
| num-traits | 0.2.19 | Copyright (c) 2014 The Rust Project Developers |
| numpy (rust-numpy; BSD-2-Clause) | 0.23.0 | Copyright (c) 2017 Toshiki Teramura |
| once_cell | 1.21.4 | The once_cell developers |
| pyo3, pyo3-ffi | 0.23.5 | Copyright (c) 2023–present PyO3 Project and Contributors; (c) 2017 Daniel Grunwald |
| rawpointer | 0.2.1 | Copyright (c) 2015; Copyright 2016–2019 bluss and rawpointer developers |
| rustc-hash | 2.1.3 | The rustc-hash developers |
| unindent | 0.2.4 | The unindent developers |

The statically linked Rust 1.98.1 standard library is MIT-licensed:
Copyright (c) The Rust Project Contributors / Developers
([contributors](https://thanks.rust-lang.org)); its backtrace code also retains
Copyright (c) 2014 Alex Crichton and its channels
Copyright (c) 2019 The Crossbeam Project Developers.
Its included libraries share the MIT permission:

| Library | Version | Copyright / authors |
| --- | --- | --- |
| addr2line | 0.25.1 | Copyright (c) 2016–2018 The gimli Developers |
| adler2 | 2.0.1 | The adler2 developers |
| cfg-if | 1.0.4 | Copyright (c) 2014 Alex Crichton |
| gimli | 0.32.3 | Copyright (c) 2015 The Rust Project Developers; Copyright 2016 The rust-dwarf Developers |
| hashbrown | 0.17.1 | Copyright (c) 2016 Amanieu d’Antras |
| libc | 0.2.185 | Copyright (c) The Rust Project Developers |
| memchr | 2.7.6 | Copyright (c) 2015 Andrew Gallant |
| miniz_oxide | 0.8.9 | Copyright 2013–2014 RAD Game Tools and Valve Software; 2010–2014 Rich Geldreich and Tenacious Software LLC; (c) 2017 Frommi; (c) 2017–2024 oyvindln |
| object | 0.37.3 | Copyright (c) 2015 The Gimli Developers |
| rustc-demangle | 0.1.27 | Copyright (c) 2014 Alex Crichton |

### MIT permission

Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

### rust-numpy: BSD-2-Clause

BSD 2-Clause License

Copyright (c) 2017, Toshiki Teramura
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

### Rust Unicode tables: Unicode-3.0

Rust's character tables retain Copyright (c) 1991–2024 Unicode, Inc.,
under the following permission:

UNICODE LICENSE V3

COPYRIGHT AND PERMISSION NOTICE

Copyright © 1991-2023 Unicode, Inc.

NOTICE TO USER: Carefully read the following legal agreement. BY
DOWNLOADING, INSTALLING, COPYING OR OTHERWISE USING DATA FILES, AND/OR
SOFTWARE, YOU UNEQUIVOCALLY ACCEPT, AND AGREE TO BE BOUND BY, ALL OF THE
TERMS AND CONDITIONS OF THIS AGREEMENT. IF YOU DO NOT AGREE, DO NOT
DOWNLOAD, INSTALL, COPY, DISTRIBUTE OR USE THE DATA FILES OR SOFTWARE.

Permission is hereby granted, free of charge, to any person obtaining a
copy of data files and any associated documentation (the "Data Files") or
software and any associated documentation (the "Software") to deal in the
Data Files or Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, and/or sell
copies of the Data Files or Software, and to permit persons to whom the
Data Files or Software are furnished to do so, provided that either (a)
this copyright and permission notice appear with all copies of the Data
Files or Software, or (b) this copyright and permission notice appear in
associated Documentation.

THE DATA FILES AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF
THIRD PARTY RIGHTS.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS INCLUDED IN THIS NOTICE
BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL INDIRECT OR CONSEQUENTIAL DAMAGES,
OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THE DATA
FILES OR SOFTWARE.

Except as contained in this notice, the name of a copyright holder shall
not be used in advertising or otherwise to promote the sale, use or other
dealings in these Data Files or Software without prior written
authorization of the copyright holder.

## Data

| Dataset | Source | Terms |
| --- | --- | --- |
| `src/prusie/data/convergence` | [susieR FinemappingConvergence](https://github.com/stephenslab/susieR/blob/1f9166cf88ed19c1d6acd4ee53a605ff9505a563/data/FinemappingConvergence.RData) | BSD-3-Clause, susieR notice above; synthetic response with real-genotype-derived sufficient statistics |
| `tests/fixtures/susier_regression` | [coloc 6.0.3 toy fixture](https://github.com/chr1swallace/coloc/tree/8f20f0bc5e60ffc99e4c2f787bd55fd30cfe7c45), Chris Wallace and contributors | Upstream declares “GPL” without a version; the supplied `LICENSE.GPL-3` and provenance are retained |
| `tests/fixtures/susier_parity_0166` | Deterministic subsets of susieR 0.16.6 N2finemapping and N3finemapping, with raw R outputs | BSD-3-Clause; the complete susieR notice is retained with the fixture |
| `src/prusie/data/toy` | Independently generated prusie toy data, Copyright (c) 2026 Wenxin Jiang | GPL-3.0-or-later; see LICENSE |

## Documentation assets

[KaTeX 0.16.22](https://github.com/KaTeX/KaTeX/tree/v0.16.22) and its fonts are
included only in the repository and documentation site, not the wheel or sdist.
The MIT notice, font copyrights and full OFL-1.1 terms are in
[`docs/assets/katex/LICENSE`](https://github.com/LucaJiang/prusie/blob/main/docs/assets/katex/LICENSE).

### libm 0.2.15

The native solver uses rust-lang/libm for special functions in inverse-gamma and slot priors. Its MIT-compatible terms and retained upstream notices are included in `licenses/libm-0.2.15-LICENSE.txt`. The package license remains GPL-3.0-or-later.

## Sparse matrix comparison rules

The sparse symmetry comparison in `src/prusie/utilities.py` adapts the comparison rules of Matrix 1.7-6. Copyright (c) 1999–2020 Douglas Bates and Martin Maechler; Copyright (c) 2021–2024 Douglas Bates, Martin Maechler and Mikael Jagan. Matrix's full license record, including GPL-3 terms for its R code, is retained in `licenses/Matrix-1.7-6-LICENCE.txt`. No compiled Matrix or SuiteSparse library is bundled or linked.

## Optional system vector mathematics

On supported GNU/Linux x86-64 systems, the exponential helper can dynamically load GNU C Library `libmvec` symbols. The system library is optional and is not bundled; unavailable symbols, unsupported CPUs and nonstandard floating-point controls use the scalar fallback. The applicable LGPL-2.1-or-later terms and retained host copyright record are in `licenses/glibc-LGPL-2.1.txt` and `licenses/glibc-system-copyright.txt`. The copyright record's version wording does not identify the runtime library version.
