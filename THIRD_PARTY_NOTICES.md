# Third-party notices

The combined prusie distribution is licensed under GPL-3.0-or-later.
Third-party components retain the licenses and copyright notices below.

## susieR

The Gaussian SuSiE algorithms and API semantics are adapted in Python and Rust
from official susieR 0.16.6, commit
`8e56a8e038e989856d106d9ca5175cc664fea9d2`. Source:
https://github.com/stephenslab/susieR/tree/8e56a8e038e989856d106d9ca5175cc664fea9d2
Archive SHA256: `c88c6324da061c83ac9fbac23972cd1e501210c916596ea7678c3115b33f6971`.
The upstream copyright file is retained as `rust/LICENSE-susieR-0.16.6`.
The earlier 0.14.2 attribution is retained in `rust/LICENSE-susieR` and
`rust/LICENSE-susieR-BSD-3-clause`; its source archive SHA256 is
`ba02322eb1f7a7cc024c9278aa7903a34d8ad5d6f3b12c168374bc6214ed2c6e`.
Source: https://cran.r-project.org/src/contrib/susieR_0.14.2.tar.gz.

susieR is BSD-3-Clause licensed. Its retained permission and disclaimer follow.

Copyright (c) 2017–2022, Gao Wang, Peter Carbonetto, Yuxin Zou,
Kaiqian Zhang, Matthew Stephens. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## R optimizer

`rust/optimizer.rs` adapts the `Brent_fmin` routine from R 4.4
`src/library/stats/src/optimize.c`, retrieved on 2026-10-01 from
https://svn.r-project.org/R/branches/R-4-4-branch/src/library/stats/src/optimize.c.
The adaptation is Rust code preserving R's step selection and stopping rules.
Copyright (C) 1995, 1996 Robert Gentleman and Ross Ihaka; (C) 2003–2004
The R Foundation; (C) 1998–2023 The R Core Team. The source header specifies
GPL-2.0-or-later; its full license is retained in `rust/LICENSE-R`.
The underlying method is Richard Brent's *Algorithms for Minimization without
Derivatives* (1973). The combined GPL-3.0-or-later distribution retains these
file-level terms. susieR's BSD license is separate from this R-derived code.

## Rust dependencies and binary distributions

Dependency versions and source checksums are in `Cargo.lock`. Copyright and
license texts for the bundled Rust dependencies are retained under
`src/prusie/_licenses/cargo/`. The wheel includes these files, the upstream R
and susieR license files, and notices under `prusie/_licenses/`; its primary
license is also included in the wheel's distribution metadata.

## Bundled teaching data

The source distribution includes `examples/data/`, converted from the existing
official coloc 6.0.3 teaching fixture, commit
`8f20f0bc5e60ffc99e4c2f787bd55fd30cfe7c45`. Attribution belongs to Chris Wallace
and the coloc contributors. The pinned package declares GPL without a version;
that declaration is retained and GPL-3 text is supplied as
`examples/data/LICENSE.GPL-3`. Conversion changes storage format only.
Source/data hashes, copied upstream provenance and the regeneration recipe are
in `examples/data/README.md` and `examples/data/reference.lock.json`.
