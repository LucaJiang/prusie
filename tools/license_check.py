"""Check substantive distribution notices and dependency-review coverage."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 remains supported.
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
PROJECT_LICENSE = "GPL-3.0-or-later"
# The sole primary license is intentionally immutable, unlike notice formatting.
PROJECT_LICENSE_SHA256 = "8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903"

# Check grants, conditions, disclaimers and specific attributions directly.
# These are substantive review requirements, not hashes of editorial formatting.
REQUIRED_TEXT = {
    "MIT.txt": (
        "Permission is hereby granted, free of charge, to any person obtaining a copy",
        "use, copy, modify, merge, publish, distribute, sublicense, and/or sell",
        "to permit persons to whom the Software is furnished to do so",
        "The above copyright notice and this permission notice shall be included",
        "in all copies or substantial portions of the Software",
        'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND',
        "MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT",
        "IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE",
        "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS",
    ),
    "susieR-BSD-3-Clause.txt": (
        "Copyright (c) 2017-2022, Gao Wang, Peter Carbonetto, Yuxin Zou, Kaiqian "
        "Zhang, Matthew Stephens.",
        "Redistribution and use in source and binary forms, with or without "
        "modification, are permitted",
        "Redistributions of source code must retain the above copyright notice, "
        "this list of conditions and the following disclaimer",
        "Redistributions in binary form must reproduce the above copyright "
        "notice, this list of conditions and the following disclaimer in the "
        "documentation and/or other materials provided with the distribution",
        "Neither the name of Gao Wang, Peter Carbonetto, Yuxin Zou, Kaiqian "
        "Zhang, Matthew Stephens",
        "without specific prior written permission",
        "nor the names of its contributors may be used to endorse or promote "
        "products derived from this software",
        'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"',
        "IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED",
        "IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE",
        "FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES",
        "HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY",
        "EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE",
    ),
    "numpy-rust-BSD-2-Clause.txt": (
        "Copyright (c) 2017, Toshiki Teramura",
        "Redistribution and use in source and binary forms, with or without "
        "modification, are permitted provided that the following conditions are met",
        "Redistributions of source code must retain the above copyright notice, "
        "this list of conditions and the following disclaimer",
        "Redistributions in binary form must reproduce the above copyright "
        "notice, this list of conditions and the following disclaimer in the "
        "documentation and/or other materials provided with the distribution",
        'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"',
        "IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED",
        "IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE",
        "FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES",
        "HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY",
        "EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE",
    ),
    "R-optimizer.txt": (
        "Copyright (C) 1995, 1996 Robert Gentleman and Ross Ihaka",
        "Copyright (C) 2003-2004 The R Foundation",
        "Copyright (C) 1998--2023 The R Core Team",
        "either version 2 of the License, or",
        "(at your option) any later version",
        "WITHOUT ANY WARRANTY",
        "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE",
        "The complete GPLv3 is the root LICENSE",
    ),
    "Rust-components.txt": (
        "Copyright (c) 2018 Josh Stone",
        "Copyright (c) 2014 Alex Crichton",
        "Copyright (c) 2015 The Rust Project Developers",
        "Copyright (c) The Rust Project Developers",
        'Copyright (c) 2016 - 2023 Ulrik Sverdrup "bluss"',
        "Copyirhgt (c) 2018 R. Janis Goldschmidt",
        "Copyright (c) 2021 DutchGhost [constparse.rs]",
        "Copyright (c) 2017 Gilad Naaman",
        "Copyright (c) 2020 Gilad Naaman, Ralf Jung",
        "Copyright (c) 2017 Daniel Grunwald",
        'Copyright 2026 Ulrik Sverdrup "bluss"',
        *(
            f'Copyright {years} Ulrik Sverdrup "bluss"'
            for years in (
                "2016 - 2021",
                "2016 - 2018",
                "2021-2023",
                "2016 - 2023",
                "2020, 2022",
            )
        ),
        "Copyright (c) 2021 matrixmultiply authors",
        "Copyright 2014-2024 bluss and ndarray developers.",
        *(
            f"Copyright {years} bluss and ndarray developers"
            for years in (
                "2014-2016",
                "2016",
                "2019",
                "2018",
                "2021",
                "2020",
                "2020-2021",
                "2014-2019",
                "2017",
                "2014-2020",
            )
        ),
        "Copyright 2019 ndarray developers.",
        "Copyright 2013 The Rust Project Developers.",
        "Copyright 2013-2014 The Rust Project Developers.",
        "Copyright 2016-2019 bluss and rawpointer developers.",
        'Copyright (c) 2015 - 2021 Ulrik Sverdrup "bluss", Jim Turner, and ndarray developers',
        "rawpointer 0.2.1 Copyright (c) 2015",
        "Copyright (c) 2014 The Rust Project Developers",
        "Copyright (c) 2023-present PyO3 Project and Contributors.",
        "Copyright (c) 2017-present PyO3 Project and Contributors.",
    ),
    "KaTeX.txt": (
        "Copyright (c) 2013-2020 Khan Academy and other contributors",
        "Copyright (c) 2009-2010, Design Science, Inc.",
        "Copyright (c) 2014-2018 Khan Academy",
        "Copyright (c) 2014-2017 Khan Academy",
        "Copyright 2013 Facebook, Inc.",
        "MIT.txt",
        "OFL-1.1.txt",
        "Apache-2.0.txt",
        *(
            "Reserved Font Name KaTeX_" + name
            for name in (
                "AMS",
                "Caligraphic",
                "Fraktur",
                "Main",
                "Math",
                "SansSerif",
                "Script",
                "Size1",
                "Size2",
                "Size3",
                "Size4",
                "Typewriter",
            )
        ),
    ),
    "OFL-1.1.txt": (
        "SIL OPEN FONT LICENSE Version 1.1 - 26 February 2007",
        "Permission is hereby granted, free of charge",
        "1) Neither the Font Software nor any of its individual components",
        "in Original or Modified Versions, may be sold by itself",
        "2) Original or Modified Versions of the Font Software",
        "provided that each copy contains the above copyright notice and this license",
        "as long as those fields can be easily viewed by the user",
        "3) No Modified Version of the Font Software may use the Reserved Font Name",
        "unless explicit written permission is granted by the corresponding Copyright Holder",
        "This restriction only applies to the primary font name as presented to the users",
        "4) The name(s) of the Copyright Holder(s) or the Author(s)",
        "shall not be used to promote, endorse or advertise any Modified Version",
        "except to acknowledge the contribution(s) of the Copyright Holder(s) and"
        " the Author(s) or with their explicit written permission",
        "5) The Font Software, modified or unmodified, in part or in whole",
        "must be distributed entirely under this license, and must not be "
        "distributed under any other license",
        "does not apply to any document created using the Font Software",
        "This license becomes null and void if any of the above conditions are not met",
        'THE FONT SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND',
        "IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE",
        "MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT",
        "INCLUDING ANY GENERAL, SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES",
    ),
    "Apache-2.0.txt": (
        "Apache License Version 2.0, January 2004",
        "1. Definitions.",
        "2. Grant of Copyright License.",
        "3. Grant of Patent License.",
        "4. Redistribution.",
        "5. Submission of Contributions.",
        "6. Trademarks.",
        "7. Disclaimer of Warranty.",
        "8. Limitation of Liability.",
        "9. Accepting Warranty or Additional Liability.",
        "copyright license to reproduce, prepare Derivative Works of, publicly "
        "display, publicly perform, sublicense, and distribute",
        "patent license to make, have made, use, offer to sell, sell, import, and"
        " otherwise transfer the Work",
        "any patent licenses granted to You under this License for that Work "
        "shall terminate as of the date such litigation is filed",
        "You must give any other recipients of the Work or Derivative Works a copy of this License",
        "You must cause any modified files to carry prominent notices stating "
        "that You changed the files",
        "You must retain, in the Source form of any Derivative Works that You "
        "distribute, all copyright, patent, trademark, and attribution notices",
        'If the Work includes a "NOTICE" text file',
        "include a readable copy of the attribution notices contained within such NOTICE file",
        "any Contribution intentionally submitted for inclusion in the Work by "
        "You to the Licensor shall be under the terms and conditions of this "
        "License",
        "nothing herein shall supersede or modify the terms of any separate license agreement",
        "does not grant permission to use the trade names, trademarks, service "
        "marks, or product names of the Licensor",
        "except as required for reasonable and customary use in describing the origin of the Work",
        "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND",
        "TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE",
        "shall any Contributor be liable to You for damages",
        "only if You agree to indemnify, defend, and hold each Contributor harmless",
    ),
    "LLVM-exception.txt": (
        "LLVM Exceptions to the Apache 2.0 License",
        "you may redistribute such embedded portions in such Object form",
        "without complying with the conditions of Sections 4(a), 4(b) and 4(d)",
        "you may retroactively and prospectively choose to deem waived",
        "only in their entirety and only with respect to the Combined Software",
    ),
    "Unicode-3.0.txt": (
        "UNICODE LICENSE V3",
        "Copyright © 1991-2023 Unicode, Inc.",
        "Permission is hereby granted, free of charge, to any person obtaining a copy",
        "this copyright and permission notice appear with all copies",
        "or (b) this copyright and permission notice appear in associated Documentation",
        'THE DATA FILES AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND',
        "IN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS INCLUDED IN THIS NOTICE BE LIABLE",
        "the name of a copyright holder shall not be used in advertising",
        "without prior written authorization of the copyright holder",
    ),
    "Rust-runtime.txt": (
        "Copyright (c) The Rust Project Contributors",
        "Copyright (c) 2019 The Crossbeam Project Developers",
        "Copyright (c) 2016-2018 The gimli Developers",
        "Copyright 2016 The rust-dwarf Developers",
        "Copyright (c) 2016 Amanieu d'Antras",
        "Copyright (c) 2015 Andrew Gallant",
        "Copyright 2013-2014 RAD Game Tools and Valve Software",
        "Copyright 2010-2014 Rich Geldreich and Tenacious Software LLC",
        "Copyright (c) 2017 Frommi",
        "Copyright (c) 2017-2024 oyvindln",
        "Copyright (c) 2015 The Gimli Developers",
        "Copyright (c) 1991-2024 Unicode, Inc.",
        "MIT AND Apache-2.0 WITH LLVM-exception",
        "Copyright (c) 2009-2016 by the contributors listed in CREDITS.TXT",
        "Copyright (c) 2018 Jorge Aparicio",
        "Copyright © 2005-2020 Rich Felker",
        "Copyright © 1993,2004 Sun Microsystems",
        "Copyright © 2003-2011 David Schultz",
        "Copyright © 2003-2009 Steven G. Kargl",
        "Copyright © 2003-2009 Bruce D. Evans",
        "Copyright © 2008 Stephen L. Moshier",
        "Copyright © 2017-2018 Arm Limited",
        "Copyright 2017 The Rust Project Developers",
    ),
    "Rust-libm.txt": (
        "Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.",
        "Developed at SunSoft, a Sun Microsystems, Inc. business.",
        "Developed at SunPro, a Sun Microsystems, Inc. business.",
        "Copyright (C) 2004 by Sun Microsystems, Inc. All rights reserved.",
        "Copyright 2004 Sun Microsystems, Inc. All Rights Reserved.",
        "Permission to use, copy, modify, and distribute this software is freely "
        "granted, provided that this notice is preserved",
        "Copyright (c) 2005 David Schultz <das@FreeBSD.ORG>",
        "Redistribution and use in source and binary forms, with or without "
        "modification, are permitted provided that the following conditions are met",
        "Redistributions of source code must retain the above copyright notice, "
        "this list of conditions and the following disclaimer",
        "Redistributions in binary form must reproduce the above copyright "
        "notice, this list of conditions and the following disclaimer in the "
        "documentation and/or other materials provided with the distribution",
        "THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS''",
        "IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE",
        "HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY",
        "EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE",
        "Copyright (c) 2021-2022 Alexei Sibidanov.",
        "Ported to Rust in 2025 by Trevor Gross.",
        "Conversion to float by Ian Lance Taylor, Cygnus Support",
        "Optimized by Bruce D. Evans.",
    ),
}


def license_paths(root: Path = ROOT) -> list[str]:
    """Return the license payload named by the human-readable inventory."""
    document = (root / "THIRD_PARTY_NOTICES.md").read_text()
    links = sorted(set(re.findall(r"\]\((licenses/[A-Za-z0-9_.-]+\.txt)\)", document)))
    return ["LICENSE", "THIRD_PARTY_NOTICES.md", *links]


def check_license_texts(read_text) -> dict:
    """Check actual text supplied by a source tree, wheel or sdist reader."""
    primary = read_text("LICENSE")
    if hashlib.sha256(primary.encode()).hexdigest() != PROJECT_LICENSE_SHA256:
        raise ValueError("Project GPL text differs from the complete primary license")
    document = read_text("THIRD_PARTY_NOTICES.md")
    links = set(re.findall(r"\]\((licenses/[A-Za-z0-9_.-]+\.txt)\)", document))
    if links != {"licenses/" + name for name in REQUIRED_TEXT}:
        raise ValueError("Inventory license links need review")
    for name, phrases in REQUIRED_TEXT.items():
        try:
            text = " ".join(read_text("licenses/" + name).replace("\n *", " ").split())
        except (FileNotFoundError, KeyError) as exc:
            raise ValueError(f"Missing required license material: {name}") from exc
        for phrase in phrases:
            if phrase not in text:
                raise ValueError(f"Incomplete required terms or copyright in {name}: {phrase}")
    if "upstream declares **GPL**, without a version" not in document:
        raise ValueError("Preserve coloc's unversioned upstream GPL declaration")
    return {"license_texts_verified": len(REQUIRED_TEXT), "primary_license": PROJECT_LICENSE}


def check_licenses(root: Path = ROOT) -> dict:
    result = check_license_texts(lambda name: (root / name).read_text())
    document = (root / "THIRD_PARTY_NOTICES.md").read_text()
    rows = re.findall(r"^\| `([\w-]+)` \| ([\d.]+) \|", document, re.M)
    reviewed = dict(rows)
    lock = tomllib.loads((root / "Cargo.lock").read_text())
    actual = {(p["name"], p["version"]) for p in lock["package"] if "source" in p}
    if (
        set(rows) != actual
        or len(rows) != len(set(rows))
        or any(
            p.get("source", "registry+https://github.com/rust-lang/crates.io-index")
            != "registry+https://github.com/rust-lang/crates.io-index"
            for p in lock["package"]
        )
    ):
        raise ValueError("Cargo.lock dependency versions and inventory differ; review licenses")
    cargo = tomllib.loads((root / "Cargo.toml").read_text())
    expected = {
        "pyo3": {"version": "=" + reviewed["pyo3"]},
        "numpy": "=" + reviewed["numpy"],
        "ndarray": "=" + reviewed["ndarray"],
    }
    if (
        cargo.get("dependencies") != expected
        or cargo.get("features")
        != {
            "vector-math": [],
            "extension-module": ["pyo3/extension-module"],
        }
        or any(
            cargo.get(name)
            for name in (
                "target",
                "build-dependencies",
                "dev-dependencies",
                "patch",
                "replace",
                "workspace",
            )
        )
    ):
        raise ValueError("Cargo dependencies, features or target scope require a license review")
    toolchain = tomllib.loads((root / "rust-toolchain.toml").read_text())["toolchain"]["channel"]
    if f"Rust toolchain: `{toolchain}`." not in document:
        raise ValueError("Rust toolchain runtime notices require review")
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    if (
        project.get("license") != PROJECT_LICENSE
        or cargo["package"].get("license") != PROJECT_LICENSE
    ):
        raise ValueError("Project license metadata must be GPL-3.0-or-later in Python and Cargo")
    if project.get("dependencies") != ["numpy>=1.26,<3"]:
        raise ValueError("Python runtime dependencies require a distribution license review")
    if not all(
        any(fnmatch.fnmatch(name, pattern) for pattern in project.get("license-files", []))
        for name in license_paths(root)
    ):
        raise ValueError("Python license-files does not cover required distribution notices")
    return result | {"cargo_components_verified": len(actual), "payload_paths": license_paths(root)}


if __name__ == "__main__":
    print(json.dumps(check_licenses(), indent=2))
