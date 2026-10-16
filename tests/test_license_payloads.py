"""Check license terms, dependency changes and distribution archives."""

import re
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from license_check import check_license_texts, check_licenses, license_paths  # noqa: E402


def source_copy(tmp_path):
    for name in [
        *license_paths(ROOT),
        "Cargo.lock",
        "Cargo.toml",
        "pyproject.toml",
        "rust-toolchain.toml",
    ]:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    return tmp_path


def test_complete_reviewed_license_content():
    check_licenses(ROOT)


def test_incomplete_primary_license_is_rejected(tmp_path):
    root = source_copy(tmp_path)
    path = root / "LICENSE"
    path.write_text(path.read_text().replace("GNU GENERAL PUBLIC LICENSE", "", 1))
    with pytest.raises(ValueError, match="complete primary license"):
        check_licenses(root)


def test_python_runtime_dependency_change_requires_review(tmp_path):
    root = source_copy(tmp_path)
    path = root / "pyproject.toml"
    path.write_text(path.read_text().replace("numpy>=1.26,<3", "numpy>=2,<4"))
    with pytest.raises(ValueError, match="Python runtime dependencies"):
        check_licenses(root)


@pytest.mark.parametrize(
    ("name", "removed"),
    [
        ("MIT.txt", "Permission is hereby granted"),
        ("MIT.txt", "copyright notice and this permission notice"),
        ("MIT.txt", "in all copies or substantial portions of the Software"),
        ("MIT.txt", 'THE SOFTWARE IS PROVIDED "AS IS"'),
        ("MIT.txt", "NONINFRINGEMENT"),
        ("susieR-BSD-3-Clause.txt", "Peter Carbonetto"),
        ("susieR-BSD-3-Clause.txt", "Neither the name"),
        (
            "susieR-BSD-3-Clause.txt",
            "documentation and/or other materials provided with the distribution",
        ),
        ("numpy-rust-BSD-2-Clause.txt", "Toshiki Teramura"),
        ("numpy-rust-BSD-2-Clause.txt", "are permitted provided that"),
        ("R-optimizer.txt", "(at your option) any later version"),
        ("Rust-components.txt", "Gilad Naaman"),
        ("Rust-components.txt", "Ralf Jung"),
        ("Rust-components.txt", "Daniel Grunwald"),
        ("KaTeX.txt", "Reserved Font Name KaTeX_AMS"),
        ("KaTeX.txt", "Copyright 2013 Facebook, Inc."),
        ("OFL-1.1.txt", "1) Neither the Font Software"),
        ("OFL-1.1.txt", "may be sold by itself"),
        (
            "OFL-1.1.txt",
            "provided that each copy contains the above copyright notice and this license",
        ),
        (
            "OFL-1.1.txt",
            "unless explicit written permission is granted by the corresponding Copyright Holder",
        ),
        ("OFL-1.1.txt", "shall not be used to promote, endorse or advertise any Modified Version"),
        ("OFL-1.1.txt", "must be distributed entirely under this license"),
        ("Apache-2.0.txt", "Grant of Patent License"),
        ("Apache-2.0.txt", "patent license to make, have made, use, offer to sell, sell, import"),
        (
            "Apache-2.0.txt",
            "only if You agree to indemnify, defend, and hold each Contributor harmless",
        ),
        ("LLVM-exception.txt", "Sections 4(a), 4(b) and 4(d)"),
        ("Unicode-3.0.txt", "copyright and permission notice"),
        ("Rust-runtime.txt", "Rich Geldreich"),
        ("Rust-runtime.txt", "rust-dwarf Developers"),
        ("Rust-libm.txt", "David Schultz"),
        ("Rust-libm.txt", "are permitted provided that"),
        ("Rust-libm.txt", "software is freely granted, provided that this notice is preserved"),
        ("Rust-libm.txt", "IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE"),
    ],
)
def test_removed_terms_or_copyright_are_rejected(tmp_path, name, removed):
    root = source_copy(tmp_path)
    path = root / "licenses" / name
    original = path.read_text()
    pattern = r"\s+".join(re.escape(word) for word in removed.split())
    changed, count = re.subn(pattern, "", original, count=1)
    assert count == 1
    path.write_text(changed)
    with pytest.raises(ValueError, match="Incomplete required terms or copyright"):
        check_licenses(root)


def test_missing_required_material_is_rejected(tmp_path):
    root = source_copy(tmp_path)
    (root / "licenses/OFL-1.1.txt").unlink()
    with pytest.raises(ValueError, match="Missing required license material"):
        check_licenses(root)


def test_inventory_link_removal_is_rejected(tmp_path):
    root = source_copy(tmp_path)
    path = root / "THIRD_PARTY_NOTICES.md"
    path.write_text(path.read_text().replace("(licenses/KaTeX.txt)", "(missing.txt)"))
    with pytest.raises(ValueError, match="Inventory license links"):
        check_licenses(root)


@pytest.mark.parametrize("extra_name", ["unreviewed-crate", "pyo3"])
def test_new_dependency_or_additional_version_requires_review(tmp_path, extra_name):
    root = source_copy(tmp_path)
    path = root / "Cargo.lock"
    with path.open("a") as stream:
        stream.write(
            f'\n[[package]]\nname = "{extra_name}"\nversion = "9.8.7"\nsource = "registry+https://github.com/rust-lang/crates.io-index"\n'
        )
    with pytest.raises(ValueError, match="dependency versions"):
        check_licenses(root)


def test_changed_dependency_source_requires_review(tmp_path):
    root = source_copy(tmp_path)
    path = root / "Cargo.lock"
    path.write_text(
        path.read_text().replace(
            "registry+https://github.com/rust-lang/crates.io-index",
            "git+https://example.invalid/other-source#12345678",
            1,
        )
    )
    with pytest.raises(ValueError, match="dependency versions"):
        check_licenses(root)


@pytest.mark.parametrize(
    "addition",
    [
        "\n[features.new_feature]\n",
        "\n[target.'cfg(windows)'.dependencies]\nlibc = \"0.2\"\n",
        '\n[build-dependencies]\nlibc = "0.2"\n',
    ],
)
def test_feature_or_target_scope_changes_require_review(tmp_path, addition):
    root = source_copy(tmp_path)
    path = root / "Cargo.toml"
    path.write_text(path.read_text() + addition)
    with pytest.raises(ValueError, match="features or target scope"):
        check_licenses(root)


@pytest.mark.parametrize("name", ["Cargo.toml", "pyproject.toml"])
def test_incorrect_primary_license_metadata_is_rejected(tmp_path, name):
    root = source_copy(tmp_path)
    path = root / name
    path.write_text(path.read_text().replace('license = "GPL-3.0-or-later"', 'license = "MIT"'))
    with pytest.raises(ValueError, match="Project license metadata"):
        check_licenses(root)


def test_toolchain_change_requires_runtime_review(tmp_path):
    root = source_copy(tmp_path)
    path = root / "rust-toolchain.toml"
    path.write_text(path.read_text().replace('channel = "1.98.1"', 'channel = "1.99.0"'))
    with pytest.raises(ValueError, match="runtime notices require review"):
        check_licenses(root)


def test_artifact_reader_rejects_missing_or_incomplete_actual_member(tmp_path):
    prefix = "prusie.dist-info/licenses/"
    for defect in (None, "missing", "incomplete"):
        wheel = tmp_path / f"licenses-{defect}.zip"
        with zipfile.ZipFile(wheel, "w") as archive:
            for name in license_paths(ROOT):
                content = (ROOT / name).read_text()
                if name == "licenses/MIT.txt":
                    if defect == "missing":
                        continue
                    if defect == "incomplete":
                        content = content.replace('THE SOFTWARE IS PROVIDED "AS IS"', "")
                archive.writestr(prefix + name, content)
        with zipfile.ZipFile(wheel) as archive:

            def read_text(name):
                return archive.read(prefix + name).decode()

            if defect is None:
                check_license_texts(read_text)
            else:
                with pytest.raises(ValueError, match=r"required license material|required terms"):
                    check_license_texts(read_text)
