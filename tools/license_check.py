"""Verify complete central notices against the retained upstream text hashes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def notice_text(document: str, identifier: str) -> str:
    """Extract the exact upstream bytes between a notice's rendering markers."""
    opening = f"<!-- {identifier}:start -->\n```text\n"
    closing = f"\n```\n<!-- {identifier}:end -->"
    if document.count(opening) != 1 or document.count(closing) != 1:
        raise ValueError(f"Missing or duplicated notice: {identifier}")
    return document.split(opening, 1)[1].split(closing, 1)[0]


def check_licenses(root: Path = ROOT) -> dict:
    manifest = json.loads((root / "tools/notice_manifest.json").read_text())
    license_bytes = (root / "LICENSE").read_bytes()
    if hashlib.sha256(license_bytes).hexdigest() != manifest["project_license_sha256"]:
        raise ValueError("Project GPL text differs from the recorded full license")
    document = (root / "THIRD_PARTY_NOTICES.md").read_text()
    for notice in manifest["notices"]:
        payload = notice_text(document, notice["id"]).encode()
        if len(payload) != notice["bytes"] or hashlib.sha256(payload).hexdigest() != notice["sha256"]:
            raise ValueError(f"Incomplete or altered upstream terms: {notice['id']}")
    cargo = tomllib.loads((root / "Cargo.lock").read_text())
    dependencies = {f"{p['name']}-{p['version']}" for p in cargo["package"] if "source" in p}
    if dependencies != set(manifest["cargo_components"]):
        raise ValueError("Cargo.lock dependency versions and central notices differ")
    return {"upstream_texts_verified": len(manifest["notices"]),
            "cargo_components_verified": len(dependencies),
            "source_hashes": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                              for name in ("LICENSE", "THIRD_PARTY_NOTICES.md")}}


if __name__ == "__main__":
    print(json.dumps(check_licenses(), indent=2))
