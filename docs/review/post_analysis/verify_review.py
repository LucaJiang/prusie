"""Verify retained review evidence without fitting, rendering or contacting AI.

Run from any directory: python docs/review/post_analysis/verify_review.py
Only the Python standard library is required. REPRODUCE.md explains recovery
of the two historical HTML presentations whose manifest hashes remain intact.
"""

import csv
import hashlib
import json
from pathlib import Path
import re
import struct
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
ARCHIVED_HTML = {"examples/stable/index.html", "examples/problem/index.html"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    errors = []
    counts = dict(manifest_files=0, archived_html=0, json_files=0, evidence_facts=0,
                  pngs=0, relative_links=0)
    manifest = json.loads((ROOT / "artifact_manifest.json").read_text())
    for row in manifest["files"]:
        path = ROOT / row["destination"]
        if row["destination"] in ARCHIVED_HTML and not path.exists():
            counts["archived_html"] += 1
            continue
        if not path.is_file() or sha(path) != row["sha256"]:
            errors.append("Artifact hash mismatch: " + row["destination"])
        counts["manifest_files"] += 1
    for path in ROOT.rglob("*.json"):
        json.loads(path.read_text())
        counts["json_files"] += 1
    for case, expected_p in [("stable", 500), ("problem", 200)]:
        directory = ROOT / "examples" / case
        evidence = json.loads((directory / "evidence.json").read_text())
        canonical = json.dumps(evidence["data"], sort_keys=True, ensure_ascii=False,
                               allow_nan=False, separators=(",", ":"))
        if hashlib.sha256(canonical.encode()).hexdigest() != evidence["sha256"]:
            errors.append(case + ": canonical evidence hash mismatch")
        for fact in evidence["facts"]:
            value = evidence["data"]
            for key in fact["path"].split("/")[1:]:
                key = key.replace("~1", "/").replace("~0", "~")
                value = value[int(key)] if isinstance(value, list) else value[key]
            expected_id = "ev_" + hashlib.sha256(fact["path"].encode()).hexdigest()[:20]
            if value != fact["value"] or expected_id != fact["id"]:
                errors.append(case + ": bad evidence reference " + fact["id"])
            counts["evidence_facts"] += 1
        fitdoc = json.loads((directory / "fits/analysis.json").read_text())
        if fitdoc["arrays_sha256"] != sha(directory / "fits/arrays.npz"):
            errors.append(case + ": fit array archive mismatch")
        with (directory / "tables/variants.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        variants = evidence["data"]["summary"]["variants"]
        if len(rows) != expected_p or len(variants) != expected_p:
            errors.append(case + ": variant count mismatch")
        for row, variant in zip(rows, variants):
            if row["variant_id"] != variant["variant_id"] or float(row["pip"]) != variant["pip"]:
                errors.append(case + ": CSV/evidence variant mismatch")
        for index in range(1, 8):
            path = directory / f"figure_{index}.png"
            data = path.read_bytes()
            if not data.startswith(b"\x89PNG\r\n\x1a\n") or min(struct.unpack(">II", data[16:24])) == 0:
                errors.append(str(path) + ": invalid PNG header")
            counts["pngs"] += 1
    for path in [REPO / "README.md", *ROOT.rglob("*.md")]:
        for target in re.findall(r'!?\[[^\]]*\]\(([^\s)]+)(?:\s+"[^"]*")?\)', path.read_text()):
            if target.startswith(("http:", "https:", "mailto:", "#")):
                continue
            target = unquote(target.split("#")[0])
            if not (path.parent / target).exists():
                errors.append("Broken link: " + str(path.relative_to(REPO)) + " -> " + target)
            counts["relative_links"] += 1
    print(json.dumps(dict(passed=not errors, counts=counts, errors=errors), indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
