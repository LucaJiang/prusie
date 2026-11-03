#!/usr/bin/env python3
"""Render documentation tables from the compact benchmark summary and the toy result."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = "docs/assets/benchmark/documentation_summary.json"
TOY_RESULT = "docs/assets/toy_result.json"


def read(path):
    return json.loads(path.read_text())


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "&#124;")

    lines = ["", "| " + " | ".join(map(cell, headers)) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(map(cell, row)) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def toy_blocks(result):
    pip_error = result["comparison"]["errors"]["pip"]["max_abs_error"]
    text = (
        f"The independent 500-variant toy example converged in {result['niter']} iterations "
        f"and returned {len(result['cs'])} credible sets. Its maximum absolute PIP difference "
        f"from the separately generated susieR 0.16.6 reference was {pip_error:.3g}.\n"
    )
    rows = [
        [
            c["component_id_0based"],
            ", ".join(f"syn{j + 1:04d}" for j in c["members_0based"]),
            f"{c['coverage']:.6f}",
            f"{c['purity']['min_abs_corr']:.6f}",
        ]
        for c in result["cs"]
    ]
    cs = table(
        ["Original component", "Synthetic variants", "Returned posterior mass", "Minimum |r|"],
        rows,
    )
    return [
        ("docs/example.md", "toy", text, "generated"),
        ("docs/numerical_accuracy.md", "toy", text, "generated"),
        ("docs/example.md", "toy-cs", cs, "generated"),
    ]


def render(root, pages=None, check=False):
    blocks = [(b["page"], b["key"], b["text"], b["prefix"]) for b in read(root / SUMMARY)["blocks"]]
    if (root / TOY_RESULT).exists():
        blocks += toy_blocks(read(root / TOY_RESULT))

    outputs = {}
    for page, key, value, prefix in blocks:
        if pages is not None and page not in pages:
            continue
        path = root / page
        text = outputs.get(path, path.read_text())
        start, end = f"<!-- {prefix}:{key}:start -->", f"<!-- {prefix}:{key}:end -->"
        assert text.count(start) == text.count(end) == 1, f"{page}: {key}"
        pattern = re.escape(start) + r".*?" + re.escape(end)
        replacement = start + "\n" + value.rstrip() + "\n" + end
        outputs[path] = re.sub(pattern, lambda _, r=replacement: r, text, flags=re.S)

    stale = [str(p.relative_to(root)) for p, text in outputs.items() if p.read_text() != text]
    if check:
        assert not stale, "Stale generated blocks: " + ", ".join(stale)
    else:
        for path, text in outputs.items():
            path.write_text(text)
    return stale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", help="Comma-separated repository-relative pages")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    pages = set(args.pages.split(",")) if args.pages else None
    changed = render(ROOT, pages=pages, check=args.check)
    print("Current blocks:", len(changed), "stale" if args.check else "updated")


if __name__ == "__main__":
    main()
