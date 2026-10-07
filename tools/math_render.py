"""Render only the math nodes protected by Arithmatex, before HTML serialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from markdown.extensions import Extension
from markdown.treeprocessors import Treeprocessor
from markdown.util import AtomicString

ROOT = Path(__file__).resolve().parents[1]


def check_assets():
    """Pin renderer, stylesheet, fonts and notices independently of npm/network."""
    lock = json.loads((ROOT / "tools/math-assets.lock.json").read_text())
    for name, expected in lock["files"].items():
        path = ROOT / name
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(
                f"Math asset differs from tools/math-assets.lock.json: {name}"
            )


class MathTreeprocessor(Treeprocessor):
    def __init__(self, md, page):
        super().__init__(md)
        self.page = str(page)

    def run(self, root):
        nodes, formulas = [], []
        for node in root.iter():
            if node.get("class") != "arithmatex":
                continue
            display = node.tag == "div"
            wrapped = node.text or ""
            opening, closing = (r"\[", r"\]") if display else (r"\(", r"\)")
            if not (wrapped.startswith(opening) and wrapped.endswith(closing)):
                raise ValueError(
                    f"{self.page}: unexpected protected math node {wrapped!r}"
                )
            nodes.append(node)
            formulas.append(dict(tex=wrapped[2:-2], display=display))
        if not formulas:
            return
        try:
            result = subprocess.run(
                ["node", "--jitless", str(ROOT / "tools/render_math.cjs")],
                input=json.dumps(formulas),
                text=True,
                capture_output=True,
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"{self.page}: Node.js is required to build math (not to read pages)"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"{self.page}: {exc.stderr.strip()}") from exc
        rendered = json.loads(result.stdout)
        if len(rendered) != len(nodes):
            raise ValueError(f"{self.page}: incomplete math renderer response")
        for node, markup in zip(nodes, rendered):
            # Stash KaTeX HTML/MathML so subsequent Markdown processors cannot
            # reinterpret underscores, tags or entity escapes inside formulas.
            node.text = AtomicString(self.md.htmlStash.store(markup))
            if node.tag == "div":
                node.set("tabindex", "0")
                node.set("role", "region")
                node.set(
                    "aria-label", "Mathematical formula; scroll horizontally if needed"
                )


class StaticMath(Extension):
    def __init__(self, page):
        self.page = page
        super().__init__()

    def extendMarkdown(self, md):
        # Arithmatex runs during block/inline parsing. Render after the inline
        # treeprocessor (20), before prettify (10) and the table of contents (5).
        md.treeprocessors.register(MathTreeprocessor(md, self.page), "static_math", 15)
