#!/usr/bin/env python3
"""Render all docs/*.md deterministically; pip install -r tools/docs-requirements.txt."""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import markdown
from pygments.formatters import HtmlFormatter
from math_render import StaticMath, check_assets

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def render_markdown(source: str, page: str):
    """Use the same protected math/highlighting pipeline for pages and tests."""
    md = markdown.Markdown(
        extensions=["fenced_code", "tables", "toc", "sane_lists", "attr_list",
                    "codehilite", "pymdownx.arithmatex", StaticMath(page)],
        extension_configs={
            "codehilite": {"guess_lang": False, "linenums": False,
                           "css_class": "highlight", "pygments_style": "friendly"},
            "toc": {"toc_depth": "2-3", "permalink": False},
            "pymdownx.arithmatex": {"generic": True, "inline_syntax": ["dollar"],
                                    "block_syntax": ["dollar"]},
        },
    )
    return md.convert(source), md


def build(check: bool = False) -> None:
    check_assets()
    config = json.loads((ROOT / "tools/site_config.json").read_text())
    version = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M)[1]
    stale = []
    for path in sorted(DOCS.glob("*.md")):
        body, md = render_markdown(path.read_text(), path.relative_to(ROOT).as_posix())
        body = body.replace('<pre>', '<pre tabindex="0" aria-label="Code example; scroll horizontally if needed">')
        # Markdown sources remain readable; Pages links resolve within docs/.
        body = re.sub(r'(href="[^"#?:]+)\.md(?=[#"])', r'\1.html', body)
        body = re.sub(r"(<table>.*?</table>)", r'<div class="table-wrap" role="region" aria-label="Scrollable data table" tabindex="0">\1</div>', body, flags=re.S)
        body = re.sub(r'(<p><img [^>]+></p>)', r'<div class="figure-wrap" role="region" aria-label="Scrollable scientific figure" tabindex="0">\1</div>', body)
        title = re.search(r"<h1[^>]*>(.*?)</h1>", body)
        title = re.sub("<[^>]+>", "", title[1]) if title else path.stem
        nav = []
        for group, entries in config["navigation"]:
            links = []
            for label, target in entries:
                active = ' aria-current="page"' if target == path.with_suffix(".html").name else ""
                links.append(f'<a href="{html.escape(target)}"{active}>{html.escape(label)}</a>')
            nav.append(f'<div class="nav-group"><p class="nav-label">{html.escape(group)}</p>{"".join(links)}</div>')
        toc = f'<details class="on-this-page"><summary>On this page</summary>{md.toc}</details>' if md.toc_tokens else ""
        output = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="{html.escape(config['description'])}">
<title>{html.escape(title)} · {html.escape(config['package'])}</title>
<link rel="stylesheet" href="assets/site.css">
<link rel="stylesheet" href="assets/highlight.css">
<link rel="stylesheet" href="assets/katex/katex.min.css">
<script src="assets/site.js" defer></script>
</head>
<body class="{html.escape(config['package'])}">
<a class="skip-link" href="#main">Skip to content</a>
<div class="site-shell">
<aside class="sidebar" aria-label="Documentation sidebar">
<a class="brand" href="index.html"><span class="brand-mark" aria-hidden="true">{config['mark']}</span><span>{config['package']}<small>{html.escape(config['tagline'])}</small></span></a>
<span class="version">{version} <span>release candidate</span></span>
<nav aria-label="Documentation">{''.join(nav)}</nav>
<div class="sidebar-note">Python software · Wenxin Jiang<br>Static documentation. Works offline.</div>
</aside>
<main id="main" tabindex="-1">
<div class="page-meta"><span>{html.escape(config['section'])}</span><a href="{path.name}">Markdown source</a></div>
<article>{toc}{body}</article>
<footer><p>{config['package']} {version} · Wenxin Jiang</p><p><a href="model.html">Statistical model</a> · <a href="citation.html">Citation and license</a></p></footer>
</main>
</div>
</body>
</html>
'''
        dest = path.with_suffix(".html")
        if check:
            if not dest.exists() or dest.read_text() != output:
                stale.append(str(dest.relative_to(ROOT)))
        else:
            dest.write_text(output)
    css = HtmlFormatter(style="friendly").get_style_defs('.highlight') + "\n"
    css_path = DOCS / "assets/highlight.css"
    if check:
        if not css_path.exists() or css_path.read_text() != css:
            stale.append("docs/assets/highlight.css")
    else:
        css_path.write_text(css)
    if stale:
        raise SystemExit("Stale rendered documentation: " + ", ".join(stale))
    if not check:
        (DOCS / ".nojekyll").write_text("# Serve this directory as static GitHub Pages content.\n")
    print(f"{'Checked' if check else 'Built'} {len(list(DOCS.glob('*.md')))} static pages")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if committed HTML differs; do not write")
    build(parser.parse_args().check)
