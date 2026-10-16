#!/usr/bin/env python3
"Render all docs/*.md deterministically; pip install -r tools/docs-requirements.txt."

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import markdown
from license_check import check_licenses, license_paths
from math_render import StaticMath, check_assets
from pygments.formatters import HtmlFormatter

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def render_markdown(source: str, page: str):
    "Use the same protected math/highlighting pipeline for pages and tests."
    md = markdown.Markdown(
        extensions=[
            "fenced_code",
            "tables",
            "toc",
            "sane_lists",
            "attr_list",
            "md_in_html",
            "codehilite",
            "pymdownx.arithmatex",
            StaticMath(page),
        ],
        extension_configs={
            "codehilite": {
                "guess_lang": False,
                "linenums": False,
                "css_class": "highlight",
                "pygments_style": "friendly",
            },
            "toc": {"toc_depth": "2-3", "permalink": False},
            "pymdownx.arithmatex": {
                "generic": True,
                "inline_syntax": ["dollar"],
                "block_syntax": ["dollar"],
            },
        },
    )
    return md.convert(source), md


def decorate_body(body: str) -> str:
    "Add local scroll regions and original-image links to rendered content."
    body = body.replace(
        "<pre>", ('<pre tabindex="0" aria-label="Code example; scroll horizontally if needed">')
    )
    body = re.sub(r'(href="[^"#?:]+)\.md(?=[#"])', r"\1.html", body)
    body = re.sub(
        r"(<table>.*?</table>)",
        (
            '<div class="table-wrap" role="region" aria-label="Data '
            'table" tabindex="0" data-scroll-label="Scroll '
            'horizontally to view all columns.">\\1</div>'
        ),
        body,
        flags=re.S,
    )

    def figure(match):
        image = match[1]
        source = re.search(r'\bsrc="([^"]+)"', image)
        if source is None:
            return match[0]
        classes = re.search(r'\bclass="([^"]*)"', image)
        complex_figure = classes is not None and "figure-complex" in classes[1].split()
        wrapper_class = "figure-wrap figure-complex" if complex_figure else "figure-wrap"
        # Source attributes have already been HTML-escaped by Markdown.
        return (
            f'<figure class="doc-figure"><div class="{wrapper_class}'
            f'" role="region" '
            'aria-label="Scientific figure" tabindex="0" '
            'data-scroll-label="Scroll horizontally to inspect the full figure.">'
            f"{image}"
            f'</div><figcaption><a href="{source[1]}'
            f'">Open original figure</a>'
            "</figcaption></figure>"
        )

    return re.sub(r"<p>\s*(<img [^>]+>)\s*</p>", figure, body)


def page_toc(filename: str, markdown_document) -> str:
    "Keep in-page navigation on long pages, after a direct homepage opening."
    if filename == "index.md" or not markdown_document.toc_tokens:
        return ""
    return (
        '<details class="on-this-page"><summary>On this page</summary>'
        f"{markdown_document.toc}"
        f"</details>"
    )


def license_site_files(root: Path) -> dict[str, str]:
    """Publish a short landing page, component inventory and separate legal texts."""

    def page(title, body):
        return (
            '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{title} · prusie</title>"
            "<style>body{max-width:90ch;margin:2em auto;padding:0 1em;font:16px/1.6 sans-serif}"
            "table{border-collapse:collapse;font-size:.9em}td,th{padding:.4em;text-align:left;"
            "border-bottom:1px solid #ddd}a{overflow-wrap:anywhere}"
            ".table-wrap{overflow-x:auto}code{overflow-wrap:anywhere}</style></head>"
            '<body><main><p><a href="citation.html">Citation and license</a></p>'
            f"{body}</main></body></html>\n"
        )

    landing = (
        "<h1>License</h1><p>Copyright © 2026 Wenxin Jiang. "
        "prusie is licensed under <strong>GPL-3.0-or-later</strong>.</p>"
        '<p>Read the <a href="https://github.com/LucaJiang/prusie/blob/main/LICENSE">'
        'project LICENSE on GitHub</a> or its <a href="licenses/LICENSE">offline copy</a>.</p>'
        '<p>The <a href="third-party.html">third-party component inventory</a> identifies '
        "upstream sources, versions, copyrights and links to their applicable license texts.</p>"
    )
    inventory, _ = render_markdown(
        (root / "THIRD_PARTY_NOTICES.md").read_text(), "THIRD_PARTY_NOTICES.md"
    )

    # Inventory links to legal texts work in the standalone Pages tree. Other
    # repository files keep a public source link rather than escaping the site root.
    def source_link(match):
        target = html.unescape(match[1])
        if target == "LICENSE":
            target = "licenses/LICENSE"
        elif not target.startswith(("licenses/", "#", "https://", "http://", "mailto:")):
            target = "https://github.com/LucaJiang/prusie/blob/main/" + target
        return f'href="{html.escape(target, quote=True)}"'

    inventory = re.sub(r'href="([^"]+)"', source_link, inventory)
    outputs = {
        "notices.html": page("License", landing),
        "third-party.html": page("Third-party notices", decorate_body(inventory)),
    }
    for path in license_paths(root):
        if path != "THIRD_PARTY_NOTICES.md":
            target = "licenses/LICENSE" if path == "LICENSE" else path
            outputs[target] = (root / path).read_text()
    return outputs


def build(check: bool = False) -> None:
    check_assets()
    check_licenses(ROOT)
    config = json.loads((ROOT / "tools/site_config.json").read_text())
    version = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M)[1]
    stale = []
    for path in sorted(DOCS.glob("*.md")):
        body, md = render_markdown(path.read_text(), path.relative_to(ROOT).as_posix())
        body = decorate_body(body)
        title = re.search(r"<h1[^>]*>(.*?)</h1>", body)
        title = re.sub("<[^>]+>", "", title[1]) if title else path.stem
        nav = []
        for group, entries in config["navigation"]:
            links = []
            for label, target in entries:
                active = ' aria-current="page"' if target == path.with_suffix(".html").name else ""
                links.append(f'<a href="{html.escape(target)}"{active}>{html.escape(label)}</a>')
            nav.append(
                f'<div class="nav-group"><p class="nav-label">{html.escape(group)}'
                f"</p>{''.join(links)}"
                f"</div>"
            )
        toc = page_toc(path.name, md)
        target = config.get("redirects", {}).get(path.with_suffix(".html").name)
        redirect = (
            (
                f'<meta http-equiv="refresh" content="0; '
                f"url={html.escape(target)}"
                f'">\n'
                f'<link rel="canonical" href="{html.escape(target)}'
                f'">'
            )
            if target
            else ""
        )
        output = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="{html.escape(config["description"])}">
{redirect}
<title>{html.escape(title)} · {html.escape(config["package"])}</title>
<link rel="stylesheet" href="assets/site.css">
<link rel="stylesheet" href="assets/highlight.css">
<link rel="stylesheet" href="assets/katex/katex.min.css">
<script src="assets/site.js" defer></script>
</head>
<body class="{html.escape(config["package"])}">
<a class="skip-link" href="#main">Skip to content</a>
<div class="site-shell">
<aside class="sidebar" aria-label="Documentation sidebar">
<a class="brand" href="index.html"><span class="brand-mark" aria-hidden="true">{config["mark"]}</span><span>{config["package"]}<small>{html.escape(config["tagline"])}</small></span></a>
<div class="version"><span class="version-number">{version}</span><span class="version-badge">{"Development" if ".dev" in version else "Release candidate"}</span></div>
<details class="site-nav" open><summary>Browse documentation</summary><nav aria-label="Documentation">{"".join(nav)}</nav></details>
<div class="sidebar-note">Python software · Wenxin Jiang<br>Static documentation. Works offline.</div>
</aside>
<main id="main" tabindex="-1">
<div class="page-meta"><span>{html.escape(config["section"])}</span><a href="{path.name}">Markdown source</a></div>
<article>{toc}{body}</article>
<footer><p>{config["package"]} {version} · Wenxin Jiang</p><p><a href="model.html">Statistical model</a> · <a href="citation.html">Citation and license</a></p></footer>
</main>
</div>
</body>
</html>
'''  # noqa: E501 - Preserve generated HTML bytes.
        dest = path.with_suffix(".html")
        if check:
            if not dest.exists() or dest.read_text() != output:
                stale.append(str(dest.relative_to(ROOT)))
        else:
            dest.write_text(output)
    css = HtmlFormatter(style="friendly").get_style_defs(".highlight") + "\n"
    css_path = DOCS / "assets/highlight.css"
    if check:
        if not css_path.exists() or css_path.read_text() != css:
            stale.append("docs/assets/highlight.css")
    else:
        css_path.write_text(css)
    # GitHub Pages serves docs/ independently; legal text copies are generated
    # from the authoritative root files, without embedding them in the landing page.
    for name, content in license_site_files(ROOT).items():
        target = DOCS / name
        if check:
            if not target.is_file() or target.read_text() != content:
                stale.append(f"docs/{name}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
    if stale:
        raise SystemExit("Stale rendered documentation: " + ", ".join(stale))
    if not check:
        (DOCS / ".nojekyll").write_text("# Serve this directory as static GitHub Pages content.\n")
    print(f"{'Checked' if check else 'Built'} {len(list(DOCS.glob('*.md')))} static pages")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if committed HTML differs; do not write",
    )
    build(parser.parse_args().check)
