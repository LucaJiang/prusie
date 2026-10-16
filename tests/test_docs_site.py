"""Website structure uses the real Markdown/template pipeline without fitting."""

import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_docs  # noqa: E402 - Import the checkout's renderer after setting its path.


class Elements(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.elements = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


def test_figures_have_original_links_and_only_explicit_complex_scroll():
    source = (
        "![Simple heatmap](simple.png)\n\n"
        "![A & B panels](complex.png){.figure-complex}\n\n"
        "| Candidate | PIP |\n| --- | --- |\n| x | 0 |\n"
    )
    rendered, _ = build_docs.render_markdown(source, "figure-example.md")
    body = build_docs.decorate_body(rendered)
    elements = Elements(body).elements
    regions = [a for tag, a in elements if tag == "div" and "figure-wrap" in a.get("class", "")]
    assert len(regions) == 2
    assert regions[0]["class"] == "figure-wrap"
    assert regions[1]["class"] == "figure-wrap figure-complex"
    assert [a["href"] for tag, a in elements if tag == "a"] == ["simple.png", "complex.png"]
    assert [a["alt"] for tag, a in elements if tag == "img"] == ["Simple heatmap", "A & B panels"]
    assert "<table>" in body and "<td>0</td>" in body
    assert "\x01" not in body
    assert all(a.get("data-scroll-label") and a.get("tabindex") == "0" for a in regions)


def test_homepage_and_tutorial_use_distinct_navigation_chrome(tmp_path, monkeypatch):
    (tmp_path / "tools").mkdir()
    docs = tmp_path / "docs"
    (docs / "assets").mkdir(parents=True)
    (tmp_path / "tools/site_config.json").write_text((ROOT / "tools/site_config.json").read_text())
    for name in [
        *build_docs.license_paths(ROOT),
        "Cargo.lock",
        "Cargo.toml",
        "rust-toolchain.toml",
        "pyproject.toml",
    ]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    (docs / "index.md").write_text((ROOT / "docs/index.md").read_text())
    (docs / "example.md").write_text("# Example\n\n## Fit\n\nRead [results](results.md).\n")
    monkeypatch.setattr(build_docs, "ROOT", tmp_path)
    monkeypatch.setattr(build_docs, "DOCS", docs)
    build_docs.build()
    build_docs.build(check=True)
    notices = (docs / "notices.html").read_text()
    assert "GPL-3.0-or-later" in notices
    assert "Copyright © 2026 Wenxin Jiang" in notices
    assert 'href="https://github.com/LucaJiang/prusie/blob/main/LICENSE"' in notices
    assert 'href="third-party.html"' in notices
    assert len(notices) < 5000
    inventory = (docs / "third-party.html").read_text()
    for name in build_docs.license_paths(ROOT):
        if name == "THIRD_PARTY_NOTICES.md":
            continue
        target = "licenses/LICENSE" if name == "LICENSE" else name
        assert (docs / target).read_bytes() == (ROOT / name).read_bytes()
        assert f'href="{target}"' in inventory or f'href="{target}"' in notices
    # Missing or changed published text must fail the generated-page check.
    published = docs / "licenses/MIT.txt"
    published.write_text("incomplete")
    import pytest

    with pytest.raises(SystemExit, match="Stale rendered documentation"):
        build_docs.build(check=True)
    build_docs.build()
    homepage = (docs / "index.html").read_text()
    guide = (docs / "example.html").read_text()
    assert 'class="on-this-page"' not in homepage
    assert 'class="on-this-page"' in guide
    assert 'href="results.html"' in guide
    elements = Elements(homepage).elements
    assert any(
        tag == "details" and a.get("class") == "site-nav" and "open" in a for tag, a in elements
    )
    assert any(tag == "span" and a.get("class") == "version-number" for tag, a in elements)
    assert '<span class="version-badge">Development</span>' in homepage
    assert homepage.count("<table>") == 2  # Current and historical evidence only.
    assert homepage.index("current:highlights:start") < homepage.index("generated:highlights:start")
    history_start = homepage.index("<summary>Historical 0.2.3rc4 comparison</summary>")
    history_end = homepage.index("</details>", history_start)
    assert history_start < homepage.index("generated:highlights:start") < history_end
    assert 'class="start-links"' in homepage
    assert homepage.index("generated:highlights:start") < homepage.index('class="start-links"')
    assert homepage.index('class="start-links"') < homepage.index('id="one-fit-a-readable-result"')
    assert "0.2.3rc4 made on 6 October 2026" in homepage


def test_homepage_and_readme_share_current_and_historical_evidence_blocks():
    import re

    home = (ROOT / "docs/index.md").read_text()
    readme = (ROOT / "README.md").read_text()
    for kind in ("current", "generated"):
        expression = rf"<!-- {kind}:highlights:start -->(.*?)<!-- {kind}:highlights:end -->"
        home_blocks = re.findall(expression, home, re.S)
        readme_blocks = re.findall(expression, readme, re.S)
        assert len(home_blocks) == len(readme_blocks) == 1
        assert home_blocks == readme_blocks
    assert "Measured **0.2.4.dev2** against susieR 0.16.6." in home
    installation = (ROOT / "docs/install.md").read_text()
    assert "git clone --branch main" in installation
    assert "git clone --branch post_analysis" not in installation
