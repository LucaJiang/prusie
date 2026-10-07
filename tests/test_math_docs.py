"""The real documentation pipeline protects TeX, highlights code and fails closed."""

from html.parser import HTMLParser
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from build_docs import render_markdown
from math_render import check_assets


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def test_assets_are_pinned():
    check_assets()


def test_protected_math_and_accessible_output():
    source = r"""An inline $M^{(2)}_{\ell j}=\mu_{\ell j}^2+\tau_{\ell j}^2$.

$$
\begin{aligned}
\mathbf Q&=\begin{pmatrix}2&1\\1&2\end{pmatrix},\\
\mathbb{E}[a^2]&=\operatorname{Var}(a)+\mu^2,\\
\alpha_j&=\frac{\pi_j e^{w_j}}{\sum_{k=1}^{p}\pi_k e^{w_k}},\\
P_j&=1-\prod_{\ell=1}^{L}(1-\alpha_{\ell j}),\quad 0\leq P_j<1.
\end{aligned}
$$
"""
    html, _ = render_markdown(source, "mixed-fixture.md")
    assert html.count('class="katex"') == 2
    assert html.count("<math xmlns=") == 2
    assert 'encoding="application/x-tex"' in html
    assert 'aria-hidden="true"' in html
    assert "scroll horizontally" in html
    assert "<em>" not in html and "katex-error" not in html
    assert r"\mu_{\ell j}" in html
    assert "&lt;1" in html


@pytest.mark.parametrize(
    "language,code",
    [
        ("sh", 'echo "$HOME" "${array[0]}"\ncd /path_with_underscores'),
        ("python", r'tex = "$\alpha_j < 1$"' + "\nvalues = [a * b for a, b in pairs]"),
        ("r", "fit$alpha\nfit <- susie_rss(z, R, n = 1000)"),
    ],
)
def test_math_and_highlighted_code_copy_original_text(language, code):
    html, _ = render_markdown(
        f"$x_j$ and `echo $PATH`\n\n```{language}\n{code}\n```", "code.md"
    )
    assert html.count('class="katex"') == 1
    assert "<code>echo $PATH</code>" in html
    pre = html[html.index("<pre>") : html.index("</pre>")]
    assert "<span class=" in pre
    text = Text()
    text.feed(pre)
    assert "".join(text.parts).rstrip("\n") == code


def test_long_formula_has_local_scroll_region():
    source = "$$\n" + r"\sum_{j=1}^{p}Q_{jj}\alpha_j M^{(2)}_j" * 12 + "\n$$"
    html, _ = render_markdown(source, "long.md")
    assert '<div aria-label="Mathematical formula;' in html
    assert 'tabindex="0"' in html


def test_protected_math_inside_optional_details():
    source = '<details markdown="1"><summary>Derivation</summary>\n\n$x_j*y_j < 1$\n\n</details>'
    html, _ = render_markdown(source, "details.md")
    assert 'class="katex"' in html
    assert "<em>" not in html and "x_j*y_j" in html


def test_error_identifies_page_formula_and_source():
    with pytest.raises(ValueError) as error:
        render_markdown(r"$\doesnotexist{x}$", "broken-model.md")
    assert "broken-model.md" in str(error.value)
    assert "formula 1" in str(error.value)
    assert "doesnotexist" in str(error.value)
