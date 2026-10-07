"""Offline syntax highlighting and the HTML-parser comparison regression."""
import importlib.util
from pathlib import Path
import re
from html.parser import HTMLParser
import markdown

ROOT=Path(__file__).resolve().parents[1]


def test_highlight_tokens_and_exact_copy_text():
    class Text(HTMLParser):
        def __init__(self):super().__init__();self.text=[]
        def handle_data(self,s):self.text.append(s)
    samples={'python':'import prusie\nfit = prusie.susie_rss(z=z, R=R, n=1000)',
             'r':'library(susieR)\nfit <- susie_rss(z, R, n=1000)',
             'sh':'python -m pip install .', 'json':'{"max_iter": 100, "synthetic": true}'}
    for language,code in samples.items():
        text=markdown.markdown(f'```{language}\n{code}\n```',extensions=['fenced_code','codehilite'],
            extension_configs={'codehilite':{'guess_lang':False,'linenums':False}})
        assert re.search(r'<span class="[a-z0-9]+">',text)
        p=Text();p.feed(text)
        assert ''.join(p.text).strip()==code
        assert 'lineno' not in text


def test_literal_comparison_remains_text():
    source=(ROOT/'docs/results.md').read_text()
    assert 'V&lt;prior_tol' in source
    html=markdown.markdown(source,extensions=['tables','fenced_code','codehilite'])
    assert 'V&lt;prior_tol' in html and '<prior_tol' not in html


def test_no_guessed_languages_or_external_runtime_resources():
    source=(ROOT/'tools/build_docs.py').read_text()
    assert '"guess_lang": False' in source
    for page in (ROOT/'docs').glob('*.html'):
        text=page.read_text()
        assert not re.search(r'<(?:script|img|link)[^>]+(?:src|href)="(?:https?:)?//',text)


def test_generated_credible_set_table_preserves_absolute_correlation_header():
    """Absolute-value bars must not become Markdown table delimiters."""
    source=(ROOT/'docs/example.md').read_text()
    html=markdown.markdown(source,extensions=['tables','fenced_code','codehilite'])
    tables=re.findall(r'<table>.*?</table>',html,re.S)
    table=next(t for t in tables if 'Original component' in t)
    assert len(re.findall(r'<th>',table))==4
    assert 'Minimum &#124;r&#124;' in table
    assert 'syn0100' in table and 'syn0350' in table
