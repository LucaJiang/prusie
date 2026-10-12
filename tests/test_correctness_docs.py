"""Execute the documented standalone and sequential workflows as copied."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]


def blocks(relative_path):
    return re.findall(r"```(python|text)\n(.*?)\n```", (ROOT / relative_path).read_text(), re.S)


def execute_examples(examples, namespace):
    """Each text block following Python is its actual displayed stdout."""
    stdout = None
    for language, source in examples:
        if language == "python":
            stdout = StringIO()
            with redirect_stdout(stdout):
                exec(compile(source, "<documented example>", "exec"), namespace)
        elif stdout is not None:
            assert stdout.getvalue().strip() == source.strip()
            stdout = None


@pytest.mark.parametrize("page", ["README.md", "docs/index.md"])
def test_landing_example_runs_as_written(page):
    execute_examples(blocks(page), {})


def test_quickstart_and_result_fragment_share_the_declared_context():
    namespace = {}
    # The first three examples load/fit, read PIPs/CS, and summarize. Later
    # sections intentionally write reports and are checked by installed examples.
    execute_examples(blocks("docs/example.md")[:6], namespace)
    execute_examples(blocks("docs/results.md"), namespace)


def test_convergence_walkthrough_runs_as_written():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        execute_examples(blocks("docs/post_analysis.md"), {})
    finally:
        plt.close("all")


def test_api_packaged_example_is_self_contained():
    source = (ROOT / "docs/api.md").read_text().split("## load_example\n", 1)[1].split("\n## ", 1)[0]
    code = re.search(r"```python\n(.*?)\n```", source, re.S)[1]
    namespace = {}
    exec(compile(code, "<API packaged example>", "exec"), namespace)
    fit = namespace["fit"]
    assert fit.converged
    assert len(fit.pip) == 500
    assert fit.variant_ids[fit.pip.argmax()] == "syn0350"
