"""Explanatory numbers are independent arithmetic, not new fitted datasets."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from model_examples import build, calculate


def test_document_examples_are_current():
    build(check=True)


def test_probability_and_matrix_examples():
    x = calculate()
    np.testing.assert_allclose(x["pip"], [0.82, 0.83, 0.145], rtol=0, atol=2e-16)
    assert x["cs_mass"] == 0.98
    np.testing.assert_allclose(x["cache"]["residual"], [0.8, 0.4], rtol=0, atol=2e-16)
    np.testing.assert_allclose(
        x["signed_product"], [0.6, 0.6, -0.6], rtol=0, atol=2e-16
    )
    assert x["matrix_bytes"] == 200_000_000
