"""PIP-based stopping, including susieR's NaN-ELBO switch, matches susieR 0.16.6."""

import json
import warnings
from pathlib import Path

import numpy as np

import prusie

FIXTURES = Path(__file__).parent / "fixtures"


def quiet(fitter, *args, **options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fitter(*args, **options)


def test_pip_history_keeps_variances_of_earlier_sweeps():
    reference = json.loads((FIXTURES / "susier_refine_0166/refine_convergence.json").read_text())
    case = reference["cases"]["pip_zero_weights"]
    result = quiet(
        prusie.susie_ss,
        **prusie.load_convergence_example()["inputs"],
        **reference["common"],
        convergence_method="pip",
        prior_weights=np.asarray(case["options"]["prior_weights"]),
    )
    np.testing.assert_allclose(result.alpha, np.asarray(case["alpha"]), rtol=0, atol=1e-5)


def test_nan_elbo_sweeps_use_the_pip_rule_by_default():
    directory = FIXTURES / "susier_parity_0166"
    case = json.loads((directory / "combination_L1.json").read_text())["cases"][
        "suff_stat_mixture_NIG_L1"
    ]
    data = json.loads((directory / "reference.json").read_text())["datasets"]["N2finemapping"]
    options = {
        key: np.asarray(value) if key in ("prior_variance_grid", "mixture_weights") else value
        for key, value in case["options"].items()
    }
    result = quiet(
        prusie.susie_ss,
        np.asarray(data["XtX"]),
        np.asarray(data["Xty"]),
        data["yty"],
        data["n"],
        **options,
    )
    assert np.isnan(result.elbo).all()
    assert result.niter == case["reference"]["niter"]
    expected = np.asarray(case["reference"]["alpha"]).reshape(result.alpha.shape)
    np.testing.assert_allclose(result.alpha, expected, rtol=0, atol=1e-5)
