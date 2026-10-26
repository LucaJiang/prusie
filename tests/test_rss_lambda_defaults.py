"""RSS-lambda effect counts use retained variants, excluding the null column."""

from inspect import signature

import numpy as np
import pytest

import prusie


def test_rss_lambda_public_L_default():
    assert signature(prusie.susie_rss_lambda).parameters["L"].default == 10


@pytest.mark.parametrize(
    "route,p,options,expected_L",
    [
        (route, p, {}, min(10, p))
        for route in ("R", "X")
        for p in (3, 10, 12)
    ]
    + [
        ("R", 12, {"L": 3}, 3),
        ("R", 12, {"L": 11}, 11),
        ("R", 3, {"L": 20}, 3),
        ("R", 3, {"null_weight": 0.2}, 3),
        ("R", 3, {"L": 20, "null_weight": 0.2}, 3),
        ("R", 3, {"L": None, "null_weight": 0.2}, 3),
        ("R", 12, {"L": None}, 10),
        ("R", 12, {"maf": [0.01] * 8 + [0.2] * 4, "maf_thresh": 0.05}, 4),
        (
            "X",
            12,
            {
                "L": 20,
                "maf": [0.01] * 8 + [0.2] * 4,
                "maf_thresh": 0.05,
                "null_weight": 0.2,
            },
            4,
        ),
    ],
)
def test_rss_lambda_retained_L_preparation_and_fit(route, p, options, expected_L):
    # Centered orthogonal reference columns give identity LD by construction.
    inputs = {"R": np.eye(p)} if route == "R" else {"X": np.vstack((np.eye(p), -np.eye(p)))}
    retained = (
        np.flatnonzero(np.asarray(options["maf"]) > options["maf_thresh"])
        if "maf" in options
        else np.arange(p)
    )
    has_null = options.get("null_weight", 0) > 0
    expected_columns = len(retained) + int(has_null)
    prepared = prusie.susie_rss_lambda(
        np.linspace(0.1, 0.8, p),
        **inputs,
        lambda_=0.2,
        prior_variance=0.2,
        estimate_prior_variance=False,
        coverage=None,
        max_iter=50,
        check_prior=False,
        variant_ids=[f"v{i}" for i in range(p)],
        init_only=True,
        **options,
    )
    assert isinstance(prepared, prusie.SusieInputs)
    assert prepared.params["L"] == expected_L
    assert prepared.params["effective_L"] == expected_L
    assert prepared._request.options["l"] == expected_L
    assert prepared.data["XtX"].shape == (expected_columns, expected_columns)
    assert prepared._request.null_index == (len(retained) if has_null else None)

    fit = prusie.susie_workhorse(prepared)
    assert isinstance(fit, prusie.SusieResult)
    assert fit.params["effective_L"] == expected_L
    assert fit.alpha.shape == (expected_L, expected_columns)
    assert list(fit.variant_ids) == [f"v{i}" for i in retained]
    assert np.isfinite(fit.alpha).all()
    np.testing.assert_allclose(fit.alpha.sum(axis=1), 1, atol=1e-12, rtol=0)


@pytest.mark.parametrize("L", [0, -1, True, 20.5, np.inf, np.nan, [20]])
def test_rss_lambda_validates_L_before_capping(L):
    with pytest.raises(ValueError, match="L must be a positive integer"):
        prusie.susie_rss_lambda([0.1, 0.2], np.eye(2), lambda_=0.2, L=L, init_only=True)
