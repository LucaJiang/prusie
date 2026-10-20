"""Check API materialization wiring and the actual native matrix operator."""

import warnings

import numpy as np
import pytest

from prusie import _native, susie_rss, susie_suff_stat


def materialized_working_statistics(args, options):
    """Construct the likelihood matrix explicitly, outside public preparation."""
    if "XtX" in args:
        gram = np.array(args["XtX"], copy=True)
        response = np.asarray(args["Xty"], dtype=float)
        yty, n = args["yty"], args["n"]
        standardize = options["standardize"]
    else:
        correlation = np.asarray(args["R"], dtype=float)
        n = args["n"]
        effects = "bhat" in args
        z = (np.asarray(args["bhat"]) / np.asarray(args["shat"])) if effects else args["z"]
        z = np.asarray(z, dtype=float)
        if n is None:
            gram, response, yty, n = correlation.copy(), z, 1.0, 2.0
            standardize = False
        else:
            adjustment = (n - 1) / (z * z + n - 2)
            adjusted = np.sqrt(adjustment) * z
            if effects:
                se = np.asarray(args["shat"])
                variance = args["var_y"]
                diagonal = variance * adjustment / se**2
                gram = correlation * np.sqrt(diagonal[:, None] * diagonal[None, :])
                response = adjusted * np.sqrt(adjustment) * variance / se
                yty = (n - 1) * variance
            else:
                gram = (n - 1) * correlation
                response, yty = np.sqrt(n - 1) * adjusted, n - 1
            standardize = options["standardize"]
    gram = (gram + gram.T) / 2
    if options["null_weight"]:
        gram = np.pad(gram, ((0, 1), (0, 1)))
        response = np.r_[response, 0.0]
    scales = np.sqrt(np.diag(gram) / (n - 1)) if standardize else np.ones(len(response))
    scales[scales == 0] = 1
    working = np.asfortranarray(gram / scales[:, None] / scales[None, :])
    return working, np.ascontiguousarray(response / scales), yty, n


def record_native_diagnostics(actual, expected, record_property):
    active = expected["V"] > actual.params["prior_tol"]
    expected_pip = 1 - np.prod(1 - expected["alpha"][active], axis=0)
    if actual.null_index is not None:
        expected_pip = np.delete(expected_pip, actual.null_index)
    record_property("pip_max_abs", float(np.max(np.abs(actual.pip - expected_pip))))
    for field in ("mu", "mu2", "V", "sigma2", "elbo", "XtXr", "KL", "lbf", "lbf_variable"):
        left, right = np.asarray(getattr(actual, field)), np.asarray(expected[field])
        if left.shape == right.shape:
            record_property(field + "_max_abs", float(np.max(np.abs(left - right))))
        else:
            record_property(field + "_shapes", [list(left.shape), list(right.shape)])
    for field in ("niter", "converged"):
        record_property(field + "_values", [getattr(actual, field), expected[field]])


def public_operator_case(kind, layout, null_weight, standardize):
    r = np.array([[1.0 + 7e-9, -0.2, -0.0], [-0.2, 1.0, 0.125], [0.0, 0.125 + 1e-14, 1.0 - 7e-9]])
    options = dict(
        L=2,
        variant_ids=["v2", "v0", "v1"],
        null_weight=null_weight,
        standardize=standardize,
        prior_weights=[0.7, 0.0, 0.3],
        coverage=None if null_weight else 0.95,
        check_input=kind in {"known_n", "no_information"},
    )
    if kind == "no_information":
        matrix = r * 99.0
        matrix[-1, :] = matrix[:, -1] = 0.0
        function = susie_suff_stat
        args = dict(Xty=[12.0, 2.0, 0.0], yty=99.0, n=100.0, estimate_residual_variance=False)
        matrix_key = "XtX"
    else:
        matrix = r
        function = susie_rss
        args = dict(n=None if kind == "missing_n" else 100.0)
        args.update(
            dict(bhat=[0.3, 0.1, -0.2], shat=[0.1, 0.2, 0.3], var_y=1.4)
            if kind == "effects"
            else dict(z=[3.0, 1.0, -0.2])
        )
        matrix_key = "R"
    if layout == "F":
        matrix = np.asfortranarray(matrix)
    elif layout == "strided":
        backing = np.empty((6, 6))
        backing[::2, ::2] = matrix
        matrix = backing[::2, ::2]
    matrix.flags.writeable = False
    before = matrix.view(np.uint64).copy()
    args[matrix_key] = matrix
    return function, args, options, matrix, before


@pytest.mark.parametrize("kind", ["known_n", "missing_n", "effects", "no_information"])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("null_weight", [0.0, 0.2])
@pytest.mark.parametrize("standardize", [False, True])
def test_default_implicit_public_boundary(
    monkeypatch, record_property, kind, layout, null_weight, standardize
):
    assert _native.MATRIX_OPERATOR_VERSION == 1
    function, args, options, matrix, before = public_operator_case(
        kind, layout, null_weight, standardize
    )
    original = _native.fit
    calls = []

    def capture(matrix, response, yty, n, settings):
        calls.append((dict(settings), response.copy(), yty, n))
        assert "matrix_global_scale" in settings
        assert "matrix_inverse_scales" in settings
        assert "matrix_scales" in settings
        return original(matrix, response, yty, n, settings)

    monkeypatch.setattr(_native, "fit", capture)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        actual = function(**args, **options)
    assert len(calls) == 1
    settings, response, yty, n = calls[0]
    working, expected_response, expected_yty, expected_n = materialized_working_statistics(
        args, options
    )
    np.testing.assert_allclose(response, expected_response, atol=1e-12, rtol=1e-12)
    assert yty == expected_yty and n == expected_n
    oracle_settings = {
        key: value for key, value in settings.items() if not key.startswith("matrix_")
    }
    expected = original(working, expected_response, expected_yty, expected_n, oracle_settings)
    shape = (2, 3 + int(null_weight > 0))
    assert actual.alpha.shape == expected["alpha"].shape == shape
    assert actual.variant_ids.tolist() == ["v2", "v0", "v1"]
    for alpha in (actual.alpha, expected["alpha"]):
        assert np.isfinite(alpha).all() and ((alpha >= 0) & (alpha <= 1)).all()
        np.testing.assert_allclose(alpha.sum(axis=1), 1, atol=1e-10, rtol=0)
    # Compare complete component rows in their original order, including null columns.
    np.testing.assert_allclose(actual.alpha, expected["alpha"], atol=1e-5, rtol=0)
    record_native_diagnostics(actual, expected, record_property)
    np.testing.assert_array_equal(matrix.view(np.uint64), before)


@pytest.mark.parametrize("capability", [0, 2])
def test_unknown_capability_retains_materialized_contract(monkeypatch, capability):
    original = _native.fit
    called = []

    def capture(matrix, response, yty, n, options):
        assert not any(key.startswith("matrix_") for key in options)
        assert matrix.flags.f_contiguous
        called.append(True)
        return original(matrix, response, yty, n, options)

    monkeypatch.setattr(_native, "MATRIX_OPERATOR_VERSION", capability, raising=False)
    monkeypatch.setattr(_native, "fit", capture)
    susie_rss([2.0, 0.0], np.eye(2), 100, L=1)
    assert called


def test_capability_keeps_public_invalid_matrix_rejection(monkeypatch):
    monkeypatch.setattr(_native, "MATRIX_OPERATOR_VERSION", 1, raising=False)
    for matrix in [-np.eye(4), np.full((4, 4), np.nan)]:
        with pytest.raises(ValueError):
            susie_rss(np.arange(4, dtype=float), matrix, 100, L=1)


def test_nonfinite_factors_keep_original_materialization(monkeypatch):
    # Standardization can produce infinite scales while the prepared matrix is
    # finite zero. Such inputs must retain the old API behavior rather than
    # violating the native operator's finite-positive factor contract.
    original = _native.fit
    calls = []

    def capture(matrix, response, yty, n, options):
        assert not any(key.startswith("matrix_") for key in options)
        assert np.isfinite(matrix).all()
        calls.append(True)
        return original(matrix, response, yty, n, options)

    monkeypatch.setattr(_native, "MATRIX_OPERATOR_VERSION", 1, raising=False)
    monkeypatch.setattr(_native, "fit", capture)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = susie_suff_stat(
            np.eye(2) * np.finfo(float).max,
            [0.0, 0.0],
            np.finfo(float).eps,
            1.0 + np.finfo(float).eps,
            L=1,
            max_iter=1,
            estimate_prior_variance=False,
            estimate_residual_variance=False,
        )
    assert calls
    assert np.isfinite(result.pip).all()
