"""Preprocessing decisions and bytes exported by pinned R constructors."""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from prusie import _native, susie_rss, susie_suff_stat

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/pinned_susier_symmetry.json").read_text(encoding="utf-8")
)
CASES = FIXTURE["cases"]


def decode(bits):
    return np.asarray(bits, dtype=np.uint64).view(np.float64).reshape(3, 3)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize("layout", ["C", "F", "strided", "reversed"])
def test_public_preprocessing_matches_actual_r(case, layout, monkeypatch):
    matrix = decode(case["input_bits"])
    expected = decode(case["expected_XtX_bits"]) if "expected_XtX_bits" in case else None
    if layout == "F":
        matrix = np.asfortranarray(matrix)
    elif layout == "strided":
        backing = np.empty((6, 6))
        backing[::2, ::2] = matrix
        matrix = backing[::2, ::2]
    elif layout == "reversed":
        matrix = matrix[::-1, ::-1]
        if expected is not None:
            expected = expected[::-1, ::-1]
    matrix.flags.writeable = False
    before = matrix.tobytes()
    seen = []
    native_fit = _native.fit

    def capture(matrix, response, yty, n, options):
        seen.append(matrix.T.copy() * options.get("matrix_global_scale", 1.0))
        return native_fit(matrix, response, yty, n, options)

    monkeypatch.setattr(_native, "fit", capture)
    kwargs = dict(
        L=1, standardize=False, max_iter=100, check_prior=False, estimate_residual_variance=False
    )

    def run():
        if case["route"] == "direct":
            return susie_suff_stat(matrix, [0.5] * 3, case["n"] - 1, case["n"], **kwargs)
        if case["route"] == "original":
            se = np.array([0.03, 0.2, 0.7])
            bhat = np.array([2.0, -1.0, 0.5]) * se
            if layout == "reversed":
                se, bhat = se[::-1], bhat[::-1]
            return susie_rss(bhat=bhat, shat=se, var_y=1.7, R=matrix, n=case["n"], **kwargs)
        return susie_rss(
            [0.5] * 3, matrix, None if case["route"] == "missing_n" else case["n"], **kwargs
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with np.errstate(over="ignore", invalid="ignore"):
            if not case["safe_public_input"] and case["id"] != "rss_range":
                with pytest.raises(ValueError):
                    run()
                assert not seen
            else:
                fit = run()
                assert np.isfinite(fit.pip).all() and ((fit.pip >= 0) & (fit.pip <= 1)).all()
                np.testing.assert_array_equal(seen[0].view(np.uint64), expected.view(np.uint64))
                messages = [str(w.message) for w in caught if "not symmetric" in str(w.message)]
                assert len(messages) == int(case["symmetry_warning"])
                assert sum("not symmetric" in w for w in fit.warnings) == len(messages)
    assert matrix.tobytes() == before and not matrix.flags.writeable


@pytest.mark.parametrize("use_simd", [False, True])
def test_scaled_underflow_can_remove_raw_asymmetry(use_simd):
    matrix = np.eye(3)
    matrix[0, 2] = np.nextafter(0.0, 1.0)
    assert _native.validate_rss_matrix(matrix, 0.5, use_simd) is True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        susie_rss([1.0] * 3, matrix, 1.5, L=1, check_prior=False)
    assert not any("not symmetric" in str(w.message) for w in caught)


def test_direct_averaging_overflow_is_rejected():
    matrix = np.array([[np.finfo(float).max, 0.2], [0.3, 1.0]])
    with (
        pytest.warns(RuntimeWarning, match="not symmetric"),
        pytest.raises(ValueError, match="finite"),
    ):
        susie_suff_stat(matrix, [0.0, 0.0], 99.0, 100, standardize=False)
