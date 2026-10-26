"""Per-fit thread requests preserve input ownership and deterministic results."""

import inspect
import os
import warnings

import numpy as np
import pytest

import prusie
from prusie import _native


def fit_small(name, **options):
    common = dict(L=1, max_iter=3, coverage=None, **options)
    if name == "susie_rss":
        return prusie.susie_rss([2.0, -1.0], np.eye(2), 100, **common)
    if name == "susie_suff_stat":
        return prusie.susie_suff_stat(
            99 * np.eye(2),
            [20.0, -10.0],
            99.0,
            100,
            residual_variance=1,
            estimate_residual_variance=False,
            **common,
        )
    if name == "susie":
        return prusie.susie(
            [[1.0, 2.0], [3.0, 1.0], [2.0, 4.0], [4.0, 3.0]],
            [1.0, 3.0, 2.0, 4.0],
            residual_variance=1,
            estimate_residual_variance=False,
            **common,
        )
    if name == "susie_rss_lambda":
        return prusie.susie_rss_lambda([2.0, -1.0], np.eye(2), 100, lambda_=0.1, **common)
    raise AssertionError(name)


FITTERS = ["susie", "susie_suff_stat", "susie_rss", "susie_rss_lambda"]


@pytest.mark.parametrize("name", FITTERS)
@pytest.mark.parametrize("count", [1, np.int64(4), 8, 32])
def test_thread_count_is_explicit_and_recorded_for_every_fitter(name, count):
    assert inspect.signature(getattr(prusie, name)).parameters["n_threads"].default == 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_small(name, n_threads=count)
    assert fit.params["n_threads"] == int(count)
    assert type(fit.params["n_threads"]) is int


@pytest.mark.parametrize("name", FITTERS)
@pytest.mark.parametrize("count", [0, -1, 33, True, np.bool_(False), 1.5, 2.0, "4", None])
def test_invalid_thread_counts_are_rejected(name, count):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="n_threads must be an integer between 1 and 32"):
            fit_small(name, n_threads=count)


@pytest.mark.parametrize("layout", ["C", "F"])
@pytest.mark.parametrize("requested", [1, 4])
def test_public_default_overrides_hidden_thread_environment_without_mutating_it(
    monkeypatch, layout, requested
):
    monkeypatch.setenv("PRUSIE_NUM_THREADS", "32")
    monkeypatch.setenv("RAYON_NUM_THREADS", "17")
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    before = {
        key: os.environ.get(key)
        for key in (
            "PRUSIE_NUM_THREADS",
            "RAYON_NUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        )
    }
    calls = []
    original_fit = _native.fit
    original_validation = _native.validate_rss_matrix
    original_fused = _native.validate_and_repair_rss

    def capture_fit(matrix, response, yty, n, options):
        calls.append(("fit", options["n_threads"]))
        return original_fit(matrix, response, yty, n, options)

    def capture_validation(*args, **kwargs):
        calls.append(("validation", kwargs.get("proof_threads")))
        return original_validation(*args, **kwargs)

    def capture_fused(*args, **kwargs):
        calls.append(("fused_validation", kwargs.get("proof_threads")))
        return original_fused(*args, **kwargs)

    monkeypatch.setattr(_native, "fit", capture_fit)
    monkeypatch.setattr(_native, "validate_rss_matrix", capture_validation)
    monkeypatch.setattr(_native, "validate_and_repair_rss", capture_fused)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Check both fused C preparation and the general F validation route.
        # Even an explicit product request does not parallelize validation.
        options = {} if requested == 1 else dict(n_threads=requested)
        fit = prusie.susie_rss(
            [2.0, -1.0],
            np.array(np.eye(2), order=layout),
            100,
            L=1,
            max_iter=3,
            coverage=None,
            **options,
        )
    assert fit.params["n_threads"] == requested
    assert ("fit", requested) in calls
    assert ("fused_validation" if layout == "C" else "validation", 1) in calls
    assert before == {key: os.environ.get(key) for key in before}


@pytest.mark.parametrize("count", [2, 4, 8])
def test_parallel_products_are_deterministic_and_do_not_mutate_inputs(count):
    # An uneven output length exercises static chunk boundaries above the
    # declared parallel-kernel threshold. This is a deterministic unit fixture.
    matrix = np.eye(1025)
    matrix.flags.writeable = False
    scores = np.linspace(-3.0, 3.0, 1025)
    before_matrix, before_scores = matrix.tobytes(), scores.tobytes()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = prusie.susie_rss(
            scores, matrix, 1000, L=2, max_iter=3, coverage=None, n_threads=count
        )
        second = prusie.susie_rss(
            scores, matrix, 1000, L=2, max_iter=3, coverage=None, n_threads=count
        )
    np.testing.assert_array_equal(first.alpha.view(np.uint64), second.alpha.view(np.uint64))
    assert matrix.tobytes() == before_matrix
    assert scores.tobytes() == before_scores
    assert np.isfinite(first.alpha).all()
    np.testing.assert_allclose(first.alpha.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_initialization_specification_retains_thread_request():
    specification = prusie.susie_rss([2.0, -1.0], np.eye(2), 100, L=1, init_only=True, n_threads=4)
    assert specification.params["n_threads"] == 4
    fit = prusie.susie_workhorse(specification)
    assert fit.params["n_threads"] == 4
