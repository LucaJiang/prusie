"""Process instrumentation cannot change matrix-preparation path or alpha."""

import contextlib
import io
import logging
import pickle
import sys
import threading
import warnings

import numpy as np
import pytest

from prusie import _execution, _native, rss, susie_rss


def request(**updates):
    p = 1025
    values = dict(
        z=np.linspace(-2.0, 2.0, p),
        R=np.eye(p),
        n=1000,
        n_threads=4,
        L=2,
        max_iter=3,
        coverage=None,
        estimate_prior_variance=False,
    )
    values.update(updates)
    return values


def assert_alpha_bits(left, right):
    np.testing.assert_array_equal(left.alpha.view(np.uint64), right.alpha.view(np.uint64))


@pytest.mark.parametrize("count", [1, 4])
@pytest.mark.parametrize("overwrite", [False, True])
def test_pickled_float64_ld_preserves_preparation_path_and_alpha(monkeypatch, count, overwrite):
    paths = []
    for name in (
        "validate_and_repair_rss",
        "_validate_and_repair_rss_with_executor",
        "validate_and_overwrite_rss",
    ):
        original = getattr(_native, name)

        def observe(*args, _original=original, _name=name, **kwargs):
            paths.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(_native, name, observe)
    ordinary = request(n_threads=count, overwrite_ld=overwrite)
    baseline = susie_rss(**ordinary)
    expected_path = paths.copy()
    paths.clear()
    restored = request(n_threads=count, overwrite_ld=overwrite)
    # Unpickling can use a bytes-backed view. Copy it to meet the separate
    # overwrite ownership contract while retaining the restored dtype object.
    restored["R"] = pickle.loads(pickle.dumps(restored["R"], protocol=4)).copy(order="C")
    assert restored["R"].dtype == ordinary["R"].dtype
    assert restored["R"].flags.owndata
    actual = susie_rss(**restored)
    assert paths == expected_path
    assert_alpha_bits(actual, baseline)


@contextlib.contextmanager
def instrumentation(name, monkeypatch):
    if name == "stderr":
        monkeypatch.setattr(sys, "stderr", io.StringIO())
    previous_trace, previous_profile = sys.gettrace(), sys.getprofile()
    if name == "trace":
        sys.settrace(lambda frame, event, argument: None)
    if name == "profile":
        sys.setprofile(lambda frame, event, argument: None)
    if name == "logging":
        logging.captureWarnings(True)
    try:
        yield
    finally:
        sys.settrace(previous_trace)
        sys.setprofile(previous_profile)
        if name == "logging":
            logging.captureWarnings(False)


@pytest.mark.parametrize("mode", ["stderr", "logging", "trace", "profile", "numpy"])
@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("asymmetric", [False, True])
def test_instrumentation_and_numpy_scalars_preserve_path_and_alpha(
    monkeypatch, caplog, mode, overwrite, asymmetric
):
    paths = []
    warning_messages = []
    original_warn = warnings.warn
    original_parallel = _native._validate_and_repair_rss_with_executor
    original_overwrite = _native.validate_and_overwrite_rss

    def parallel(matrix, scale, owner):
        result = original_parallel(matrix, scale, owner)
        paths.append(("parallel", owner.has_started))
        return result

    def destructive(*args, **kwargs):
        result = original_overwrite(*args, **kwargs)
        paths.append(("overwrite", result[3]))
        return result

    def warn(message, *args, **kwargs):
        warning_messages.append(str(message))
        return original_warn(message, *args, **kwargs)

    def case(**updates):
        result = request(**updates)
        if asymmetric:
            result["R"][0, 1], result["R"][1, 0] = 0.125, 0.25
        return result

    monkeypatch.setattr(_native, "_validate_and_repair_rss_with_executor", parallel)
    monkeypatch.setattr(_native, "validate_and_overwrite_rss", destructive)
    monkeypatch.setattr(warnings, "warn", warn)
    expected_path = (
        ("overwrite", "ordered_scaled_xtx" if asymmetric else "symmetric_no_write")
        if overwrite
        else ("parallel", _native._parallel_preparation_available())
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        baseline = susie_rss(**case(overwrite_ld=overwrite))
        assert paths == [expected_path]
        baseline_warnings = warning_messages.copy()
        paths.clear()
        warning_messages.clear()
        updates = dict(overwrite_ld=overwrite)
        if mode == "numpy":
            updates.update(
                n_threads=np.int64(4),
                n=np.int64(1000),
                L=np.int64(2),
                max_iter=np.int64(3),
                scaled_prior_variance=np.float64(0.2),
                prior_tol=np.float64(1e-9),
                tol=np.float64(1e-4),
                standardize=np.bool_(True),
                z_method=np.str_("wald"),
            )
        args = case(**updates)
        original_matrix = args["R"].copy()
        with instrumentation(mode, monkeypatch):
            actual = susie_rss(**args)
    assert paths == [expected_path]
    assert warning_messages == baseline_warnings
    if asymmetric:
        assert any("not symmetric" in message for message in warning_messages)
        if mode == "logging":
            assert any("not symmetric" in record.message for record in caplog.records)
    if overwrite and asymmetric:
        assert args["R"].tobytes() != original_matrix.tobytes()
    else:
        np.testing.assert_array_equal(args["R"].view(np.uint64), original_matrix.view(np.uint64))
    assert_alpha_bits(actual, baseline)


def test_warning_dispatch_is_after_join_on_calling_thread_and_in_serial_order(monkeypatch):
    events = []
    caller = threading.get_ident()
    original_close = _execution.close_executor
    original_warn = warnings.warn

    def close(owner, **kwargs):
        original_close(owner, **kwargs)
        events.append(("joined", threading.get_ident()))

    def warn(message, *args, **kwargs):
        events.append((str(message), threading.get_ident()))
        return original_warn(message, *args, **kwargs)

    monkeypatch.setattr(rss, "close_executor", close)
    monkeypatch.setattr(warnings, "warn", warn)
    messages = []
    for count in (1, 4):
        args = request(n_threads=count, estimate_residual_variance=True)
        args["z"][0] = np.nan
        args["R"][0, 1] = 0.125
        events.clear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            susie_rss(**args)
        messages.append([str(item.message) for item in caught])
        assert all(thread == caller for _, thread in events)
        if count == 4:
            assert "NA values" in events[0][0]
            assert events[1][0] == "joined"
            assert "not symmetric" in events[2][0]
    assert messages[0] == messages[1]
    assert messages[0][:3] == [
        "NA values in z-scores are replaced with 0.",
        "XtX is not symmetric; using (XtX + t(XtX))/2.",
        "Estimating residual variance requires in-sample LD consistent with the summary statistics",
    ]


def test_warning_as_error_precedes_deferred_repair_error_and_only_first_warning_escapes(
    monkeypatch,
):
    observed = []
    original_prepare = _native._validate_and_repair_rss_with_executor

    def prepare(matrix, scale, owner):
        result = original_prepare(matrix, scale, owner)
        observed.append((owner.has_started, result[2] is not None))
        return result

    monkeypatch.setattr(_native, "_validate_and_repair_rss_with_executor", prepare)
    for count in (1, 4):
        observed.clear()
        args = request(n_threads=count, n=1e308)
        args["R"][0, 1], args["R"][1, 0] = 1.5, 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(RuntimeWarning, match="not symmetric"):
                susie_rss(**args)
        assert observed == (
            [(_native._parallel_preparation_available(), True)] if count == 4 else []
        )


@pytest.mark.parametrize("underflow", ["warn", "call", "raise"])
def test_preparation_joins_before_numpy_error_handling_and_separate_fit(monkeypatch, underflow):
    events = []
    original_prepare = _native._validate_and_repair_rss_with_executor
    original_close = _execution.close_executor
    original_fit = _native.fit

    def prepare(matrix, scale, owner):
        events.append("prepare")
        return original_prepare(matrix, scale, owner)

    def close(owner, **kwargs):
        original_close(owner, **kwargs)
        events.append("joined")

    def fit(*args, **kwargs):
        events.append("fit")
        return original_fit(*args, **kwargs)

    def callback(error, flag):
        assert events[:2] == ["prepare", "joined"] and "fit" not in events
        events.append(error)

    monkeypatch.setattr(_native, "_validate_and_repair_rss_with_executor", prepare)
    monkeypatch.setattr(rss, "close_executor", close)
    monkeypatch.setattr(_native, "fit", fit)
    args = request(n=1.5, z_method="score", check_prior=False)
    args["R"] *= 3 * np.nextafter(0.0, 1.0)
    args["z"][:] = 0.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with np.errstate(under=underflow, call=callback):
            if underflow == "raise":
                with pytest.raises(FloatingPointError, match="underflow"):
                    susie_rss(**args)
            else:
                susie_rss(**args)
    assert events[:2] == ["prepare", "joined"]
    if underflow == "warn":
        assert any("underflow" in str(item.message) for item in caught)
        assert events[-1] == "fit"
    elif underflow == "call":
        assert "underflow" in events
        assert events[-1] == "fit"
    else:
        assert "fit" not in events


def test_thread_count_uses_index_protocol_but_rejects_booleans():
    from prusie._validation import _thread_count

    class Count:
        def __index__(self):
            return 4

    assert _thread_count(Count()) == _thread_count(np.int64(4)) == 4
    for value in (True, np.bool_(False), 4.0, np.float64(4), 0, 33):
        with pytest.raises(ValueError, match="n_threads"):
            _thread_count(value)
