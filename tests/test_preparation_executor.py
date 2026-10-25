"""Private preparation ownership, exact parent fallbacks and exception cleanup."""

import ctypes
import inspect
import json
import os
import pickle
import platform
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pytest

from prusie import _native, api, susie_rss
from prusie import _execution


def arguments(p=1025, **updates):
    result = {name: parameter.default
              for name, parameter in inspect.signature(susie_rss).parameters.items()}
    result.update(z=np.linspace(-2., 2., p), R=np.eye(p), n=1000., n_threads=4)
    result.update(updates)
    return result


def native_options(p, count=4):
    return dict(l=2, prior_variance=0.2, residual_variance=1.,
                prior_weights=[1 / p] * p, estimate_prior_variance=False,
                estimate_prior_method="optim", estimate_residual_variance=False,
                check_null_threshold=0., max_iter=3, tol=1e-4,
                check_prior=False, prior_tol=1e-9, null_index=None, n_threads=count)


def task_ids():
    return {int(path.name) for path in Path("/proc/self/task").iterdir()}


def parse_task_stat(raw):
    """Parse identity/exit flags without assuming comm contains no spaces or ')'."""
    first, last = raw.index("("), raw.rindex(")")
    fields = raw[last + 1:].split()
    return dict(tid=int(raw[:first].strip()), comm=raw[first + 1:last],
                state=fields[0], flags=int(fields[6]), starttime=int(fields[19]),
                raw_stat=raw)


def read_task_record(tid):
    try:
        return parse_task_stat(Path(f"/proc/self/task/{tid}/stat").read_text())
    except FileNotFoundError:
        # Listing a task and opening stat are separate procfs observations.
        return None


def worker_identity(record):
    return record["tid"], record["starttime"], record["comm"]


def capture_worker_records(tids):
    records = {tid: read_task_record(tid) for tid in sorted(tids)}
    assert all(record is not None for record in records.values()), records
    for tid, record in records.items():
        assert record["tid"] == tid, records
        assert re.fullmatch(r"prusie-kernel-\d+", record["comm"]), records
    return records


def assert_worker_tasks_disappear(tids, known=None, *, read=None, now=None, pause=None):
    """Allow only an already-exiting, same-identity kernel teardown tail.

    A successful join can precede removal from /proc/self/task. No readable
    live worker gets a grace period: every sample must already have PF_EXITING.
    Known identities are captured before close where the test has access to
    the started owner. Public exception tests first observe identities after
    close, so they additionally require the native kernel-worker name.
    """
    read = read_task_record if read is None else read
    now = time.monotonic if now is None else now
    pause = time.sleep if pause is None else pause
    pending = set(tids)
    identities = {} if known is None else {
        tid: worker_identity(record) for tid, record in known.items()}
    assert known is None or set(known) == pending
    started = now()
    history = []
    while pending:
        for tid in sorted(pending):
            record = read(tid)
            history.append((now() - started, tid, record))
            if record is None:
                pending.remove(tid)
                continue
            assert record["tid"] == tid, history
            assert re.fullmatch(r"prusie-kernel-\d+", record["comm"]), history
            identity = worker_identity(record)
            assert identity == identities.setdefault(tid, identity), history
            assert record["flags"] & 0x4, history  # Linux PF_EXITING.
        if pending:
            remaining = 1.0 - (now() - started)
            assert remaining > 0, ("exiting tasks remain after 1 s", history)
            pause(min(0.001, remaining))


def observer_record(**updates):
    record = dict(tid=123, starttime=456, comm="prusie-kernel-0", flags=0x4,
                  state="R", raw_stat="deterministic observer fixture")
    record.update(updates)
    return record


def unexpected_observer_pause(duration):
    # This exception must escape pytest.raises(AssertionError): a regressed
    # rejection guard should fail the test, not loop on its constant clock.
    raise RuntimeError(f"observer unexpectedly waited {duration} s for an invalid task")


def test_task_stat_parser_preserves_identity_and_exit_flags():
    fields = ["R"] + ["0"] * 19
    fields[6], fields[19] = "4194372", "456"
    raw = "123 (name with ) and spaces) " + " ".join(fields)
    record = parse_task_stat(raw)
    assert worker_identity(record) == (123, 456, "name with ) and spaces")
    assert record["flags"] & 0x4 and record["state"] == "R"
    assert record["raw_stat"] == raw


@pytest.mark.parametrize("record", [
    observer_record(flags=0), observer_record(tid=124),
    observer_record(comm="unrelated-worker"),
])
def test_teardown_observer_rejects_live_or_unowned_tasks_without_waiting(record):
    with pytest.raises(AssertionError):
        assert_worker_tasks_disappear({123}, read=lambda tid: record,
                                      now=lambda: 0., pause=unexpected_observer_pause)


@pytest.mark.parametrize("updates", [{"starttime": 457}, {"comm": "prusie-kernel-1"}])
def test_teardown_observer_rejects_known_identity_changes_without_waiting(updates):
    with pytest.raises(AssertionError):
        assert_worker_tasks_disappear({123}, {123: observer_record()},
                                      read=lambda tid: observer_record(**updates),
                                      now=lambda: 0., pause=unexpected_observer_pause)


@pytest.mark.parametrize("updates", [{"starttime": 457}, {"flags": 0}])
def test_teardown_observer_rejects_changed_or_live_task_after_exiting_sample(updates):
    records = iter([observer_record(), observer_record(**updates)])
    pauses = []
    with pytest.raises(AssertionError):
        assert_worker_tasks_disappear({123}, read=lambda tid: next(records),
                                      now=lambda: 0., pause=pauses.append)
    assert pauses == [0.001]


@pytest.mark.parametrize("known", [None, {123: observer_record()}])
def test_teardown_observer_accepts_only_exiting_tail_then_disappearance(known):
    records = iter([observer_record(), None])
    pauses = []
    assert_worker_tasks_disappear({123}, known, read=lambda tid: next(records),
                                  now=lambda: 0., pause=pauses.append)
    assert pauses == [0.001]


def test_teardown_observer_accepts_listed_then_absent_without_inventing_flags():
    pauses = []
    assert_worker_tasks_disappear({123}, read=lambda tid: None,
                                  now=lambda: 0., pause=pauses.append)
    assert pauses == []


def test_teardown_observer_rejects_persistent_exiting_task_at_fixed_deadline():
    clock, pauses = [0.], []

    def advance(duration):
        pauses.append(duration)
        clock[0] += duration

    with pytest.raises(AssertionError, match="remain after 1 s"):
        assert_worker_tasks_disappear({123}, read=lambda tid: observer_record(),
                                      now=lambda: clock[0], pause=advance)
    assert clock[0] == 1.0
    assert 0 < len(pauses) <= 1001 and max(pauses) <= 0.001


def bits_equal(actual, expected):
    if isinstance(actual, np.ndarray):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        if actual.dtype.kind == "f":
            np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        else:
            np.testing.assert_array_equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            bits_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            bits_equal(left, right)
    elif isinstance(actual, float):
        assert actual.hex() == expected.hex()
    else:
        assert actual == expected


def assert_fit_bits(actual, expected):
    for field in ("alpha", "mu", "mu2", "pip", "V", "elbo", "KL", "XtXr",
                  "lbf", "lbf_variable", "sigma2", "niter", "converged", "sets",
                  "params", "qc", "warnings", "variant_ids", "convergence"):
        bits_equal(getattr(actual, field), getattr(expected, field))


def test_all_public_arguments_have_an_explicit_admission_partition():
    assert set(inspect.signature(susie_rss).parameters) == _execution._ARGUMENTS
    partition = (_execution._NONE | _execution._FALSE | _execution._ZERO
                 | _execution._BOOL | _execution._NUMERIC | _execution._NULLABLE_NUMERIC
                 | set(_execution._CHOICES)
                 | {"z", "R", "n", "n_threads", "n_purity", "estimate_prior_method", "variant_ids"})
    assert partition == _execution._ARGUMENTS
    assert _execution._allowed(arguments(2))


class Hook:
    def __float__(self):
        raise AssertionError("admission must not coerce")

    def __eq__(self, other):
        raise AssertionError("admission must check exact type before comparison")


@pytest.mark.parametrize("field", sorted(_execution._ARGUMENTS - {"z", "R", "n", "n_threads"}))
def test_custom_option_objects_fall_back_without_calling_hooks(field):
    assert not _execution._allowed(arguments(2, **{field: Hook()}))


@pytest.mark.parametrize("field,value", [
    ("overwrite_ld", True), ("track_fit", "compact"), ("retain_data", True),
    ("init_only", True), ("refine", True), ("check_input", True),
    ("L_greedy", 1), ("R_finite", 200.), ("R_mismatch", "eb"),
    ("maf", np.ones(2)), ("null_weight", 0.1), ("prior_weights", np.ones(2)),
    ("scaled_prior_variance", np.ones(2)), ("estimate_residual_method", "NIG"),
    ("prior_variance_grid", [0.1]), ("variant_metadata", {"x": [1, 2]}),
    ("variant_ids", ["a", Hook()]), ("coverage", np.float64(0.95)),
])
def test_first_scope_exclusions_keep_parent_preparation(field, value):
    assert not _execution._allowed(arguments(2, **{field: value}))


def test_invalid_plain_controls_still_reach_original_validation():
    assert _execution._allowed(arguments(2, L=-1, max_iter=0, coverage=2.,
                                        estimate_prior_method="unknown", tol=-1.))


@pytest.mark.parametrize("count,p", [(1, 1025), (4, 2), (np.int64(4), 1025)])
def test_default_small_and_numpy_counts_do_not_create_an_owner(monkeypatch, count, p):
    def forbidden(*args, **kwargs):
        raise AssertionError("parent route must not create a preparation owner")

    monkeypatch.setattr(_native, "_new_fit_executor", forbidden)
    monkeypatch.setattr(_execution, "create_executor", forbidden)
    monkeypatch.setattr(_execution, "_ordinary_hooks", forbidden)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        fit = susie_rss(**arguments(p, n_threads=count, L=1, max_iter=2, coverage=None))
    assert fit.params["n_threads"] == int(count)


def test_warning_recording_is_distinct_from_custom_display_and_numpy_hooks(monkeypatch):
    args = arguments()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        owner = _execution.create_executor(args)
        assert owner is not None
        owner.close()
        with monkeypatch.context() as patch:
            patch.setattr(warnings, "showwarning", lambda *a, **k: None)
            assert _execution.create_executor(args) is None
        with monkeypatch.context() as patch:
            patch.setattr(warnings, "_showwarnmsg_impl", lambda *a, **k: None)
            assert _execution.create_executor(args) is None
        previous = np.geterrcall()
        try:
            np.seterrcall(lambda *a: None)
            assert _execution.create_executor(args) is None
        finally:
            np.seterrcall(previous)


def test_private_factory_is_inert_and_not_serializable():
    before = task_ids() if sys.platform == "linux" else None
    owner = _native._new_fit_executor(4)
    assert not owner.has_started
    if before is not None:
        assert task_ids() <= before
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(owner)
    owner.close()
    owner.close()


@pytest.mark.skipif(sys.platform != "linux" or platform.machine() != "x86_64",
                    reason="early parallel preparation has a Linux x86_64 contract")
@pytest.mark.parametrize("count", [2, 4, 8])
@pytest.mark.parametrize("asymmetric", [False, True])
def test_shared_preparation_matches_same_count_parent_and_serializes(count, asymmetric):
    args = arguments(n_threads=count, variant_ids=[str(i) for i in range(1025)],
                     L=2, max_iter=3, coverage=None)
    if asymmetric:
        args["R"][0, -1], args["R"][-1, 0] = 0.025, -0.03125
    before = (args["R"].tobytes(), args["z"].tobytes())
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        # NumPy integer is an accepted count whose preparation intentionally
        # stays on the unmodified parent path at this exact same native count.
        parent = susie_rss(**{**args, "n_threads": np.int64(count)})
        actual = susie_rss(**args)
        repeated = susie_rss(**args)
    assert_fit_bits(actual, parent)
    assert_fit_bits(actual, repeated)
    assert before == (args["R"].tobytes(), args["z"].tobytes())
    assert "executor" not in json.dumps(actual.params)
    restored = pickle.loads(pickle.dumps(actual))
    assert_fit_bits(actual, restored)


@pytest.mark.skipif(sys.platform != "linux" or platform.machine() != "x86_64",
                    reason="native early worker ownership is Linux x86_64")
def test_native_helper_workers_are_joined_before_close_returns():
    matrix = np.eye(1025)
    before = task_ids()
    owner = _native._new_fit_executor(4)
    assert _native._validate_and_repair_rss_with_executor(matrix, 999., owner) == (True, None, None)
    assert owner.has_started
    own_workers = task_ids() - before
    assert len(own_workers) == 4
    identities = capture_worker_records(own_workers)
    owner.close()
    assert_worker_tasks_disappear(own_workers, identities)


@pytest.mark.parametrize("count", [2, 4, 8])
def test_owned_output_and_deferred_repair_error_match_original_bits(count):
    matrix = np.eye(1025)
    matrix[0, 1], matrix[1, 0] = 1., 0.9
    matrix[31, 35], matrix[35, 31] = -0., 0.
    matrix[1023, 1024], matrix[1024, 1023] = np.nextafter(0., 1.), -np.nextafter(0., 1.)
    before = matrix.tobytes()
    for scale in (999., 1e308):
        owner = _native._new_fit_executor(count)
        expected = _native.validate_and_repair_rss(matrix, scale, proof_threads=1)
        try:
            actual = _native._validate_and_repair_rss_with_executor(matrix, scale, owner)
            bits_equal(actual, expected)
            assert matrix.tobytes() == before
            assert (actual[2] is not None) == (scale == 1e308)
        finally:
            owner.close()


@pytest.mark.parametrize("fault,scale,error", [
    ("raw_nan", 999., "R must contain only finite"),
    ("raw_negative", 999., "R diagonal must be nonnegative"),
    ("scaled_overflow", 1e308, "XtX must contain only finite"),
])
def test_private_helper_preserves_parent_error_priority_and_closes(fault, scale, error):
    matrix = np.eye(1025)
    matrix[0, 1] = 0.125
    if fault == "raw_nan":
        matrix[1, 0], matrix[-1, -1] = np.nan, -1.
    elif fault == "raw_negative":
        matrix[0, 0] = -1.
    else:
        matrix[0, 1] = 2.
    owner = _native._new_fit_executor(4)
    with pytest.raises(ValueError, match=error):
        _native.validate_and_repair_rss(matrix, scale, proof_threads=1)
    with pytest.raises(ValueError, match=error):
        _native._validate_and_repair_rss_with_executor(matrix, scale, owner)
    owner.close()


@pytest.mark.parametrize("problem", ["parse", "dimensions", "closed", "count"])
def test_private_native_failure_does_not_leave_helper_workers(problem):
    matrix, scores = np.eye(1025), np.linspace(-1., 1., 1025)
    before = task_ids() if sys.platform == "linux" else set()
    owner = _native._new_fit_executor(4)
    _native._validate_and_repair_rss_with_executor(matrix, 1., owner)
    own_workers = task_ids() - before if before else set()
    identities = capture_worker_records(own_workers) if before else None
    options = native_options(1025)
    if problem == "parse":
        del options["l"]
    elif problem == "dimensions":
        scores = scores[:-1]
    elif problem == "closed":
        owner.close()
    else:
        options["n_threads"] = 8
    with pytest.raises((ValueError, KeyError, RuntimeError)):
        _native._fit_with_executor(matrix, scores, 999., 1000., options, owner)
    owner.close()
    if before:
        assert_worker_tasks_disappear(own_workers, identities)


def test_public_warning_error_closes_owner_without_replacing_primary(monkeypatch):
    owners = []
    original = _native._new_fit_executor

    def capture(count):
        owner = original(count)
        owners.append(owner)
        return owner

    monkeypatch.setattr(_native, "_new_fit_executor", capture)
    args = arguments(L=1, max_iter=2, coverage=None)
    args["R"][0, 1] = 0.125
    before = task_ids() if sys.platform == "linux" else set()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("error")
        with pytest.raises(RuntimeWarning, match="symmetric") as exception:
            susie_rss(**args)
    assert len(owners) == 1
    # A saved traceback can retain the closed wrapper; explicit cleanup must
    # not wait for reference counting or garbage collection.
    assert exception.value.__traceback__ is not None
    owners[0].close()
    if before:
        assert_worker_tasks_disappear(task_ids() - before)


def test_late_public_validation_error_closes_started_owner(monkeypatch):
    owners = []
    original = _native._new_fit_executor

    def capture(count):
        owner = original(count)
        owners.append(owner)
        return owner

    monkeypatch.setattr(_native, "_new_fit_executor", capture)
    before = task_ids() if sys.platform == "linux" else set()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="L must be a positive integer"):
            susie_rss(**arguments(L=0, coverage=None))
    assert len(owners) == 1
    owners[0].close()
    if before:
        assert_worker_tasks_disappear(task_ids() - before)


def test_parallel_and_parent_warning_stacks_use_the_same_original_callsite():
    args = arguments(L=1, max_iter=2, coverage=None)
    args["R"][0, 1] = 0.125
    signatures = []
    for count in (np.int64(4), 4):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            susie_rss(**{**args, "n_threads": count})
        signatures.append([(str(row.message), row.category, row.filename, row.lineno)
                           for row in caught])
    assert signatures[0] == signatures[1]
    assert any("symmetric" in row[0] for row in signatures[0])


def test_cleanup_preserves_baseexception_and_reports_unmasked_cleanup_failure():
    class FailingClose:
        def close(self):
            raise RuntimeError("injected worker join failure")

    with pytest.raises(RuntimeError, match="join failure"):
        _execution.close_executor(FailingClose())
    with pytest.raises(KeyboardInterrupt, match="primary"):
        try:
            raise KeyboardInterrupt("primary")
        finally:
            _execution.close_executor(FailingClose())


@pytest.mark.skipif(sys.platform != "linux" or platform.machine() != "x86_64",
                    reason="Linux x86_64 caller-control boundary")
def test_nonstandard_rounding_keeps_parent_and_changed_controls_replace_workers():
    library = ctypes.CDLL(None)
    library.fegetround.restype = ctypes.c_int
    library.fesetround.argtypes = [ctypes.c_int]
    original = library.fegetround()
    matrix, scores = np.eye(1025), np.linspace(-1., 1., 1025)
    owner = _native._new_fit_executor(4)
    try:
        assert library.fesetround(0x400) == 0  # Linux FE_DOWNWARD.
        expected = _native.validate_and_repair_rss(matrix, 999., proof_threads=1)
        actual = _native._validate_and_repair_rss_with_executor(matrix, 999., owner)
        assert actual == expected and not owner.has_started
        owner.close()
        assert library.fegetround() == 0x400
        assert library.fesetround(0) == 0  # FE_TONEAREST before early preparation.
        owner = _native._new_fit_executor(4)
        before = task_ids()
        _native._validate_and_repair_rss_with_executor(matrix, 1., owner)
        assert owner.has_started
        early_workers = task_ids() - before
        identities = capture_worker_records(early_workers)
        assert library.fesetround(0x400) == 0
        options = native_options(1025)
        expected = _native.fit(matrix, scores, 999., 1000., options)
        actual = _native._fit_with_executor(matrix, scores, 999., 1000., options, owner)
        bits_equal(actual, expected)
        assert library.fegetround() == 0x400
        assert_worker_tasks_disappear(early_workers, identities)
    finally:
        try:
            owner.close()
        finally:
            assert library.fesetround(original) == 0
