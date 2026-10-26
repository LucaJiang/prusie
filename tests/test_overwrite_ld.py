"""Exact storage, alias and failure contracts of explicit RSS overwrite."""
import inspect
import subprocess
import sys
import warnings

import numpy as np
import pytest

from prusie import susie_rss
from prusie import _native
from prusie import api
_PUBLIC_ARGUMENTS = set(inspect.signature(susie_rss).parameters)


def matrix(p=9, late=False):
    r = np.eye(p)
    i = p - 2 if late else 0
    r[i, i + 1], r[i + 1, i] = 0.125, 0.1250001
    if p > 3:
        r[0, -1], r[-1, 0] = -0.0, 0.0
    return r


def repair(r, n):
    x = np.multiply(r, float(n) - 1)
    y = np.multiply(r.T, float(n) - 1)
    np.add(x, y, out=x)
    np.divide(x, 2., out=x)
    return x


def bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.tobytes() == b.tobytes()


def same_fit(a, b):
    for field in ("alpha", "mu", "mu2", "V", "pip", "elbo", "KL", "XtXr",
                  "lbf", "lbf_variable", "X_column_scale_factors"):
        bits(getattr(a, field), getattr(b, field))
    assert (a.niter, a.converged, a.sigma2) == (b.niter, b.converged, b.sigma2)
    assert a.convergence == b.convergence
    bits(a.sets["cs_index"], b.sets["cs_index"])
    for left, right in zip(a.sets["cs"], b.sets["cs"]):
        bits(left, right)
    assert a.params["n"] == b.params["n"] == a.params["input_n"]


@pytest.mark.parametrize("p,late", [(2, False), (3, True), (4, False), (7, True),
    (8, False), (9, True), (31, False), (32, True), (33, False), (65, True)])
@pytest.mark.parametrize("standardize", [False, True])
def test_exact_posterior_and_consumed_buffer_match_owned_parent(p, late, standardize):
    original = matrix(p, late)
    r = original.copy()
    alias = r.view()
    z = np.zeros(p)
    z[:2] = [2., -0.5]
    prior = np.ones(p)
    args = dict(z=z, n=100, L=2, max_iter=10, check_prior=False,
                prior_weights=prior, standardize=standardize,
                variant_ids=[f"s{i}" for i in range(p)],
                variant_metadata={"position": np.arange(p)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        parent = susie_rss(R=original, **args)
        child = susie_rss(R=r, overwrite_ld=np.bool_(True), **args)
    same_fit(parent, child)
    bits(r, repair(original, 100))
    bits(alias, r)
    bits(z[:2], [2., -0.5])
    bits(prior, np.ones(p))
    assert not parent.params["overwrite_ld"] and child.params["overwrite_ld"] is True
    assert child.qc["ld_overwrite"] == dict(action="ordered_scaled_xtx", original_rss_scale=99.)


@pytest.mark.parametrize("threads", [1, 2, 4, 8])
def test_overwrite_retains_thread_option_and_same_count_determinism(threads):
    r = matrix(1025)
    original = r.copy()
    args = dict(z=np.zeros(len(r)), n=100, L=1, max_iter=2, check_prior=False,
                n_threads=threads)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        parent = susie_rss(R=original, **args)
        child = susie_rss(R=r, overwrite_ld=True, **args)
        repeat = susie_rss(R=original.copy(), overwrite_ld=True, **args)
    same_fit(parent, child)
    same_fit(child, repeat)
    bits(r, repair(original, 100))
    assert child.params["n_threads"] == threads


@pytest.mark.parametrize("kind,action", [("unit", "symmetric_no_write"),
    ("diagonal", "no_witness_eager_no_write"), ("rounded", "no_witness_eager_no_write")])
def test_all_successful_no_write_actions_preserve_original_bits(kind, action):
    r = np.eye(2)
    n, method = 100, "wald"
    if kind == "diagonal":
        r *= 2.
    if kind == "rounded":
        r[0, 1] = np.nextafter(0., 1.)
        n, method = 1.5, "score"
    original = r.copy()
    args = dict(z=[0., 0.], n=n, z_method=method, L=1, max_iter=2)
    parent = susie_rss(R=original, **args)
    child = susie_rss(R=r, overwrite_ld=True, **args)
    same_fit(parent, child)
    bits(r, original)
    assert child.qc["ld_overwrite"] == dict(action=action, original_rss_scale=n - 1)


def test_no_witness_keeps_valid_symmetric_extreme_despite_speculative_repair_overflow():
    r = np.eye(2) * 1e308
    original = r.copy()
    symmetric, prepared, error, action = _native.validate_and_overwrite_rss(r, 1.)
    assert symmetric and prepared is None and error is None
    assert action == "no_witness_eager_no_write"
    bits(r, original)


def test_snapshot_covers_every_public_argument_and_default_never_walks(monkeypatch):
    captured = []
    original = api._overwrite_prepare

    def capture(arguments):
        captured.append(set(arguments))
        return original(arguments)

    monkeypatch.setattr(api, "_overwrite_prepare", capture)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        susie_rss(np.zeros(2), np.eye(2), 100, L=1, overwrite_ld=True)
    assert captured == [_PUBLIC_ARGUMENTS]
    def forbidden(*args):
        raise AssertionError("default must not snapshot or walk aliases")
    monkeypatch.setattr(api, "_overwrite_prepare", forbidden)
    susie_rss([0., 0.], np.eye(2), 100, L=1, overwrite_ld=False)
    prepared = susie_rss([0., 0.], np.eye(2), 100, L=1, init_only=True)
    assert prepared.params["overwrite_ld"] is False


@pytest.mark.parametrize("flag", [0, 1, "yes", None, np.int64(1), np.array(True), [True]])
def test_flag_is_strict_boolean(flag):
    r = matrix()
    before = r.copy()
    with pytest.raises(ValueError, match="overwrite_ld must"):
        susie_rss(np.zeros(9), r, 100, overwrite_ld=flag)
    bits(r, before)


@pytest.mark.parametrize("kind", ["F", "strided", "readonly", "float32", "endian",
    "metadata", "external", "subclass", "misaligned", "empty", "nonsquare"])
def test_unsupported_storage_rejected_before_writes(kind):
    r = matrix(9)
    if kind == "F": r = np.asfortranarray(r)
    elif kind == "strided": r = np.eye(18)[::2, ::2]
    elif kind == "readonly": r.flags.writeable = False
    elif kind == "float32": r = r.astype(np.float32)
    elif kind == "endian": r = r.astype(">f8")
    elif kind == "metadata": r = r.view(np.dtype("f8", metadata={"source": "x"}))
    elif kind == "external": r = np.ndarray((9, 9), dtype="f8", buffer=bytearray(r.tobytes()))
    elif kind == "subclass": r = r.view(type("Special", (np.ndarray,), {}))
    elif kind == "misaligned": r = np.ndarray((9, 9), dtype="f8", buffer=bytearray(9 * 9 * 8 + 1), offset=1)
    elif kind == "empty": r = np.empty((0, 0))
    elif kind == "nonsquare": r = np.zeros((9, 8))
    before = r.tobytes()
    with pytest.raises(ValueError, match="overwrite_ld"):
        susie_rss(np.zeros(len(r)), r, 100, overwrite_ld=True)
    assert r.tobytes() == before


@pytest.mark.parametrize("name", ["z", "prior_weights", "variant_ids", "tol", "variant_metadata"])
def test_all_argument_aliases_including_zero_dimensional_and_nested_are_rejected(name):
    r = matrix()
    original = r.copy()
    value = r[0]
    if name == "tol": value = r[:1, :1].reshape(())
    if name == "prior_weights": value = [r[0]]
    if name == "variant_metadata": value = {"nested": [r[0]]}
    args = dict(z=np.zeros(len(r)), R=r, n=100, overwrite_ld=True)
    args[name] = value
    with pytest.raises(ValueError, match="overlaps"):
        susie_rss(**args)
    bits(r, original)


@pytest.mark.parametrize("name", sorted(_PUBLIC_ARGUMENTS - {"R", "overwrite_ld"}))
def test_snapshot_rejects_overlap_in_every_other_public_argument(name):
    r = matrix()
    original = r.copy()
    args = dict(z=np.zeros(9), R=r, n=100, overwrite_ld=True)
    args[name] = r[0]
    with pytest.raises(ValueError, match="overlaps"):
        susie_rss(**args)
    bits(r, original)


def test_disjoint_views_of_same_numpy_owner_are_allowed():
    owner = np.empty(9 * 9 + 9)
    r, z = owner[:81].reshape(9, 9), owner[81:]
    r[:] = matrix()
    z[:] = 0.
    original = r.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        child = susie_rss(z, r, 100, overwrite_ld=True, L=1)
    bits(r, repair(original, 100))
    bits(z, np.zeros(9))
    assert child.qc["ld_overwrite"]["action"] == "ordered_scaled_xtx"


@pytest.mark.parametrize("as_plain_view", [False, True])
def test_mmap_and_plain_views_of_mmap_are_rejected(tmp_path, as_plain_view):
    mapped = np.memmap(tmp_path / "ld.bin", dtype="f8", mode="w+", shape=(9, 9))
    mapped[:] = matrix()
    r = np.asarray(mapped) if as_plain_view else mapped
    original = r.tobytes()
    with pytest.raises(ValueError, match="overwrite_ld"):
        susie_rss(np.zeros(9), r, 100, overwrite_ld=True)
    assert r.tobytes() == original


@pytest.mark.parametrize("option,value", [("track_fit", "compact"), ("retain_data", True),
    ("check_input", True), ("refine", True), ("init_only", True), ("var_y", 1.),
    ("null_weight", 0.1), ("R_mismatch", "eb"), ("unmappable_effects", False),
    ("n_threads", True), ("max_iter", 0), ("tol", np.array(1e-4)),
    ("variant_metadata", {"x": np.zeros(())}), ("tol", -1.),
    ("prior_weights", [1.]), ("estimate_prior_method", "unknown"),
    ("variant_ids", ["one"]), ("min_abs_corr", 2.), ("R_finite", np.bool_(False))])
def test_unsupported_options_rejected_before_write(option, value):
    r = matrix()
    before = r.copy()
    with pytest.raises((ValueError, TypeError)):
        susie_rss(np.zeros(9), r, 100, overwrite_ld=True, **{option: value})
    bits(r, before)


def test_conversion_and_truthiness_hooks_are_never_invoked():
    called = []
    class Hook:
        def __array__(self, *args, **kwargs):
            called.append("array")
            return np.zeros(9)
        def __bool__(self):
            called.append("bool")
            return True
    r = matrix()
    before = r.copy()
    for args in (dict(z=Hook(), overwrite_ld=True), dict(z=np.zeros(9), overwrite_ld=Hook()),
                 dict(z=np.zeros(9), overwrite_ld=True, prior_weights=[Hook()])):
        with pytest.raises(ValueError):
            susie_rss(R=r, n=100, **args)
    assert called == []
    bits(r, before)


@pytest.mark.parametrize("attribute", ["warn", "showwarning", "_showwarnmsg", "_showwarnmsg_impl"])
def test_custom_warning_display_does_not_disable_overwrite(monkeypatch, attribute):
    r = matrix()
    before = r.copy()
    invoked = []
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        monkeypatch.setattr(warnings, attribute, lambda *a, **k: invoked.append(True))
        fit = susie_rss(np.zeros(9), r, 100, overwrite_ld=True, L=1)
    assert invoked
    assert fit.qc["ld_overwrite"]["action"] == "ordered_scaled_xtx"
    bits(r, repair(before, 100))


def test_import_under_recording_keeps_standard_display_supported():
    code = """
import warnings
import numpy as np
with warnings.catch_warnings(record=True):
    import prusie
r = np.eye(2)
r[0, 1] = 0.1
r[1, 0] = 0.2
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    fit = prusie.susie_rss([0., 0.], r, 100, L=1, overwrite_ld=True)
assert fit.qc['ld_overwrite']['action'] == 'ordered_scaled_xtx'
"""
    subprocess.run([sys.executable, "-B", "-c", code], check=True)


def test_warning_as_error_consumes_complete_ordered_xtx_before_abort():
    r = matrix()
    expected = repair(r, 100)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeWarning, match="not symmetric"):
            susie_rss(np.zeros(9), r, 100, overwrite_ld=True)
    bits(r, expected)


@pytest.mark.parametrize("defect,scale,error", [("nan", 99., "R must contain only finite"),
    ("negative", 99., "R diagonal must be nonnegative"),
    ("scaled", 1e308, "XtX must contain only finite"),
    ("repair", 1., "XtX must contain only finite")])
def test_error_priority_and_deferred_warning_match_parent(defect, scale, error):
    r = matrix()
    if defect == "nan": r[-1, -1] = np.nan; r[2, 2] = -1.
    elif defect == "negative": r[-1, -1] = -1.
    elif defect == "scaled": r[-1, -1] = 2.
    else: r[0, 1], r[1, 0] = 1e308, 1.1e308
    outputs = []
    for overwrite in (False, True):
        trial = r.copy()
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter("always", RuntimeWarning)
            with pytest.raises(ValueError, match=error):
                susie_rss(np.zeros(9), trial, scale + 1., overwrite_ld=overwrite)
        outputs.append([str(w.message) for w in messages])
        if not overwrite: bits(trial, r)
    assert outputs[0] == outputs[1]
    assert bool(outputs[1]) == (defect == "repair")


def test_default_never_mutates_on_warning_as_error():
    r = matrix()
    before = r.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeWarning, match="not symmetric"):
            susie_rss(np.zeros(9), r, 100)
    bits(r, before)


def test_later_numerical_fit_failure_may_consume_workspace():
    r = matrix()
    expected = repair(r, 100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(ValueError, match="prior variance is unreasonably large"):
            susie_rss(np.ones(9), r, 100, overwrite_ld=True,
                      scaled_prior_variance=1e12, estimate_prior_variance=False)
    bits(r, expected)
