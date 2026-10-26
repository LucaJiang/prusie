"""Every preparation path preserves all caller-owned array bytes."""

import warnings

import numpy as np
import pytest

from prusie import susie_rss


@pytest.mark.parametrize("layout", ["C", "F", "strided", "reverse", "unaligned"])
@pytest.mark.parametrize("asymmetric", [False, True])
@pytest.mark.parametrize("check_input", [False, True])
def test_rss_never_mutates_caller_buffers(layout, asymmetric, check_input):
    R = np.eye(5)
    if asymmetric:
        R[1, 3] = 0.2
        R[3, 1] = 0.1
    if layout == "F":
        R = np.asfortranarray(R)
    elif layout == "strided":
        base = np.zeros((10, 10))
        base[::2, ::2] = R
        R = base[::2, ::2]
    elif layout == "reverse":
        R = R[::-1, ::-1]
    elif layout == "unaligned":
        base = np.empty(R.nbytes + 1, dtype=np.uint8)
        view = np.ndarray(R.shape, dtype=np.float64, buffer=base, offset=1)
        view[:] = R
        R = view
    z = np.array([2.0, -1.0, 0.5, 0.0, 1.0])
    before_R, before_z = R.tobytes(), z.tobytes()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        susie_rss(z=z, R=R, n=100, L=2, max_iter=3, check_input=check_input)
    assert R.tobytes() == before_R
    assert z.tobytes() == before_z


@pytest.mark.parametrize("check_input", [False, True])
def test_redirected_warning_stream_preserves_display_and_caller_input(monkeypatch, check_input):
    import sys

    R = np.array([[1.0, 0.4], [0.2, 1.0]])
    displayed = []

    class CallbackStream:
        def write(self, text):
            displayed.append(text)

        def flush(self):
            pass

    with warnings.catch_warnings():
        warnings.simplefilter("always")

        # Pytest records warnings by replacing _showwarnmsg_impl. Restore the
        # normal display flow explicitly so the redirected write is exercised.
        def display(message):
            stream = sys.stderr if message.file is None else message.file
            stream.write(warnings._formatwarnmsg(message))

        monkeypatch.setattr(warnings, "_showwarnmsg_impl", display)
        monkeypatch.setattr(sys, "stderr", CallbackStream())
        susie_rss(z=np.array([1.0, 2.0]), R=R, n=1000, L=1, max_iter=2, check_input=check_input)
    assert R[0, 0] == 1.0
    assert "XtX is not symmetric" in "".join(displayed)


def test_module_warning_filter_keeps_original_helper_stack():
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        warnings.filterwarnings(
            "error", message="XtX is not symmetric", category=RuntimeWarning, module=r"prusie\.api"
        )
        with pytest.raises(RuntimeWarning, match="XtX is not symmetric"):
            susie_rss(
                z=np.array([1.0, 2.0]),
                R=np.array([[1.0, 0.4], [0.2, 1.0]]),
                n=1000,
                L=1,
                max_iter=2,
            )


@pytest.mark.parametrize("hook", ["category", "message", "module"])
@pytest.mark.parametrize("check_input", [False, True])
def test_warning_filter_callbacks_run_after_owned_repair(hook, check_input):
    R = np.array([[1.0, 0.4], [0.2, 1.0]])
    events = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        baseline = susie_rss(
            z=np.array([1.0, 2.0]),
            R=R.copy(),
            n=1000,
            L=1,
            max_iter=2,
            check_input=check_input,
        )

    def callback():
        events.append(hook)
        R[0, 0] = -1.0

    class CategoryMeta(type):
        def __subclasscheck__(cls, subclass):
            callback()
            return True

    class FilterCategory(Warning, metaclass=CategoryMeta):
        pass

    class Pattern:
        def match(self, value):
            callback()
            return True

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if hook == "category":
            warnings.filterwarnings("always", category=FilterCategory)
        else:
            warnings.filters.insert(
                0,
                (
                    "always",
                    Pattern() if hook == "message" else None,
                    Warning,
                    Pattern() if hook == "module" else None,
                    0,
                ),
            )
            warnings._filters_mutated()
        # Warnings dispatch after owned matrix preparation. This deliberately
        # adversarial callback mutates the caller's buffer only after the valid
        # working matrix has been built; it cannot alter the prepared alpha.
        actual = susie_rss(
            z=np.array([1.0, 2.0]),
            R=R,
            n=1000,
            L=1,
            max_iter=2,
            check_input=check_input,
        )
    assert R[0, 0] == -1.0
    np.testing.assert_array_equal(actual.alpha.view(np.uint64), baseline.alpha.view(np.uint64))
    assert events == [hook]
    assert [str(w.message) for w in caught] == ["XtX is not symmetric; using (XtX + t(XtX))/2."]


@pytest.mark.parametrize("check_input", [False, True])
def test_builtin_text_stream_codec_preserves_display_and_caller_input(
    monkeypatch, tmp_path, check_input
):
    import codecs
    import io
    import sys

    R = np.array([[1.0, 0.4], [0.2, 1.0]])
    events = []

    class Encoder(codecs.IncrementalEncoder):
        def encode(self, value, final=False):
            events.append(value)
            return value.encode("utf-8")

    def search(name):
        if name == "prusie_warning_test_codec":
            return codecs.CodecInfo(
                name=name,
                encode=lambda s, errors="strict": (s.encode("utf-8"), len(s)),
                decode=lambda b, errors="strict": (b.decode("utf-8"), len(b)),
                incrementalencoder=Encoder,
                incrementaldecoder=codecs.getincrementaldecoder("utf-8"),
            )

    def display(message):
        stream = sys.stderr if message.file is None else message.file
        stream.write(warnings._formatwarnmsg(message))

    codecs.register(search)
    stream = io.TextIOWrapper(
        io.BufferedWriter(io.FileIO(tmp_path / "warning.txt", "w")),
        encoding="prusie_warning_test_codec",
    )
    try:
        with warnings.catch_warnings(), monkeypatch.context() as patch:
            warnings.simplefilter("always")
            patch.setattr(warnings, "_showwarnmsg_impl", display)
            patch.setattr(sys, "stderr", stream)
            susie_rss(z=np.array([1.0, 2.0]), R=R, n=1000, L=1, max_iter=2, check_input=check_input)
            assert R[0, 0] == 1.0
    finally:
        stream.close()
        codecs.unregister(search)
    assert events and "XtX is not symmetric" in "".join(events)


@pytest.mark.parametrize("hook", ["array_conversion", "z_method_comparison"])
@pytest.mark.parametrize("check_input", [False, True])
def test_preparation_callbacks_keep_original_repair_order(hook, check_input):
    R = np.array([[1.0, 0.4], [0.2, 1.0]])
    events = []

    class Pattern:
        def match(self, value):
            events.append("filter")
            R[0, 0] = -1.0
            return True

    class ArrayLike:
        def __array__(self, dtype=None, copy=None):
            events.append("array")
            warnings.filters.insert(0, ("always", Pattern(), Warning, None, 0))
            warnings._filters_mutated()
            return R

    class Method(str):
        def __eq__(self, other):
            events.append("method")
            if len(events) >= 2:
                R[0, 0] = -1.0
            return str.__eq__(self, other)

    with warnings.catch_warnings(record=True) as caught:
        warnings.resetwarnings()
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="XtX diagonal must be nonnegative"):
            susie_rss(
                z=np.array([1.0, 2.0]),
                R=ArrayLike() if hook == "array_conversion" else R,
                n=1000,
                L=1,
                max_iter=2,
                check_input=check_input,
                z_method=Method("wald") if hook == "z_method_comparison" else "wald",
            )
    assert events == (["array", "filter"] if hook == "array_conversion" else ["method", "method"])
    assert [str(w.message) for w in caught] == ["XtX is not symmetric; using (XtX + t(XtX))/2."]
