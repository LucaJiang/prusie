"""In-memory timing harness for the isolated timing-instrumented prusie build.

Usage: timer = install(); fit = prusie.susie_rss(...); timer.latest().
Call latest() only after the outer complete-call clock has stopped.
"""
from __future__ import annotations
import importlib
import time


class NativeStageTimer:
    def __init__(self):
        import prusie
        from prusie import _native
        self._latest = None
        self._active = False
        self._parts = None
        self._native_done_ns = None
        self._index = 0
        self.original_public = prusie.susie_rss
        self.original_native = _native.fit

        def native_call(*args, **kwargs):
            result = self.original_native(*args, **kwargs)
            if self._active:
                self._native_done_ns = time.perf_counter_ns()
                assert self._parts is None, "Expected exactly one native fit per public call"
                self._parts = (float(result["_benchmark_native_fitting_seconds"]),
                               float(result["_benchmark_native_export_seconds"]))
            return result

        def public_call(*args, **kwargs):
            assert not self._active, "Nested fits are outside the frozen panel contract"
            self._latest = self._parts = self._native_done_ns = None
            assert kwargs.get("n_threads", 1) == 1
            assert kwargs.get("refine", False) is False
            assert kwargs.get("s_init") is None and kwargs.get("model_init") is None
            assert kwargs.get("L_greedy") is None and kwargs.get("R_finite") is None
            assert kwargs.get("R_mismatch", "none") == "none"
            assert not isinstance(kwargs.get("scaled_prior_variance", 0.2), list)
            self._active = True
            try:
                result = self.original_public(*args, **kwargs)
                end = time.perf_counter_ns()
                assert self._parts is not None and self._native_done_ns is not None, "Instrumented native timing keys missing"
                core, export = self._parts
                finalize = (end - self._native_done_ns) / 1e9
                assert core > 0 and export >= 0 and finalize >= 0
                self._index += 1
                self._latest = dict(call_index=self._index, fitting_seconds=core + export + finalize,
                                    native_solver_seconds=core, native_export_seconds=export,
                                    python_finalize_seconds=finalize,
                                    definition="Sum of directly timed solver initialization/IBSS/finalization spans, native result export and Python posterior/CS/RSS result construction; all input preparation excluded")
                return result
            finally:
                self._active = False

        self.public_wrapper = public_call
        self.native_wrapper = native_call
        prusie.susie_rss = public_call
        importlib.import_module("prusie.rss").susie_rss = public_call
        _native.fit = native_call

    def latest(self):
        assert self._latest is not None, "No successful timed fit has completed"
        return dict(self._latest)


def install():
    return NativeStageTimer()
