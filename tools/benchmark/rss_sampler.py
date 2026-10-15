"""Linux current-RSS sampler for the optional fine_mapping memory handshake.

Sampling reports a lower bound on the fit-stage peak, independently of the
backend's full-process high-water mark. This module never changes fitting inputs.
"""

import time
from itertools import pairwise
from pathlib import Path


def resident_bytes(pid):
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Process RSS is unavailable")


class Sampler:
    """Call poll regularly while a fresh backend process is running."""

    def __init__(self, prefix, target_interval_seconds=0.002):
        self.prefix = Path(prefix)
        self.target_interval_seconds = target_interval_seconds
        self.pid = None
        self.samples = []
        self.baseline = None
        self.finished = False
        self.process_error = None

    def marker(self, suffix):
        return Path(str(self.prefix) + "." + suffix)

    def poll(self):
        try:
            self._poll()
        except (FileNotFoundError, ProcessLookupError) as error:
            # A backend killed by its resource limit may disappear between
            # Popen.poll and the /proc read. Preserve the failed measurement.
            self.process_error = str(error)

    def _poll(self):
        if self.process_error is not None:
            return
        if self.finished:
            return
        if self.pid is None:
            ready = self.marker("ready")
            if not ready.exists() or not ready.read_text().strip():
                return
            self.pid = int(ready.read_text())
            self.baseline = resident_bytes(self.pid)
            self.samples.append((time.monotonic(), self.baseline))
            self.marker("go").touch(exist_ok=False)
        self.samples.append((time.monotonic(), resident_bytes(self.pid)))
        if self.marker("done").exists():
            self.finished = True
            self.marker("release").touch(exist_ok=False)

    def result(self):
        if not self.finished:
            return dict(
                status="incomplete", samples=len(self.samples), process_error=self.process_error
            )
        times, rss = zip(*self.samples, strict=True)
        return dict(
            status="ok",
            baseline_rss_bytes=self.baseline,
            sampled_fit_peak_rss_bytes=max(rss),
            sampled_incremental_peak_bytes=max(rss) - self.baseline,
            returned_fit_rss_bytes=rss[-1],
            samples=len(rss),
            target_interval_seconds=self.target_interval_seconds,
            max_observed_interval_seconds=max(b - a for a, b in pairwise(times)),
            sampled_boundary_seconds=times[-1] - times[0],
            definition=(
                "Current RSS after imports, inputs and GC; sampled until a "
                "returned fit is retained, before export. Increment = sampled "
                "peak minus this current-RSS baseline."
            ),
            limitation=(
                "Sampled peak and increment are lower bounds; short allocation "
                "spikes between samples can be missed. Handshake waits are "
                "outside timing mode. Full-process HWM is a separate backend "
                "metric."
            ),
        )
