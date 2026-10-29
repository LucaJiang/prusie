"""Small helpers for reading the published 0.3.0 benchmark records."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def distribution(values):
    values = sorted(v for v in values if finite(v))
    if not values:
        return dict(n=0, minimum=None, q1=None, median=None, q3=None, maximum=None)

    # Linear interpolation agrees with numpy.quantile(method="linear").
    def quantile(p):
        position = (len(values) - 1) * p
        lo = math.floor(position)
        hi = math.ceil(position)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)

    return dict(
        n=len(values),
        minimum=values[0],
        q1=quantile(0.25),
        median=quantile(0.5),
        q3=quantile(0.75),
        maximum=values[-1],
    )
