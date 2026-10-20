"""Linear-storage trend-filter design and cumulative-sum products."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class _TrendTranspose:
    design: TrendDesign

    def __matmul__(self, response: NDArray[np.float64]) -> NDArray[np.float64]:
        value = np.asarray(response, dtype=float)
        for _ in range(self.design.order + 1):
            value = -np.cumsum(value, axis=0)
        return value


@dataclass(frozen=True)
class TrendDesign:
    """Structured ``(-upper_triangular_ones) ** (order + 1)`` design."""

    n: int
    order: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.n, self.n

    @property
    def T(self) -> _TrendTranspose:
        return _TrendTranspose(self)

    def __matmul__(self, coefficients: NDArray[np.float64]) -> NDArray[np.float64]:
        value = np.asarray(coefficients, dtype=float)
        for _ in range(self.order + 1):
            value = -np.cumsum(value[::-1], axis=0)[::-1]
        return value

    def __getitem__(self, index: Any) -> NDArray[np.float64]:
        rows, columns = index
        selected = np.arange(self.n)[columns]
        selected = np.atleast_1d(selected)
        # Only requested columns are constructed for CS purity calculations.
        matrix = np.zeros((self.n, len(selected)))
        matrix[selected, np.arange(len(selected))] = 1
        return (self @ matrix)[rows]

    def column_statistics(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        sums = np.ones(self.n)
        for _ in range(self.order + 1):
            sums = -np.cumsum(sums)
        base = np.ones(self.n)
        for _ in range(self.order):
            base = np.cumsum(base)
        raw_squares = np.cumsum(base**2)
        means = sums / self.n
        return means, np.maximum(raw_squares - self.n * means**2, 0)
