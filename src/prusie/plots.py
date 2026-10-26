"""Compatibility imports for figures now organized in :mod:`prusie.viz`."""

from .viz import (
    plot_convergence,
    plot_ld_diagnostics,
    plot_locus,
    plot_prior_influence,
    plot_refinement,
    plot_sensitivity,
    plot_signal,
    read_gene_track,
)
from .viz._common import _variant_ticks as _variant_ticks

__all__ = [
    "plot_convergence",
    "plot_ld_diagnostics",
    "plot_locus",
    "plot_prior_influence",
    "plot_refinement",
    "plot_sensitivity",
    "plot_signal",
    "read_gene_track",
]
