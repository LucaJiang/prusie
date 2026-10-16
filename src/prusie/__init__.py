"""prusie: Sum of Single Effects (SuSiE) fine-mapping with a native IBSS solver.

The fitting functions :func:`susie_rss` and :func:`susie_suff_stat` implement
the Gaussian SuSiE and SuSiE-RSS models of susieR 0.16.6. The remaining
functions summarize, diagnose and compare fitted results.
"""

from .analysis import summarize
from .api import susie_rss, susie_suff_stat
from .comparison import compare_fits, run_sensitivity
from .datasets import load_convergence_example, load_example
from .diagnostics import ConvergenceConfig, diagnose_convergence
from .ld import diagnose_ld, estimate_s_rss, kriging_rss
from .refinement import RefinementConfig, continue_fit, evaluate_objective, refine_fit
from .report import summary_text
from .result import SusieResult

__all__ = [
    "ConvergenceConfig",
    "RefinementConfig",
    "SusieResult",
    "compare_fits",
    "continue_fit",
    "diagnose_convergence",
    "diagnose_ld",
    "estimate_s_rss",
    "evaluate_objective",
    "kriging_rss",
    "load_convergence_example",
    "load_example",
    "refine_fit",
    "run_sensitivity",
    "summarize",
    "summary_text",
    "susie_rss",
    "susie_suff_stat",
]
__version__ = "0.2.4.dev2"
