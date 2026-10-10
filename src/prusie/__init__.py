"""Native SuSiE-RSS inference, referenced to pinned official susieR 0.16.6."""
from .api import susie_rss, susie_suff_stat
from .result import SusieResult
from .datasets import load_example

__all__ = ["susie_rss", "susie_suff_stat", "SusieResult", "load_example"]
__version__ = "0.2.3rc6"

from .diagnostics import ConvergenceConfig, diagnose_convergence
from .refinement import RefinementConfig, refine_fit, continue_fit, evaluate_objective
__all__ += ["ConvergenceConfig", "diagnose_convergence", "RefinementConfig", "refine_fit", "continue_fit", "evaluate_objective"]
from .analysis import summarize
from .comparison import compare_fits, run_sensitivity
from .ld import diagnose_ld, estimate_s_rss, kriging_rss
from .datasets import load_convergence_example
__all__ += ["summarize", "compare_fits", "run_sensitivity", "diagnose_ld", "estimate_s_rss", "kriging_rss", "load_convergence_example"]
