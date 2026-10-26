"""Sum of Single Effects (SuSiE) fine-mapping with a native IBSS solver.

Fit individual-level, sufficient-statistics, RSS and RSS-lambda models.
Options include Gaussian or Gaussian-mixture effect priors,
normal-inverse-gamma inference and slot-activity priors. Additional functions
construct credible sets, summarize and compare results, diagnose LD
consistency, continue or refine fits, and run sensitivity analyses.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from ._types import CredibleSetAudit, FitParameters, InputChecks
from .constructors import susie_auto, susie_trendfilter
from .control import (
    RSSControl,
    SlotPrior,
    slot_prior_betabinom,
    slot_prior_poisson,
    susie_rss_control,
)
from .datasets import load_convergence_example, load_example
from .diagnostics import ConvergenceConfig, diagnose_convergence
from .getters import (
    coef,
    get_cs_correlation,
    get_objective,
    predict,
    summary,
    susie_get_cs,
    susie_get_cs_attainable,
    susie_get_lfsr,
    susie_get_niter,
    susie_get_objective,
    susie_get_pip,
    susie_get_posterior_mean,
    susie_get_posterior_samples,
    susie_get_posterior_sd,
    susie_get_prior_variance,
    susie_get_residual_variance,
    susie_init_coef,
)
from .individual import calc_z, compute_suff_stat, susie, univariate_regression
from .refinement import RefinementConfig, continue_fit, evaluate_objective, refine_fit
from .result import InputQuality, SusieResult
from .rss import susie_rss
from .suff_stat import susie_ss, susie_suff_stat
from .workhorse import SusieInputs, susie_workhorse

if TYPE_CHECKING:
    from . import (
        analysis as analysis,
    )
    from . import (
        comparison as comparison,
    )
    from . import (
        ld as ld,
    )
    from . import (
        plots as plots,
    )
    from . import (
        report as report,
    )
    from . import (
        viz as viz,
    )
    from .analysis import summarize as summarize
    from .comparison import compare_fits as compare_fits
    from .comparison import run_sensitivity as run_sensitivity
    from .ld import diagnose_ld as diagnose_ld
    from .ld import estimate_s_rss as estimate_s_rss
    from .ld import kriging_rss as kriging_rss
    from .report import summary_text as summary_text

__all__ = [
    "ConvergenceConfig",
    "CredibleSetAudit",
    "FitParameters",
    "InputChecks",
    "InputQuality",
    "RSSControl",
    "RefinementConfig",
    "SlotPrior",
    "SusieInputs",
    "SusieResult",
    "absolute_order",
    "block_coordinate_ascent",
    "calc_z",
    "coef",
    "compare_fits",
    "compute_marginal_bhat_shat",
    "compute_suff_stat",
    "continue_fit",
    "diagnose_convergence",
    "diagnose_ld",
    "estimate_s_rss",
    "evaluate_objective",
    "get_cs_correlation",
    "get_objective",
    "is_symmetric_matrix",
    "kriging_rss",
    "load_convergence_example",
    "load_example",
    "path_order",
    "predict",
    "refine_fit",
    "run_sensitivity",
    "slot_prior_betabinom",
    "slot_prior_poisson",
    "summarize",
    "summary",
    "summary_text",
    "susie",
    "susie_auto",
    "susie_get_cs",
    "susie_get_cs_attainable",
    "susie_get_lfsr",
    "susie_get_niter",
    "susie_get_objective",
    "susie_get_pip",
    "susie_get_posterior_mean",
    "susie_get_posterior_samples",
    "susie_get_posterior_sd",
    "susie_get_prior_variance",
    "susie_get_residual_variance",
    "susie_init_coef",
    "susie_post_outcome_configuration",
    "susie_rss",
    "susie_rss_control",
    "susie_rss_lambda",
    "susie_ser",
    "susie_ss",
    "susie_suff_stat",
    "susie_trendfilter",
    "susie_workhorse",
    "univar_order",
    "univariate_regression",
]
__version__ = "0.2.4.dev2"
from .configuration import susie_post_outcome_configuration
from .rss_lambda import susie_rss_lambda
from .ser import susie_ser
from .utilities import (
    absolute_order,
    block_coordinate_ascent,
    compute_marginal_bhat_shat,
    is_symmetric_matrix,
    path_order,
    univar_order,
)

_OPTIONAL_EXPORTS = {
    "summarize": "analysis",
    "compare_fits": "comparison",
    "run_sensitivity": "comparison",
    "diagnose_ld": "ld",
    "estimate_s_rss": "ld",
    "kriging_rss": "ld",
    "summary_text": "report",
}
_OPTIONAL_MODULES = {"analysis", "comparison", "ld", "plots", "report", "viz"}


def __getattr__(name: str) -> Any:
    if name in _OPTIONAL_EXPORTS:
        module = import_module(f".{_OPTIONAL_EXPORTS[name]}", __name__)
        value = getattr(module, name)
    elif name in _OPTIONAL_MODULES:
        value = import_module(f".{name}", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | _OPTIONAL_MODULES)
