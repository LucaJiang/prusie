"""Sum of Single Effects (SuSiE) fine-mapping with a native IBSS solver.

Fit individual-level, sufficient-statistics, RSS and RSS-lambda models.
Options include Gaussian or Gaussian-mixture effect priors,
normal-inverse-gamma inference and slot-activity priors. Additional functions
construct credible sets, summarize and compare results, diagnose LD
consistency, continue or refine fits, and run sensitivity analyses.
"""

from .analysis import summarize
from .api import susie_rss, susie_ss, susie_suff_stat
from .comparison import compare_fits, run_sensitivity
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
from .individual import (
    SusieInputs,
    calc_z,
    compute_suff_stat,
    susie,
    susie_workhorse,
    univariate_regression,
)
from .ld import diagnose_ld, estimate_s_rss, kriging_rss
from .refinement import RefinementConfig, continue_fit, evaluate_objective, refine_fit
from .report import summary_text
from .result import InputQuality, SusieResult

__all__ = [
    "ConvergenceConfig",
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
