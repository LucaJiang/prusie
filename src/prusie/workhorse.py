"""Prepared SuSiE inputs and shared model fitting and result assembly."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import _native
from ._request import (
    _FitRequest,
    _legacy_analysis_limitations,
    _require_legacy_analysis,
)
from ._types import CredibleSetAudit
from ._validation import (
    _warn,
)
from .diagnostics import (
    convergence_summary,
    nonconvergence_message,
    tracking_level,
)
from .posterior import credible_sets, marginal_pip
from .refinement import RefinementConfig, initial_options, refine_fit
from .result import SusieResult


def _fit(request: _FitRequest, *, _executor=None) -> SusieResult | SusieInputs:
    """Run the prepared model and assemble the common public result."""
    limitations = tuple(
        dict.fromkeys(
            request.analysis_limitations
            + _legacy_analysis_limitations(
                request.options,
                design=request.design,
                dynamic=request.dynamic,
                L_greedy=request.params.get("L_greedy"),
                median_abs_corr=request.params.get("median_abs_corr"),
                cs_extension_corr=request.params.get("cs_extension_corr"),
            )
        )
    )
    if limitations:
        # Dynamic models can be attached after ordinary preparation. Never expose
        # their static snapshot to Gaussian-only objective/replay helpers.
        request = replace(request, prepared=None, signature=None, analysis_limitations=limitations)
    crossproduct = request.crossproduct
    crossresponse = request.crossresponse
    yty = request.yty
    n = request.n
    options = request.options
    XtX = request.XtX
    diagonal = request.diagonal
    scales = request.scales
    means = request.means
    y_mean_value = request.y_mean_value
    ids = request.ids
    metadata = request.metadata
    coverage = request.coverage
    min_abs_corr = request.min_abs_corr
    null_index = request.null_index
    n_purity = request.n_purity
    prior_tol = request.prior_tol
    collected = request.collected
    qc = request.qc
    params = request.params
    control = request.control
    resolution = request.resolution
    prepared = request.prepared
    signature = request.signature
    refine = request.refine
    _rss_matrix_scale = request._rss_matrix_scale
    if params.get("init_only", False):
        return SusieInputs(
            data={
                "XtX": XtX,
                "Xty": crossresponse,
                "yty": yty,
                "n": n,
                "X": None if request.design is None else request.design.X,
            },
            params=params.copy(),
            _request=request,
        )
    if request.deferred_model_init is not None:
        raise ValueError(
            "model_init must be a fitted SusieResult; prepared SusieInputs has no posterior state"
        )
    # refine=True on these models uses the refitting search below; a
    # RefinementConfig still requires the retained-analysis engine.
    _require_legacy_analysis(
        bool(params.get("retain_data")),
        refine if isinstance(refine, RefinementConfig) else False,
        limitations,
    )
    if params.get("L_greedy") is not None:
        return _fit_greedy(request)
    if request.dynamic is not None:
        from .rss_models import fit_dynamic

        native = fit_dynamic(request)
    elif request.design is None:
        if _executor is None:
            native = _native.fit(crossproduct, crossresponse, yty, n, options)
        else:
            native = _native._fit_with_executor(
                crossproduct, crossresponse, yty, n, options, _executor
            )
    else:
        design = request.design
        arguments = (
            np.ascontiguousarray(design.means),
            np.ascontiguousarray(design.scales),
            crossresponse,
            yty,
            n,
            options,
        )
        from ._trend import TrendDesign

        if isinstance(design.X, TrendDesign):
            native = _native.fit_design_trend(design.X.n, design.X.order, *arguments)
        elif hasattr(design.X, "indptr"):
            native = _native.fit_design_csc(
                design.X.data,
                design.X.indices.astype(np.int64),
                design.X.indptr.astype(np.int64),
                design.X.shape[0],
                *arguments,
            )
        else:
            native = _native.fit_design_dense(design.X, *arguments)
    for message in native["warnings"]:
        _warn(message, collected)
    if not native["converged"]:
        _warn(
            nonconvergence_message(native),
            collected,
        )
    alpha, V = native["alpha"], native["V"]
    cs_audit: list[CredibleSetAudit] = []
    sets = credible_sets(
        alpha,
        V,
        XtX,
        coverage=coverage,
        min_abs_corr=min_abs_corr,
        median_abs_corr=params.get("median_abs_corr"),
        cs_extension_corr=params.get("cs_extension_corr"),
        null_index=null_index,
        n_purity=n_purity,
        X=None if request.design is None else request.design.X,
        covariance_scales=None if request.design is not None else np.sqrt(diagonal),
        covariance_multiplier=_rss_matrix_scale,
        audit=cs_audit,
    )
    intercept = y_mean_value - float(np.sum(means * np.sum(alpha * native["mu"], axis=0) / scales))
    summary, trace = convergence_summary(native, options, control)
    summary["config_resolution"] = resolution.copy()
    summary["legacy_requested_config"] = control.options()
    summary["requested_convergence_method"] = request.requested_convergence_method
    summary["requested_config"] = {
        name: options[name]
        for name in ("criterion", "alpha_tol", "pip_tol", "min_iter", "cycle_lag")
    }
    summary["requested_criterion"] = options["criterion"]
    summary["alpha_tol"] = options["alpha_tol"]
    summary["pip_tol"] = options["pip_tol"]
    summary["retained_analysis_available"] = prepared is not None
    summary["retained_analysis_limitations"] = list(limitations)
    fit = SusieResult(
        variant_ids=ids,
        alpha=alpha,
        mu=native["mu"],
        mu2=native["mu2"],
        lbf_variable=native["lbf_variable"],
        lbf=native["lbf"],
        V=V,
        sigma2=native["sigma2"],
        pip=marginal_pip(
            alpha,
            V,
            null_index=null_index,
            prior_tol=prior_tol,
            slot_weights=native.get("slot_weights"),
        ),
        elbo=native["elbo"],
        niter=native["niter"],
        converged=native["converged"],
        sets=sets,
        null_index=null_index,
        params=params,
        backend_version=_native.backend_version(),
        warnings=collected,
        qc=qc,
        variant_metadata=metadata,
        X_column_scale_factors=scales,
        intercept=intercept,
        KL=native["KL"],
        XtXr=native["XtXr"],
        convergence=summary,
        trace=trace,
        cs_audit=cs_audit,
        data_signature=signature,
        _prepared=prepared,
        slot_weights=native.get("slot_weights"),
        lbf_grid=native.get("lbf_grid"),
        rv=native.get("rv"),
        marginal_loglik=native.get("marginal_loglik"),
        R_finite_diagnostics=native.get("R_finite_diagnostics", {}),
    )
    if fit.R_finite_diagnostics:
        if "R_bf_attenuation" in native:
            from ._rss_qc import summarize_attenuation

            summarize_attenuation(
                fit, native["R_bf_attenuation"], request.dynamic["control"].sensitivity_threshold
            )
        else:
            fit.R_finite_diagnostics["R_reliability_flag"] = bool(
                fit.R_finite_diagnostics.get("artifact_flag", False)
            )
        if fit.R_finite_diagnostics["R_reliability_flag"]:
            _warn(
                "Fine-mapping is sensitive to LD-reference uncertainty; "
                "inspect R_finite_diagnostics",
                collected,
            )
    if request.design is not None and request.design.fitted_response:
        fit.fitted = np.asarray(request.design.X @ fit.coef).ravel() + fit.intercept
    if params.get("verbose"):
        import logging

        logging.getLogger("prusie").info(
            "SuSiE: %d sweeps, converged=%s, %d credible sets",
            fit.niter,
            fit.converged,
            len(fit.sets["cs"]),
        )
    if refine is not False:
        if limitations:
            fit = _refine_by_refitting(request, fit)
        else:
            fit = refine_fit(fit, refine if isinstance(refine, RefinementConfig) else None)
    return fit


def _refine_by_refitting(request: _FitRequest, model: SusieResult) -> SusieResult:
    """Credible-set refinement by block coordinate ascent.

    For each credible set, refit with its members' prior weights set to zero,
    then refit from that posterior with the original weights. The candidate
    with the largest ELBO is accepted if it improves the objective by more than
    ``tol``; the search stops when no candidate does or when the relative ELBO
    change falls below ``tol``.
    """
    tol = float(request.params["tol"])
    null_index = request.null_index
    null_weight = float(request.params["null_weight"])
    weights = np.asarray(request.options["prior_weights"], dtype=np.float64)
    base = weights if null_index is None else np.delete(weights, null_index) / (1 - null_weight)

    def full_weights(values: NDArray[np.float64]) -> NDArray[np.float64]:
        if null_index is not None:
            values = np.r_[values * (1 - null_weight), null_weight]
        return values / values.sum()

    def refit(prior_weights: NDArray[np.float64], init: SusieResult | None) -> SusieResult:
        options = {k: v for k, v in request.options.items() if not k.startswith("init_")}
        options.update(prior_weights=prior_weights, track_level=tracking_level(False))
        if init is not None:
            options.update(
                initial_options(
                    {"alpha": init.alpha, "mu": init.mu, "mu2": init.mu2},
                    ids=request.ids,
                    metadata=request.metadata,
                    scales=request.scales,
                    null_index=null_index,
                    signature=None,
                    L=options["l"],
                    weights=prior_weights,
                    estimate_prior_variance=options["estimate_prior_variance"],
                    estimate_residual_variance=options["estimate_residual_variance"],
                    prior_variance=options.get(
                        "component_prior_variance", options["prior_variance"]
                    ),
                )
            )
        params = {
            **request.params,
            "prior_weights": prior_weights.tolist(),
            "refine": False,
            "track_fit": False,
            "verbose": False,
            "s_init": None if init is None else "posterior_state",
        }
        return _fit(
            replace(
                request,
                options=options,
                params=params,
                refine=False,
                collected=[],
                qc=dict(request.qc),
            )
        )

    if not model.sets["cs"]:
        return model
    history: list[dict[str, Any]] = []
    total_fits, converged, stop_reason, accepted = 0, False, "max_iter", False
    original_weights = full_weights(base)
    for iteration in range(1, 101):
        candidates = []
        for members in model.sets["cs"]:
            excluded = base.copy()
            excluded[np.asarray(members, dtype=np.intp)] = 0.0
            if not np.any(excluded):
                break
            explored = refit(full_weights(excluded), None)
            candidates.append(refit(original_weights, explored))
            total_fits += 2
        current = float(model.elbo[-1])
        if not candidates:
            converged, stop_reason = True, "no_credible_sets"
            break
        objectives = [float(candidate.elbo[-1]) for candidate in candidates]
        best = int(np.argmax(objectives))
        history.append(dict(iteration=iteration, elbo=current, candidate_elbos=objectives))
        if objectives[best] - current <= tol:
            converged, stop_reason = True, "no_improvement"
            break
        change = objectives[best] - current
        model, accepted = candidates[best], True
        if change / max(1.0, abs(objectives[best])) < tol:
            converged, stop_reason = True, "relative_elbo_change"
            break
    if accepted:
        model.warnings[:0] = request.collected
    model.params["refine"] = True
    model.refinement = dict(
        status="converged" if converged else "not_converged",
        stop_reason=stop_reason,
        method="block_coordinate_ascent",
        history=history,
        total_fits=total_fits,
    )
    if not converged:
        _warn("Credible-set refinement did not converge in 100 iterations", model.warnings)
    return model


def _model_init_state(model: SusieResult | Mapping[str, Any]) -> dict[str, Any]:
    """Extract initialization fields, clearing CS metadata as susieR does."""
    if isinstance(model, Mapping):
        return {**model, "sets": None}
    return dict(
        alpha=model.alpha,
        mu=model.mu,
        mu2=model.mu2,
        V=model.V,
        sigma2=model.sigma2,
        variant_ids=model.variant_ids,
        variant_metadata=model.variant_metadata,
        scales=model.X_column_scale_factors,
        null_index=model.null_index,
        data_signature=model.data_signature,
        sets=None,
    )


def _fit_greedy(request: _FitRequest) -> SusieResult:
    """Grow L in fixed increments, matching susieR's workhorse outer loop."""
    maximum, step = request.options["l"], request.params["L_greedy"]
    current, previous, history = min(step, maximum), None, []
    while True:
        options = {
            key: value for key, value in request.options.items() if not key.startswith("init_")
        }
        options["l"] = current
        if "component_prior_variance" in options:
            options["component_prior_variance"] = options["component_prior_variance"][:current]
        if previous is not None:
            options.update(
                initial_options(
                    _model_init_state(previous),
                    ids=request.ids,
                    metadata=request.metadata,
                    scales=request.scales,
                    null_index=request.null_index,
                    signature=request.signature,
                    L=current,
                    weights=options["prior_weights"],
                    estimate_prior_variance=options["estimate_prior_variance"],
                    estimate_residual_variance=options["estimate_residual_variance"],
                    prior_variance=options.get(
                        "component_prior_variance", options["prior_variance"]
                    ),
                )
            )
        else:
            for key, value in request.options.items():
                if key.startswith("init_"):
                    options[key] = value
        params = {**request.params, "L_greedy": None, "effective_L": current}
        previous = _fit(replace(request, options=options, params=params))
        minimum = float(np.nanmin(previous.lbf))
        action = (
            "L_max reached"
            if current >= maximum
            else "saturated"
            if minimum < request.params["greedy_lbf_cutoff"]
            else "grow"
        )
        history.append(dict(L=current, min_lbf=minimum, action=action))
        if action != "grow":
            previous.params.update(L_greedy=step, greedy_history=history)
            return previous
        current = min(current + step, maximum)


@dataclass(frozen=True)
class SusieInputs:
    """Validated data and fitting parameters returned by ``init_only=True``.

    ``data`` and ``params`` are inspectable mappings. Pass this object to
    :func:`susie_workhorse` to fit the same prepared inputs. A trend-filter
    MAD preparation may retain another preparation as ``model_init``;
    this is not a fitted warm start and cannot be used by the workhorse.
    """

    data: dict[str, Any]
    params: dict[str, Any]
    _request: Any = field(repr=False)

    def __getitem__(self, key: str) -> dict[str, Any]:
        if key not in ("data", "params"):
            raise KeyError(key)
        return getattr(self, key)


def susie_workhorse(inputs: SusieInputs) -> SusieResult:
    """Fit a request returned by an entry point with ``init_only=True``.

    Parameters
    ----------
    inputs : SusieInputs
        Validated data and model options returned by a public fitter.

    Returns
    -------
    SusieResult
        Posterior fit using the stored request without repeating preprocessing.

    Raises
    ------
    ValueError
        A deferred ``model_init`` contains prepared inputs instead of a fitted
        posterior, as in the default-MAD trend-filter preparation-only path.
    """
    from dataclasses import replace

    if not isinstance(inputs, SusieInputs):
        raise TypeError("inputs must be SusieInputs returned by init_only=True")
    request = replace(inputs._request, params={**inputs._request.params, "init_only": False})
    return _fit(request)
