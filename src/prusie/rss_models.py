"""Dynamic RSS models with native single-effect updates.

Finite-reference uncertainty, mismatch estimation and panel-mixture updates
follow susieR 0.16.6 (BSD-3-Clause). The ordinary Gaussian fit stays in Rust.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from . import _native
from .posterior import marginal_pip

_EPS = np.finfo(float).eps


def _bounded(
    function: Any, lower: float, upper: float, *, tolerance: float = 1e-8
) -> tuple[float, float]:
    from scipy.optimize import minimize_scalar

    result = minimize_scalar(
        function, bounds=(lower, upper), method="bounded", options={"xatol": tolerance}
    )
    return float(result.x), float(result.fun)


def _lambda_bias(r: np.ndarray, s: np.ndarray, sigma2: float, B: float, estimator: str) -> float:
    keep = np.isfinite(r) & np.isfinite(s) & (s > _EPS)
    if not keep.any() or sigma2 <= _EPS:
        return 0.0
    r2, signal = r[keep] ** 2, s[keep]
    base = sigma2 + signal / B
    positive = (r2 - base) / signal
    positive = positive[np.isfinite(positive) & (positive > 0)]
    information = 0.5 * np.sum((signal / base) ** 2)
    boundary_se = np.sqrt(1 / information) if information > 0 else np.inf
    upper = max(1.0, 100 / B, 10 * float(np.max(positive, initial=0)))
    if estimator == "mle":

        def objective(value: float) -> float:
            variance = base + value * signal
            return float(0.5 * np.sum(np.log(variance) + r2 / variance))

        while (
            np.isfinite(upper)
            and upper < np.finfo(float).max / 10
            and objective(upper) < objective(upper / 2)
        ):
            upper *= 10
        value, score = _bounded(objective, 0, upper)
        if objective(0) <= score:
            return 0.0
    else:
        scale = np.sqrt(max(1 / B, 1 / 10000))
        upper = max(upper, 100 * scale**2)

        def objective(root: float) -> float:
            variance = base + root**2 * signal
            return float(
                0.5 * np.sum(np.log(variance) + r2 / variance) + np.log1p((root / scale) ** 2)
            )

        root, _ = _bounded(objective, 0, np.sqrt(upper))
        value = root**2
    return 0.0 if value < 0.1 * boundary_se else value


def _mixture_gate(
    residual: np.ndarray,
    eta2: np.ndarray,
    B: float,
    bias: float,
    sigma2: float,
    n: float,
    z: np.ndarray,
    reference_p: float,
) -> np.ndarray:
    from scipy.special import expit, ndtri

    rz = residual / np.sqrt(n - 1)
    tau = np.maximum(sigma2 + (1 / B + bias) * eta2, _EPS)
    reference_z = -ndtri(reference_p / 2)
    odds = 0.5 * (np.max(z**2) - reference_z**2)
    signal_prior = float(expit(-odds))
    scale = max(1.0, float(np.quantile(rz**2, 0.99)), float(np.max(tau)), float(np.max(rz**2)))
    lower = max(_EPS, float(np.min(tau)) * 0.01)
    upper = max(1.0, 10 * scale, 100 * float(np.max(rz**2)), 100 * float(np.max(tau)))
    if len(rz) < 2 or signal_prior <= 0:
        wide = scale
    else:

        def objective(log_variance: float) -> float:
            variance = np.exp(log_variance)
            with np.errstate(divide="ignore"):
                log0 = np.log1p(-signal_prior) - 0.5 * (np.log(2 * np.pi * tau) + rz**2 / tau)
                log1 = np.log(signal_prior) - 0.5 * (
                    np.log(2 * np.pi * variance) + rz**2 / variance
                )
            return float(-np.logaddexp(log0, log1).sum())

        log_variance, _ = _bounded(objective, np.log(lower), np.log(upper), tolerance=1e-6)
        wide = float(np.exp(log_variance))
    llr = -0.5 * (np.log(tau / wide) + rz**2 / tau - rz**2 / wide)
    probability = expit(odds + llr)
    return np.where(np.isfinite(probability), probability, 1.0)


def _working_gram(request: Any) -> np.ndarray:
    if request.design is None:
        matrix = np.asarray(request.crossproduct)
        if "matrix_global_scale" in request.options:
            inverse = request.options["matrix_inverse_scales"]
            matrix = (
                (matrix * request.options["matrix_global_scale"])
                * inverse[:, None]
                * inverse[None, :]
            )
        return matrix
    design = request.design
    product = design.X.T @ design.X
    if hasattr(product, "toarray"):
        product = product.toarray()
    matrix = np.asarray(product) - design.X.shape[0] * design.means[:, None] * design.means[None, :]
    matrix = matrix / design.scales[:, None] / design.scales[None, :]
    if request.null_index is not None:
        matrix = np.pad(matrix, ((0, 1), (0, 1)))
    return matrix


def _expected_residual(
    gram: np.ndarray,
    response: np.ndarray,
    yty: float,
    alpha: np.ndarray,
    mu: np.ndarray,
    mu2: np.ndarray,
    slot: np.ndarray,
) -> float:
    component_mean = alpha * mu
    mean = np.sum(slot[:, None] * component_mean, axis=0)
    quadratic = float(mean @ gram @ mean)
    quadratic -= float(
        np.sum(slot**2 * np.einsum("li,ij,lj->l", component_mean, gram, component_mean))
    )
    quadratic += float(np.sum(slot[:, None] * alpha * mu2 * np.diag(gram)))
    return float(yty - 2 * mean @ response + quadratic)


def _lambda_quantities(
    R: np.ndarray, z: np.ndarray, ridge: float, sigma2: float, r_tol: float
) -> tuple[np.ndarray, np.ndarray, float, float]:
    values, vectors = np.linalg.eigh(R)
    values[values < r_tol] = 0
    covariance = sigma2 * values + ridge
    inverse = np.divide(1.0, covariance, out=np.zeros_like(covariance), where=covariance > 0)
    gram = (vectors * (inverse * values**2)) @ vectors.T
    response = vectors @ (inverse * values * (vectors.T @ z))
    yty = float(np.sum(inverse * (vectors.T @ z) ** 2))
    positive = covariance[covariance > 0]
    constant = -len(positive) / 2 * np.log(2 * np.pi) - 0.5 * np.log(positive).sum()
    return gram, response, yty, float(constant)


def _omega_objective(
    panels: list[np.ndarray],
    weights: np.ndarray,
    z: np.ndarray,
    alpha: np.ndarray,
    mu: np.ndarray,
    mu2: np.ndarray,
    slot: np.ndarray,
    sigma2: float,
    nm1: float,
) -> float:
    matrix = sum(weight * panel for weight, panel in zip(weights, panels, strict=True))
    gram, response, yty, constant = _lambda_quantities(matrix, z, 1 / nm1, sigma2, 0)
    transformed_mu, transformed_mu2 = mu * np.sqrt(nm1), mu2 * nm1
    return constant - 0.5 * _expected_residual(
        gram, response, yty, alpha, transformed_mu, transformed_mu2, slot
    )


def _update_omega(
    panels: list[np.ndarray], old: np.ndarray, objective: Any
) -> tuple[np.ndarray, bool]:
    weights = old.copy()
    current = objective(weights)
    count = len(weights)
    if count == 2:
        grid = np.arange(0, 1.0001, 0.25)
        scores = np.array([objective(np.array([w, 1 - w])) for w in grid])
        best = int(np.argmax(scores))
        weights, current = np.array([grid[best], 1 - grid[best]]), scores[best]
    for _ in range(5):
        scores = np.array([objective(np.eye(count)[k]) for k in range(count)])
        vertex = np.eye(count)[int(np.argmax(scores))]
        gamma, negative = _bounded(
            lambda fraction, weights=weights, vertex=vertex: (
                -objective((1 - fraction) * weights + fraction * vertex)
            ),
            0,
            1,
            tolerance=np.finfo(float).eps ** 0.25,
        )
        if -negative - current < 1e-6:
            break
        weights, current = (1 - gamma) * weights + gamma * vertex, -negative
    return weights, bool(np.max(np.abs(weights - old)) < 1e-3)


def fit_dynamic(request: Any) -> dict[str, Any]:
    """Run exact sequential SER updates for a dynamic RSS covariance model."""
    from scipy.special import digamma, expit

    config, settings = request.dynamic, dict(request.options)
    count, p = settings["l"], len(request.crossresponse)
    shape = (count, p)
    alpha = np.asarray(settings.get("init_alpha", np.full(shape, 1 / p))).reshape(shape).copy()
    mu = np.asarray(settings.get("init_mu", np.zeros(shape))).reshape(shape).copy()
    mu2 = np.asarray(settings.get("init_mu2", np.zeros(shape))).reshape(shape).copy()
    variances = np.asarray(
        settings.get(
            "init_V",
            settings.get("component_prior_variance", np.full(count, settings["prior_variance"])),
        )
    ).copy()
    sigma2 = float(settings.get("init_sigma2", settings["residual_variance"]))
    raw_gram = _working_gram(request)
    gram, response, yty = raw_gram, request.crossresponse, request.yty
    lbf_variable, lbf, kl = np.zeros(shape), np.zeros(count), np.zeros(count)
    rv, marginal = np.zeros(count), np.zeros(count)
    grid_values = None
    attenuation = None
    mismatch_init = None
    slot_spec = settings.get("slot_prior")
    slot = np.ones(count)
    gamma_shape, gamma_rate, skip = 0.0, 1.0, 0.0
    if slot_spec:
        if slot_spec["family"] == "betabinom":
            activity = slot_spec.get("a_beta", 1) / (
                slot_spec.get("a_beta", 1) + slot_spec.get("b_beta", 2)
            )
        else:
            activity = min(slot_spec["C"] / count, 1 - 1e-10)
            gamma_shape = slot_spec.get("nu", 8) + slot_spec["C"]
            gamma_rate = slot_spec.get("nu", 8) / max(slot_spec["C"], 1e-6) + 1
        slot[:] = activity
        if slot_spec.get("c_hat_init") is not None:
            slot = np.array(slot_spec["c_hat_init"], dtype=float)
            if slot_spec["family"] != "betabinom":
                gamma_shape = slot_spec.get("nu", 8) + slot.sum()
    bias = 0.0
    trace, objectives = [], []
    history_alpha, history_pip = (
        [alpha.copy()],
        [
            marginal_pip(
                alpha,
                variances,
                null_index=request.null_index,
                prior_tol=settings["prior_tol"],
                slot_weights=slot,
            )
        ],
    )
    method = config.get("mismatch", "none")
    ridge = config.get("lambda", 0.0)
    B = config.get("reference_n", np.inf)
    panels = config.get("panels")
    omega = np.asarray(config.get("omega", [1.0]), dtype=float)
    omega_converged = False
    final_inflation = None
    lambda_difference = 0.0
    alpha_difference = pip_difference = np.inf
    converged, reason = False, "max_iter"

    def fit_bias() -> float:
        if method == "none":
            return 0.0
        mean = np.sum(slot[:, None] * alpha * mu, axis=0)
        fitted = gram @ mean
        residual = (response - fitted) / np.sqrt(request.n - 1)
        signal = fitted**2 / (request.n - 1) + max(float(mean @ fitted), 0)
        return _lambda_bias(residual, signal, sigma2, B, config["control"].mismatch_estimator)

    def single_effect(component: int) -> None:
        nonlocal grid_values, final_inflation, alpha, mu, mu2, variances, kl, attenuation
        mean = np.sum(slot[:, None] * alpha * mu, axis=0)
        without = mean - slot[component] * alpha[component] * mu[component]
        fitted_without = gram @ without
        residual = response - fitted_without
        inflation = None
        if config["kind"] != "lambda":
            eta2 = fitted_without**2 / (request.n - 1)
            signal = eta2 + max(float(without @ fitted_without), 0)
            local_bias: float | np.ndarray = bias
            if method == "eb_mix" and bias > 0 and panels is None:
                local_bias = (
                    _mixture_gate(
                        residual,
                        eta2,
                        B,
                        bias,
                        sigma2,
                        request.n,
                        config["z"],
                        config["control"].mixture_reference_p,
                    )
                    * bias
                )
            inflation = 1 + (1 / B + local_bias) * signal / sigma2
        final_inflation = inflation
        local_settings = {
            **settings,
            "l": 1,
            "prior_variance": float(variances[component]),
            "residual_variance": 1.0 if config["kind"] == "lambda" else sigma2,
        }
        yy = max(float(yty - 2 * without @ response + without @ fitted_without), _EPS)
        update = _native.ser_update(
            np.ascontiguousarray(residual),
            np.ascontiguousarray(np.diag(gram)),
            yy,
            request.n,
            local_settings,
            inflation,
        )
        for name, target in (
            ("alpha", alpha),
            ("mu", mu),
            ("mu2", mu2),
            ("lbf_variable", lbf_variable),
        ):
            target[component] = update[name]
        lbf[component], variances[component], kl[component] = (
            update["lbf"],
            update["V"],
            update.get("KL", 0.0),
        )
        if "KL" not in update:
            local_sigma = local_settings["residual_variance"]
            sampling = np.ones(p) if inflation is None else inflation
            kl[component] = -lbf[component] + np.sum(
                (
                    2 * residual * alpha[component] * mu[component]
                    - np.diag(gram) * alpha[component] * mu2[component]
                )
                / (2 * local_sigma * sampling)
            )
        if count > 1 and settings.get("estimate_residual_method") == "NIG":
            kl[component] = 0.0
        if (
            settings.get("estimate_residual_method") != "NIG"
            and inflation is not None
            and np.any(np.isfinite(inflation) & (inflation > 1))
        ):
            diagonal = np.diag(gram)
            with np.errstate(divide="ignore", invalid="ignore"):
                sampling = sigma2 / diagonal
                estimate = residual / diagonal
                shrinkage = variances[component] / (variances[component] + sampling)
                nominal = 0.5 * (
                    np.log(sampling / (sampling + variances[component]))
                    + estimate**2 * shrinkage / sampling
                )
            delta = np.maximum(nominal - lbf_variable[component], 0)
            delta[~np.isfinite(delta)] = np.nan
            if attenuation is None:
                attenuation = np.full(shape, np.nan)
            attenuation[component] = delta
        if "rv" in update:
            rv[component] = update["rv"]
            marginal[component] = update.get("marginal_loglik", 0)
        if "lbf_grid" in update:
            if grid_values is None:
                grid_values = np.zeros((count, p, update["lbf_grid"].shape[1]))
            grid_values[component] = update["lbf_grid"]

    if method in ("eb", "eb_mix", "eb_force_init") or (method == "eb_ser_init" and np.isinf(B)):
        if "init_alpha" not in settings:
            single_effect(0)
            bias = fit_bias()
            coherence = 1.0
            if method in ("eb", "eb_mix"):
                prior = np.asarray(settings["prior_weights"])
                keep = (alpha[0] > 0) & (alpha[0] >= 2 * prior)
                if keep.any():
                    mass = alpha[0, keep] / alpha[0, keep].sum()
                    correlation = gram[np.ix_(keep, keep)] / (request.n - 1)
                    coherence = float(np.clip(mass @ (correlation**2) @ mass, 0, 1))
                    bias *= coherence
            from ._rss_qc import finite_reference_diagnostics

            init_mean = np.sum(slot[:, None] * alpha * mu, axis=0)
            init_residual = (response - gram @ init_mean) / np.sqrt(request.n - 1)
            init_diagnostics = finite_reference_diagnostics(
                gram / (request.n - 1),
                B,
                bias=bias,
                inflation=final_inflation,
                residual=init_residual,
                control=config["control"],
                mismatch=method,
            )
            mismatch_init = dict(
                method="ser",
                ld_coherence=coherence,
                lambda_bias=bias,
                Q_art=init_diagnostics["Q_art"],
                artifact_flag=init_diagnostics["artifact_flag"],
            )
        history_alpha, history_pip = (
            [alpha.copy()],
            [
                marginal_pip(
                    alpha,
                    variances,
                    null_index=request.null_index,
                    prior_tol=settings["prior_tol"],
                    slot_weights=slot,
                )
            ],
        )

    for iteration in range(settings["max_iter"]):
        if config["kind"] == "lambda":
            gram, response, yty, constant = _lambda_quantities(
                config["R"], config["z"], ridge, sigma2, config["r_tol"]
            )
        elif panels is not None:
            gram = (request.n - 1) * sum(
                weight * panel for weight, panel in zip(omega, panels, strict=True)
            )
            if config.get("panel_sizes") is not None:
                B = 1 / float(np.sum(omega**2 / config["panel_sizes"]))
        for component in range(count):
            if slot[component] < skip:
                continue
            single_effect(component)
            if slot_spec:
                if slot_spec["family"] == "betabinom":
                    others = float(slot.sum() - slot[component])
                    logodds = np.log(slot_spec.get("a_beta", 1) + others) - np.log(
                        slot_spec.get("b_beta", 2) + count - 1 - others
                    )
                else:
                    logodds = digamma(gamma_shape) - np.log(gamma_rate) - np.log(count)
                old_slot = slot[component]
                slot[component] = expit(np.clip(logodds + lbf[component], -20, 20))
                if (
                    slot_spec["family"] != "betabinom"
                    and slot_spec.get("update_schedule", "sequential") == "sequential"
                ):
                    gamma_shape += slot[component] - old_slot
        if (
            slot_spec
            and slot_spec["family"] != "betabinom"
            and slot_spec.get("update_schedule") == "batch"
        ):
            gamma_shape = slot_spec.get("nu", 8) + slot.sum()
        if slot_spec and slot_spec.get("skip_threshold_multiplier", 0) > 0:
            if slot_spec["family"] == "betabinom":
                total = slot.sum()
                a, b = slot_spec.get("a_beta", 1), slot_spec.get("b_beta", 2)
                approximate = np.log(a + total) - np.log(b + count - 1 - total)
                others = total - expit(approximate)
                baseline = np.log(a + others) - np.log(b + count - 1 - others)
            else:
                baseline = digamma(gamma_shape) - np.log(gamma_rate) - np.log(count)
            skip = slot_spec["skip_threshold_multiplier"] * expit(baseline)
        new_bias = fit_bias()
        lambda_difference, bias = abs(new_bias - bias), new_bias
        er2 = _expected_residual(gram, response, yty, alpha, mu, mu2, slot)
        if settings.get("estimate_residual_method") == "NIG":
            if count == 1:
                objective = marginal[0]
            else:
                # Expected Gaussian log likelihood under the conjugate residual
                # posterior; this is the NIG diagnostic objective, not an ELBO.
                a_post = (settings["alpha0"] + request.n) / 2
                b_post = (settings["beta0"] + er2) / 2
                sampling = np.ones(p) if final_inflation is None else final_inflation
                diagonal = np.diag(gram)
                scaled = variances[:, None] * diagonal
                shrinkage = scaled / (scaled + sampling)
                conditional = float(np.sum(alpha * shrinkage * sampling))
                residual_fixed = er2 - b_post / (a_post - 1) * conditional
                objective = (
                    -request.n / 2 * np.log(2 * np.pi)
                    - request.n / 2 * (np.log(b_post) - digamma(a_post))
                    - 0.5 * (residual_fixed * a_post / b_post + conditional)
                )
        elif config["kind"] == "lambda":
            objective = constant - 0.5 * er2 - kl.sum()
        elif final_inflation is not None:
            am = alpha * mu
            residual = response - gram @ np.sum(slot[:, None] * am, axis=0)
            variance = np.sum(
                slot[:, None] ** 2 * ((alpha * mu2) @ (gram**2) - (am @ gram) ** 2), axis=0
            )
            diagonal = np.diag(gram)
            valid = diagonal > 0
            inflation = final_inflation[valid]
            objective = (
                -valid.sum() / 2 * np.log(2 * np.pi)
                - 0.5 * np.log(sigma2 * inflation / diagonal[valid]).sum()
                - 0.5
                * np.sum(
                    (residual[valid] ** 2 + variance[valid])
                    / (diagonal[valid] * sigma2 * inflation)
                )
                - kl.sum()
            )
        else:
            objective = -request.n / 2 * np.log(2 * np.pi * sigma2) - er2 / (2 * sigma2) - kl.sum()
        objectives.append(float(objective))
        current_pip = marginal_pip(
            alpha,
            variances,
            null_index=request.null_index,
            prior_tol=settings["prior_tol"],
            slot_weights=slot,
        )
        criterion = settings["criterion"]
        if criterion == "pip" or not np.isfinite(objective):
            for lag in range(1, min(5, len(history_alpha)) + 1):
                alpha_difference = float(np.max(np.abs(alpha - history_alpha[-lag])))
                pip_difference = float(np.max(np.abs(current_pip - history_pip[-lag])))
                if max(alpha_difference, pip_difference) < settings["tol"]:
                    reason = "alpha_pip_fixed_point" if lag == 1 else f"alpha_pip_cycle_{lag}"
                    if lag > 1:
                        alpha = np.mean([*history_alpha[-(lag - 1) :], alpha], axis=0)
                    converged = lambda_difference <= settings["tol"]
                    break
        elif iteration > 0:
            change = objectives[-1] - objectives[-2]
            converged = 0 <= change < settings["tol"] and lambda_difference <= settings["tol"]
            if converged:
                reason = "elbo_tolerance"
        history_alpha.append(alpha.copy())
        history_pip.append(
            marginal_pip(
                alpha,
                variances,
                null_index=request.null_index,
                prior_tol=settings["prior_tol"],
                slot_weights=slot,
            )
        )
        history_alpha, history_pip = history_alpha[-5:], history_pip[-5:]
        if settings.get("track_level", 0):
            trace.append(
                dict(
                    iteration=iteration + 1,
                    elbo=float(objective),
                    sigma2=sigma2,
                    lambda_bias=bias,
                    alpha_delta=alpha_difference,
                    pip_delta=pip_difference,
                )
            )
        if converged:
            break
        if settings["estimate_residual_variance"]:
            if config["kind"] == "lambda":

                def objective_sigma(value: float, alpha: np.ndarray = alpha) -> float:
                    g, r, yy, c = _lambda_quantities(
                        config["R"], config["z"], ridge, value, config["r_tol"]
                    )
                    return -(c - 0.5 * _expected_residual(g, r, yy, alpha, mu, mu2, slot))

                sigma2, negative = _bounded(
                    objective_sigma, 1e-4, 1 - ridge, tolerance=np.finfo(float).eps ** 0.25
                )
                if objective_sigma(1 - ridge) < negative:
                    sigma2 = 1 - ridge
            elif settings.get("estimate_residual_method") == "NIG":
                # Mean of IG((alpha0+n)/2, (beta0+ER2)/2), as in
                # susieR update_variance_components.default.
                sigma2 = (settings["beta0"] + er2) / (settings["alpha0"] + request.n - 2)
            else:
                sigma2 = er2 / request.n
            sigma2 = float(
                np.clip(
                    sigma2,
                    settings["residual_variance_lowerbound"],
                    settings["residual_variance_upperbound"],
                )
            )
        if panels is not None and len(panels) > 1 and not omega_converged:

            def objective_omega(
                weights: np.ndarray, alpha: np.ndarray = alpha, sigma2: float = sigma2
            ) -> float:
                return _omega_objective(
                    panels, weights, config["z"], alpha, mu, mu2, slot, sigma2, request.n - 1
                )

            omega, omega_converged = _update_omega(panels, omega, objective_omega)
    if settings.get("estimate_residual_method") == "NIG":
        variances *= rv
    inactive = variances < settings["prior_tol"]
    for target in (mu, mu2, lbf_variable):
        target[inactive] = 0
    alpha[inactive] = settings["prior_weights"]
    variances[inactive], lbf[inactive], kl[inactive] = 0, 0, 0
    output = dict(
        alpha=alpha,
        mu=mu,
        mu2=mu2,
        lbf_variable=lbf_variable,
        lbf=lbf,
        V=variances,
        KL=kl,
        elbo=np.asarray(objectives),
        sigma2=sigma2,
        niter=len(objectives),
        converged=converged,
        warnings=[],
        XtXr=gram @ np.sum(slot[:, None] * alpha * mu, axis=0),
        diagnostics=dict(
            trace=trace,
            stop_reason=reason,
            effective_criterion=settings["criterion"],
            alpha_delta=alpha_difference,
            pip_delta=pip_difference,
            lambda_bias=bias,
            omega=omega.tolist(),
        ),
    )
    if settings.get("estimate_residual_method") == "NIG":
        output.update(rv=rv, marginal_loglik=marginal)
    if slot_spec:
        output["slot_weights"] = slot
    if grid_values is not None:
        output["lbf_grid"] = grid_values
    if config["kind"] != "lambda":
        from ._rss_qc import finite_reference_diagnostics

        final_mean = np.sum(slot[:, None] * alpha * mu, axis=0)
        residual = (response - gram @ final_mean) / np.sqrt(request.n - 1)
        output["R_finite_diagnostics"] = finite_reference_diagnostics(
            gram / (request.n - 1),
            B,
            bias=bias,
            inflation=final_inflation,
            residual=residual,
            control=config["control"],
            mismatch=method,
        )
        if mismatch_init is not None:
            output["R_finite_diagnostics"]["R_mismatch_init"] = mismatch_init
        if attenuation is not None:
            output["R_bf_attenuation"] = attenuation
    return output


def prepare_multipanel(arguments: dict[str, Any]) -> Any:
    """Construct panel sub-fits and the outcome-independent initial mixture."""
    import inspect

    from .api import _fit, susie_rss
    from .posterior import _design_correlation

    panels = arguments["panels"]
    if arguments["n"] is None:
        raise ValueError("n is required for multi-panel RSS fitting")
    z = np.asarray(arguments["z"], dtype=float)
    if z.ndim != 1:
        raise ValueError("Multi-panel RSS requires a single shared z-score vector")
    accepted = set(inspect.signature(susie_rss).parameters) - {"R", "X", "R_finite", "init_only"}
    options = {key: value for key, value in arguments.items() if key in accepted}
    sizes = arguments["R_finite"]
    if sizes is None or sizes is False:
        panel_sizes = None
    elif sizes is True:
        if arguments["X"] is None:
            raise ValueError("R_finite=True requires X panels")
        panel_sizes = np.array([np.shape(panel)[0] for panel in panels], dtype=float)
    else:
        panel_sizes = np.asarray(sizes, dtype=float)
        if panel_sizes.ndim == 0:
            panel_sizes = np.full(len(panels), float(panel_sizes))
        if (
            panel_sizes.shape != (len(panels),)
            or np.any(panel_sizes <= 0)
            or not np.isfinite(panel_sizes).all()
        ):
            raise ValueError("R_finite requires one positive size per panel")
    fits, correlations = [], []
    for k, panel in enumerate(panels):
        if arguments["R"] is not None:
            matrix = np.asarray(panel, dtype=float)
            denominator = np.sqrt(np.maximum(np.diag(matrix), 0))
            inverse = np.divide(
                1, denominator, out=np.zeros_like(denominator), where=denominator > 0
            )
            matrix = matrix * inverse[:, None] * inverse[None, :]
        else:
            from .individual import _validated_design

            raw, _ = _validated_design(panel, np.zeros(np.shape(panel)[0]), na_rm=False)
            matrix = _design_correlation(raw, np.arange(raw.shape[1]), np.arange(raw.shape[1]))
        correlations.append(matrix)
        fits.append(
            susie_rss(R=matrix, R_finite=None if panel_sizes is None else panel_sizes[k], **options)
        )
    scores = [float(fit.elbo[-1]) for fit in fits]
    selected = int(np.argmax(scores))
    options["convergence_method"] = "pip"
    inputs = susie_rss(R=correlations[selected], R_finite=None, init_only=True, **options)
    selected_variables = np.arange(len(z))
    if options.get("maf") is not None:
        selected_variables = np.flatnonzero(np.asarray(options["maf"]) > options["maf_thresh"])
    correlations = [
        matrix[np.ix_(selected_variables, selected_variables)] for matrix in correlations
    ]
    if inputs._request.null_index is not None:
        correlations = [np.pad(matrix, ((0, 1), (0, 1))) for matrix in correlations]
    omega = np.zeros(len(panels))
    omega[selected] = 1
    adjusted = z[selected_variables]
    if arguments["z_method"] == "wald":
        adjusted = adjusted * np.sqrt((arguments["n"] - 1) / (adjusted**2 + arguments["n"] - 2))
    if inputs._request.null_index is not None:
        adjusted = np.r_[adjusted, 0.0]
    dynamic = dict(
        kind="multipanel",
        panels=correlations,
        omega=omega,
        panel_sizes=panel_sizes,
        reference_n=np.inf if panel_sizes is None else panel_sizes[selected],
        mismatch=arguments["R_mismatch"],
        control=arguments["rss_control"],
        z=adjusted,
    )
    request = replace(
        inputs._request,
        dynamic=dynamic,
        params={**inputs._request.params, "init_only": arguments["init_only"]},
    )
    if arguments["init_only"]:
        return replace(inputs, _request=request)
    result = _fit(request)
    result.params.update(
        single_panel_elbo=scores, initial_panel=selected, omega=result.convergence.get("omega")
    )
    return result
