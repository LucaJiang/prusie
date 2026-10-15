"""Deterministic posterior tables; no fitting, eigen decomposition or network I/O."""
from __future__ import annotations

import copy
import math

import numpy as np

from .posterior import credible_sets

_UNSET = object()


def _ids(fit):
    ids = [str(x) for x in fit.variant_ids]
    if len(set(ids)) != len(ids):
        raise ValueError("variant IDs must be unique")
    return ids


def _biological(fit, values):
    values = np.asarray(values)
    return np.delete(values, fit.null_index, axis=-1) if fit.null_index is not None else values


def _correlation(R, p):
    R = np.asarray(R, dtype=float)
    if R.shape != (p, p) or not np.all(np.isfinite(R)):
        raise ValueError("R must be a finite p by p correlation matrix in variant order")
    if not np.allclose(R, R.T, atol=1e-10, rtol=0) or not np.allclose(np.diag(R), 1, atol=1e-8, rtol=0):
        raise ValueError("R must be symmetric with unit diagonal")
    if np.any(np.abs(R) > 1 + 1e-8):
        raise ValueError("R entries must lie in [-1, 1]")
    return R


def _value(v):
    if isinstance(v, np.generic):
        v = v.item()
    return None if isinstance(v, float) and not math.isfinite(v) else v


def _normal_log10p(z):
    """Two-sided normal tail, including large z where erfc underflows."""
    z = abs(float(z))
    if not math.isfinite(z):
        return None
    if z < 35:
        return -math.log10(math.erfc(z / math.sqrt(2)))
    # Mills' expansion; at z >= 35 the first omitted term is < 1e-14.
    inv = 1 / (z * z)
    series = 1 - inv + 3 * inv**2 - 15 * inv**3 + 105 * inv**4 - 945 * inv**5
    logp = math.log(2) - z*z/2 - math.log(z) - math.log(2*math.pi)/2 + math.log(series)
    return -logp / math.log(10)


def _column_summary(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return dict(status="assessed" if len(finite) else "not_assessed", count=len(values),
                missing=int(len(values) - len(finite)), minimum=float(finite.min()) if len(finite) else None,
                median=float(np.median(finite)) if len(finite) else None,
                maximum=float(finite.max()) if len(finite) else None)


def summarize(fit, R=None, *, coverage=_UNSET, min_abs_corr=_UNSET, proxy_r=0.99,
              z=None, pvalues=None, metadata=None, input_info=None):
    """Return variant, component and CS tables without modifying ``fit``.

    ``R`` must be signed correlation in fit variant order. It is used only for
    explicit posterior CS re-extraction, direct proxy edges and CS relations.
    Supplying coverage/purity re-extracts CS from the same posterior, without
    IBSS. None explicitly disables the respective CS option. Unspecified
    thresholds preserve the fit's formal CS. No LD eigen decomposition occurs.

    Component entropy uses biological alpha normalized to its biological mass;
    null mass is separate. Prior odds use current *conditional* component Bayes
    factors, not external independent evidence. The caller may supply per-variant
    annotation in ``metadata`` and study information in ``input_info``.
    """
    ids = _ids(fit)
    p = len(ids)
    if not 0 <= proxy_r <= 1:
        raise ValueError("proxy_r must lie in [0, 1]")
    corr = None if R is None else _correlation(R, p)
    alpha = _biological(fit, fit.alpha)
    if alpha.shape != (len(fit.V), p):
        raise ValueError("fit alpha and variant IDs do not align")
    columns = {k: np.asarray(v).copy() for k, v in fit.variant_metadata.items()}
    if metadata:
        columns.update({k: np.asarray(v).copy() for k, v in metadata.items()})
    if any(v.shape != (p,) for v in columns.values()):
        raise ValueError("metadata columns must each have one value per variant")
    if z is None:
        z = columns.get("z")
    if pvalues is None:
        pvalues = columns.get("pvalue", columns.get("p"))
    if z is not None:
        z = np.asarray(z, dtype=float)
        if z.shape != (p,):
            raise ValueError("z must have one value per variant")
    if pvalues is not None:
        pvalues = np.asarray(pvalues, dtype=float)
        if pvalues.shape != (p,) or np.any((pvalues < 0) | (pvalues > 1)):
            raise ValueError("pvalues must have one value in [0, 1] per variant (NaN allowed)")
    variants = []
    order = sorted(range(p), key=lambda j: (-float(fit.pip[j]), ids[j]))
    ranks = {j: i+1 for i, j in enumerate(order)}
    for j, variant in enumerate(ids):
        row = dict(variant_id=variant, index=j, pip=float(fit.pip[j]), rank=ranks[j],
                   position=_value(columns.get("position", [None]*p)[j]),
                   z=_value(z[j]) if z is not None else None,
                   neg_log10_p=None, pvalue_source="not_assessed",
                   metadata={key: _value(value[j]) for key, value in columns.items()})
        if pvalues is not None and np.isfinite(pvalues[j]):
            row.update(pvalue=float(pvalues[j]), pvalue_source="provided",
                       neg_log10_p=-math.log10(pvalues[j]) if pvalues[j] > 0 else None,
                       pvalue_status="provided_zero_underflow_or_rounding" if pvalues[j] == 0 else "available")
        elif z is not None and np.isfinite(z[j]):
            row.update(neg_log10_p=_normal_log10p(z[j]), pvalue_source="two_sided_normal_approximation")
        variants.append(row)
    sets = fit.sets
    audit = copy.deepcopy(getattr(fit, "cs_audit", []))
    extraction = "original_fit"
    if coverage is not _UNSET or min_abs_corr is not _UNSET:
        cov = fit.params.get("coverage", .95) if coverage is _UNSET else coverage
        purity = fit.params.get("min_abs_corr", .5) if min_abs_corr is _UNSET else min_abs_corr
        if cov is not None and (not np.isfinite(cov) or not 0 < cov <= 1):
            raise ValueError("coverage must lie in (0, 1]")
        if purity is not None and (not np.isfinite(purity) or not 0 <= purity <= 1):
            raise ValueError("min_abs_corr must lie in [0, 1]")
        if corr is None and cov is not None and purity is not None:
            raise ValueError("explicit R is required to re-extract credible sets")
        matrix = corr
        if matrix is not None and fit.null_index is not None:
            matrix = np.insert(np.insert(matrix, fit.null_index, 0, axis=0), fit.null_index, 0, axis=1)
        audit = []
        sets = credible_sets(fit.alpha, fit.V, matrix, coverage=cov, min_abs_corr=purity,
                             null_index=fit.null_index, audit=audit)
        extraction = "posterior_only_no_IBSS"
    if not audit:
        audit = [dict(component_id=i, reasons=["unavailable"], stages=[],
                      explanation="The fitting pipeline did not record a CS audit.") for i in range(len(alpha))]
    components, cs_rows, proxies, relations, prior_rows = [], [], [], [], []
    formal = {int(i): (members, k) for k, (i, members) in enumerate(zip(sets["cs_index"], sets["cs"], strict=True))}
    weights = np.asarray(fit.params.get("prior_weights", []), dtype=float)
    weights = _biological(fit, weights) if len(weights) == fit.alpha.shape[1] else weights
    prior_available = weights.shape == (p,)
    effective_weights = weights + np.sqrt(np.finfo(float).eps)
    if fit.params.get("hard_exclusion", False):
        effective_weights = np.where(weights == 0, 0., effective_weights)
    log_bf = _biological(fit, fit.lbf_variable)
    for i, row in enumerate(alpha):
        # Lexical ID tie-breaking makes presentation independent of input permutation.
        ranking = sorted(range(p), key=lambda j: (-float(row[j]), ids[j]))
        mass = float(row.sum())
        normalized = row[row > 0] / mass if mass > 0 else np.array([])
        entropy = float(-np.sum(normalized * np.log(normalized))) if mass > 0 else None
        component = dict(component_id=i, active=bool(fit.V[i] > fit.params.get("prior_tol", 1e-9)),
                         V=float(fit.V[i]), top_variant=ids[ranking[0]] if ranking else None,
                         top_alpha=float(row[ranking[0]]) if ranking else None,
                         entropy=entropy, effective_candidates=math.exp(entropy) if entropy is not None else None,
                         biological_mass=mass, null_mass=float(fit.alpha[i, fit.null_index]) if fit.null_index is not None else 0.,
                         ranking=[ids[j] for j in ranking], alpha=[float(row[j]) for j in ranking],
                         cumulative_posterior=np.cumsum(row[ranking]).tolist(),
                         component_log_bf=_value(fit.lbf[i]), cs_retained=i in formal)
        components.append(component)
        if i in formal:
            members, k = formal[i]
            # Returned sets exclude null; map internal indices back to biological order.
            members = [int(j) - int(fit.null_index is not None and j > fit.null_index) for j in members]
            representative = min(members, key=lambda j: (-float(row[j]), ids[j]))
            csrow = dict(component_id=i, members=[ids[j] for j in members], member_indices=members,
                         size=len(members), requested_coverage=sets.get("requested_coverage"),
                         actual_coverage=float(np.sum(row[members])), representative=ids[representative],
                         representative_index=representative, representative_rule="highest alpha; lexical variant ID breaks ties",
                         **{key: float(sets["purity"][key][k]) for key in ("min_abs_corr", "mean_abs_corr", "median_abs_corr")})
            cs_rows.append(csrow)
            if corr is not None:
                membership = set(members)
                for j in range(p):
                    if j not in membership:
                        for m in members:
                            r = float(corr[j, m])
                            if abs(r) >= proxy_r:
                                proxies.append(dict(component_id=i, variant_id=ids[j], cs_variant_id=ids[m],
                                                    signed_r=r, r_squared=r*r, abs_r=abs(r)))
        if prior_available:
            support = [j for j in ranking if effective_weights[j] > 0 and row[j] > 0 and np.isfinite(log_bf[i, j])]
            reference = support[0] if support else None
            for j in range(p):
                entry = dict(component_id=i, variant_id=ids[j], prior_weight=float(weights[j]),
                             log_prior=math.log(weights[j]) if weights[j] > 0 else None,
                             effective_prior_weight=float(effective_weights[j]),
                             log_effective_prior=math.log(effective_weights[j]) if effective_weights[j] > 0 else None,
                             conditional_log_bf=_value(log_bf[i, j]), alpha=float(row[j]),
                             reference_variant=ids[reference] if reference is not None else None,
                             log_prior_odds=None, conditional_log_bf_difference=None,
                             log_posterior_odds=None, odds_decomposition_error=None,
                             log_effective_prior_odds=None, raw_odds_decomposition_error=None,
                             zero_prior_policy="hard_exclusion" if fit.params.get("hard_exclusion",False) else "epsilon_added_in_ordinary_SER",
                             status="zero_prior" if weights[j] == 0 else "not_evaluable")
                if reference is not None and effective_weights[j] > 0 and row[j] > 0 and np.isfinite(log_bf[i, j]):
                    lp = math.log(weights[j]) - math.log(weights[reference]) if weights[j] > 0 and weights[reference] > 0 else None
                    le = math.log(effective_weights[j]) - math.log(effective_weights[reference])
                    lb = float(log_bf[i, j] - log_bf[i, reference])
                    la = math.log(row[j]) - math.log(row[reference])
                    entry.update(log_prior_odds=lp, conditional_log_bf_difference=lb,
                                 log_effective_prior_odds=le, log_posterior_odds=la,
                                 raw_odds_decomposition_error=la-lp-lb if lp is not None else None,
                                 odds_decomposition_error=la-le-lb,
                                 status="trimmed_prior_state" if fit.V[i] == 0 else
                                 ("effective_odds_verified" if abs(la-le-lb) < 1e-8 else "saved_state_odds_discrepancy"))
                prior_rows.append(entry)
    for k, a in enumerate(cs_rows):
        for b in cs_rows[k+1:]:
            common = sorted(set(a["members"]) & set(b["members"]))
            r = float(corr[a["representative_index"], b["representative_index"]]) if corr is not None else None
            relations.append(dict(component_a=a["component_id"], component_b=b["component_id"],
                                  shared_members=common, overlap_count=len(common),
                                  representative_a=a["representative"], representative_b=b["representative"],
                                  signed_r=r, r_squared=r*r if r is not None else None))
    qc = copy.deepcopy(fit.qc)
    for key, aliases in {"N": ("N", "n", "per_variant_n"), "INFO": ("INFO", "info"),
                         "EAF": ("EAF", "eaf")}.items():
        column = next((name for name in aliases if name in columns), None)
        qc[key] = _column_summary(columns[column]) if column is not None else {"status": "not_assessed"}
        qc[key]["source_column"] = column
    qc["study_information"] = copy.deepcopy(input_info or {})
    qc["study_information_status"] = {key: "provided" if key in (input_info or {}) else "not_assessed"
        for key in ("genome_build", "ld_ancestry", "ld_reference_n", "ld_source", "in_sample_ld")}
    qc["external_flags_status"] = "provided" if "qc_flags" in columns else "not_assessed"
    return dict(schema_version="prusie.analysis.v1", variants=variants, components=components,
                credible_sets=cs_rows, cs_audit=audit, cs_extraction=extraction,
                proxies=proxies, proxy_threshold=proxy_r, proxy_rule="direct pairwise edges only; no transitive closure",
                proxy_status="assessed" if corr is not None else "not_assessed",
                cs_relations=relations, cs_count=len(cs_rows), independent_causal_signal_count=None,
                prior_influence=prior_rows, prior_influence_status="assessed" if prior_available else "not_assessed",
                input_qc=qc, convergence=copy.deepcopy(getattr(fit, "convergence", {})),
                trace=copy.deepcopy(getattr(fit, "trace", [])), refinement=copy.deepcopy(getattr(fit, "refinement", {})),
                warnings=list(fit.warnings), definitions={
                    "alpha": "conditional selection probability within one component",
                    "pip": "marginal inclusion probability across active components",
                    "effective_candidates": "entropy-based resolution of normalized biological alpha; not a causal count",
                    "cs_count": "number of retained component credible sets, not number of independent causal signals",
                    "prior_influence": "conditional BF depends on other components/settings; ordinary SER uses pi + sqrt(machine epsilon), including nominal zero pi; hard-exclusion proposals retain exact zeros. Raw and effective prior odds are distinct. Trimmed V=0 rows are prior state, not SER evidence."})
