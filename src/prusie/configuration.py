"""Post-hoc outcome configurations for supported single-outcome SuSiE fits.

The activation and pairwise ABF models are ported from susieR 0.16.6
(BSD-3-Clause). They summarize existing fits without refitting.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .result import SusieResult


def susie_post_outcome_configuration(
    input: SusieResult | Sequence[SusieResult] | Mapping[str, SusieResult],
    *,
    by: str = "fit",
    outcome_names: Sequence[str] | None = None,
    method: str = "susiex",
    prob_thresh: float = 0.8,
    single_effect_lfsr_cutoff: float = 0.05,
    cs_only: bool = True,
    p1: float = 1e-4,
    p2: float = 1e-4,
    p12: float = 5e-6,
) -> dict[str, Any] | None:
    """Summarize shared-outcome activation or pairwise colocalization evidence.

    Parameters
    ----------
    input : SusieResult, sequence, or mapping of SusieResult
        Existing single-outcome fits. Variant IDs align their effect evidence.
    by : {'fit', 'outcome'}, default 'fit'
        Enumerate all credible-set tuples, or match component IDs across fits.
    outcome_names : sequence of str, optional
        Names for the fits; mapping keys are used otherwise.
    method : {'susiex', 'coloc_pairwise'}, default 'susiex'
        Shared-variant activation configurations or pairwise H0-H4 probabilities.
    prob_thresh : float, default 0.8
        Minimum marginal activation probability for the returned active flag.
    single_effect_lfsr_cutoff : float, default 0.05
        Multi-output screening threshold retained for interface compatibility;
        single-outcome fits do not require this screen.
    cs_only : bool, default True
        Remove fit views without credible sets before comparing outcomes.
    p1, p2, p12 : float
        Variant-level priors for pairwise trait-specific and shared association.

    Returns
    -------
    dict or None
        ``susiex`` contains configuration probability and summary records for
        each component tuple. ``coloc_pairwise`` contains H0-H4 records. Component
        indices use Python's zero-based convention. Fewer than two usable fits
        returns ``None``. This function supports prusie's single-outcome models.
    """
    from scipy.special import logsumexp

    if by not in ("fit", "outcome") or method not in ("susiex", "coloc_pairwise"):
        raise ValueError("Invalid by or method")
    for name, value in (
        ("prob_thresh", prob_thresh),
        ("single_effect_lfsr_cutoff", single_effect_lfsr_cutoff),
    ):
        if not np.isscalar(value) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    for name, value in (("p1", p1), ("p2", p2), ("p12", p12)):
        if not np.isscalar(value) or not np.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"{name} must be in (0, 1)")
    if not isinstance(cs_only, (bool, np.bool_)):
        raise ValueError("cs_only must be boolean")
    if isinstance(input, SusieResult):
        fits, names = [input], ["trait_1"]
    elif isinstance(input, Mapping):
        fits, names = list(input.values()), list(input)
    else:
        fits = list(input)
        names = [f"trait_{i + 1}" for i in range(len(fits))]
    if not fits or not all(isinstance(fit, SusieResult) for fit in fits):
        raise ValueError("input must contain SuSiE fits")
    if outcome_names is not None:
        names = list(outcome_names)
    if len(names) != len(fits) or not all(isinstance(name, str) and name for name in names):
        raise ValueError("outcome_names must contain one nonempty name per fit")
    if len(set(names)) != len(names):
        raise ValueError("Outcome names must be unique")
    if cs_only:
        retained = [k for k, fit in enumerate(fits) if fit.sets and fit.sets.get("cs")]
        if len(retained) < len(fits):
            warnings.warn("Skipping fits without credible sets", RuntimeWarning, stacklevel=2)
        fits, names = [fits[k] for k in retained], [names[k] for k in retained]
    if len(fits) < 2:
        return None
    components = [
        [int(k) for k in fit.sets.get("cs_index", [])] if fit.sets else [] for fit in fits
    ]
    keys = [list(map(str, fit.variant_ids)) for fit in fits]
    maps = [{key: j for j, key in enumerate(variant_keys)} for variant_keys in keys]
    if any(len(index) != len(variant_keys) for index, variant_keys in zip(maps, keys, strict=True)):
        raise ValueError("Variant IDs must be unique within each fit")
    if method == "susiex":
        if len(fits) > 20:
            raise ValueError("SuSiEx configuration enumeration supports at most 20 outcomes")
        if by == "outcome":
            tuples = [
                tuple([component] * len(fits))
                for component in sorted(set.intersection(*map(set, components)))
            ]
        else:
            # R expand.grid varies its first argument fastest.
            tuples = [tuple(reversed(row)) for row in itertools.product(*reversed(components))]
        configs = np.array(
            [
                [int((index >> bit) & 1) for bit in range(len(fits))]
                for index in range(2 ** len(fits))
            ]
        )
        union_size = len(set.union(*map(set, keys)))
        summaries: dict[str, Any] = {}
        for number, components_tuple in enumerate(tuples):
            if any(
                np.all(fit.alpha[component] == 0)
                for fit, component in zip(fits, components_tuple, strict=True)
            ):
                continue
            logbf = np.zeros(len(configs))
            for index, config in enumerate(configs):
                active = np.flatnonzero(config)
                if not len(active):
                    continue
                common = set.intersection(*(set(keys[k]) for k in active))
                ordered = [key for key in keys[active[0]] if key in common]
                if not ordered:
                    logbf[index] = -np.inf
                    continue
                combined = np.full(len(ordered), -np.log(union_size))
                for k in active:
                    combined += fits[k].lbf_variable[
                        components_tuple[k], [maps[k][key] for key in ordered]
                    ]
                logbf[index] = logsumexp(combined)
            probabilities = np.exp(logbf - logsumexp(logbf))
            marginal = configs.T @ probabilities
            summaries[f"config_{number + 1}"] = {
                "config_probability": [
                    {
                        **dict(zip(names, map(int, row), strict=True)),
                        "config_prob": float(probability),
                    }
                    for row, probability in zip(configs, probabilities, strict=True)
                ],
                "config_summary": [
                    {
                        "outcome": name,
                        "cs_indices": f"L{components_tuple[k] + 1}",
                        "logBF_outcome": float(logbf[1 << k]),
                        "posthoc_prob": float(marginal[k]),
                        "active": bool(marginal[k] >= prob_thresh),
                    }
                    for k, name in enumerate(names)
                ],
                "component_indices": dict(zip(names, components_tuple, strict=True)),
            }
        return {"susiex": summaries}
    records = []
    for a, b in itertools.combinations(range(len(fits)), 2):
        common = [key for key in keys[a] if key in maps[b]]
        if not common:
            raise ValueError("Pairwise fits have no overlapping variant IDs")
        ia, ib = [maps[a][key] for key in common], [maps[b][key] for key in common]
        for first, second in itertools.product(components[a], components[b]):
            one, two = fits[a].lbf_variable[first, ia], fits[b].lbf_variable[second, ib]
            sums, shared = logsumexp(one) + logsumexp(two), logsumexp(one + two)
            log_distinct = (
                sums + np.log(-np.expm1(min(0.0, shared - sums))) if shared < sums else -np.inf
            )
            hypotheses = np.array(
                [
                    0,
                    np.log(p1) + logsumexp(one),
                    np.log(p2) + logsumexp(two),
                    np.log(p1) + np.log(p2) + log_distinct,
                    np.log(p12) + shared,
                ]
            )
            posterior = np.exp(hypotheses - logsumexp(hypotheses))
            records.append(
                dict(
                    trait1=names[a],
                    trait2=names[b],
                    l1=first,
                    l2=second,
                    hit1=common[int(np.argmax(one))],
                    hit2=common[int(np.argmax(two))],
                    **{f"PP.H{k}": float(value) for k, value in enumerate(posterior)},
                )
            )
    return {"coloc_pairwise": records}
