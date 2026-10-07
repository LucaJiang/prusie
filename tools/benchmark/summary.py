"""Statistics for completed, settings-matched full fine-mapping calls."""
from __future__ import annotations
import math
import numpy as np

FIELDS = ('pip', 'alpha', 'mu', 'mu2', 'lbf_variable', 'lbf', 'V', 'sigma2', 'elbo', 'KL')


def distribution(values):
    a = np.asarray(values, dtype=float)
    if not a.size:
        return None
    if not np.isfinite(a).all():
        raise ValueError('Summary values must be finite')
    q1, med, q3 = np.quantile(a, [.25, .5, .75])
    return dict(count=len(a), min=float(a.min()), q1=float(q1), median=float(med),
                q3=float(q3), max=float(a.max()), iqr=float(q3-q1),
                iqr_over_median=float((q3-q1)/med) if med else None)


def classify(case_id, panel, metadata):
    if panel == 'real':
        trait = metadata['trait']
        if trait not in ('gwas', 'eqtl') or case_id.rsplit('/', 1)[-1] != trait:
            raise ValueError('Case ID and independently supplied trait metadata disagree')
        return {'gwas': 'GWAS', 'eqtl': 'eQTL'}[trait]
    return 'Synthetic regression'


def paired(row):
    """Separate timing inclusion from numerical acceptance, convergence and CS."""
    row['timing_eligible'] = False
    row['memory_eligible'] = False
    row['speedup'] = row['memory_ratio'] = row['memory_reduction'] = None
    if row.get('stage') != 'C' or not row.get('settings_match') or row.get('measurement_invalid'):
        return row
    backends = row.get('backends', {})
    if any(backends.get(k, {}).get('status') != 'ok' for k in ('prusie', 'susieR')):
        return row
    for backend in backends.values():
        times = backend.get('repeat_seconds', [])
        if len(times) < 7 or any(t is None or not math.isfinite(t) or t <= 0 for t in times):
            return row
        backend['runtime_seconds'] = distribution(times)
    p, r = (backends[k] for k in ('prusie', 'susieR'))
    row['timing_eligible'] = True
    row['speedup'] = r['runtime_seconds']['median'] / p['runtime_seconds']['median']
    peaks = [b.get('peak_memory_mib') for b in (p, r)]
    if all(b.get('memory_status') == 'ok' for b in (p, r)) and all(
            x is not None and math.isfinite(x) and x > 0 for x in peaks):
        row['memory_eligible'] = True
        row['memory_ratio'] = peaks[1] / peaks[0]
        row['memory_reduction'] = 1 - peaks[0] / peaks[1]
    return row


def array_error(left, right):
    a, b = np.asarray(left, float), np.asarray(right, float)
    record = dict(shape_match=a.shape == b.shape, prusie_shape=list(a.shape), susieR_shape=list(b.shape),
                  max_abs_error=None, index_0based=None)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return record
    delta = abs(a-b)
    index = np.unravel_index(int(np.argmax(delta)), delta.shape) if delta.size else ()
    record.update(max_abs_error=float(delta[index]) if delta.size else 0.,
                  index_0based=list(map(int, index)),
                  prusie_at_max=float(a[index]) if a.size else None,
                  susieR_at_max=float(b[index]) if b.size else None,
                  max_abs_susieR=float(np.max(abs(b), initial=0)))
    return record


def summarize(rows):
    groups = {}
    rows = [r for r in rows if r['stage'] == 'C']
    for group in dict.fromkeys(r['group'] for r in rows):
        panel = [r for r in rows if r['group'] == group]
        times = [r for r in panel if r['timing_eligible']]
        memory = [r for r in panel if r['memory_eligible']]
        valid = [r for r in panel if r.get('numerical', {}).get('outputs_valid')]
        ratios = [r['speedup'] for r in times]
        errors = {}
        for field in FIELDS:
            candidates = [r for r in valid if r['numerical']['errors'][field]['max_abs_error'] is not None]
            if candidates:
                worst = max(candidates, key=lambda r:r['numerical']['errors'][field]['max_abs_error'])
                errors[field] = dict(worst['numerical']['errors'][field], case_id=worst['case_id'],
                                     source=worst['numerical']['sources'], valid_case_count=len(candidates))
        groups[group] = dict(analyses=len(panel), valid_numerical_analyses=len(valid),
            variant_count_range=[min(r['n_variants'] for r in panel), max(r['n_variants'] for r in panel)],
            timing_analyses=len(times), memory_analyses=len(memory),
            runtime_speedup_geometric_mean=math.exp(math.fsum(math.log(x) for x in ratios)/len(ratios)) if ratios else None,
            runtime_speedup_distribution=distribution(ratios),
            memory_ratio_distribution=distribution([r['memory_ratio'] for r in memory]),
            memory_reduction_distribution=distribution([r['memory_reduction'] for r in memory]),
            runtime_seconds={k:distribution([r['backends'][k]['runtime_seconds']['median'] for r in times]) for k in ('prusie','susieR')},
            peak_memory_mib={k:distribution([r['backends'][k]['peak_memory_mib'] for r in memory]) for k in ('prusie','susieR')},
            repeat_iqr_over_median={k:distribution([r['backends'][k]['runtime_seconds']['iqr_over_median'] for r in times]) for k in ('prusie','susieR')},
            errors=errors,
            iteration_limit_cases=[r['case_id'] for r in panel if all(r.get('backends',{}).get(k,{}).get('niter') == r['max_iter'] and not r['backends'][k].get('converged') for k in ('prusie','susieR'))],
            no_cs_analyses={k:sum(r.get('backends',{}).get(k,{}).get('cs_count') == 0 for r in panel) for k in ('prusie','susieR')})
    return groups
