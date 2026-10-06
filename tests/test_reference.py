"""Redistributable analytical cases from pinned official susieR 0.16.6."""
import json
from pathlib import Path
import warnings

import numpy as np
import pytest
import prusie

REFERENCE = json.loads((Path(__file__).parent / 'fixtures/pinned_susieR_gaussian.json').read_text())


@pytest.mark.parametrize('case', REFERENCE['cases'], ids=[c['label'] for c in REFERENCE['cases']])
def test_pinned_gaussian_analytical_case(case):
    function = prusie.susie_rss if case['kind'] == 'rss' else prusie.susie_suff_stat
    with warnings.catch_warnings(record=True):
        fit = function(**case['args'])
    expected = case['expected']
    np.testing.assert_allclose(fit.pip, expected['pip'], atol=1e-5, rtol=0)
    assert np.isfinite(fit.pip).all() and ((fit.pip >= 0) & (fit.pip <= 1)).all()
    assert np.isfinite(fit.alpha).all() and ((fit.alpha >= 0) & (fit.alpha <= 1)).all()
    np.testing.assert_allclose(fit.alpha.sum(axis=1), 1, atol=1e-12, rtol=0)
    assert fit.variant_ids.tolist() == expected['variant_ids']
    assert fit.null_index == expected['null_index']
    assert 1 <= fit.niter <= case['args']['max_iter']
    for members, component, mass in zip(fit.sets['cs'], fit.sets['cs_index'], fit.sets['coverage']):
        np.testing.assert_allclose(fit.alpha[component, members].sum(), mass, atol=1e-12, rtol=0)
