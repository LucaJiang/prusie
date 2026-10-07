"""Installed-resource teaching example and independent reference checks."""
from importlib.resources import files
import io,json
import numpy as np
import prusie


def test_resources_are_independent_owned_arrays(monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path)
    a=prusie.load_example();b=prusie.load_example()
    assert a['inputs']['R'].shape==(500,500)
    assert a['metadata']['synthetic'] and a['metadata']['genome_build'] is None
    assert a['metadata']['seed']==20261007
    a['inputs']['z'][0]=1e99
    assert b['inputs']['z'][0]!=1e99


def test_packaged_reference_and_probability_identities():
    example=prusie.load_example()
    fit=prusie.susie_rss(**example['inputs'],**example['parameters'])
    folder=files('prusie').joinpath('data/teaching')
    with np.load(io.BytesIO(folder.joinpath('reference.npz').read_bytes()),allow_pickle=False) as ref:
        np.testing.assert_allclose(fit.pip,ref['pip'],atol=1e-5,rtol=0)
    meta=json.loads(folder.joinpath('reference.json').read_text())
    assert meta['susieR_version']=='0.16.6'
    assert fit.converged and fit.niter<=100
    np.testing.assert_allclose(fit.alpha.sum(axis=1),1.,atol=1e-10,rtol=0)
    for k,c in enumerate(fit.sets['cs_index']):
        members=fit.sets['cs'][k]
        np.testing.assert_allclose(fit.sets['coverage'][k],fit.alpha[c,members].sum(),atol=1e-10,rtol=0)
        reference=next(x for x in meta['cs'] if x['component_id_0based']==int(c))
        assert sorted(members)==sorted(reference['members_0based'])
