"""Installed native RSS fits preserve variant identity in alternative plans."""
import copy

import numpy as np
import pytest

from prusie import susie_rss
from prusie.comparison import run_sensitivity


def inputs(order=None):
    data = dict(z=np.array([3., 2., 1.]),
                R=np.array([[1., .2, .1], [.2, 1., .15], [.1, .15, 1.]]),
                n=200, variant_ids=['rsA', 'rsB', 'rsC'])
    if order is not None:
        data.update(z=data['z'][order], R=data['R'][np.ix_(order, order)],
                    variant_ids=[data['variant_ids'][i] for i in order])
    return data


def options(prior=None, null=0.):
    return dict(L=1, prior_weights=prior, null_weight=null,
                estimate_prior_variance=False, scaled_prior_variance=.1, tol=1e-8)


@pytest.mark.parametrize('provided', [False, True])
@pytest.mark.parametrize('null', [0., .2])
def test_nonuniform_permutation_preserves_prior_and_posterior(provided, null):
    data, settings = inputs(), options([8., 1.5, .5], null)
    baseline = susie_rss(**data, **settings)
    prior_before, alpha_before = baseline.params['prior_weights'][:], baseline.alpha.copy()
    out = run_sensitivity(data, [dict(name='permuted', inputs=inputs([2, 0, 1]))],
                          baseline=baseline if provided else None,
                          baseline_settings=None if provided else settings)
    permuted = out['fits']['permuted']
    columns = [2, 0, 1] + ([3] if null else [])
    np.testing.assert_allclose(permuted.params['prior_weights'],
                               np.asarray(baseline.params['prior_weights'])[columns], atol=1e-15, rtol=0)
    np.testing.assert_allclose(permuted.pip[[1, 2, 0]], baseline.pip, atol=1e-12, rtol=0)
    np.testing.assert_allclose(permuted.alpha, baseline.alpha[:, columns], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(baseline.alpha, alpha_before)
    assert baseline.params['prior_weights'] == prior_before
    assert data['variant_ids'] == ['rsA', 'rsB', 'rsC']
    resolution = out['records'][-1]['prior_resolution']
    assert resolution['source_indices'] == [2, 0, 1]
    assert resolution['policy'] == 'reindex_existing_weights'


@pytest.mark.parametrize('prior', [None, [1., 1., 1.], [.8, .15, .05], [8., 1.5, .5]])
@pytest.mark.parametrize('null', [0., .2])
@pytest.mark.parametrize('provided', [False, True])
def test_subset_conditions_original_augmented_prior_once(prior, null, provided):
    settings = options(prior, null)
    baseline = susie_rss(**inputs(), **settings)
    out = run_sensitivity(inputs(), [dict(name='subset', inputs=inputs([2, 0]))],
                          baseline=baseline if provided else None,
                          baseline_settings=None if provided else settings)
    expected = np.asarray(baseline.params['prior_weights'])[[2, 0] + ([3] if null else [])]
    expected /= expected.sum()
    actual = out['fits']['subset'].params['prior_weights']
    np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0)
    resolution = out['records'][-1]['prior_resolution']
    np.testing.assert_allclose(resolution['normalized_weights'], expected, atol=1e-15, rtol=0)
    assert resolution['policy'] == 'condition_on_retained_variants_and_null'
    assert resolution['null_weight'] == null


def test_explicit_baseline_prior_with_supplied_fit_and_prior_in_inputs():
    settings = options([.8, .15, .05])
    baseline = susie_rss(**inputs(), **settings)
    for baseline_data, baseline_settings in [
        (inputs(), settings),
        (dict(inputs(), prior_weights=settings['prior_weights']),
         {k: v for k, v in settings.items() if k != 'prior_weights'}),
    ]:
        out = run_sensitivity(baseline_data, [dict(name='reorder', inputs=inputs([2, 0, 1]))],
                              baseline=baseline, baseline_settings=baseline_settings)
        np.testing.assert_allclose(out['fits']['reorder'].params['prior_weights'], [.05, .8, .15], atol=1e-15)


def extended():
    return dict(z=np.array([3., 2., 1., .5]), R=np.eye(4), n=200,
                variant_ids=['rsA', 'rsB', 'rsC', 'rsD'])


@pytest.mark.parametrize('provided', [False, True])
def test_new_ids_under_nonuniform_prior_require_explicit_complete_weights(provided):
    calls = []
    baseline = susie_rss(**inputs(), **options([.8, .15, .05])) if provided else None
    with pytest.raises(ValueError, match='new.*explicit.*prior_weights'):
        run_sensitivity(inputs(), [dict(name='valid', changes={'L': 2}),
                                   dict(name='new', inputs=extended())],
                        baseline=baseline, baseline_settings=options([.8, .15, .05]),
                        fit_function=lambda **kw: calls.append(kw))
    assert calls == []


@pytest.mark.parametrize('prior', [[.2, .3, .4, .1], None])
def test_explicit_alternative_prior_is_part_of_input_factor_and_cache(prior):
    cache = {}
    plan = [dict(name='new', inputs=dict(extended(), prior_weights=prior))]
    settings = options([.8, .15, .05], .2)
    first = run_sensitivity(inputs(), plan, baseline_settings=settings, cache=cache)
    again = run_sensitivity(inputs(), plan, baseline_settings=settings, cache=cache)
    direct = susie_rss(**extended(), **dict(settings, prior_weights=prior))
    np.testing.assert_allclose(first['fits']['new'].pip, direct.pip, atol=1e-12, rtol=0)
    assert again['records'][-1]['cache_hit']
    policy = first['records'][-1]['prior_resolution']['policy']
    assert policy == ('explicit_uniform' if prior is None else 'explicit_alternative_weights')
    changed = run_sensitivity(inputs(), [dict(name='new', inputs=dict(extended(), prior_weights=[1., 0., 0., 0.]))],
                              baseline_settings=settings, cache=cache)
    assert changed['records'][-1]['cache_key'] != first['records'][-1]['cache_key']
    assert not changed['records'][-1]['cache_hit']


@pytest.mark.parametrize('prior', [None, [2., 2., 2.]])
@pytest.mark.parametrize('provided', [False, True])
def test_uniform_new_ids_extend_equal_raw_weights(prior, provided):
    settings = options(prior, .2)
    baseline = susie_rss(**inputs(), **settings) if provided else None
    out = run_sensitivity(inputs(), [dict(name='new', inputs=extended())],
                          baseline=baseline, baseline_settings=None if provided else settings)
    expected = np.r_[np.repeat((1/3 if prior is None else 2.)*.8, 4), .2]
    expected /= expected.sum()
    np.testing.assert_allclose(out['fits']['new'].params['prior_weights'], expected, atol=1e-15, rtol=0)
    assert out['records'][-1]['prior_resolution']['policy'] == 'extend_equal_existing_weights'


@pytest.mark.parametrize('prior', [[0., 0.], [-1., 1.], [np.nan, 1.], [np.inf, 1.],
                                 [1e308, 1e308], [1.], [[.5, .5]]])
def test_invalid_alternative_weights_reject_entire_plan_before_any_fit(prior):
    calls = []
    with pytest.raises(ValueError, match='prior_weights'):
        run_sensitivity(inputs(), [dict(name='valid', changes={'L': 2}),
                                   dict(name='bad', inputs=dict(inputs([0, 1]), prior_weights=prior))],
                        fit_function=lambda **kw: calls.append(kw))
    assert calls == []


@pytest.mark.parametrize('null', [0., .2])
def test_zero_support_subset_is_rejected_even_with_null(null):
    calls = []
    with pytest.raises(ValueError, match='prior_weights'):
        run_sensitivity(inputs(), [dict(name='zero', inputs=inputs([1, 2]))],
                        baseline_settings=options([1., 0., 0.], null),
                        fit_function=lambda **kw: calls.append(kw))
    assert calls == []


@pytest.mark.parametrize('change', [dict(variant_ids=['rsA', 'rsA']), dict(variant_ids=['', 'rsB']),
    dict(variant_ids=[['rsA', 'rsB']]), dict(z=[1.]), dict(R=np.eye(3)),
    dict(variant_metadata={'position': [1]}), dict(z=[np.nan, 1.])])
def test_input_dimensions_ids_and_metadata_validated_before_any_fit(change):
    calls = []
    with pytest.raises(ValueError):
        run_sensitivity(inputs(), [dict(name='valid', changes={'L': 2}),
                                   dict(name='bad', inputs=dict(inputs([0, 1]), **change))],
                        fit_function=lambda **kw: calls.append(kw))
    assert calls == []


@pytest.mark.parametrize('null', [-.1, 1., np.nan, np.inf])
def test_invalid_null_weight_is_preflighted(null):
    calls = []
    with pytest.raises(ValueError, match='null_weight'):
        run_sensitivity(inputs(), [], baseline_settings=options(None, null),
                        fit_function=lambda **kw: calls.append(kw))
    assert calls == []


@pytest.mark.parametrize('metadata_in_settings', [False, True])
def test_inherited_metadata_is_reindexed_without_mutation(metadata_in_settings):
    data = inputs()
    metadata = {'effect_allele': np.array(['A', 'C', 'G']),
                'other_allele': np.array(['G', 'T', 'A']), 'position': np.array([10, 20, 30])}
    settings = options([.8, .15, .05])
    (settings if metadata_in_settings else data)['variant_metadata'] = copy.deepcopy(metadata)
    out = run_sensitivity(data, [dict(name='subset', inputs=inputs([2, 0]))], baseline_settings=settings)
    fit = out['fits']['subset']
    for key, values in metadata.items():
        np.testing.assert_array_equal(fit.variant_metadata[key], values[[2, 0]])
        np.testing.assert_array_equal((settings if metadata_in_settings else data)['variant_metadata'][key], values)
    assert out['records'][-1]['input_resolution']['metadata_policy'] == 'reindex_existing_columns'
    with pytest.raises(ValueError, match='new.*variant_metadata'):
        run_sensitivity(data, [dict(name='new', inputs=dict(extended(), prior_weights=None))], baseline_settings=settings)


def test_effect_input_scalar_shat_remains_scalar_and_metadata_can_be_explicit():
    data = dict(bhat=[.3, .2, .1], shat=.1, R=np.eye(3), n=200, variant_ids=['rsA', 'rsB', 'rsC'])
    alt = dict(bhat=[.1, .3], shat=.1, R=np.eye(2), n=200, variant_ids=['rsC', 'rsA'],
               variant_metadata={'position': [30, 10]})
    out = run_sensitivity(data, [dict(name='subset', inputs=alt)], baseline_settings=options([.8, .15, .05]))
    np.testing.assert_allclose(out['fits']['subset'].params['prior_weights'], np.array([.05, .8])/.85)
    assert out['fits']['subset'].variant_metadata['position'].tolist() == [30, 10]
