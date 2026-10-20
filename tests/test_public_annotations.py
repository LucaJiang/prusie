"""Public fitter annotations resolve without custom type-hint namespaces."""

from typing import get_type_hints

import pytest

import prusie


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("susie", prusie.SusieResult | prusie.SusieInputs),
        ("susie_suff_stat", prusie.SusieResult | prusie.SusieInputs),
        ("susie_ss", prusie.SusieResult | prusie.SusieInputs),
        ("susie_rss", prusie.SusieResult | prusie.SusieInputs),
        ("susie_rss_lambda", prusie.SusieResult | prusie.SusieInputs),
        ("susie_trendfilter", prusie.SusieResult | prusie.SusieInputs),
        ("susie_auto", prusie.SusieResult),
        ("susie_ser", prusie.SusieResult),
        ("susie_workhorse", prusie.SusieResult),
    ],
)
def test_exported_fitter_return_annotations_resolve(name, expected):
    annotations = get_type_hints(getattr(prusie, name))
    assert annotations["return"] == expected
