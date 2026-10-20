"""Input identities belong to owned snapshots, never mutable caller storage."""

import numpy as np
import pytest

from prusie import (
    RefinementConfig,
    continue_fit,
    evaluate_objective,
    refine_fit,
    susie_rss,
    susie_suff_stat,
)


def inputs():
    return dict(
        z=np.array([6.0, 3.0, 0.25]),
        R=np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.25], [0.0, 0.25, 1.0]]),
        n=200,
        L=2,
        check_prior=False,
        variant_ids=np.array(["a", "b", "c"]),
        variant_metadata={
            "effect_allele": np.array(["A", "C", "G"]),
            "other_allele": np.array(["G", "T", "A"]),
        },
    )


def assert_same_fit(left, right):
    for name in (
        "alpha",
        "mu",
        "mu2",
        "lbf_variable",
        "lbf",
        "V",
        "pip",
        "elbo",
        "KL",
        "XtXr",
    ):
        assert getattr(left, name).tobytes() == getattr(right, name).tobytes(), name
    assert left.sigma2 == right.sigma2
    assert left.niter == right.niter and left.converged == right.converged
    assert [a.tolist() for a in left.sets["cs"]] == [a.tolist() for a in right.sets["cs"]]
    np.testing.assert_array_equal(left.sets["cs_index"], right.sets["cs_index"])


@pytest.mark.parametrize("interface", ["rss", "suff_stat"])
def test_unretained_fit_does_not_hash_or_snapshot(monkeypatch, interface):
    import prusie.api as api
    import prusie.refinement as module

    def forbidden(*args, **kwargs):
        raise AssertionError("Ordinary fits must not hash or retain their matrix")

    monkeypatch.setattr(module, "data_signature", forbidden)
    monkeypatch.setattr(api, "make_prepared", forbidden)
    values = inputs()
    original = values["R"].copy()
    if interface == "rss":
        fit = susie_rss(**values)
    else:
        fit = susie_suff_stat(
            values["R"] * 199,
            values["z"] * np.sqrt(199),
            199,
            200,
            L=2,
            estimate_residual_variance=False,
        )
    assert fit.data_signature is None and fit._prepared is None
    assert fit.as_dict()["data_signature"] is None
    assert values["R"].flags.writeable
    np.testing.assert_array_equal(values["R"], original)
    with pytest.raises(ValueError, match="retained data"):
        continue_fit(fit)


@pytest.mark.parametrize(
    "options",
    [
        {"retain_data": True},
        {"track_fit": "compact"},
        {"track_fit": "dense"},
        {"refine": RefinementConfig(candidate_budget=0)},
    ],
)
def test_owned_snapshot_is_hashed_once_and_reused(monkeypatch, options):
    import prusie.refinement as module

    original = module.data_signature
    calls = []

    def counted(matrix, response, *args):
        assert not matrix.flags.writeable and not response.flags.writeable
        calls.append(matrix)
        return original(matrix, response, *args)

    monkeypatch.setattr(module, "data_signature", counted)
    values = inputs()
    fit = susie_rss(**values, **options)
    data = fit._prepared
    assert len(calls) == 1 and calls[0] is data.matrix
    assert not np.shares_memory(data.matrix, values["R"])
    for name in ("matrix", "response", "cs_matrix", "diagonal", "scales", "means", "variant_ids"):
        assert not getattr(data, name).flags.writeable
    assert all(not allele.flags.writeable for allele in data.alleles.values())
    continued = continue_fit(fit)
    refined = refine_fit(fit, RefinementConfig(candidate_budget=1))
    evaluate_objective(fit)
    assert data is continued._prepared is refined._prepared
    assert fit.data_signature == data.signature == continued.data_signature
    assert len(calls) == 1


def test_caller_mutation_preserves_snapshot_labels_and_continuation():
    values = inputs()
    fit = susie_rss(**values, retain_data=True)
    signature = fit.data_signature
    objective = evaluate_objective(fit)
    continued = continue_fit(fit)
    retained_matrix = fit._prepared.matrix.copy()
    values["R"][0, 1] = values["R"][1, 0] = 0.25
    values["z"][0] = -100.0
    values["variant_ids"][0] = "x"
    values["variant_metadata"]["effect_allele"][0] = "G"
    assert fit.data_signature == fit._prepared.signature == signature
    assert fit.variant_ids.tolist() == fit._prepared.variant_ids.tolist() == ["a", "b", "c"]
    assert fit.variant_metadata["effect_allele"].tolist() == ["A", "C", "G"]
    assert fit._prepared.alleles["effect_allele"].tolist() == ["A", "C", "G"]
    np.testing.assert_array_equal(fit._prepared.matrix, retained_matrix)
    assert evaluate_objective(fit)["elbo"] == objective["elbo"]
    assert_same_fit(continue_fit(fit), continued)


def test_initialization_callback_fits_the_hashed_snapshot(monkeypatch):
    import prusie.refinement as module

    values = inputs()
    baseline = susie_rss(**values, retain_data=True)
    expected = susie_rss(**values, s_init=baseline)
    original_matrix = values["R"].copy()
    original = module.data_signature
    calls = []

    def counted(*args):
        calls.append(args[0])
        return original(*args)

    class MutatingInitialization:
        def __getattr__(self, name):
            if name == "alpha":
                values["R"][0, 1] = values["R"][1, 0] = 0.0
            return getattr(baseline, name)

    monkeypatch.setattr(module, "data_signature", counted)
    actual = susie_rss(**values, s_init=MutatingInitialization())
    assert values["R"][0, 1] == 0.0
    assert len(calls) == 1 and calls[0] is actual._prepared.matrix
    assert actual._prepared is not baseline._prepared
    assert_same_fit(actual, expected)
    np.testing.assert_array_equal(actual._prepared.cs_matrix, original_matrix)
    assert actual.data_signature == baseline.data_signature


@pytest.mark.parametrize("invalid", [None, "", "not-a-sha256", "x" * 64, "A" * 64, 42])
@pytest.mark.parametrize("side", ["posterior", "snapshot", "both"])
def test_missing_or_malformed_signatures_fail_closed(invalid, side):
    values = inputs()
    fit = susie_rss(**values, retain_data=True)
    if side in ("posterior", "both"):
        fit.data_signature = invalid
    if side in ("snapshot", "both"):
        # Explicit corruption covers the forbidden None == None shortcut.
        object.__setattr__(fit._prepared, "signature", invalid)
    with pytest.raises(ValueError, match="signature"):
        continue_fit(fit)
    with pytest.raises(ValueError, match="identity"):
        evaluate_objective(fit)
    assert refine_fit(fit).refinement_stop_reason == "invalid_baseline_state"
    if side != "snapshot":
        initialized = susie_rss(**values, s_init=fit)
        assert initialized.data_signature == initialized._prepared.signature
        assert isinstance(initialized.data_signature, str) and len(initialized.data_signature) == 64


def test_unretained_source_can_initialize_a_new_fit_but_not_continue():
    values = inputs()
    fit = susie_rss(**values)
    initialized = susie_rss(**values, s_init=fit)
    assert np.isfinite(initialized.pip).all()
    with pytest.raises(ValueError, match="retained data"):
        continue_fit(fit)


@pytest.mark.parametrize("change", ["matrix", "response", "allele"])
def test_new_inputs_allow_new_content_but_reject_incompatible_alleles(change):
    values = inputs()
    fit = susie_rss(**values, retain_data=True)
    if change == "matrix":
        values["R"][0, 1] = np.nextafter(values["R"][0, 1], 1.0)
    elif change == "response":
        values["z"][0] += 0.125
    else:
        values["variant_metadata"]["effect_allele"][0] = "G"
    if change == "allele":
        with pytest.raises(ValueError, match="allele"):
            susie_rss(**values, s_init=fit)
    else:
        initialized = susie_rss(**values, s_init=fit)
        assert np.isfinite(initialized.pip).all()
        assert initialized.data_signature != fit.data_signature
