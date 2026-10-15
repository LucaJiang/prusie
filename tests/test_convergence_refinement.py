"""Independent phase, state and search invariants for bounded Gaussian examples."""

import copy

import numpy as np
import pytest

from prusie import (
    ConvergenceConfig,
    RefinementConfig,
    continue_fit,
    evaluate_objective,
    refine_fit,
    susie_rss,
    susie_suff_stat,
)
from prusie.posterior import marginal_pip


def small(**kwargs):
    args = dict(
        z=[5.0, 3.0, 0.1],
        R=[[1.0, 0.6, 0.0], [0.6, 1.0, 0.1], [0.0, 0.1, 1.0]],
        n=200,
        L=2,
        check_prior=False,
        retain_data=True,
    )
    args.update(kwargs)
    return susie_rss(**args)


def exact(a, b):
    for field in ("alpha", "mu", "mu2", "V", "pip", "elbo", "KL", "XtXr", "lbf", "lbf_variable"):
        x, y = getattr(a, field), getattr(b, field)
        assert x.tobytes() == y.tobytes(), field
    assert a.sigma2 == b.sigma2 and a.niter == b.niter and a.converged == b.converged
    assert [x.tolist() for x in a.sets["cs"]] == [x.tolist() for x in b.sets["cs"]]


def test_tracking_does_not_change_fit_and_metrics_are_phase_exact():
    a = small()
    b = small(track_fit="dense")
    c = small(track_fit=True)
    exact(a, b)
    exact(a, c)
    assert not a.trace and len(b.trace) == b.niter
    assert b.trace[0]["max_abs_delta_alpha"] is None
    # Adjacent iterations intentionally leave the final left-hand element unpaired.
    for prev, cur in zip(b.trace, b.trace[1:], strict=False):
        aa, bb = np.array(prev["alpha"]), np.array(cur["alpha"])
        pa = marginal_pip(aa, prev["V"], prior_tol=b.params["prior_tol"])
        pb = marginal_pip(bb, cur["V"], prior_tol=b.params["prior_tol"])
        np.testing.assert_allclose(cur["max_abs_delta_alpha"], np.max(abs(aa - bb)), atol=0, rtol=0)
        np.testing.assert_allclose(
            cur["max_abs_delta_pip"], np.max(abs(pa - pb)), atol=1e-16, rtol=0
        )
        mean_a = (aa * np.array(prev["mu"])).sum(0)
        mean_b = (bb * np.array(cur["mu"])).sum(0)
        np.testing.assert_allclose(
            cur["max_abs_delta_bbar"], max(abs(mean_a - mean_b)), atol=1e-16, rtol=0
        )
        assert cur["delta_elbo"] == cur["elbo"] - prev["elbo"]
    for row in c.trace:
        assert row["lossy"] and "alpha" not in row
        for component in range(len(b.V)):
            kept = sum(v for i, j, v in row["alpha_entries"] if i == component)
            assert abs(kept + row["omitted_mass"][component] - 1) < 1e-12


def test_no_active_null_and_component_swaps():
    a = np.array([[0.8, 0.2], [0.2, 0.8]])
    np.testing.assert_array_equal(marginal_pip(a, [1, 1]), marginal_pip(a[::-1], [1, 1]))
    assert np.max(abs(a - a[::-1])) > 0.5
    np.testing.assert_array_equal(marginal_pip(a, [0, 0]), [0, 0])
    np.testing.assert_allclose(marginal_pip(a, [1, 0], null_index=1), [0.8], rtol=0, atol=0)
    assert np.max(abs(marginal_pip(a, [1, 0]) - marginal_pip(a, [0, 1]))) > 0.5
    f = small(z=[0, 0, 0], null_weight=0.2, track_fit="dense")
    np.testing.assert_array_equal(f.pip, np.zeros(3))
    assert len(f.trace[-1]["pip"]) == 3


def test_criteria_and_cap_do_not_claim_convergence():
    f = small(max_iter=1, track_fit=True)
    assert not f.converged and f.convergence["stop_reason"] == "iteration_limit"
    for kind in ("alpha_pip", "joint"):
        f = small(convergence=ConvergenceConfig(kind, alpha_tol=1e-8, pip_tol=1e-8))
        assert f.convergence["requested_criterion"] == kind
        if f.converged:
            assert f.convergence["last"]["max_abs_delta_alpha"] < 1e-8
            assert f.convergence["last"]["max_abs_delta_pip"] < 1e-8
    with pytest.raises(ValueError):
        small(convergence=ConvergenceConfig(min_iter=1))


def test_warm_start_reconstructs_products_and_is_immutable():
    f = small(max_iter=2, track_fit=True)
    saved = copy.deepcopy(f.as_dict())
    g = continue_fit(f, max_iter=100, tol=1e-7)
    h = small(max_iter=100, tol=1e-7, s_init=f)
    exact(g, h)
    original = g.alpha.copy()
    f.KL[:] = 999.0
    f.XtXr[:] = 1e30
    f.lbf[:] = -1e20
    h = continue_fit(f, max_iter=100, tol=1e-7)
    np.testing.assert_array_equal(h.alpha, original)
    np.testing.assert_array_equal(f.alpha, saved["alpha"])
    assert not f._prepared.matrix.flags.writeable
    with pytest.raises(TypeError):
        f._prepared.options["l"] = 99


@pytest.mark.parametrize(
    "change,match",
    [
        ("ids", "IDs/order"),
        ("scale", "scale"),
        ("null", "null"),
        ("moments", "mu2"),
        ("alpha", "normalized"),
        ("signature", "signature"),
    ],
)
def test_warm_rejects_invalid_state(change, match):
    f = small()
    g = copy.copy(f)
    if change == "ids":
        g.variant_ids = f.variant_ids[::-1]
    elif change == "scale":
        g.X_column_scale_factors = f.X_column_scale_factors * 2
    elif change == "null":
        g.null_index = 0
    elif change == "moments":
        g.mu2 = np.zeros_like(f.mu2)
    elif change == "alpha":
        g.alpha = f.alpha * 2
    else:
        g.data_signature = "wrong"
    with pytest.raises(ValueError, match=match):
        small(s_init=g)


def test_resize_L_and_minimal_explicit_state():
    f = small(L=1)
    g = small(L=3, s_init=f)
    assert g.alpha.shape == (3, 3)
    h = small(L=1, s_init=g)
    assert h.alpha.shape == (1, 3)
    with pytest.raises(ValueError, match=r"metadata"):
        small(s_init=dict(alpha=f.alpha, mu=f.mu, mu2=f.mu2))
    state = dict(
        alpha=f.alpha,
        mu=f.mu,
        mu2=f.mu2,
        variant_ids=f.variant_ids,
        scales=f.X_column_scale_factors,
        null_index=f.null_index,
        data_signature=f.data_signature,
    )
    h = small(L=1, s_init=state)
    assert h.converged


def test_objective_recomputation_and_final_phase():
    f = small(L=1)
    v = evaluate_objective(f)
    assert abs(v["difference_from_logged"]) < 1e-8
    f.KL[:] = 1e40
    f.XtXr[:] = 1e40
    assert evaluate_objective(f)["elbo"] == v["elbo"]
    g = susie_suff_stat(
        [[99.0, 0.0], [0.0, 99.0]],
        [20.0, 0.0],
        99.0,
        100.0,
        L=1,
        max_iter=1,
        estimate_residual_variance=True,
        retain_data=True,
    )
    r = evaluate_objective(g)
    assert g.sigma2 != g.convergence["last"]["sigma2"]
    assert abs(r["difference_from_logged"]) > 1e-6
    t = small(scaled_prior_variance=1e-12, estimate_prior_variance=False, max_iter=1)
    assert t.convergence["finalization"]["trimmed_components"] == [0, 1]
    assert np.all(t.V == 0) and np.isfinite(evaluate_objective(t)["elbo"])


def test_refinement_failure_no_cs_budget_and_original_prior():
    limited = small(max_iter=1)
    r = refine_fit(limited)
    assert r.refinement_status == "unsupported" and not r.converged
    f = small(coverage=None)
    r = refine_fit(f)
    assert r.refinement_status == "no_cs" and r.converged == f.converged
    f = small()
    r = refine_fit(f, RefinementConfig(candidate_budget=0))
    assert r.refinement_status == "budget_exhausted"
    exact(f, r)
    f = susie_rss([5.0], [[1.0]], 100, L=1, retain_data=True)
    r = refine_fit(f)
    assert r.refinement_status == "all_candidates_failed"
    assert r.refinement["history"][0]["reason"] == "no_prior_support"
    f = small(prior_weights=[0.7, 0.0, 0.3], null_weight=0.1)
    before = f.alpha.copy()
    r = refine_fit(f, RefinementConfig(max_iter=1))
    np.testing.assert_array_equal(f.alpha, before)
    np.testing.assert_array_equal(r.params["prior_weights"], f.params["prior_weights"])
    for row in r.refinement["history"]:
        if "A" in row.get("stages", {}):
            assert row["stages"]["A"].get("excluded_max_alpha", 0) == 0
    assert r.refinement_status in ("all_candidates_failed", "no_cs")


def test_objective_rejects_mutated_model_identity():
    f = small()
    f.params["prior_weights"] = [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match=r"prior"):
        evaluate_objective(f)
    r = refine_fit(f)
    assert r.refinement_status == "unsupported"
    f = small()
    g = small(z=[4.0, 2.0, 0.0])
    with pytest.raises(ValueError, match=r"identity"):
        evaluate_objective(f, g._prepared)
    f = small()
    f.sigma2 *= 2
    with pytest.raises(ValueError, match=r"Fixed residual"):
        evaluate_objective(f)


@pytest.mark.parametrize("kind", ["ids", "alleles"])
def test_retained_identity_cannot_be_relabelled(kind):
    f = small(
        variant_ids=["a", "b", "c"],
        variant_metadata={"effect_allele": ["A", "C", "A"], "other_allele": ["G", "T", "C"]},
    )
    if kind == "ids":
        f.variant_ids[:] = f.variant_ids[::-1]
    else:
        f.variant_metadata["effect_allele"][0] = "G"
    with pytest.raises(ValueError, match=r"IDs/order|allele"):
        continue_fit(f)
    assert refine_fit(f).refinement_status == "unsupported"


def test_official_local_optimum_improves_and_preserves_tracking():
    from prusie.datasets import load_convergence_example

    example = load_convergence_example()
    f = susie_suff_stat(**example["inputs"], **example["parameters"], track_fit="compact")
    g = refine_fit(f, RefinementConfig(inner_tol=0.001))
    assert g.refinement_status == "completed" and g.converged
    assert evaluate_objective(g)["elbo"] - evaluate_objective(f)["elbo"] > 17
    assert len(g.trace) == g.niter and len(g.refinement["before"]["trace"]) == f.niter
    for phase, result in [("baseline", f), ("refined", g)]:
        np.testing.assert_allclose(
            result.pip, example["reference"]["latest_" + phase + "_pip"], atol=1e-5, rtol=0
        )
    accepted = [r for r in g.refinement["history"] if r["accepted"]]
    assert accepted and all(r["improvement"] > 0.001 for r in accepted)
    assert g.refinement["after"]["elbo"] == max(r["objective"] for r in g.refinement["history"])
    assert all(r["stages"]["A"]["excluded_max_alpha"] == 0 for r in g.refinement["history"])
    # Stage A commonly has a higher (different-prior) objective; it never wins.
    assert g.refinement["after"]["elbo"] != max(
        r["stages"]["A"]["objective"] for r in g.refinement["history"]
    )
    np.testing.assert_array_equal(g.params["prior_weights"], f.params["prior_weights"])


def test_failed_last_candidate_cannot_replace_accepted_best(monkeypatch):
    import prusie.refinement as module
    from prusie.datasets import load_convergence_example

    e = load_convergence_example()
    f = susie_suff_stat(**e["inputs"], **e["parameters"], retain_data=True)
    original = module._run
    calls = 0

    def bounded_failure(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 6:
            raise RuntimeError("deliberate last-candidate failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_run", bounded_failure)
    g = refine_fit(f, RefinementConfig(inner_tol=0.001, max_rounds=1))
    assert evaluate_objective(g)["elbo"] > evaluate_objective(f)["elbo"] + 17
    assert "deliberate" in g.refinement["history"][-1]["reason"]
    assert g.refinement_status == "budget_exhausted" and g.converged


def test_em_objective_uses_saved_V_not_stale_SER_cache():
    f = susie_suff_stat(
        [[99.0]],
        [20.0],
        99.0,
        100.0,
        L=1,
        max_iter=1,
        estimate_prior_method="EM",
        estimate_residual_variance=False,
        retain_data=True,
    )
    m = float(f.mu[0, 0])
    m2 = float(f.mu2[0, 0])
    v = float(f.V[0])
    s2 = m2 - m * m
    er2 = 99.0 - 40.0 * m + 99.0 * m2
    categorical = -np.log(1 + np.sqrt(np.finfo(float).eps))
    kl = categorical + 0.5 * (m2 / v - 1 + np.log(v / s2))
    expected = -50 * np.log(2 * np.pi) - er2 / 2 - kl
    actual = evaluate_objective(f)
    np.testing.assert_allclose(actual["elbo"], expected, rtol=0, atol=3e-14)
    assert abs(actual["difference_from_logged"]) > 1e-4


def test_nonfinite_candidates_are_rejected_without_losing_baseline(monkeypatch):
    import prusie.refinement as module

    original = module._run

    def invalid_B(*args, **kwargs):
        fit = original(*args, **kwargs)
        if kwargs.get("reset_variances"):
            fit.mu[0, 0] = np.nan
        return fit

    monkeypatch.setattr(module, "_run", invalid_B)
    f = small(L=1)
    r = refine_fit(f)
    assert r.refinement_status == "all_candidates_failed"
    assert any("Nonfinite" in row["reason"] for row in r.refinement["history"])
    exact(f, r)
