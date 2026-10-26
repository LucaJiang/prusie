"""Argument checks that must act before fitting and missing z-score handling."""

import numpy as np
import pytest

import prusie
import prusie.rss

R = np.array(
    [
        [1.0, 0.6, 0.2, 0.0],
        [0.6, 1.0, 0.3, 0.1],
        [0.2, 0.3, 1.0, 0.4],
        [0.0, 0.1, 0.4, 1.0],
    ]
)
R2 = np.array(
    [
        [1.0, 0.5, 0.1, 0.0],
        [0.5, 1.0, 0.2, 0.1],
        [0.1, 0.2, 1.0, 0.5],
        [0.0, 0.1, 0.5, 1.0],
    ]
)
Z = np.array([np.nan, 4.0, 1.0, -0.5])
FILLED = np.where(np.isnan(Z), 0.0, Z)
OPTIONS = dict(n=500, L=2, max_iter=20, coverage=None, check_prior=False)
MESSAGE = "NA values in z-scores are replaced with 0."


def test_missing_z_scores_are_replaced_with_a_recorded_warning():
    with pytest.warns(RuntimeWarning, match="NA values in z-scores are replaced with 0"):
        fit = prusie.susie_rss(z=Z, R=R, **OPTIONS)
    assert fit.warnings.count(MESSAGE) == 1
    np.testing.assert_array_equal(fit.alpha, prusie.susie_rss(z=FILLED, R=R, **OPTIONS).alpha)


@pytest.mark.filterwarnings("ignore:IBSS did not converge:RuntimeWarning")
def test_multipanel_joint_fit_uses_zero_for_missing_z_scores():
    with pytest.warns(RuntimeWarning, match="NA values in z-scores are replaced with 0"):
        fit = prusie.susie_rss(z=Z, R=[R, R2], **OPTIONS)
    filled = prusie.susie_rss(z=FILLED, R=[R, R2], **OPTIONS)
    assert np.isfinite(fit.alpha).all()
    assert fit.warnings.count(MESSAGE) == 1
    np.testing.assert_array_equal(fit.alpha, filled.alpha)
    np.testing.assert_array_equal(fit.params["omega"], filled.params["omega"])


@pytest.mark.parametrize(
    "config",
    [
        prusie.ConvergenceConfig(criterion="alpha_pip"),
        prusie.ConvergenceConfig(criterion="joint"),
        prusie.ConvergenceConfig(min_iter=3),
        prusie.ConvergenceConfig(cycle_lag=3),
    ],
)
@pytest.mark.parametrize("model", [dict(R_mismatch="eb"), dict(R_finite=400.0)])
def test_finite_and_mismatch_rss_reject_unimplemented_stopping_controls(monkeypatch, config, model):
    monkeypatch.setattr(prusie.rss, "_fit", pytest.fail)
    with pytest.raises(ValueError, match="support only the default"):
        prusie.susie_rss(z=FILLED, R=R, convergence=config, **model, **OPTIONS)


def test_multipanel_rejects_unimplemented_controls_before_panel_fits(monkeypatch):
    monkeypatch.setattr(prusie.rss, "_fit", pytest.fail)
    with pytest.raises(ValueError, match="support only the default"):
        prusie.susie_rss(
            z=FILLED, R=[R, R2], convergence=prusie.ConvergenceConfig(min_iter=3), **OPTIONS
        )
    with pytest.raises(ValueError, match="support only the default"):
        prusie.susie_rss(
            z=FILLED,
            R=[R, R2],
            R_finite=400.0,
            convergence=prusie.ConvergenceConfig(criterion="alpha_pip"),
            **OPTIONS,
        )


@pytest.mark.filterwarnings("ignore:IBSS did not converge:RuntimeWarning")
def test_explicit_elbo_rule_remains_available_for_mismatch_rss():
    fit = prusie.susie_rss(
        z=FILLED, R=R, R_mismatch="eb", convergence=prusie.ConvergenceConfig(), **OPTIONS
    )
    assert np.isfinite(fit.alpha).all()


@pytest.mark.parametrize("refine", ["yes", 1, None])
@pytest.mark.parametrize("entry", ["suff_stat", "rss", "individual"])
def test_invalid_refine_is_rejected_before_fitting(monkeypatch, entry, refine):
    monkeypatch.setattr(getattr(prusie, entry), "_fit", pytest.fail)
    with pytest.raises(TypeError, match="refine must be False, True, or a RefinementConfig"):
        if entry == "suff_stat":
            prusie.susie_suff_stat(499 * R, FILLED, 499.0, 500, L=2, refine=refine)
        elif entry == "rss":
            prusie.susie_rss(z=FILLED, R=R, refine=refine, **OPTIONS)
        else:
            rng = np.random.default_rng(1)
            X = rng.normal(size=(30, 4))
            prusie.susie(X, X[:, 1] + rng.normal(size=30), L=2, refine=refine)
