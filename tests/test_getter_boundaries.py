"""Deterministic boundary checks for core posterior summaries without fitting."""

import numpy as np
import pytest

import prusie


def _model(alpha):
    alpha = np.asarray(alpha, dtype=float)
    count, p = alpha.shape
    return prusie.SusieResult(
        variant_ids=np.asarray([str(index) for index in range(p)]),
        alpha=alpha,
        mu=np.ones_like(alpha),
        mu2=np.full_like(alpha, 2.0),
        lbf_variable=np.zeros_like(alpha),
        lbf=np.zeros(count),
        V=np.ones(count),
        sigma2=1.0,
        pip=1 - np.prod(1 - alpha, axis=0),
        elbo=np.zeros(1),
        niter=1,
        converged=True,
        sets={},
        null_index=None,
        params={},
        backend_version="manual-boundary-object",
        warnings=[],
        qc={},
        variant_metadata={},
        X_column_scale_factors=np.ones(p),
        intercept=1.25,
        KL=np.zeros(count),
        XtXr=np.zeros(p),
    )


def test_attainable_ignores_ld_keywords_and_preserves_original_effect_ids():
    model = _model([[0.5, 0.5, 0], [0.8, 0.1, 0.1], [0, 0.1, 0.9]])

    class UnusedLD:
        def __array__(self, *args, **kwargs):
            pytest.fail("attainable CS must ignore both supplied LD sources")

    with pytest.warns(RuntimeWarning, match="purity filtering is skipped"):
        result = prusie.susie_get_cs_attainable(
            model,
            coverage=0.7,
            X=UnusedLD(),
            Xcorr=UnusedLD(),
            min_abs_corr=1.0,
            median_abs_corr=1.0,
            squared=True,
            check_symmetric=False,
            n_purity=1,
            cs_extension_corr=0.0,
            seed=93,
        )
    np.testing.assert_array_equal(result["cs_index"], [1, 2])
    assert [members.tolist() for members in result["cs"]] == [[0], [2]]
    np.testing.assert_allclose(result["coverage"], [0.8, 0.9], atol=0, rtol=0)
    assert result["requested_coverage"] == 0.7
    assert all(np.isnan(values).all() for values in result["purity"].values())


def test_attainable_positive_infinite_entropy_threshold_disables_filter():
    model = _model([[1 / 3, 1 / 3, 1 / 3]])
    with pytest.warns(RuntimeWarning, match="purity filtering is skipped"):
        restricted = prusie.susie_get_cs_attainable(model, coverage=0.5, ethres=2)
    with pytest.warns(RuntimeWarning, match="purity filtering is skipped"):
        unlimited = prusie.susie_get_cs_attainable(model, coverage=0.5, ethres=np.inf)
    assert restricted["cs"] == []
    np.testing.assert_array_equal(unlimited["cs_index"], [0])
    np.testing.assert_array_equal(unlimited["cs"][0], [0, 1])
    np.testing.assert_allclose(unlimited["coverage"], [2 / 3], atol=1e-15, rtol=0)


@pytest.mark.parametrize("dedup, expected", [(True, [0]), (False, [0, 1])])
def test_attainable_forwards_dedup(dedup, expected):
    model = _model([[0.8, 0.2, 0], [0.8, 0.2, 0]])
    with pytest.warns(RuntimeWarning, match="purity filtering is skipped"):
        result = prusie.susie_get_cs_attainable(model, coverage=0.7, dedup=dedup)
    np.testing.assert_array_equal(result["cs_index"], expected)


@pytest.mark.parametrize("sparse_input", [False, True])
def test_cs_correlation_constant_columns_have_unit_diagonal(sparse_input):
    model = _model([[0.8, 0.2, 0], [0, 0.1, 0.9]])
    model.sets = {"cs": [np.array([0]), np.array([2])], "cs_index": np.array([0, 1])}
    design = np.array([[1, 0, -1], [1, 1, 1], [1, 2, -1], [1, 3, 1]], dtype=float)
    if sparse_input:
        sparse = pytest.importorskip("scipy.sparse")
        design = sparse.csc_matrix(design)
    np.testing.assert_allclose(prusie.get_cs_correlation(model, X=design), np.eye(2))
    assert prusie.get_cs_correlation(model, X=design, max=True) == 0


@pytest.mark.parametrize("kind", ["list", "array", "sparse"])
def test_predict_accepts_array_like_and_preserves_sparse(kind):
    model = _model([[0.8, 0.2, 0], [0, 0.1, 0.9]])
    observations = [[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]]
    expected = np.asarray(observations) @ prusie.coef(model)[1:] + model.intercept
    if kind == "array":
        observations = np.asarray(observations)
    elif kind == "sparse":
        sparse = pytest.importorskip("scipy.sparse")

        class ProtectedCSC(sparse.csc_matrix):
            def __array__(self, *args, **kwargs):
                pytest.fail("predict must preserve the sparse input")

        observations = ProtectedCSC(observations)
    np.testing.assert_allclose(prusie.predict(model, observations), expected, atol=1e-14, rtol=0)
