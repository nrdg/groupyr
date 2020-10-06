import numpy as np
import pytest

from groupyr import SGL, SGLCV, LogisticSGLCV
from groupyr.datasets import make_group_regression
from groupyr.sgl import _alpha_grid

from sklearn.linear_model.tests.test_coordinate_descent import build_dataset
from sklearn.utils._testing import assert_almost_equal, assert_array_almost_equal


def test_sgl_input_validation():
    X = [[0], [0], [0]]
    y = [0, 0, 0]

    with pytest.raises(ValueError):
        SGL(l1_ratio=1.0, alpha=0.1, warm_start="error").fit(X, y)

    with pytest.raises(ValueError):
        SGL(l1_ratio=1.0, alpha=0.1).fit(X, y, loss="error")

    with pytest.raises(ValueError):
        SGL(l1_ratio=100, alpha=0.1).fit(X, y)

    with pytest.raises(ValueError):
        SGL(l1_ratio=0.5, alpha=0.1, scale_l2_by="error").fit(X, y)

    with pytest.raises(ValueError):
        _alpha_grid(X, y, l1_ratio=0.5, scale_l2_by="error")


@pytest.mark.parametrize("Estimator", [SGL, SGLCV, LogisticSGLCV])
def test_sgl_masks(Estimator):
    groups = [np.arange(5), np.arange(5, 10)]
    model = Estimator(groups=groups)
    coefs_0 = np.concatenate([np.ones(5), np.zeros(5)])
    model.coef_ = coefs_0
    assert_array_almost_equal(model.chosen_features_, np.arange(5))
    assert_array_almost_equal(model.sparsity_mask_, coefs_0 != 0)
    assert_array_almost_equal(model.chosen_groups_, np.array([0]))

    model.groups = None
    assert_array_almost_equal(model.chosen_groups_, np.arange(5))

    coefs_1 = np.concatenate([np.ones(5), np.ones(5) * 1e-7])
    model.coef_ = coefs_1
    assert_array_almost_equal(model.like_nonzero_mask_(rtol=2.1e-7), coefs_0 != 0)
    assert all(model.like_nonzero_mask_(rtol=1e-8))  # nosec


@pytest.mark.parametrize("suppress_warnings", [True, False])
def test_sgl_zero(suppress_warnings):
    # Check that SGL can handle zero data without crashing
    X = [[0], [0], [0]]
    y = [0, 0, 0]
    clf = SGL(l1_ratio=1.0, alpha=0.1, suppress_solver_warnings=suppress_warnings).fit(
        X, y
    )
    pred = clf.predict([[1], [2], [3]])
    assert_array_almost_equal(clf.coef_, [0])
    assert_array_almost_equal(pred, [0, 0, 0])


# When l1_ratio=1, SGL should behave like the lasso. These next tests
# replicate the unit testing for sklearn.linear_model.Lasso
@pytest.mark.parametrize("loss", ["squared_loss", "huber"])
def test_sgl_toy(loss):
    # Test on a toy example for various values of l1_ratio.
    # When validating this against glmnet notice that glmnet divides it
    # against n_obs.

    X = [[-1], [0], [1]]
    y = [-1, 0, 1]  # just a straight line
    T = [[2], [3], [4]]  # test sample

    tol = 1e-6

    clf = SGL(alpha=1e-8, tol=tol)
    clf.fit(X, y, loss=loss)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [1])
    assert_array_almost_equal(pred, [2, 3, 4])

    clf = SGL(alpha=0.1, tol=tol)
    clf.fit(X, y, loss=loss)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.85])
    assert_array_almost_equal(pred, [1.7, 2.55, 3.4])

    clf = SGL(alpha=0.5, tol=tol)
    clf.fit(X, y, loss=loss)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.25])
    assert_array_almost_equal(pred, [0.5, 0.75, 1.0])

    clf = SGL(alpha=1, tol=tol)
    clf.fit(X, y, loss=loss)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.0])
    assert_array_almost_equal(pred, [0, 0, 0])


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_warm_start(fit_intercept):
    X, y, _, _ = build_dataset()
    max_iter = 5
    sgl = SGL(
        l1_ratio=0.5,
        alpha=0.1,
        warm_start=True,
        max_iter=max_iter,
        fit_intercept=fit_intercept,
        include_solver_trace=True,
    )
    sgl.fit(X, y)
    tr_0 = sgl.solver_trace_
    sgl.fit(X, y)
    tr_1 = sgl.solver_trace_

    sgl = SGL(
        l1_ratio=0.5,
        alpha=0.1,
        max_iter=max_iter + 1,
        fit_intercept=fit_intercept,
        include_solver_trace=True,
    )
    sgl.fit(X, y)
    tr_2 = sgl.solver_trace_

    fx_cold_start = np.array(tr_2.trace_fx)
    fx_warm_start = np.array(tr_0.trace_fx + tr_1.trace_fx)[: fx_cold_start.size]

    assert_array_almost_equal(fx_warm_start, fx_cold_start)


def test_sgl_cv():
    X, y, X_test, y_test = build_dataset()
    max_iter = 150
    clf = SGLCV(n_alphas=10, eps=1e-3, max_iter=max_iter, cv=3).fit(X, y)
    assert_almost_equal(clf.alpha_, 0.056, 2)
    assert clf.score(X_test, y_test) > 0.99  # nosec

    alphas = np.copy(clf.alphas_)
    np.random.default_rng().shuffle(alphas)
    clf2 = SGLCV(alphas=alphas, max_iter=max_iter, cv=3, n_jobs=2).fit(X, y)
    assert_array_almost_equal(clf.alphas_, clf2.alphas_)


@pytest.mark.parametrize("execution_number", range(5))
@pytest.mark.parametrize("l1_ratio", np.random.default_rng(seed=42).uniform(size=4))
@pytest.mark.parametrize("compute_Xy", [True, "ndim2", False])
def test_alpha_grid_starts_at_zero(execution_number, l1_ratio, compute_Xy):
    X, y, groups = make_group_regression()

    if compute_Xy is True:
        Xy = X.T.dot(y)
    elif compute_Xy == "ndim2":
        Xy = X.T.dot(y)[:, np.newaxis]
    else:
        Xy = None

    agrid = _alpha_grid(X, y, Xy=Xy, groups=groups, l1_ratio=l1_ratio, n_alphas=10)
    model = SGL(groups=groups, l1_ratio=l1_ratio, alpha=agrid[0]).fit(X, y)

    assert model.chosen_features_.size == 0  # nosec


def test_alpha_grid_float_resolution():
    X = np.zeros((3, 3))
    y = [1, 2, 3]
    agrid = _alpha_grid(X, y, l1_ratio=0.5, n_alphas=5)
    assert_array_almost_equal(agrid, np.ones(5) * np.finfo(float).resolution)


def test_sglcv_value_errors():
    X = [[0], [0], [0]]
    y = [0, 0, 0]

    with pytest.raises(ValueError):
        SGLCV().fit(X, y + [0])
