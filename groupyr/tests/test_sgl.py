import numpy as np
import pytest

from groupyr import SGL, SGLCV
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


def test_sgl_zero():
    # Check that SGL can handle zero data without crashing
    X = [[0], [0], [0]]
    y = [0, 0, 0]
    clf = SGL(l1_ratio=1.0, alpha=0.1).fit(X, y)
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


@pytest.mark.parametrize("execution_number", range(5))
@pytest.mark.parametrize("l1_ratio", np.random.default_rng(seed=42).uniform(size=4))
def test_alpha_grid_starts_at_zero(execution_number, l1_ratio):
    X, y, groups = make_group_regression()
    agrid = _alpha_grid(X, y, groups=groups, l1_ratio=l1_ratio, n_alphas=10)
    model = SGL(groups=groups, l1_ratio=l1_ratio, alpha=agrid[0]).fit(X, y)

    assert model.chosen_features_.size == 0  # nosec
