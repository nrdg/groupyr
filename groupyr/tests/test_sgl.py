import numpy as np
import pytest

from groupyr import SGL, SGLCV
from groupyr.datasets import make_group_regression
from groupyr.sgl import _alpha_grid

from sklearn.utils._testing import assert_array_almost_equal


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

    clf = SGL(alpha=1e-8)
    clf.fit(X, y, loss=loss)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [1])
    assert_array_almost_equal(pred, [2, 3, 4])

    clf = SGL(alpha=0.1)
    clf.fit(X, y, loss=loss)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.85])
    assert_array_almost_equal(pred, [1.7, 2.55, 3.4])

    clf = SGL(alpha=0.5)
    clf.fit(X, y, loss=loss)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.25])
    assert_array_almost_equal(pred, [0.5, 0.75, 1.0])

    clf = SGL(alpha=1)
    clf.fit(X, y, loss=loss)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.0])
    assert_array_almost_equal(pred, [0, 0, 0])


def build_dataset(n_samples=50, n_features=200, n_informative_features=10, n_targets=1):
    """
    build an ill-posed linear regression problem with many noisy features and
    comparatively few samples
    """
    random_state = np.random.RandomState(0)
    if n_targets > 1:
        w = random_state.randn(n_features, n_targets)
    else:
        w = random_state.randn(n_features)
    w[n_informative_features:] = 0.0
    X = random_state.randn(n_samples, n_features)
    y = np.dot(X, w)
    X_test = random_state.randn(n_samples, n_features)
    y_test = np.dot(X_test, w)
    return X, y, X_test, y_test


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
    X, y, groups = make_group_regression(random_state=42)

    sglcv = SGLCV(l1_ratio=[0.5, 0.9, 1.0], groups=groups, cv=3).fit(X, y)

    assert sglcv.score(X, y) > 0.99  # nosec


@pytest.mark.parametrize("execution_number", range(5))
@pytest.mark.parametrize("l1_ratio", np.random.default_rng(seed=42).uniform(size=4))
def test_alpha_grid_starts_at_zero(execution_number, l1_ratio):
    X, y, groups = make_group_regression()
    agrid = _alpha_grid(X, y, groups=groups, l1_ratio=l1_ratio, n_alphas=10)
    model = SGL(groups=groups, l1_ratio=l1_ratio, alpha=agrid[0]).fit(X, y)

    assert model.chosen_features_.size == 0  # nosec
