import pytest

from sklearn.utils.estimator_checks import check_estimator

from groupyr._base import SGLBaseEstimator
from groupyr import LogisticSGL, LogisticSGLCV, SGL, SGLCV


@pytest.mark.parametrize("Estimator", [SGLBaseEstimator, SGL, LogisticSGL])
def test_all_estimators(Estimator):
    return check_estimator(Estimator(max_iter=50, tol=1e-2))


@pytest.mark.parametrize("Estimator", [SGLCV, LogisticSGLCV])
def test_all_cv_estimators(Estimator):
    return check_estimator(Estimator(max_iter=50, tol=1e-2, cv=2, n_alphas=3))
