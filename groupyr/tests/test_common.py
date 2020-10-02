import pytest

from sklearn.utils.estimator_checks import check_estimator

from groupyr._base import SGLBaseEstimator
from groupyr import LogisticSGL, LogisticSGLCV, SGL, SGLCV


@pytest.mark.parametrize(
    "Estimator", [SGLBaseEstimator, SGL, LogisticSGL, SGLCV, LogisticSGLCV]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator())
