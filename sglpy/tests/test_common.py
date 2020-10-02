import pytest

from sklearn.utils.estimator_checks import check_estimator

from sglpy._base import SGLBaseEstimator
from sglpy import LogisticSGL, LogisticSGLCV, SGL, SGLCV


@pytest.mark.parametrize(
    "Estimator", [SGLBaseEstimator, SGL, LogisticSGL, SGLCV, LogisticSGLCV]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator())
