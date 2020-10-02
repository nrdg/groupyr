import pytest

from sklearn.utils.estimator_checks import check_estimator

from sglpy import TemplateEstimator
from sglpy import TemplateClassifier
from sglpy import TemplateTransformer


@pytest.mark.parametrize(
    "Estimator", [TemplateEstimator, TemplateTransformer, TemplateClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
