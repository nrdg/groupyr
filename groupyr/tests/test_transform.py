import numpy as np
import pytest

from groupyr.transform import isiterable, GroupExtractor, GroupRemover
from sklearn.utils.estimator_checks import check_estimator


def test_isiterable():
    assert isiterable(range(10))  # nosec
    assert not isiterable(5)  # nosec
    assert isiterable(np.arange(10))  # nosec


def test_GroupExtractor():
    X = np.array([list(range(10))] * 10)
    groups = [
        np.array([0, 1, 2]),
        np.array([3, 4, 5]),
        np.array([6, 7, 8]),
        np.array([9]),
    ]

    extract = 2
    ge = GroupExtractor(groups=groups, extract=extract)
    X_tr = ge.fit_transform(X)
    assert np.allclose(X[:, groups[extract]], X_tr)  # nosec

    extract = [0, 3]
    ge = GroupExtractor(groups=groups, extract=extract)
    X_tr = ge.fit_transform(X)
    idx = np.concatenate([groups[e] for e in extract])
    assert np.allclose(X[:, idx], X_tr)  # nosec

    ge = GroupExtractor()
    X_tr = ge.fit_transform(X)
    assert np.allclose(X_tr, X)  # nosec

    with pytest.raises(ValueError):
        GroupExtractor(extract="error").fit_transform(X)


def test_GroupRemover():
    X = np.array([list(range(10))] * 10)
    groups = [
        np.array([0, 1, 2]),
        np.array([3, 4, 5]),
        np.array([6, 7, 8]),
        np.array([9]),
    ]

    remove = 2
    idx = np.concatenate([grp for idx, grp in enumerate(groups) if idx != remove])
    ge = GroupRemover(groups=groups, remove=remove)
    X_tr = ge.fit_transform(X)
    assert np.allclose(X[:, idx], X_tr)  # nosec

    remove = [0, 3]
    ge = GroupRemover(groups=groups, remove=remove)
    X_tr = ge.fit_transform(X)
    idx = np.concatenate([grp for idx, grp in enumerate(groups) if idx not in remove])
    assert np.allclose(X[:, idx], X_tr)  # nosec

    ge = GroupRemover()
    X_tr = ge.fit_transform(X)
    assert np.allclose(X_tr, X)  # nosec

    with pytest.raises(ValueError):
        GroupRemover(remove="error").fit_transform(X)


@pytest.mark.parametrize("Transformer", [GroupExtractor, GroupRemover])
def test_all_estimators(Transformer):
    return check_estimator(Transformer())
