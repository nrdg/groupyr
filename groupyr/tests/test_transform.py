import numpy as np
import pytest

from groupyr.transform import isiterable, GroupExtractor
from groupyr.transform import GroupRemover, GroupShuffler
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils import shuffle as util_shuffle


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
    group_names = ["one", "two", "three", "four"]
    group_tuple_names = [
        ("one", "alpha"),
        ("one", "beta"),
        ("two", "alpha"),
        ("three", "beta"),
    ]

    select = 2
    ge = GroupExtractor(groups=groups, select=select)
    X_tr = ge.fit_transform(X)
    assert np.allclose(X[:, groups[select]], X_tr)  # nosec

    select = "two"
    idx = np.concatenate([grp for grp, t in zip(groups, group_names) if t == select])
    ge = GroupExtractor(groups=groups, group_names=group_names, select=select)
    X_tr = ge.fit_transform(X)
    assert np.allclose(X[:, idx], X_tr)  # nosec

    select = ["two", "three"]
    idx = np.concatenate([grp for grp, t in zip(groups, group_names) if t in select])
    ge = GroupExtractor(groups=groups, group_names=group_names, select=select)
    X_tr = ge.fit_transform(X)
    assert np.allclose(X[:, idx], X_tr)  # nosec

    select = "one"
    idx = np.concatenate(
        [grp for grp, t in zip(groups, group_tuple_names) if t[0] == select]
    )
    ge = GroupExtractor(groups=groups, group_names=group_tuple_names, select=select)
    X_tr = ge.fit_transform(X)
    assert np.allclose(X[:, idx], X_tr)  # nosec

    select = ["alpha", "three"]
    idx = np.concatenate(
        [
            grp
            for grp, t in zip(groups, group_tuple_names)
            if any([r in t for r in select])
        ]
    )
    ge = GroupExtractor(groups=groups, group_names=group_tuple_names, select=select)
    X_tr = ge.fit_transform(X)
    assert np.allclose(X[:, idx], X_tr)  # nosec

    select = [0, 3]
    ge = GroupExtractor(groups=groups, select=select)
    X_tr = ge.fit_transform(X)
    idx = np.concatenate([groups[e] for e in select])
    assert np.allclose(X[:, idx], X_tr)  # nosec

    ge = GroupExtractor()
    X_tr = ge.fit_transform(X)
    assert np.allclose(X_tr, X)  # nosec

    with pytest.raises(ValueError):
        GroupExtractor(select="error").fit_transform(X)


def test_GroupRemover():
    X = np.array([list(range(10))] * 10)
    groups = [
        np.array([0, 1, 2]),
        np.array([3, 4, 5]),
        np.array([6, 7, 8]),
        np.array([9]),
    ]
    group_names = ["one", "two", "three", "four"]
    group_tuple_names = [
        ("one", "alpha"),
        ("one", "beta"),
        ("two", "alpha"),
        ("three", "beta"),
    ]

    select = 2
    idx = np.concatenate([grp for idx, grp in enumerate(groups) if idx != select])
    ge = GroupRemover(groups=groups, select=select)
    X_tr = ge.fit_transform(X)
    assert np.allclose(X[:, idx], X_tr)  # nosec

    select = "two"
    idx = np.concatenate([grp for grp, t in zip(groups, group_names) if t != select])
    ge = GroupRemover(groups=groups, group_names=group_names, select=select)
    X_tr = ge.fit_transform(X)
    assert np.allclose(X[:, idx], X_tr)  # nosec

    select = ["two", "three"]
    idx = np.concatenate(
        [grp for grp, t in zip(groups, group_names) if t not in select]
    )
    ge = GroupRemover(groups=groups, group_names=group_names, select=select)
    X_tr = ge.fit_transform(X)
    assert np.allclose(X[:, idx], X_tr)  # nosec

    select = "one"
    idx = np.concatenate(
        [grp for grp, t in zip(groups, group_tuple_names) if t[0] != select]
    )
    ge = GroupRemover(groups=groups, group_names=group_tuple_names, select=select)
    X_tr = ge.fit_transform(X)
    assert np.allclose(X[:, idx], X_tr)  # nosec

    select = ["alpha", "three"]
    idx = np.concatenate(
        [
            grp
            for grp, t in zip(groups, group_tuple_names)
            if all([r not in t for r in select])
        ]
    )
    ge = GroupRemover(groups=groups, group_names=group_tuple_names, select=select)
    X_tr = ge.fit_transform(X)
    assert np.allclose(X[:, idx], X_tr)  # nosec

    select = [0, 3]
    ge = GroupRemover(groups=groups, select=select)
    X_tr = ge.fit_transform(X)
    idx = np.concatenate([grp for idx, grp in enumerate(groups) if idx not in select])
    assert np.allclose(X[:, idx], X_tr)  # nosec

    ge = GroupRemover()
    X_tr = ge.fit_transform(X)
    assert np.allclose(X_tr, X)  # nosec

    with pytest.raises(ValueError):
        GroupRemover(select=object()).fit_transform(X)

    with pytest.raises(ValueError):
        GroupRemover(group_names=group_names).fit_transform(X)

    with pytest.raises(ValueError):
        GroupRemover(groups=groups, group_names=["a", "b"]).fit_transform(X)

    with pytest.raises(ValueError):
        GroupRemover(groups=groups, select="b").fit_transform(X)

    with pytest.raises(ValueError):
        GroupRemover(groups=groups, select=["a", "b"]).fit_transform(X)

    with pytest.raises(TypeError):
        GroupRemover(groups=groups, group_names=0).fit_transform(X)


def assert_shuffle_match(
    X, X_tr, groups=None, group_names=None, select=None, random_state=0
):
    ge = GroupExtractor(select=select, groups=groups, group_names=group_names)
    shuffled = {"orig": ge.fit_transform(X), "tran": ge.fit_transform(X_tr)}
    gr = GroupRemover(select=select, groups=groups, group_names=group_names)
    same = {"orig": gr.fit_transform(X), "tran": gr.fit_transform(X_tr)}
    assert np.allclose(same["orig"], same["tran"])
    assert np.allclose(
        util_shuffle(shuffled["orig"], random_state=random_state), shuffled["tran"]
    )


def test_GroupShuffler():
    X = np.arange(100).reshape(10, 10)
    groups = [
        np.array([0, 1, 2]),
        np.array([3, 4, 5]),
        np.array([6, 7, 8]),
        np.array([9]),
    ]
    group_names = ["one", "two", "three", "four"]
    group_tuple_names = [
        ("one", "alpha"),
        ("one", "beta"),
        ("two", "alpha"),
        ("three", "beta"),
    ]

    select = 2
    gs = GroupShuffler(groups=groups, select=select, random_state=0)
    X_tr = gs.fit_transform(X)
    assert_shuffle_match(X, X_tr, groups=groups, select=select, random_state=0)

    select = "two"
    gs = GroupShuffler(
        groups=groups, group_names=group_names, select=select, random_state=0
    )
    X_tr = gs.fit_transform(X)
    assert_shuffle_match(
        X, X_tr, groups=groups, group_names=group_names, select=select, random_state=0
    )

    select = ["two", "three"]
    gs = GroupShuffler(
        groups=groups, group_names=group_names, select=select, random_state=0
    )
    X_tr = gs.fit_transform(X)
    assert_shuffle_match(
        X, X_tr, groups=groups, group_names=group_names, select=select, random_state=0
    )

    select = "one"
    gs = GroupShuffler(
        groups=groups, group_names=group_tuple_names, select=select, random_state=0
    )
    X_tr = gs.fit_transform(X)
    assert_shuffle_match(
        X,
        X_tr,
        groups=groups,
        group_names=group_tuple_names,
        select=select,
        random_state=0,
    )

    select = ["alpha", "three"]
    gs = GroupShuffler(
        groups=groups, group_names=group_tuple_names, select=select, random_state=0
    )
    X_tr = gs.fit_transform(X)
    assert_shuffle_match(
        X,
        X_tr,
        groups=groups,
        group_names=group_tuple_names,
        select=select,
        random_state=0,
    )

    select = [0, 3]
    gs = GroupShuffler(groups=groups, select=select, random_state=0)
    X_tr = gs.fit_transform(X)
    assert_shuffle_match(X, X_tr, groups=groups, select=select, random_state=0)

    gs = GroupShuffler()
    X_tr = gs.fit_transform(X)
    assert np.allclose(X_tr, X)  # nosec


@pytest.mark.parametrize("Transformer", [GroupExtractor, GroupRemover, GroupShuffler])
def test_all_estimators(Transformer):
    return check_estimator(Transformer())
