import numpy as np
import pytest

from groupyr.transform import isiterable, GroupExtractor
from groupyr.transform import (
    GroupRemover,
    GroupShuffler,
    GroupAggregator,
    GroupResampler,
)
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

    # Test the same as before but with group intersection instead of group union
    select = ["alpha", "two"]
    idx = np.concatenate(
        [
            grp
            for grp, t in zip(groups, group_tuple_names)
            if all([r in t for r in select])
        ]
    )
    ge = GroupExtractor(
        groups=groups,
        group_names=group_tuple_names,
        select=select,
        select_intersection=True,
    )
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


def test_GroupAggregator():
    X = np.arange(100).reshape(10, 10)
    groups = [
        np.array([0, 1, 2]),
        np.array([3, 4, 5]),
        np.array([6, 7, 8]),
        np.array([9]),
    ]
    group_names = ["one", "two", "three", "four"]

    ga = GroupAggregator(groups=groups)
    X_tr = ga.fit_transform(X)
    assert ga.feature_names_out_ == [
        f"group{i}__mean" for i in range(len(groups))
    ]  # nosec

    X_ref = np.array([np.array([1, 4, 7, 9]) + i * 10 for i in range(10)])
    assert np.allclose(X_tr, X_ref)  # nosec

    ga = GroupAggregator(func=["mean", np.amax], groups=groups, group_names=group_names)
    X_tr = ga.fit_transform(X)
    feature_names_ref = []
    for grp in group_names:
        feature_names_ref.append("__".join([grp, "mean"]))
        feature_names_ref.append("__".join([grp, "amax"]))

    assert ga.feature_names_out_ == feature_names_ref  # nosec
    X_ref = np.array([np.array([1, 2, 4, 5, 7, 8, 9, 9]) + i * 10 for i in range(10)])
    assert np.allclose(X_tr, X_ref)  # nosec

    ga = GroupAggregator(func="median", groups=groups, group_names=group_names)
    X_tr = ga.fit_transform(X)
    feature_names_ref = []
    for grp in group_names:
        feature_names_ref.append("__".join([grp, "median"]))

    assert ga.feature_names_out_ == feature_names_ref  # nosec
    X_ref = np.array([np.array([1, 4, 7, 9]) + i * 10 for i in range(10)])
    assert np.allclose(X_tr, X_ref)  # nosec

    # Check that the axis kwarg is ignored
    ga = GroupAggregator(
        func=np.median, groups=groups, group_names=group_names, kw_args=dict(axis=99)
    )
    X_tr = ga.fit_transform(X)
    assert ga.feature_names_out_ == feature_names_ref  # nosec
    assert np.allclose(X_tr, X_ref)  # nosec

    assert ga.get_feature_names() == ga.feature_names_out_


def test_GroupResampler():
    X = np.arange(100).reshape(5, 20)
    groups = [
        np.array([0, 1, 2, 3, 4]),
        np.array([5, 6, 7, 8, 9]),
        np.array([10, 11, 12, 13]),
        np.array([14, 15, 16]),
        np.array([17, 18, 19]),
    ]
    group_names = [
        ("one", "alpha"),
        ("one", "beta"),
        ("two", "alpha"),
        ("two", "beta"),
        ("three", "alpha"),
    ]

    X_ref = np.array(
        [
            [
                0.0,
                1.33333333,
                2.66666667,
                4.0,
                5.0,
                6.33333333,
                7.66666667,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                14.66666667,
                15.33333333,
                16.0,
                17.0,
                17.66666667,
                18.33333333,
                19.0,
            ],
            [
                20.0,
                21.33333333,
                22.66666667,
                24.0,
                25.0,
                26.33333333,
                27.66666667,
                29.0,
                30.0,
                31.0,
                32.0,
                33.0,
                34.0,
                34.66666667,
                35.33333333,
                36.0,
                37.0,
                37.66666667,
                38.33333333,
                39.0,
            ],
            [
                40.0,
                41.33333333,
                42.66666667,
                44.0,
                45.0,
                46.33333333,
                47.66666667,
                49.0,
                50.0,
                51.0,
                52.0,
                53.0,
                54.0,
                54.66666667,
                55.33333333,
                56.0,
                57.0,
                57.66666667,
                58.33333333,
                59.0,
            ],
            [
                60.0,
                61.33333333,
                62.66666667,
                64.0,
                65.0,
                66.33333333,
                67.66666667,
                69.0,
                70.0,
                71.0,
                72.0,
                73.0,
                74.0,
                74.66666667,
                75.33333333,
                76.0,
                77.0,
                77.66666667,
                78.33333333,
                79.0,
            ],
            [
                80.0,
                81.33333333,
                82.66666667,
                84.0,
                85.0,
                86.33333333,
                87.66666667,
                89.0,
                90.0,
                91.0,
                92.0,
                93.0,
                94.0,
                94.66666667,
                95.33333333,
                96.0,
                97.0,
                97.66666667,
                98.33333333,
                99.0,
            ],
        ]
    )

    gs = GroupResampler(resample_to=4, groups=groups)
    X_tr = gs.fit_transform(X)
    assert gs.feature_names_out_ == [
        (f"group{i}", idx) for i in range(len(groups)) for idx in range(4)
    ]  # nosec
    assert np.allclose(X_tr, X_ref)  # nosec

    X_ref = np.array(
        [
            [0.0, 2.0, 4.0, 5.0, 7.0, 9.0, 10.0, 13.0, 14.0, 16.0, 17.0, 19.0],
            [20.0, 22.0, 24.0, 25.0, 27.0, 29.0, 30.0, 33.0, 34.0, 36.0, 37.0, 39.0],
            [40.0, 42.0, 44.0, 45.0, 47.0, 49.0, 50.0, 53.0, 54.0, 56.0, 57.0, 59.0],
            [60.0, 62.0, 64.0, 65.0, 67.0, 69.0, 70.0, 73.0, 74.0, 76.0, 77.0, 79.0],
            [80.0, 82.0, 84.0, 85.0, 87.0, 89.0, 90.0, 93.0, 94.0, 96.0, 97.0, 99.0],
        ]
    )
    gs = GroupResampler(resample_to=0.6, groups=groups, group_names=group_names)
    X_tr = gs.fit_transform(X)
    assert np.allclose(X_tr, X_ref)  # nosec

    feature_names_ref = [
        ("one", "alpha", 0),
        ("one", "alpha", 1),
        ("one", "alpha", 2),
        ("one", "beta", 0),
        ("one", "beta", 1),
        ("one", "beta", 2),
        ("two", "alpha", 0),
        ("two", "alpha", 1),
        ("two", "beta", 0),
        ("two", "beta", 1),
        ("three", "alpha", 0),
        ("three", "alpha", 1),
    ]
    assert feature_names_ref == gs.feature_names_out_

    with pytest.raises(ValueError):
        GroupResampler(resample_to=3.0).fit_transform(X)


@pytest.mark.parametrize(
    "Transformer",
    [GroupExtractor, GroupRemover, GroupShuffler, GroupAggregator, GroupResampler],
)
def test_all_estimators(Transformer):
    return check_estimator(Transformer())
