"""Transform feature matrices with grouped covariates."""
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_random_state
from sklearn.utils import shuffle as util_shuffle

from .utils import check_groups


def isiterable(obj):
    """Return True if obj is an iterable, False otherwise."""
    try:
        _ = iter(obj)  # noqa F841
    except TypeError:
        return False
    else:
        return True


def _check_group_names(groups, group_names):
    if group_names is not None:
        if groups is None:
            raise ValueError(
                "you provided group_names but not groups. "
                "Please provide groups also."
            )
        if isiterable(group_names) and all(isinstance(s, str) for s in group_names):
            group_names_ = np.array([set([s]) for s in group_names])
        elif isiterable(group_names) and all(isiterable(s) for s in group_names):
            group_names_ = np.array([set(s) for s in group_names])
        else:
            raise TypeError(
                "group_names must be a sequence of strings or a sequence "
                "of sequences. got {0} instead".format(group_names)
            )
        if len(group_names) != len(groups):
            raise ValueError("group_names must have the same length as groups.")
    else:
        group_names_ = None

    return group_names_


def _check_select(select, group_names_):
    missing_grp_names_msg = "if ``select`` is a string, you must provide group_names"
    if select is None:
        select_ = select
    elif isiterable(select) and all(isinstance(e, int) for e in select):
        select_ = np.array(select)
    elif isinstance(select, int):
        select_ = np.array([select])
    elif isinstance(select, str):
        if group_names_ is None:
            raise ValueError(missing_grp_names_msg)
        mask = np.array(set([select]) <= group_names_, dtype=bool)
        select_ = np.where(mask)[0]
    elif isiterable(select) and all(isinstance(e, str) for e in select):
        if group_names_ is None:
            raise ValueError(missing_grp_names_msg)

        mask = np.zeros_like(group_names_, dtype=bool)
        for label in select:
            mask = np.logical_or(mask, set([label]) <= group_names_)

        select_ = np.where(mask)[0]
    else:
        raise ValueError(
            f"``select`` must be an int, string or sequence of ints or "
            f"strings; got {select} instead."
        )

    return select_


class GroupExtractor(BaseEstimator, TransformerMixin):
    """An sklearn-compatible group extractor.

    Given a sequence of all group indices and a subsequence of desired
    group indices, this transformer returns the columns of the feature
    matrix, `X`, that are in the desired subgroups.

    Parameters
    ----------
    select : numpy.ndarray, int, or str, optional
        subsequence of desired groups to extract from feature matrix
        If int or sequence of ints, these will be treated as group indices.
        If str or sequence of str, these will be treated as labels for any
        level of the (potentially multi-indexed) group names, which must be
        specified in ``group_names``

    groups : list of numpy.ndarray
        list of arrays of non-overlapping indices for each group. For
        example, if nine features are grouped into equal contiguous groups of
        three, then groups would be ``[array([0, 1, 2]), array([3, 4, 5]),
        array([6, 7, 8])]``. If the feature matrix contains a bias or
        intercept feature, do not include it as a group. If None, all
        features will belong to one group.

    group_names : sequence of str or sequences, optional
        The names of the groups of X. If this is a sequence of strings, then
        this transformer will extract groups whose names match ``select``. If
        this is a sequence of sequences, then this transformer will extract
        groups that have labels that match ``select`` at any level of their
        multi-index.

    copy_X : bool, default=False
        if ``True``, X will be copied; else, ``transform`` may return a view
    """

    def __init__(self, select=None, groups=None, group_names=None, copy_X=False):
        self.select = select
        self.groups = groups
        self.group_names = group_names
        self.copy_X = copy_X

    def transform(self, X, y=None):
        """Transform the input data, extracting the desired groups.

        Parameters
        ----------
        X : numpy.ndarray
            The feature matrix
        """
        X = check_array(
            X,
            copy=self.copy_X,
            dtype=[np.float32, np.float64, int],
            force_all_finite=False,
        )
        groups = check_groups(groups=self.groups_, X=X, allow_overlap=True)
        if self.select_ is None:
            return X

        idx = np.concatenate([groups[e] for e in self.select_])
        return X[:, idx]

    def fit(self, X=None, y=None):
        """Learn the groups and number of features from the input data."""
        X = check_array(
            X,
            copy=self.copy_X,
            dtype=[np.float32, np.float64, int],
            force_all_finite=False,
        )

        _, self.n_features_in_ = X.shape
        self.groups_ = check_groups(groups=self.groups, X=X, allow_overlap=True)
        self.group_names_ = _check_group_names(self.groups, self.group_names)
        self.select_ = _check_select(self.select, self.group_names_)

        return self

    def _more_tags(self):  # pylint: disable=no-self-use
        return {"allow_nan": True, "multilabel": True, "multioutput": True}


class GroupRemover(BaseEstimator, TransformerMixin):
    """An sklearn-compatible group remover.

    Given a sequence of all group indices and a subsequence of unwanted
    group indices, this transformer returns the columns of the feature
    matrix, `X`, that DO NOT include the unwanted subgroups.

    Parameters
    ----------
    select : numpy.ndarray, int, or str, optional
        subsequence of desired groups to remove from feature matrix
        If int or sequence of ints, these will be treated as group indices.
        If str or sequence of str, these will be treated as labels for any
        level of the (potentially multi-indexed) group names, which must be
        specified in ``group_names``

    groups : list of numpy.ndarray
        list of arrays of non-overlapping indices for each group. For
        example, if nine features are grouped into equal contiguous groups of
        three, then groups would be ``[array([0, 1, 2]), array([3, 4, 5]),
        array([6, 7, 8])]``. If the feature matrix contains a bias or
        intercept feature, do not include it as a group. If None, all
        features will belong to one group.

    group_names : sequence of str or sequences, optional
        The names of the groups of X. If this is a sequence of strings, then
        this transformer will remove groups whose names match ``select``. If
        this is a sequence of sequences, then this transformer will remove
        groups that have labels that match ``select`` at any level of their
        multi-index.

    copy_X : bool, default=False
        if ``True``, X will be copied; else, ``transform`` may return a view
    """

    def __init__(self, select=None, groups=None, group_names=None, copy_X=False):
        self.select = select
        self.groups = groups
        self.group_names = group_names
        self.copy_X = copy_X

    def transform(self, X, y=None):
        """Transform the input data, removing the unwanted groups.

        Parameters
        ----------
        X : numpy.ndarray
            The feature matrix
        """
        X = check_array(
            X,
            copy=self.copy_X,
            dtype=[np.float32, np.float64, int],
            force_all_finite=False,
        )
        groups = check_groups(groups=self.groups_, X=X, allow_overlap=True)
        if self.select_ is None:
            return X

        idx = np.concatenate(
            [grp for idx, grp in enumerate(groups) if idx not in self.select_]
        )
        return X[:, idx]

    def fit(self, X=None, y=None):
        """Learn the groups and number of features from the input data."""
        X = check_array(
            X,
            copy=self.copy_X,
            dtype=[np.float32, np.float64, int],
            force_all_finite=False,
        )

        _, self.n_features_in_ = X.shape
        self.groups_ = check_groups(groups=self.groups, X=X, allow_overlap=True)
        self.group_names_ = _check_group_names(self.groups, self.group_names)
        self.select_ = _check_select(self.select, self.group_names_)

        return self

    def _more_tags(self):  # pylint: disable=no-self-use
        return {"allow_nan": True, "multilabel": True, "multioutput": True}


class GroupShuffler(BaseEstimator, TransformerMixin):
    """Shuffle some groups of a feature matrix, leaving others as is.

    Given a sequence of all group indices and a subsequence of
    group indices, this transformer returns the feature
    matrix, `X`, with the subset of groups shuffled.

    Parameters
    ----------
    select : numpy.ndarray, int, or str, optional
        subsequence of desired groups to shuffle in the feature matrix
        If int or sequence of ints, these will be treated as group indices.
        If str or sequence of str, these will be treated as labels for any
        level of the (potentially multi-indexed) group names, which must be
        specified in ``group_names``

    groups : list of numpy.ndarray
        list of arrays of non-overlapping indices for each group. For
        example, if nine features are grouped into equal contiguous groups of
        three, then groups would be ``[array([0, 1, 2]), array([3, 4, 5]),
        array([6, 7, 8])]``. If the feature matrix contains a bias or
        intercept feature, do not include it as a group. If None, all
        features will belong to one group.

    group_names : sequence of str or sequences, optional
        The names of the groups of X. If this is a sequence of strings, then
        this transformer will shuffle groups whose names match ``select``. If
        this is a sequence of sequences, then this transformer will shuffle
        groups that have labels that match ``select`` at any level of their
        multi-index.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """

    def __init__(self, select=None, groups=None, group_names=None, random_state=None):
        self.select = select
        self.groups = groups
        self.group_names = group_names
        self.random_state = random_state

    def transform(self, X, y=None):
        """Transform the input data, removing the unwanted groups.

        Parameters
        ----------
        X : numpy.ndarray
            The feature matrix
        """
        X = check_array(
            X, copy=True, dtype=[np.float32, np.float64, int], force_all_finite=False
        )
        groups = check_groups(groups=self.groups_, X=X, allow_overlap=True)
        if self.select_ is None:
            return X

        generator = check_random_state(self.random_state)
        idx = np.concatenate([groups[e] for e in self.select_])
        shuffle_view = X[:, idx]
        shuffle_view = util_shuffle(shuffle_view, random_state=generator)
        X[:, idx] = shuffle_view
        return X

    def fit(self, X=None, y=None):
        """Learn the groups and number of features from the input data."""
        X = check_array(
            X, copy=True, dtype=[np.float32, np.float64, int], force_all_finite=False
        )

        _, self.n_features_in_ = X.shape
        self.groups_ = check_groups(groups=self.groups, X=X, allow_overlap=True)
        self.group_names_ = _check_group_names(self.groups, self.group_names)
        self.select_ = _check_select(self.select, self.group_names_)

        return self

    def _more_tags(self):  # pylint: disable=no-self-use
        return {"allow_nan": True, "multilabel": True, "multioutput": True}
