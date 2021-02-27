"""Transform feature matrices with grouped covariates."""
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

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


def _check_group_subselection(selection, group_names_, var_name):
    missing_grp_names_msg = f"if {var_name} is a string, you must provide group_names"
    if selection is None:
        selection_ = selection
    elif isiterable(selection) and all(isinstance(e, int) for e in selection):
        selection_ = np.array(selection)
    elif isinstance(selection, int):
        selection_ = np.array([selection])
    elif isinstance(selection, str):
        if group_names_ is None:
            raise ValueError(missing_grp_names_msg)
        mask = np.array(set([selection]) <= group_names_, dtype=bool)
        selection_ = np.where(mask)[0]
    elif isiterable(selection) and all(isinstance(e, str) for e in selection):
        if group_names_ is None:
            raise ValueError(missing_grp_names_msg)

        mask = np.zeros_like(group_names_, dtype=bool)
        for label in selection:
            mask = np.logical_or(mask, set([label]) <= group_names_)

        selection_ = np.where(mask)[0]
    else:
        raise ValueError(
            f"{var_name} must be an int, string or sequence of ints or "
            f"strings; got {selection} instead."
        )

    return selection_


class GroupExtractor(BaseEstimator, TransformerMixin):
    """An sklearn-compatible group extractor.

    Given a sequence of all group indices and a subsequence of desired
    group indices, this transformer returns the columns of the feature
    matrix, `X`, that are in the desired subgroups.

    Parameters
    ----------
    extract : numpy.ndarray, int, or str, optional
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
        this transformer will extract groups whose names match ``extract``. If
        this is a sequence of sequences, then this transformer will extract
        groups that have labels that match ``extract`` at any level of their
        multi-index.

    copy_X : bool, default=False
        if ``True``, X will be copied; else, ``transform`` may return a view
    """

    def __init__(self, extract=None, groups=None, group_names=None, copy_X=False):
        self.extract = extract
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
        if self.extract_ is None:
            return X

        idx = np.concatenate([groups[e] for e in self.extract_])
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
        self.extract_ = _check_group_subselection(
            self.extract, self.group_names_, "extract"
        )

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
    remove : numpy.ndarray, int, or str, optional
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
        this transformer will remove groups whose names match ``remove``. If
        this is a sequence of sequences, then this transformer will remove
        groups that have labels that match ``remove`` at any level of their
        multi-index.

    copy_X : bool, default=False
        if ``True``, X will be copied; else, ``transform`` may return a view
    """

    def __init__(self, remove=None, groups=None, group_names=None, copy_X=False):
        self.remove = remove
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
        if self.remove_ is None:
            return X

        idx = np.concatenate(
            [grp for idx, grp in enumerate(groups) if idx not in self.remove_]
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
        self.remove_ = _check_group_subselection(
            self.remove, self.group_names_, "remove"
        )

        return self

    def _more_tags(self):  # pylint: disable=no-self-use
        return {"allow_nan": True, "multilabel": True, "multioutput": True}
