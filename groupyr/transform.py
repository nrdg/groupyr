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


class GroupExtractor(BaseEstimator, TransformerMixin):
    """An sklearn-compatible group extractor.

    Given a sequence of all group indices and a subsequence of desired
    group indices, this transformer returns the columns of the feature
    matrix, `X`, that are in the desired subgroups.

    Parameters
    ----------
    extract : numpy.ndarray or int, optional
        subsequence of desired groups to extract from feature matrix

    groups : list of numpy.ndarray
        list of arrays of non-overlapping indices for each group. For
        example, if nine features are grouped into equal contiguous groups of
        three, then groups would be ``[array([0, 1, 2]), array([3, 4, 5]),
        array([6, 7, 8])]``. If the feature matrix contains a bias or
        intercept feature, do not include it as a group. If None, all
        features will belong to one group.

    copy_X : bool, default=False
        if ``True``, X will be copied; else, ``transform`` may return a view
    """

    def __init__(self, extract=None, groups=None, copy_X=False):
        self.extract = extract
        self.groups = groups
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
        if self.extract is None:
            return X
        elif isiterable(self.extract) and all(isinstance(e, int) for e in self.extract):
            extract = np.array(self.extract)
        elif isinstance(self.extract, int):
            extract = np.array([self.extract])
        else:
            raise ValueError(
                "extract must be an int or sequence of ints; got "
                "{0} instead".format(self.extract)
            )

        idx = np.concatenate([groups[e] for e in extract])
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
    remove : numpy.ndarray or int, optional
        subsequence of desired groups to remove from feature matrix

    groups : list of numpy.ndarray
        list of arrays of non-overlapping indices for each group. For
        example, if nine features are grouped into equal contiguous groups of
        three, then groups would be ``[array([0, 1, 2]), array([3, 4, 5]),
        array([6, 7, 8])]``. If the feature matrix contains a bias or
        intercept feature, do not include it as a group. If None, all
        features will belong to one group.

    copy_X : bool, default=False
        if ``True``, X will be copied; else, ``transform`` may return a view
    """

    def __init__(self, remove=None, groups=None, copy_X=False):
        self.remove = remove
        self.groups = groups
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
        if self.remove is None:
            return X
        elif isiterable(self.remove) and all(isinstance(e, int) for e in self.remove):
            remove = np.array(self.remove)
        elif isinstance(self.remove, int):
            remove = np.array([self.remove])
        else:
            raise ValueError(
                "remove must be an int or sequence of ints; got "
                "{0} instead".format(self.remove)
            )

        idx = np.concatenate(
            [grp for idx, grp in enumerate(groups) if idx not in remove]
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
        return self

    def _more_tags(self):  # pylint: disable=no-self-use
        return {"allow_nan": True, "multilabel": True, "multioutput": True}
