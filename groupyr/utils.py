"""Utility functions for SGL-based estimators."""
import numpy as np


def check_groups(groups, X, allow_overlap=False, fit_intercept=False, kwarg_name="X"):
    """Validate group indices.

    Verify that all features in ``X`` are accounted for in groups,
    that all groups refer to features that actually exist in ``X``,
    and, if ``allow_overlap=False``, that all groups are distinct.

    Parameters
    ----------
    groups : list of numpy.ndarray
        list of arrays of non-overlapping indices for each group. For
        example, if nine features are grouped into equal contiguous groups of
        three, then groups would be ``[array([0, 1, 2]), array([3, 4, 5]),
        array([6, 7, 8])]``. If the feature matrix contains a bias or
        intercept feature, do not include it as a group. If None, all
        features will belong to one group.

    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The training input samples. If ``X`` includes a bias or intercept
        feature, it must be in the last column and ``fit_intercept`` should
        be ``True``.

    allow_overlap : bool, default=False
        If True, allow groups to overlap. i.e. each feature may belong to
        multiple groups

    fit_intercept : bool, default=False
        If True, assume that the last column of the feature matrix
        corresponds to the bias or intercept.

    kwarg_name : str, default="X"
        The keyword argument name used to customize error messages. Defaults to
        "X" as this function is most commonly used to check feature matrices
        during fit, transform, and predict functions.

    Returns
    -------
    groups : list of numpy.ndarray
        The validated groups.
    """
    n_features = X.shape[-1]

    if fit_intercept:
        n_features -= 1

    if groups is None:
        # If no groups provided, put all features in one group
        return [np.arange(n_features)]

    all_indices = np.concatenate(groups)

    if set(all_indices) < set(range(n_features)):
        raise ValueError(
            "Some features are unaccounted for in groups; Columns "
            "{0} are absent from groups.".format(
                set(range(n_features)) - set(all_indices)
            )
        )

    if set(all_indices) > set(range(n_features)):
        raise ValueError(
            "There are feature indices in groups that exceed the dimensions "
            "of {0}; {0} has {1} features but groups refers to indices {2}".format(
                kwarg_name, n_features, set(all_indices) - set(range(n_features))
            )
        )

    if not allow_overlap:
        _, counts = np.unique(all_indices, return_counts=True)
        if set(counts) != {1}:
            raise ValueError("Overlapping groups detected.")

    return tuple(groups)
