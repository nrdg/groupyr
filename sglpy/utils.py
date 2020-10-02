from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple
from joblib import Parallel
from tqdm.auto import tqdm

__all__ = []


def registered(fn):
    __all__.append(fn.__name__)
    return fn


canonical_tract_names = [
    "Left Arcuate",
    "Left SLF",
    "Left Uncinate",
    "Left ILF",
    "Left IFOF",
    "Left Cingulum Hippocampus",
    "Left Thalamic Radiation",
    "Left Corticospinal",
    "Left Cingulum Cingulate",
    "Callosum Forceps Minor",
    "Callosum Forceps Major",
    "Right Cingulum Cingulate",
    "Right Corticospinal",
    "Right Thalamic Radiation",
    "Right Cingulum Hippocampus",
    "Right IFOF",
    "Right ILF",
    "Right Uncinate",
    "Right SLF",
    "Right Arcuate",
]


@registered
def ecdf(data, reverse=False):
    """Compute ECDF for a one-dimensional array of measurements.

    Parameters
    ----------
    data : np.ndarray
        one-dimensional array of measurements

    reverse : bool, default=False
        If True, reverse the sorted data so that ecdf runs from top-left
        to bottom-right.

    Returns
    -------
    collections.namedtuple
        namedtuple with fields:
        x - sorted data
        y - cumulative probability
    """
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)
    if reverse:
        x = np.flip(x)

    # y-data for the ECDF: y
    y = np.arange(1, n + 1) / n

    ECDF = namedtuple("ECDF", "x y")
    return ECDF(x=x, y=y)


@registered
def plot_ecdf(data, reverse=False):
    """Plot ECDF for a one-dimensional array of measurements.

    Parameters
    ----------
    data : np.ndarray
        one-dimensional array of measurements

    reverse : bool, default=False
        If True, reverse the sorted data so that ecdf runs from top-left
        to bottom-right.
    """
    cdf = ecdf(data, reverse=reverse)

    # Generate plot
    plt.plot(cdf.x, cdf.y, marker=".", linestyle="none")

    # Make the margins nice
    plt.margins(0.02)

    # Label the axes
    plt.xlabel("data")
    plt.ylabel("ECDF")

    # Display the plot
    plt.show()


def check_groups(groups, X, allow_overlap=False, fit_intercept=True):
    """Validate group indices"""
    _, n_features = X.shape

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
            "of X; X has {0} features but groups refers to indices {1}".format(
                n_features, set(all_indices) - set(range(n_features))
            )
        )

    if not allow_overlap:
        _, counts = np.unique(all_indices, return_counts=True)
        if set(counts) != {1}:
            raise ValueError("Overlapping groups detected.")

    return tuple(groups)


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, desc=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        self._desc = desc
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(
            disable=not self._use_tqdm, desc=self._desc, total=self._total
        ) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
