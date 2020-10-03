import numpy as np

from joblib import Parallel
from tqdm.auto import tqdm


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
