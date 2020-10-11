"""Define custom proximal operators for use with copt package."""
from __future__ import absolute_import, division, print_function

import numpy as np

from itertools import compress

__all__ = ["SparseGroupL1"]


def _soft_threshold(z, alpha):
    r"""Apply the element-wise soft thresholding operator.

    The soft-thresholding operator is

    .. math::
        S(z_i, alpha) = \begin{cases}
            z_i + alpha & z_i < -T \\
            0 & -alpha \le z_i \le alpha \\
            z_i - alpha & alpha < z_i
        \end{cases}
        (
        \text{prox}_{\sigma P_2} \circ \text{prox}_{\sigma P_1}
        \right)(u)


    Parameters
    ----------
    z : array-like
        Input array

    alpha : float
        threshold value

    Returns
    -------
    np.ndarray
        Element-wise soft-thresholded array
    """  # noqa: W605
    return np.fmax(z - alpha, 0) - np.fmax(-z - alpha, 0)


class SparseGroupL1(object):
    r"""Sparse group lasso penalty class for use with openopt/copt package.

    Implements the sparse group lasso penalty [1]_

    .. math::
        (1 - \rho) * \alpha \displaystyle \sum_{g \in G} || \beta_g ||_2
        + \rho * \alpha || \beta ||_1

    where :math:`G` is a partition of the features into groups.

    Parameters
    ----------
    l1_ratio : float
        Combination between group lasso and lasso. l1_ratio = 0 gives the
        group lasso and l1_ratio = 1 gives the lasso. ``l1_ratio``
        corresponds to `\rho` in the equation above

    alpha : float
        Regularization parameter, overall strength of regularization.

    groups : list of numpy.ndarray
        list of arrays of non-overlapping indices for each group. For
        example, if nine features are grouped into equal contiguous groups of
        three, then groups would be ``[array([0, 1, 2]), array([3, 4, 5]),
        array([6, 7, 8])]``. If the feature matrix contains a bias or
        intercept feature, do not include it as a group.

    scale_l2_by : ["group_length", None], default="group_length"
        Scaling technique for the group-wise L2 penalty.
        By default, ``scale_l2_by="group_length`` and the L2 penalty is
        scaled by the square root of the group length so that each variable
        has the same effect on the penalty. This may not be appropriate for
        one-hot encoded features and ``scale_l2_by=None`` would be more
        appropriate for that case. ``scale_l2_by=None`` will also reproduce
        ElasticNet results when all features belong to one group.

    bias_index : int or None, default=None
        If None, regularize all coefficients. Otherwise, this is the index
        of the bias (i.e. intercept) feature, which should not be regularized.
        Exclude this index from the penalty term. And force the proximal
        operator for this index to return the result of the identity function.

    References
    ----------
    .. [1]  Noah Simon, Jerome Friedman, Trevor Hastie & Robert Tibshirani,
        "A Sparse-Group Lasso," Journal of Computational and Graphical
        Statistics, vol. 22:2, pp. 231-245, 2012
        DOI: 10.1080/10618600.2012.681250
    """  # noqa: W605

    def __init__(
        self, l1_ratio, alpha, groups, bias_index=None, scale_l2_by="group_length"
    ):
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.groups = groups
        self.feature2group_map = np.concatenate(
            [
                [grp_idx] * feature_indices.size
                for grp_idx, feature_indices in enumerate(groups)
            ]
        )

        n_features = np.unique(np.concatenate(groups)).size
        if bias_index is not None:
            n_features += 1

        self.group_masks = np.full((len(groups), n_features), 0)
        for row, grp in zip(self.group_masks, groups):
            row[grp] = 1

        if scale_l2_by == "group_length":
            self.group_scale = np.sqrt([grp.size for grp in groups])
        else:
            self.group_scale = np.array([1.0 for grp in groups])

        self.bias_index = bias_index

    def __call__(self, x):
        """Return the Sparse L1 penalty."""
        penalty = (
            (1.0 - self.l1_ratio)
            * self.alpha
            * self.group_scale
            * np.sum([np.linalg.norm(x[g]) for g in self.groups])
        )

        ind = np.ones(len(x), bool)
        if self.bias_index is not None:
            ind[self.bias_index] = False

        penalty += self.l1_ratio * self.alpha * np.abs(x[ind]).sum()
        return penalty

    def prox(self, x, step_size):
        r"""Return the proximal operator of the sparse group lasso penalty.

        For the sparse group lasso, we can decompose the penalty into

        .. math::
            P(\beta) = P_1(\beta) + P_2(\beta)

        where :math:`P_2 = \rho \alpha || \beta ||_1` is the lasso
        penalty and :math:`P_1 = (1 - \rho) \alpha \displaystyle
        \sum_{g \in G} || \beta_g ||_2` is the group lasso penalty.

        Then the proximal operator is given by

        .. math::
            \text{prox}_{\sigma P_1 + \sigma P_2} (u) = \left(
            \text{prox}_{\sigma P_2} \circ \text{prox}_{\sigma P_1}
            \right)(u)

        where :math:`sigma` is a step size

        Parameters
        ----------
        x : np.ndarray
            Argument for proximal operator.

        step_size : float
            Step size for proximal operator

        Returns
        -------
        np.ndarray
            proximal operator of sparse group lasso penalty evaluated on
            input `x` with step size `step_size`
        """  # noqa: W605
        l1_prox = _soft_threshold(x, self.l1_ratio * self.alpha * step_size)
        out = l1_prox.copy()

        if self.bias_index is not None:
            out[self.bias_index] = x[self.bias_index]

        norms = np.sqrt(self.group_masks.dot(l1_prox ** 2)) / self.group_scale

        norm_mask = norms > (1.0 - self.l1_ratio) * self.alpha * step_size
        all_norm = all(norm_mask)
        if not all_norm:
            idx_true = np.array([], dtype=int)
        else:
            idx_true = np.concatenate(list(compress(self.groups, norm_mask)))

        if all_norm:
            idx_false = np.array([], dtype=int)
        else:
            idx_false = np.concatenate(
                list(compress(self.groups, np.logical_not(norm_mask)))
            )

        out[idx_true] -= (
            step_size
            * (1.0 - self.l1_ratio)
            * self.alpha
            * out[idx_true]
            / norms[self.feature2group_map][idx_true]
        )
        out[idx_false] = 0.0

        return out
