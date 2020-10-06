"""Generate samples of synthetic data sets."""
import numpy as np

from sklearn.datasets import make_classification, make_regression
from sklearn.utils import check_random_state
from sklearn.utils import shuffle as util_shuffle

__all__ = ["make_group_classification", "make_group_regression"]


def make_group_classification(
    n_samples=100,
    n_groups=20,
    n_informative_groups=2,
    n_features_per_group=20,
    n_informative_per_group=2,
    n_redundant_per_group=2,
    n_repeated_per_group=0,
    n_classes=2,
    n_clusters_per_class=2,
    weights=None,
    flip_y=0.01,
    class_sep=1.0,
    hypercube=True,
    shift=0.0,
    scale=1.0,
    shuffle=True,
    useful_indices=False,
    random_state=None,
):
    """Generate a random n-class sparse group classification problem.

    This initially creates clusters of points normally distributed (std=1)
    about vertices of an ``n_informative``-dimensional hypercube with sides of
    length ``2*class_sep`` and assigns an equal number of clusters to each
    class. It introduces interdependence between these features and adds
    various types of further noise to the data.

    Prior to shuffling, ``X`` stacks a number of these primary "informative"
    features, "redundant" linear combinations of these, "repeated" duplicates
    of sampled features, and arbitrary noise for and remaining features.
    This method uses sklearn.datasets.make_classification to construct a
    giant unshuffled classification problem of size
    ``n_groups * n_features_per_group`` and then distributes the returned
    features to each group. It then optionally shuffles each group.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples.

    n_groups : int, optional (default=10)
        The number of feature groups.

    n_informative_groups : int, optional (default=2)
        The total number of informative groups. All other groups will be
        just noise.

    n_features_per_group : int, optional (default=20)
        The total number of features_per_group. These comprise `n_informative`
        informative features, `n_redundant` redundant features, `n_repeated`
        duplicated features and `n_features-n_informative-n_redundant-
        n_repeated` useless features drawn at random.

    n_informative_per_group : int, optional (default=2)
        The number of informative features_per_group. Each class is composed
        of a number of gaussian clusters each located around the vertices of a
        hypercube in a subspace of dimension `n_informative_per_group`. For
        each cluster, informative features are drawn independently from
        N(0, 1) and then randomly linearly combined within each cluster in
        order to add covariance. The clusters are then placed on the vertices
        of the hypercube.

    n_redundant_per_group : int, optional (default=2)
        The number of redundant features per group. These features are
        generated as random linear combinations of the informative features.

    n_repeated_per_group : int, optional (default=0)
        The number of duplicated features per group, drawn randomly from the
        informative and the redundant features.

    n_classes : int, optional (default=2)
        The number of classes (or labels) of the classification problem.

    n_clusters_per_class : int, optional (default=2)
        The number of clusters per class.

    weights : list of floats or None (default=None)
        The proportions of samples assigned to each class. If None, then
        classes are balanced. Note that if `len(weights) == n_classes - 1`,
        then the last class weight is automatically inferred.
        More than `n_samples` samples may be returned if the sum of `weights`
        exceeds 1.

    flip_y : float, optional (default=0.01)
        The fraction of samples whose class are randomly exchanged. Larger
        values introduce noise in the labels and make the classification
        task harder.

    class_sep : float, optional (default=1.0)
        The factor multiplying the hypercube size.  Larger values spread
        out the clusters/classes and make the classification task easier.

    hypercube : boolean, optional (default=True)
        If True, the clusters are put on the vertices of a hypercube. If
        False, the clusters are put on the vertices of a random polytope.

    shift : float, array of shape [n_features] or None, optional (default=0.0)
        Shift features by the specified value. If None, then features
        are shifted by a random value drawn in [-class_sep, class_sep].

    scale : float, array of shape [n_features] or None, optional (default=1.0)
        Multiply features by the specified value. If None, then features
        are scaled by a random value drawn in [1, 100]. Note that scaling
        happens after shifting.

    shuffle : boolean, optional (default=True)
        Shuffle the samples and the features.

    useful_indices : boolean, optional (default=False)
        If True, a boolean array indicating useful features is returned

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The generated samples.

    y : array of shape [n_samples]
        The integer labels for class membership of each sample.

    groups : list of arrays
        Each element is an array of feature indices that belong to that group

    indices : array of shape [n_features]
        A boolean array indicating which features are useful. Returned only
        if `useful_indices` is True.

    Notes
    -----
    The algorithm is adapted from Guyon [1] and was designed to generate
    the "Madelon" dataset.

    References
    ----------
    .. [1] I. Guyon, "Design of experiments for the NIPS 2003 variable
           selection benchmark", 2003.

    See Also
    --------
    sklearn.datasets.make_classification: non-group-sparse version
    sklearn.datasets.make_blobs: simplified variant
    sklearn.datasets.make_multilabel_classification: unrelated generator for multilabel tasks
    """
    generator = check_random_state(random_state)

    total_features = n_groups * n_features_per_group
    total_informative = n_informative_groups * n_informative_per_group
    total_redundant = n_informative_groups * n_redundant_per_group
    total_repeated = n_informative_groups * n_repeated_per_group

    # Count features, clusters and samples
    if (
        n_informative_per_group + n_redundant_per_group + n_repeated_per_group
        > n_features_per_group
    ):
        raise ValueError(
            "Number of informative, redundant and repeated features per group"
            " must sum to less than the number of total features per group."
        )

    # Generate a big classification problem for the total number of features
    # The `shuffle` argument is False so that the feature matrix X has
    # features stacked in the order: informative, redundant, repeated, useless
    X, y = make_classification(
        n_samples=n_samples,
        n_features=total_features,
        n_informative=total_informative,
        n_redundant=total_redundant,
        n_repeated=total_repeated,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class,
        weights=weights,
        flip_y=flip_y,
        class_sep=class_sep,
        hypercube=hypercube,
        shift=shift,
        scale=scale,
        shuffle=False,
        random_state=generator,
    )

    total_useful = total_informative + total_redundant + total_repeated
    idx = np.arange(total_features) < total_useful

    # Evenly distribute the first `n_informative_groups * n_features_per_group`
    # features into the first `n_informative_groups` groups
    n_info_grp_features = n_informative_groups * n_features_per_group
    idx_range = np.arange(n_info_grp_features)

    idx_map_consolidated_2_grouped = (
        np.concatenate(
            [np.arange(0, n_info_grp_features, n_informative_groups)]
            * n_informative_groups
        )
        + idx_range // n_features_per_group
    )

    X = np.concatenate(
        [X[:, idx_map_consolidated_2_grouped], X[:, n_info_grp_features:]], axis=1
    )

    if useful_indices:
        idx = np.concatenate(
            [idx[idx_map_consolidated_2_grouped], idx[n_info_grp_features:]]
        )

    if shuffle:
        # Randomly permute samples
        X, y = util_shuffle(X, y, random_state=generator)

        # Permute the groups, maintaining the order within them group_idx_map
        # maps feature indices to group indices. The group order is random
        # but all features in a single group are adjacent
        group_idx_map = np.concatenate(
            [
                np.ones(n_features_per_group, dtype=np.int32) * i
                for i in generator.choice(
                    np.arange(n_groups), size=n_groups, replace=False
                )
            ]
        )

        permute_group_map = (
            np.concatenate(
                [
                    generator.choice(
                        np.arange(n_features_per_group),
                        size=n_features_per_group,
                        replace=False,
                    )
                    for _ in range(n_groups)
                ]
            )
            + group_idx_map * n_features_per_group
        )

        X = X[:, permute_group_map]

        if useful_indices:
            idx = idx[permute_group_map]
    else:
        group_idx_map = np.concatenate(
            [np.ones(n_features_per_group, dtype=np.int32) * i for i in range(n_groups)]
        )

    groups = [np.where(group_idx_map == idx)[0] for idx in range(n_groups)]

    X = np.ascontiguousarray(X)
    if useful_indices:
        return X, y, groups, idx
    else:
        return X, y, groups


def make_group_regression(
    n_samples=100,
    n_groups=20,
    n_informative_groups=5,
    n_features_per_group=20,
    n_informative_per_group=5,
    effective_rank=None,
    noise=0.0,
    shift=0.0,
    scale=1.0,
    shuffle=False,
    coef=False,
    random_state=None,
):
    """Generate a sparse group regression problem.

    Prior to shuffling, ``X`` stacks a number of these primary "informative"
    features, and arbitrary noise for and remaining features.
    This method uses sklearn.datasets.make_regression to construct a
    giant unshuffled regression problem of size
    ``n_groups * n_features_per_group`` and then distributes the returned
    features to each group. It then optionally shuffles each group.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples.

    n_groups : int, optional (default=10)
        The number of feature groups.

    n_informative_groups : int, optional (default=2)
        The total number of informative groups. All other groups will be
        just noise.

    n_features_per_group : int, optional (default=20)
        The total number of features_per_group. These comprise `n_informative`
        informative features, and `n_features-n_informative` useless
        features drawn at random.

    n_informative_per_group : int, optional (default=2)
        The number of informative features_per_group that have a
        non-zero regression coefficient.

    effective_rank : int or None, optional (default=None)
        If not None, provides the number of singular vectors to explain the
        input data.

    noise : float, optional (default=0.0)
         The standard deviation of the gaussian noise applied to the output.

    shuffle : boolean, optional (default=False)
        Shuffle the samples and the features.

    coef : boolean, optional (default=False)
        If True, returns coefficient values used to generate samples via
        sklearn.datasets.make_regression.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The generated samples.

    y : array of shape [n_samples]
        The integer labels for class membership of each sample.

    groups : list of arrays
        Each element is an array of feature indices that belong to that group

    coef : array of shape [n_features]
        A numpy array containing true regression coefficient values. Returned only if `coef` is True.

    See Also
    --------
    sklearn.datasets.make_regression: non-group-sparse version
    """
    generator = check_random_state(random_state)

    total_features = n_groups * n_features_per_group
    total_informative = n_informative_groups * n_informative_per_group

    if coef:
        X, y, reg_coefs = make_regression(
            n_samples=n_samples,
            n_features=total_features,
            n_informative=total_informative,
            effective_rank=effective_rank,
            bias=0.0,
            noise=noise,
            shuffle=False,
            coef=True,
            random_state=generator,
        )
    else:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=total_features,
            n_informative=total_informative,
            effective_rank=effective_rank,
            bias=0.0,
            noise=noise,
            shuffle=False,
            coef=False,
            random_state=generator,
        )

    # Evenly distribute the first `n_informative_groups * n_features_per_group`
    # features into the first `n_informative_groups` groups
    n_info_grp_features = n_informative_groups * n_features_per_group
    idx_range = np.arange(n_info_grp_features)

    idx_map_consolidated_2_grouped = (
        np.concatenate(
            [np.arange(0, n_info_grp_features, n_informative_groups)]
            * n_informative_groups
        )
        + idx_range // n_features_per_group
    )

    X = np.concatenate(
        [X[:, idx_map_consolidated_2_grouped], X[:, n_info_grp_features:]], axis=1
    )

    group_idx_map = np.concatenate(
        [np.ones(n_features_per_group, dtype=np.int32) * i for i in range(n_groups)]
    )

    if coef:
        reg_coefs = np.concatenate(
            [reg_coefs[idx_map_consolidated_2_grouped], reg_coefs[n_info_grp_features:]]
        )

    # Randomly permute samples and features
    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

        indices = np.arange(total_features)
        generator.shuffle(indices)
        X[:, :] = X[:, indices]
        group_idx_map = group_idx_map[indices]
        if coef:
            reg_coefs = reg_coefs[indices]

    X = np.ascontiguousarray(X)
    groups = [np.where(group_idx_map == idx)[0] for idx in range(n_groups)]
    if coef:
        return X, y, groups, reg_coefs
    else:
        return X, y, groups
