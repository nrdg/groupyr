"""Perform group-wise functional PCA of feature matrices with grouped covariates."""
import inspect
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.utils.multiclass import type_of_target
from sklearn.utils import check_array, check_scalar

try:
    from skfda import FDataGrid
    from skfda.representation.basis import Basis
    from skfda.representation.basis import BSpline, Monomial, Constant, Fourier
    from skfda.preprocessing.dim_reduction.projection import FPCA

    HAS_SKFDA = True
except ImportError:  # pragma: no cover
    HAS_SKFDA = False

from .utils import check_groups


class GroupPCA(BaseEstimator, TransformerMixin):
    """An sklearn-compatible grouped PCA transformer.

    Given a feature matrix `X` and grouping information,
    this transformer returns the groupwise PCA matrix.

    Parameters
    ----------
    n_components : int, float or 'mle', default=None
        Number of components to keep for each group.
        if n_components is not set all components are kept::

            n_components == min(n_samples, n_features_in_group)

        If ``n_components == 'mle'`` and ``svd_solver == 'full'``, Minka's
        MLE is used to guess the dimension. Use of ``n_components == 'mle'``
        will interpret ``svd_solver == 'auto'`` as ``svd_solver == 'full'``.

        If ``0 < n_components < 1`` and ``svd_solver == 'full'``, select the
        number of components such that the amount of variance that needs to be
        explained is greater than the percentage specified by n_components.

        If ``svd_solver == 'arpack'``, the number of components must be
        strictly less than the minimum of n_features_in_group and n_samples.

        Hence, the None case results in::

            n_components == min(n_samples, n_features_in_group) - 1

    whiten : bool, default=False
        When True (False by default) the `components_` vectors are multiplied
        by the square root of n_samples and then divided by the singular values
        to ensure uncorrelated outputs with unit component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometimes
        improve the predictive accuracy of the downstream estimators by
        making their data respect some hard-wired assumptions.

    svd_solver : {'auto', 'full', 'arpack', 'randomized'}, default='auto'
        If auto :
            The solver is selected by a default policy based on `X.shape` and
            `n_components`: if the input data is larger than 500x500 and the
            number of components to extract is lower than 80% of the smallest
            dimension of the data, then the more efficient 'randomized'
            method is enabled. Otherwise the exact full SVD is computed and
            optionally truncated afterwards.
        If full :
            run exact full SVD calling the standard LAPACK solver via
            `scipy.linalg.svd` and select the components by postprocessing
        If arpack :
            run SVD truncated to n_components calling ARPACK solver via
            `scipy.sparse.linalg.svds`. It requires strictly
            0 < n_components < min(X.shape)
        If randomized :
            run randomized SVD by the method of Halko et al.

    tol : float, default=0.0
        Tolerance for singular values computed by svd_solver == 'arpack'.
        Must be of range [0.0, infinity).

    iterated_power : int or 'auto', default='auto'
        Number of iterations for the power method computed by
        svd_solver == 'randomized'.
        Must be of range [0, infinity).

    random_state : int, RandomState instance or None, default=None
        Used when the 'arpack' or 'randomized' solvers are used. Pass an int
        for reproducible results across multiple function calls.

    groups : numpy.ndarray or int, optional
        all group indices for feature matrix

    exclude_groups : sequence of int or int, optional
        a sequence of group indices (or a single group index) to exclude from
        PCA transformatiton. These groups will be left untouched and
        transferred to the output matrix.

    Attributes
    ----------
    components_ : list of ndarray of shape (n_components, n_features) for each group
        List of each group's principal axes in feature space, representing
        the directions of maximum variance in the data. The components are
        sorted by ``explained_variance_``.

    explained_variance_ : list of ndarray of shape (n_components,) for each group
        List of each group's amount of variance explained by each of the
        selected components. Equal to n_components largest eigenvalues of the
        covariance matrix of X.

    explained_variance_ratio_ : list of ndarray of shape (n_components,) for each group
        Percentage of variance explained by each of the selected components.
        If ``n_components`` is not set then all components are stored and the
        sum of the ratios is equal to 1.0.

    singular_values_ : list of ndarray of shape (n_components,)
        The singular values corresponding to each of the selected components.
        The singular values are equal to the 2-norms of the ``n_components``
        variables in the lower-dimensional space.

    mean_ : list of ndarray of shape (n_features,)
        Per-feature empirical mean, estimated from the training set.
        Equal to `X.mean(axis=0)`.

    n_components_ : list of int
        The estimated number of components. When n_components is set
        to 'mle' or a number between 0 and 1 (with svd_solver == 'full') this
        number is estimated from input data. Otherwise it equals the parameter
        n_components, or the lesser value of n_features_per_group and n_samples
        if n_components is None.

    noise_variance_ : float
        The estimated noise covariance following the Probabilistic PCA model
        from Tipping and Bishop 1999. See "Pattern Recognition and
        Machine Learning" by C. Bishop, 12.2.1 p. 574 or
        http://www.miketipping.com/papers/met-mppca.pdf. It is required to
        compute the estimated data covariance and score samples.

        Equal to the average of (min(n_features, n_samples) - n_components)
        smallest eigenvalues of the covariance matrix of X.

    groups_in_ : list of numpy.ndarray
        explicit group indices created from the potentially more implicit
        ``groups`` input parameter

    groups_out_ : list of numpy.ndarray
        group indices for the transformed features

    n_features_in_ : int
        number of input features

    n_features_out_ : int
        number of transformed features

    n_samples_ : int
        Number of samples in the training data.

    """

    def __init__(
        self,
        n_components=None,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        random_state=None,
        groups=None,
        exclude_groups=None,
    ):
        self.n_components = n_components
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state
        self.groups = groups
        self.exclude_groups = exclude_groups

    def _check_exclude_groups(self):
        if self.exclude_groups is None:
            return []
        elif isinstance(self.exclude_groups, int):
            return [self.exclude_groups]
        elif all(isinstance(e, int) for e in self.exclude_groups):
            return list(self.exclude_groups)
        else:
            raise TypeError(
                f"exclude_groups must be and int or sequence of ints. "
                f"Got {self.exclude_groups} instead."
            )

    def fit(self, X, y=None):
        """Fit the GroupPCA model with X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = check_array(
            X, copy=True, dtype=[np.float32, np.float64, int], force_all_finite=True
        )

        self.n_samples_, self.n_features_in_ = X.shape
        self.groups_in_ = check_groups(groups=self.groups, X=X, allow_overlap=False)

        exclude_grp_idx = self._check_exclude_groups()

        self.pca_models_ = [
            PCA(
                n_components=self.n_components,
                copy=True,
                whiten=self.whiten,
                svd_solver=self.svd_solver,
                tol=self.tol,
                iterated_power=self.iterated_power,
                random_state=self.random_state,
            )
            for grp in self.groups_in_
        ]
        self.components_ = [None for grp in self.groups_in_]
        self.explained_variance_ = [None for grp in self.groups_in_]
        self.explained_variance_ratio_ = [None for grp in self.groups_in_]
        self.singular_values_ = [None for grp in self.groups_in_]
        self.mean_ = [None for grp in self.groups_in_]
        self.n_components_ = [None for grp in self.groups_in_]
        self.noise_variance_ = [None for grp in self.groups_in_]
        self.groups_out_ = []

        feature_start_idx = 0
        for idx, grp in enumerate(self.groups_in_):
            if idx not in exclude_grp_idx:
                group_X = np.copy(X[:, grp])
                self.pca_models_[idx].fit(group_X)
                self.components_[idx] = self.pca_models_[idx].components_
                self.explained_variance_[idx] = self.pca_models_[
                    idx
                ].explained_variance_
                self.explained_variance_ratio_[idx] = self.pca_models_[
                    idx
                ].explained_variance_ratio_
                self.singular_values_[idx] = self.pca_models_[idx].singular_values_
                self.mean_[idx] = self.pca_models_[idx].mean_
                self.n_components_[idx] = self.pca_models_[idx].n_components_
                self.noise_variance_[idx] = self.pca_models_[idx].noise_variance_
                self.groups_out_.append(
                    np.arange(
                        feature_start_idx,
                        feature_start_idx + self.pca_models_[idx].n_components_,
                    )
                )
                feature_start_idx += self.pca_models_[idx].n_components_
            else:
                self.pca_models_[idx] = None
                # self.components_ is already set to None above
                self.groups_out_.append(
                    np.arange(feature_start_idx, feature_start_idx + len(grp))
                )
                feature_start_idx += len(grp)

        self.n_features_out_ = np.sum([len(grp) for grp in self.groups_out_])

        return self

    def transform(self, X, y=None):
        """Apply dimensionality reduction to X.

        X is projected on the principal components previously extracted
        from a training set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        X = check_array(
            X, copy=True, dtype=[np.float32, np.float64, int], force_all_finite=True
        )
        groups = check_groups(groups=self.groups_in_, X=X, allow_overlap=False)

        X_out = []

        for idx, grp in enumerate(groups):
            if self.pca_models_[idx] is not None:
                X_out.append(self.pca_models_[idx].transform(X[:, grp]))
            else:
                X_out.append(X[:, grp])

        return np.hstack(X_out)

    def inverse_transform(self, X):
        """Transform data back to its original space.

        In other words, return an input X_original whose transform would be X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            New data, where n_samples is the number of samples
            and n_components is the number of components.

        Returns
        -------
        X_original array-like, shape (n_samples, n_features)

        Notes
        -----
        If whitening is enabled, inverse_transform will compute the
        exact inverse operation, which includes reversing whitening.
        """
        X = check_array(
            X, copy=True, dtype=[np.float32, np.float64, int], force_all_finite=True
        )
        groups = check_groups(groups=self.groups_out_, X=X, allow_overlap=False)

        X_out = []

        for idx, grp in enumerate(groups):
            if self.pca_models_[idx] is not None:
                X_out.append(self.pca_models_[idx].inverse_transform(X[:, grp]))
            else:
                X_out.append(X[:, grp])

        return np.hstack(X_out)

    def _more_tags(self):  # pylint: disable=no-self-use
        return {
            "allow_nan": False,
            "multilabel": True,
            "multioutput": True,
            "_skip_test": True,
        }


def _get_score_func(y):
    target_type = type_of_target(y)
    if target_type == "continuous":
        return f_regression
    elif target_type in ["binary", "multiclass"]:
        return f_classif
    else:
        raise NotImplementedError(
            "The type of target is unsupported. It must be continuous, "
            "binary, or multiclass. Got {0} instead.".format(target_type)
        )


class GroupSDRMixin:
    """Mixin class for grouped supervised dimensionality reduction (SDR)."""

    def _compute_screening_mask(self, X, y):
        y = check_array(
            y,
            copy=True,
            dtype=[np.float32, np.float64, int],
            force_all_finite=True,
            ensure_2d=False,
        )
        if y.ndim != 1:
            raise NotImplementedError("y must be one-dimensional.")

        self.univariate_score_func_ = _get_score_func(y)
        self.groups_in_ = check_groups(groups=self.groups, X=X, allow_overlap=False)
        exclude_grp_idx = self._check_exclude_groups()

        check_scalar(
            self.theta,
            name="theta",
            target_type=(float, np.float64, np.float32, int),
            min_val=0.0,
            max_val=1.0,
        )
        if self.theta == 0.0:
            self.screening_mask_ = np.ones(X.shape[1])
            return

        screener = SelectKBest(self.univariate_score_func_)
        screening_masks = []
        for idx, grp in enumerate(self.groups_in_):
            if idx not in exclude_grp_idx:
                screener.fit(X[:, grp], y)
                univariate_coefs = np.abs(screener.scores_)

                # Create a mask where the univariate coefs are above the theta threshold
                if self.absolute_threshold:
                    mask = univariate_coefs >= self.theta
                else:
                    mask = univariate_coefs >= self.theta * np.max(univariate_coefs)

                # Find where mask transitions between 0 and 1
                m = mask.astype(np.float64)
                lower = m < 0.5
                higher = m > 0.5
                transition_indices = np.sort(
                    np.concatenate(
                        [
                            np.where(lower[:-1] & higher[1:])[0],
                            np.where(higher[:-1] & lower[1:])[0],
                        ]
                    )
                )

                # Create a "soft" mask where values above the theta threshold
                # are one and they gradually fall to zero outside
                if len(transition_indices) and self.smooth_screening:
                    dist_from_transition = np.min(
                        np.abs(np.arange(len(m)) - transition_indices[:, np.newaxis]),
                        axis=0,
                    )
                    dist_from_transition[mask] = 0
                    soft_mask = np.exp(-np.square(dist_from_transition) / 4)
                else:
                    soft_mask = np.copy(m)

                screening_masks.append(soft_mask)
            else:
                screening_masks.append(np.ones(len(grp)))

        self.screening_mask_ = np.concatenate(screening_masks)


class SupervisedGroupPCA(GroupPCA, GroupSDRMixin):
    """A grouped supervised PCA transformer.

    Given a feature matrix `X`, grouping information, and a supervision
    target this transformer returns the groupwise supervised PCA
    matrix.

    Parameters
    ----------
    n_components : int, float or 'mle', default=None
        Number of components to keep for each group.
        if n_components is not set all components are kept::

            n_components == min(n_samples, n_features_in_group)

        If ``n_components == 'mle'`` and ``svd_solver == 'full'``, Minka's
        MLE is used to guess the dimension. Use of ``n_components == 'mle'``
        will interpret ``svd_solver == 'auto'`` as ``svd_solver == 'full'``.

        If ``0 < n_components < 1`` and ``svd_solver == 'full'``, select the
        number of components such that the amount of variance that needs to be
        explained is greater than the percentage specified by n_components.

        If ``svd_solver == 'arpack'``, the number of components must be
        strictly less than the minimum of n_features_in_group and n_samples.

        Hence, the None case results in::

            n_components == min(n_samples, n_features_in_group) - 1

    theta : float, default=0.0
        the threshold for each features univariate regression coefficient. If
        the absolute value of an individual feature's univariate regression
        coefficient is below this threshold, its value for all samples is set
        to the mean feature value. Unless ``absolute_threshold`` is True,
        theta is scaled by the max absolute value of the regression
        coefficients in each group.

    absolute_threshold : boolean, default=False
        If True, treat ``theta`` as an absolute threshold. i.e. do not scale
        by the max absolute regression coefficient in each group.

    smooth_screening : bool, default=False
        If True, values below the `theta` threshold fall smoothly to the mean
        feature value. If False, they are set abruptly to the mean.

    whiten : bool, default=False
        When True (False by default) the `components_` vectors are multiplied
        by the square root of n_samples and then divided by the singular values
        to ensure uncorrelated outputs with unit component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometimes
        improve the predictive accuracy of the downstream estimators by
        making their data respect some hard-wired assumptions.

    svd_solver : {'auto', 'full', 'arpack', 'randomized'}, default='auto'
        If auto :
            The solver is selected by a default policy based on `X.shape` and
            `n_components`: if the input data is larger than 500x500 and the
            number of components to extract is lower than 80% of the smallest
            dimension of the data, then the more efficient 'randomized'
            method is enabled. Otherwise the exact full SVD is computed and
            optionally truncated afterwards.
        If full :
            run exact full SVD calling the standard LAPACK solver via
            `scipy.linalg.svd` and select the components by postprocessing
        If arpack :
            run SVD truncated to n_components calling ARPACK solver via
            `scipy.sparse.linalg.svds`. It requires strictly
            0 < n_components < min(X.shape)
        If randomized :
            run randomized SVD by the method of Halko et al.

    tol : float, default=0.0
        Tolerance for singular values computed by svd_solver == 'arpack'.
        Must be of range [0.0, infinity).

    iterated_power : int or 'auto', default='auto'
        Number of iterations for the power method computed by
        svd_solver == 'randomized'.
        Must be of range [0, infinity).

    random_state : int, RandomState instance or None, default=None
        Used when the 'arpack' or 'randomized' solvers are used. Pass an int
        for reproducible results across multiple function calls.

    groups : numpy.ndarray or int, optional
        all group indices for feature matrix

    exclude_groups : sequence of int or int, optional
        a sequence of group indices (or a single group index) to exclude from
        PCA transformatiton. These groups will be left untouched and
        transferred to the output matrix.

    Attributes
    ----------
    components_ : list of ndarray of shape (n_components, n_features) for each group
        List of each group's principal axes in feature space, representing
        the directions of maximum variance in the data. The components are
        sorted by ``explained_variance_``.

    explained_variance_ : list of ndarray of shape (n_components,) for each group
        List of each group's amount of variance explained by each of the
        selected components. Equal to n_components largest eigenvalues of the
        covariance matrix of X.

    explained_variance_ratio_ : list of ndarray of shape (n_components,) for each group
        Percentage of variance explained by each of the selected components.
        If ``n_components`` is not set then all components are stored and the
        sum of the ratios is equal to 1.0.

    singular_values_ : list of ndarray of shape (n_components,)
        The singular values corresponding to each of the selected components.
        The singular values are equal to the 2-norms of the ``n_components``
        variables in the lower-dimensional space.

    mean_ : list of ndarray of shape (n_features,)
        Per-feature empirical mean, estimated from the training set.
        Equal to `X.mean(axis=0)`.

    n_components_ : list of int
        The estimated number of components. When n_components is set
        to 'mle' or a number between 0 and 1 (with svd_solver == 'full') this
        number is estimated from input data. Otherwise it equals the parameter
        n_components, or the lesser value of n_features_per_group and n_samples
        if n_components is None.

    noise_variance_ : float
        The estimated noise covariance following the Probabilistic PCA model
        from Tipping and Bishop 1999. See "Pattern Recognition and
        Machine Learning" by C. Bishop, 12.2.1 p. 574 or
        http://www.miketipping.com/papers/met-mppca.pdf. It is required to
        compute the estimated data covariance and score samples.

        Equal to the average of (min(n_features, n_samples) - n_components)
        smallest eigenvalues of the covariance matrix of X.

    groups_in_ : list of numpy.ndarray
        explicit group indices created from the potentially more implicit
        ``groups`` input parameter

    groups_out_ : list of numpy.ndarray
        group indices for the transformed features

    n_features_in_ : int
        number of input features

    n_features_out_ : int
        number of transformed features

    n_samples_ : int
        Number of samples in the training data.

    """

    def __init__(
        self,
        n_components=None,
        theta=0.0,
        absolute_threshold=False,
        smooth_screening=False,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        random_state=None,
        groups=None,
        exclude_groups=None,
    ):
        self.n_components = n_components
        self.theta = theta
        self.absolute_threshold = absolute_threshold
        self.smooth_screening = smooth_screening
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state
        self.groups = groups
        self.exclude_groups = exclude_groups

    def fit(self, X, y):
        """Fit the GroupPCA model with X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._compute_screening_mask(X, y)
        return super().fit(X=X * self.screening_mask_)

    def transform(self, X, y=None):
        """Apply dimensionality reduction to X.

        X is projected on the principal components previously extracted
        from a training set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        return super().transform(X * self.screening_mask_)

    def _more_tags(self):  # pylint: disable=no-self-use
        return {
            "allow_nan": False,
            "multilabel": True,
            "multioutput": True,
            "_skip_test": True,
            "requires_y": True,
        }


def _check_skfda():
    if not HAS_SKFDA:
        raise ImportError(
            "To use functional data analysis in groupyr, you will need to have "
            "scikit-fda installed. You can do this by installing groupyr with "
            "`pip install groupyr[fda]`, or by separately installing scikit-fda "
            "with `pip install scikit-fda`."
        )


def _has_nbasis_kwarg(_basis):
    try:
        # Not all bases have an ``n_basis`` kwarg
        # If we just tried to instantiate the basis, then
        # we'd catch all TypeErrors raised by ``basis``.
        # Instead, we use ``inspect``
        inspect.signature(_basis).bind(n_basis=7)
        return True
    except TypeError:
        return False


def _check_basis(basis, arg_name, n_basis, domain_range=None):
    _check_skfda()
    allowed_bases = {
        "bspline": BSpline,
        "monomial": Monomial,
        "constant": Constant,
        "fourier": Fourier,
    }

    err_msg = (
        f"{arg_name} must be one of {list(allowed_bases.keys())} or be an "
        f"instance of a basis class from `skfda.representation.basis`."
    )

    if inspect.isclass(basis):
        if not issubclass(basis, Basis):
            raise ValueError(err_msg)

        if _has_nbasis_kwarg(basis):
            return basis(n_basis=n_basis, domain_range=domain_range)
        else:
            return basis(domain_range=domain_range)
    elif basis is not None:
        if basis.lower() not in allowed_bases.keys():
            raise ValueError(err_msg)

        if _has_nbasis_kwarg(allowed_bases[basis.lower()]):
            return allowed_bases[basis.lower()](
                n_basis=n_basis, domain_range=domain_range
            )
        else:
            return allowed_bases[basis.lower()](domain_range=domain_range)

    return None


def _get_group_fd(X, group_mask, basis, grid_points=None):
    _check_skfda()
    group_X = np.copy(X[:, group_mask])
    if grid_points is None:
        group_grid_points = np.arange(0, group_X.shape[1])
    else:
        group_grid_points = np.copy(grid_points[group_mask])
    fd = FDataGrid(group_X, group_grid_points)
    if basis is not None:
        fd = fd.to_basis(basis)
    return fd


class GroupFPCA(BaseEstimator, TransformerMixin):
    """An sklearn-compatible grouped functional PCA transformer.

    Given a feature matrix `X` and grouping information,
    this transformer returns the groupwise functional PCA
    matrix.

    Parameters
    ----------
    n_components : int, default=3
        number of principal components to obtain from functional
        principal component analysis.

    centering : bool, default=True
        if True then calculate the mean of the functional data object and
        center the data first.

    basis : str, instance of `skfda.representation.basis` class, optional
        the basis in which to represent each group's function before fPCA.

    n_basis : int, default=4
        the number of functions in the basis. Only used if ``basis`` is not
        None.

    basis_domain_range : tuple, optional
        a tuple of length 2 containing the initial and end values of the
        interval over which the basis can be evaluated.

    groups : numpy.ndarray or int, optional
        all group indices for feature matrix

    exclude_groups : sequence of int or int, optional
        a sequence of group indices (or a single group index) to exclude from
        fPCA transformatiton. These groups will be left untouched and
        transferred to the output matrix.

    Attributes
    ----------
    components_ : list of skfda.FData
        a list where each element contains a group's principal components
        in a basis representation.

    fitted_grid_points_ : numpy.ndarray or None
        The dicretization points for the input ``X`` matrix
        provided during ``fit``.

    fpca_models_ : list of fitted skfda.FPCA models
        a list where each element contains a group's fitted FPCA model

    groups_in_ : list of numpy.ndarray
        explicit group indices created from the potentially more implicit
        ``groups`` input parameter

    groups_out_ : list of numpy.ndarray
        group indices for the transformed features

    n_features_in_ : int
        number of input features

    n_features_out_ : int
        number of transformed features

    """

    def __init__(
        self,
        n_components=3,
        centering=True,
        basis=None,
        n_basis=4,
        basis_domain_range=None,
        groups=None,
        exclude_groups=None,
    ):
        self.n_components = n_components
        self.centering = centering
        self.basis = basis
        self.n_basis = n_basis
        self.basis_domain_range = basis_domain_range
        self.groups = groups
        self.exclude_groups = exclude_groups

    def _check_exclude_groups(self):
        if self.exclude_groups is None:
            return []
        elif isinstance(self.exclude_groups, int):
            return [self.exclude_groups]
        elif all(isinstance(e, int) for e in self.exclude_groups):
            return list(self.exclude_groups)
        else:
            raise TypeError(
                f"exclude_groups must be and int or sequence of ints. "
                f"Got {self.exclude_groups} instead."
            )

    def fit(self, X, y=None, grid_points=None):
        """Fit the fPCA transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.

        grid_points : numpy.ndarray, optional
            The points of dicretisation for the input ``X`` matrix. If None, this will use the
            grid_points supplied during the fit method.
        """
        _check_skfda()
        X = check_array(
            X, copy=True, dtype=[np.float32, np.float64, int], force_all_finite=True
        )
        basis = _check_basis(self.basis, "basis", self.n_basis, self.basis_domain_range)

        _, self.n_features_in_ = X.shape
        self.groups_in_ = check_groups(groups=self.groups, X=X, allow_overlap=False)

        if grid_points is None:
            self.fitted_grid_points_ = grid_points
        else:
            _ = check_groups(
                groups=self.groups_in_,
                X=grid_points,
                allow_overlap=False,
                kwarg_name="grid_points",
            )
            self.fitted_grid_points_ = np.array(grid_points)

        exclude_grp_idx = self._check_exclude_groups()

        self.fpca_models_ = [
            FPCA(n_components=self.n_components, centering=self.centering)
            for grp in self.groups_in_
        ]
        self.components_ = [None for grp in self.groups_in_]
        self.groups_out_ = []

        feature_start_idx = 0
        for idx, grp in enumerate(self.groups_in_):
            if idx not in exclude_grp_idx:
                fd = _get_group_fd(X, grp, basis, self.fitted_grid_points_)
                self.fpca_models_[idx].fit(fd)
                self.components_[idx] = self.fpca_models_[idx].components_
                self.groups_out_.append(
                    np.arange(feature_start_idx, feature_start_idx + self.n_components)
                )
                feature_start_idx += self.n_components
            else:
                self.fpca_models_[idx] = None
                # self.components_ is already set to None above
                self.groups_out_.append(
                    np.arange(feature_start_idx, feature_start_idx + len(grp))
                )
                feature_start_idx += len(grp)

        self.n_features_out_ = np.sum([len(grp) for grp in self.groups_out_])

        return self

    def transform(self, X, y=None, grid_points=None):
        """Transform the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            samples to transform.

        grid_points : numpy.ndarray, optional
            The points of dicretisation for the input ``X`` matrix. If None, this will use the
            grid_points supplied during the fit method.
        """
        X = check_array(
            X, copy=True, dtype=[np.float32, np.float64, int], force_all_finite=True
        )
        basis = _check_basis(self.basis, "basis", self.n_basis, self.basis_domain_range)
        groups = check_groups(groups=self.groups_in_, X=X, allow_overlap=False)

        X_out = []

        if grid_points is None:
            transform_grid_points = self.fitted_grid_points_
        else:
            _ = check_groups(
                groups=self.groups_in_,
                X=grid_points,
                allow_overlap=False,
                kwarg_name="grid_points",
            )
            transform_grid_points = np.array(grid_points)

        for idx, grp in enumerate(groups):
            if self.fpca_models_[idx] is not None:
                fd = _get_group_fd(X, grp, basis, transform_grid_points)
                X_out.append(self.fpca_models_[idx].transform(fd))
            else:
                X_out.append(X[:, grp])

        return np.hstack(X_out)

    def _more_tags(self):  # pylint: disable=no-self-use
        return {
            "allow_nan": False,
            "multilabel": True,
            "multioutput": True,
            "_skip_test": True,
        }


class SupervisedGroupFPCA(GroupFPCA, GroupSDRMixin):
    """A grouped supervised functional PCA transformer.

    Given a feature matrix `X`, grouping information, and a supervision
    target this transformer returns the groupwise supervised functional PCA
    matrix.

    Parameters
    ----------
    n_components : int, default=3
        number of principal components to obtain from functional
        principal component analysis.

    theta : float, default=0.0
        the threshold for each features univariate regression coefficient. If
        the absolute value of an individual feature's univariate regression
        coefficient is below this threshold, its value for all samples is set
        to the mean feature value. Unless ``absolute_threshold`` is True,
        theta is scaled by the max absolute value of the regression
        coefficients in each group.

    absolute_threshold : boolean, default=False
        If True, treat ``theta`` as an absolute threshold. i.e. do not scale
        by the max absolute regression coefficient in each group.

    smooth_screening : bool, default=True
        If True, values below the `theta` threshold fall smoothly to the mean
        feature value. If False, they are set abruptly to the mean.

    centering : bool, default=True
        if True then calculate the mean of the functional data object and
        center the data first.

    basis : str, instance of `skfda.representation.basis` class, optional
        the basis in which to represent each group's function before fPCA.

    n_basis : int, default=4
        the number of functions in the basis. Only used if ``basis`` is not
        None.

    basis_domain_range : tuple, optional
        a tuple of length 2 containing the initial and end values of the
        interval over which the basis can be evaluated.

    groups : numpy.ndarray or int, optional
        all group indices for feature matrix

    exclude_groups : sequence of int or int, optional
        a sequence of group indices (or a single group index) to exclude from
        fPCA transformatiton. These groups will be left untouched and
        transferred to the output matrix.

    Attributes
    ----------
    components_ : list of skfda.FData
        a list where each element contains a group's principal components
        in a basis representation.

    fitted_grid_points_ : numpy.ndarray or None
        The dicretization points for the input ``X`` matrix
        provided during ``fit``.

    fpca_models_ : list of fitted skfda.FPCA models
        a list where each element contains a group's fitted FPCA model

    groups_in_ : list of numpy.ndarray
        explicit group indices created from the potentially more implicit
        ``groups`` input parameter

    groups_out_ : list of numpy.ndarray
        group indices for the transformed features

    n_features_in_ : int
        number of input features

    n_features_out_ : int
        number of transformed features

    screening_mask_ : np.ndarray

    """

    def __init__(
        self,
        n_components=3,
        theta=0.0,
        absolute_threshold=False,
        smooth_screening=True,
        centering=True,
        basis=None,
        n_basis=4,
        basis_domain_range=None,
        groups=None,
        exclude_groups=None,
    ):
        self.theta = theta
        self.absolute_threshold = absolute_threshold
        self.smooth_screening = smooth_screening
        super().__init__(
            n_components=n_components,
            centering=centering,
            basis=basis,
            n_basis=n_basis,
            basis_domain_range=basis_domain_range,
            groups=groups,
            exclude_groups=exclude_groups,
        )

    def fit(self, X, y, grid_points=None):
        """Fit the fPCA transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.

        y : array-like of shape (n_samples,)
            target vectors, where n_samples is the number of samples

        grid_points : numpy.ndarray
            The points of dicretisation for the input ``X`` matrix. If None, this will use the
            grid_points supplied during the fit method.
        """
        self._compute_screening_mask(X, y)
        return super().fit(X=X * self.screening_mask_, grid_points=grid_points)

    def transform(self, X, y=None, grid_points=None):
        """Transform the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            samples to transform.

        grid_points : numpy.ndarray, optional
            The points of dicretisation for the input ``X`` matrix. If None, this will use the
            grid_points supplied during the fit method.
        """
        return super().transform(X * self.screening_mask_, grid_points=grid_points)

    def _more_tags(self):  # pylint: disable=no-self-use
        return {
            "allow_nan": False,
            "multilabel": True,
            "multioutput": True,
            "_skip_test": True,
            "requires_y": True,
        }
