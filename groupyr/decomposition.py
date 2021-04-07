"""Perform group-wise functional PCA of feature matrices with grouped covariates."""
import inspect
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.utils.multiclass import type_of_target
from sklearn.utils import check_array, check_scalar

try:
    from skfda import FDataGrid
    from skfda.representation.basis import Basis
    from skfda.representation.basis import BSpline, Monomial, Constant, Fourier
    from skfda.preprocessing.dim_reduction.projection import FPCA
except ImportError:  # pragma: no cover
    raise ImportError(
        "To use functional data analysis in groupyr, you will need to have "
        "scikit-fda installed. You can do this by installing groupyr with "
        "`pip install groupyr[fda]`, or by separately installing scikit-fda "
        "with `pip install scikit-fda`."
    )

from .utils import check_groups


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

    def fit(self, X=None, y=None, grid_points=None):
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

    def _more_tags(self):  # pylint: disable=no-self-use
        return {
            "allow_nan": False,
            "multilabel": True,
            "multioutput": True,
            "_skip_test": True,
        }


class SupervisedGroupFPCA(GroupFPCA):
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
        centering=True,
        basis=None,
        n_basis=4,
        basis_domain_range=None,
        groups=None,
        exclude_groups=None,
    ):
        self.theta = theta
        self.absolute_threshold = absolute_threshold
        super().__init__(
            n_components=n_components,
            centering=centering,
            basis=basis,
            n_basis=n_basis,
            basis_domain_range=basis_domain_range,
            groups=groups,
            exclude_groups=exclude_groups,
        )

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
        y = check_array(
            y,
            copy=True,
            dtype=[np.float32, np.float64, int],
            force_all_finite=True,
            ensure_2d=False,
        )
        if y.ndim != 1:
            raise NotImplementedError("y must be one-dimensional.")

        target_type = type_of_target(y)
        if target_type == "continuous":
            self.univariate_score_func_ = f_regression
        elif target_type in ["binary", "multiclass"]:
            self.univariate_score_func_ = f_classif
        else:
            raise NotImplementedError(
                "The type of target is unsupported. It must be continuous, "
                "binary, or multiclass. Got {0} instead.".format(target_type)
            )

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
            return super().fit(X=X, grid_points=grid_points)

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
                if len(transition_indices):
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
        X = X * self.screening_mask_
        return super().fit(X=X, grid_points=grid_points)

    def _more_tags(self):  # pylint: disable=no-self-use
        return {
            "allow_nan": False,
            "multilabel": True,
            "multioutput": True,
            "_skip_test": True,
            "requires_y": True,
        }
