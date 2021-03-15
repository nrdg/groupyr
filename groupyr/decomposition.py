"""Perform group-wise functional PCA of feature matrices with grouped covariates."""
import inspect
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

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
        X : numpy.ndarray
            The feature matrix

        grid_points : numpy.ndarray
            The points of dicretisation for the input ``X`` matrix. If None, this will use the
            grid_points supplied during the fit method.
        """
        X = check_array(
            X, copy=True, dtype=[np.float32, np.float64, int], force_all_finite=True
        )
        basis = _check_basis(self.basis, "basis", self.n_basis, self.basis_domain_range)
        groups = check_groups(groups=self.groups_, X=X, allow_overlap=False)

        X_out = []

        if grid_points is None:
            transform_grid_points = self.fitted_grid_points_
        else:
            _ = check_groups(
                groups=self.groups_,
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

    def fit(self, X=None, y=None, grid_points=None):
        """Fit the fPCA transformer.

        Parameters
        ----------
        X : numpy.ndarray
            The feature matrix

        grid_points : numpy.ndarray
            The points of dicretisation for the input ``X`` matrix. If None, this will use the
            grid_points supplied during the fit method.
        """
        X = check_array(
            X, copy=True, dtype=[np.float32, np.float64, int], force_all_finite=True
        )
        basis = _check_basis(self.basis, "basis", self.n_basis, self.basis_domain_range)

        _, self.n_features_in_ = X.shape
        self.groups_ = check_groups(groups=self.groups, X=X, allow_overlap=False)

        if grid_points is None:
            self.fitted_grid_points_ = grid_points
        else:
            _ = check_groups(
                groups=self.groups_,
                X=grid_points,
                allow_overlap=False,
                kwarg_name="grid_points",
            )
            self.fitted_grid_points_ = np.array(grid_points)

        if self.exclude_groups is None:
            exclude_grp_idx = []
        elif isinstance(self.exclude_groups, int):
            exclude_grp_idx = [self.exclude_groups]
        elif all(isinstance(e, int) for e in self.exclude_groups):
            exclude_grp_idx = list(self.exclude_groups)
        else:
            raise TypeError(
                f"exclude_groups must be and int or sequence of ints. "
                f"Got {self.exclude_groups} instead."
            )

        self.fpca_models_ = [
            FPCA(n_components=self.n_components, centering=self.centering)
            for grp in self.groups_
        ]
        self.components_ = [None for grp in self.groups_]
        self.groups_out_ = []

        feature_start_idx = 0
        for idx, grp in enumerate(self.groups_):
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
