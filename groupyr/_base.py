"""Create base classes based on the sparse group lasso."""
import contextlib
import copt as cp
import numpy as np
import warnings

from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    is_classifier,
    is_regressor,
)
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
)

from ._prox import SparseGroupL1
from .utils import check_groups


class SGLBaseEstimator(BaseEstimator, TransformerMixin):
    """
    An sklearn compatible sparse group lasso estimator.

    This solves the sparse group lasso [1]_ problem for a feature matrix
    partitioned into groups using the proximal gradient descent (PGD)
    algorithm.

    Parameters
    ----------
    l1_ratio : float, default=1.0
        Hyper-parameter : Combination between group lasso and lasso. l1_ratio=0
        gives the group lasso and l1_ratio=1 gives the lasso.

    alpha : float, default=0.0
        Hyper-parameter : overall regularization strength.

    groups : list of numpy.ndarray
        list of arrays of non-overlapping indices for each group. For
        example, if nine features are grouped into equal contiguous groups of
        three, then groups would be ``[array([0, 1, 2]), array([3, 4, 5]),
        array([6, 7, 8])]``. If the feature matrix contains a bias or
        intercept feature, do not include it as a group. If None, all
        features will belong to one group. We set groups in ``__init__`` so
        that it can be reused in model selection and CV routines.

    scale_l2_by : ["group_length", None], default="group_length"
        Scaling technique for the group-wise L2 penalty.
        By default, ``scale_l2_by="group_length`` and the L2 penalty is
        scaled by the square root of the group length so that each variable
        has the same effect on the penalty. This may not be appropriate for
        one-hot encoded features and ``scale_l2_by=None`` would be more
        appropriate for that case. ``scale_l2_by=None`` will also reproduce
        ElasticNet results when all features belong to one group.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the linear predictor (X @ coef + intercept).

    max_iter : int, default=1000
        Maximum number of iterations for PGD solver.

    tol : float, default=1e-7
        Stopping criterion. Convergence tolerance for PGD algorithm.

    warm_start : bool, default=False
        If set to ``True``, reuse the solution of the previous call to ``fit``
        as initialization for ``coef_`` and ``intercept_``.

    verbose : int, default=0
        Verbosity flag for PGD solver. Any positive integer will produce
        verbose output

    suppress_solver_warnings : bool, default=True
        If True, suppress convergence warnings from PGD solver.
        This is useful for hyperparameter tuning when some combinations
        of hyperparameters may not converge.

    include_solver_trace : bool, default=False
        If True, include copt.utils.Trace() object in the attribue ``solver_trace_``.

    Attributes
    ----------
    coef_ : array of shape (n_features,)
        Estimated coefficients for the linear predictor (`X @ coef_ +
        intercept_`).

    intercept_ : float
        Intercept (a.k.a. bias) added to linear predictor.

    n_iter_ : int
        Actual number of iterations used in the solver.

    solver_trace_ : copt.utils.Trace
        This object traces convergence of the solver and can be useful for
        debugging. If the ``include_solver_trace`` parameter is False, this
        attribute is ``None``.

    References
    ----------
    .. [1]  Noah Simon, Jerome Friedman, Trevor Hastie & Robert Tibshirani,
        "A Sparse-Group Lasso," Journal of Computational and Graphical
        Statistics, vol. 22:2, pp. 231-245, 2012
        DOI: 10.1080/10618600.2012.681250

    """

    def __init__(
        self,
        l1_ratio=1.0,
        alpha=0.0,
        groups=None,
        scale_l2_by="group_length",
        fit_intercept=True,
        max_iter=1000,
        tol=1e-7,
        warm_start=False,
        verbose=0,
        suppress_solver_warnings=True,
        include_solver_trace=False,
    ):
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.groups = groups
        self.scale_l2_by = scale_l2_by
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.verbose = verbose
        self.suppress_solver_warnings = suppress_solver_warnings
        self.include_solver_trace = include_solver_trace

    def fit(self, X, y, loss="squared_loss"):
        """Fit a linear model using the sparse group lasso.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        loss : ["squared_loss", "huber", "log"]
            The type of loss function to use in the PGD solver.

        Returns
        -------
        self : object
            Returns self.
        """
        if not isinstance(self.warm_start, bool):
            raise ValueError(
                "The argument warm_start must be bool;"
                " got {0}".format(self.warm_start)
            )

        allowed_losses = ["squared_loss", "huber"]
        if is_regressor(self) and loss.lower() not in allowed_losses:
            raise ValueError(
                "For regression, the argument loss must be one of {0};"
                "got {1}".format(allowed_losses, loss)
            )

        if not 0 <= self.l1_ratio <= 1:
            raise ValueError(
                "The parameter l1_ratio must satisfy 0 <= l1_ratio <= 1;"
                "got {0}".format(self.l1_ratio)
            )

        if y is None:
            raise ValueError("requires y to be passed, but the target y is None")

        X, y = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=[np.float64, np.float32],
            y_numeric=not is_classifier(self),
            multi_output=False,
        )

        _, self.n_features_in_ = X.shape

        if is_classifier(self):
            check_classification_targets(y)
            self.classes_ = np.unique(y)
            y = np.logical_not(y == self.classes_[0]).astype(int)

        n_samples, n_features = X.shape
        if self.fit_intercept:
            X = np.hstack([X, np.ones((n_samples, 1))])

        if self.warm_start and hasattr(self, "coef_"):
            # pylint: disable=access-member-before-definition
            if self.fit_intercept:
                coef = np.concatenate((self.coef_, np.array([self.intercept_])))
            else:
                coef = self.coef_
        else:
            if self.fit_intercept:
                coef = np.zeros(n_features + 1)
                # Initial bias condition gives 50/50 for binary classification
                coef[-1] = 0.5
            else:
                coef = np.zeros(n_features)

        if loss == "huber":
            f = cp.utils.HuberLoss(X, y)
        elif loss == "log":
            f = cp.utils.LogLoss(X, y)
        else:
            f = cp.utils.SquareLoss(X, y)

        if self.include_solver_trace:
            self.solver_trace_ = cp.utils.Trace(f)
        else:
            self.solver_trace_ = None

        if self.suppress_solver_warnings:
            ctx_mgr = warnings.catch_warnings()
        else:
            ctx_mgr = contextlib.suppress()

        groups = check_groups(
            self.groups, X, allow_overlap=False, fit_intercept=self.fit_intercept
        )

        if self.scale_l2_by not in ["group_length", None]:
            raise ValueError(
                "scale_l2_by must be 'group_length' or None; "
                "got {0}".format(self.scale_l2_by)
            )

        bias_index = n_features if self.fit_intercept else None
        sg1 = SparseGroupL1(
            l1_ratio=self.l1_ratio,
            alpha=self.alpha,
            groups=groups,
            bias_index=bias_index,
            scale_l2_by=self.scale_l2_by,
        )

        with ctx_mgr:
            # For some metaparameters, minimize_PGD might not reach the desired
            # tolerance level. This might be okay during hyperparameter
            # optimization. So ignore the warning if the user specifies
            # suppress_solver_warnings=True
            if self.suppress_solver_warnings:
                warnings.filterwarnings("ignore", category=RuntimeWarning)

            pgd = cp.minimize_proximal_gradient(
                f.f_grad,
                coef,
                sg1.prox,
                jac=True,
                step="backtracking",
                max_iter=self.max_iter,
                tol=self.tol,
                verbose=self.verbose,
                callback=self.solver_trace_,
                accelerated=False,
            )

        if self.fit_intercept:
            self.intercept_ = pgd.x[-1]
            self.coef_ = pgd.x[:-1]
        else:
            # set intercept to zero as the other linear models do
            self.intercept_ = 0.0
            self.coef_ = pgd.x

        self.n_iter_ = pgd.nit

        self.is_fitted_ = True
        return self

    @property
    def chosen_features_(self):
        """Return an index array of chosen features."""
        return np.nonzero(self.coef_)[0]

    @property
    def sparsity_mask_(self):
        """Return boolean array indicating which features survived regularization."""
        return self.coef_ != 0

    def like_nonzero_mask_(self, rtol=1e-8):
        """Return boolean array indicating which features are zero or close to zero.

        Parameters
        ----------
        rtol : float
            Relative tolerance. Any features that are larger in magnitude
            than ``rtol`` times the mean coefficient value are considered
            nonzero-like.
        """
        mean_abs_coef = abs(self.coef_.mean())
        return np.abs(self.coef_) > rtol * mean_abs_coef

    @property
    def chosen_groups_(self):
        """Return set of the group IDs that survived regularization."""
        if self.groups is not None:
            group_mask = [
                bool(set(grp).intersection(set(self.chosen_features_)))
                for grp in self.groups
            ]
            return np.nonzero(group_mask)[0]
        else:
            return self.chosen_features_

    def transform(self, X):
        """Remove columns corresponding to zeroed-out coefficients."""
        # Check is fit had been called
        check_is_fitted(self, "is_fitted_")

        # Input validation
        X = check_array(X, accept_sparse=True)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.coef_.size:
            raise ValueError("Shape of input is different from what was seen in `fit`")

        return X[:, self.sparsity_mask_]

    def _more_tags(self):  # pylint: disable=no-self-use
        return {"requires_y": True}
