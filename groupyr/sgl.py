"""Create regression estimators based on the sparse group lasso."""
import contextlib
import logging
import numpy as np
import warnings

from functools import partial
from joblib import delayed, effective_n_jobs, Parallel
from scipy import sparse
from scipy.optimize import root_scalar
from skopt import BayesSearchCV
from tqdm.auto import tqdm

from sklearn.base import RegressorMixin, TransformerMixin
from sklearn.linear_model._base import LinearModel, _preprocess_data
from sklearn.linear_model._coordinate_descent import _alpha_grid as _lasso_alpha_grid
from sklearn.metrics import get_scorer
from sklearn.model_selection import check_cv
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.utils.validation import check_array, check_is_fitted, column_or_1d

from ._base import SGLBaseEstimator
from ._prox import _soft_threshold
from .utils import check_groups

__all__ = ["SGL", "sgl_path", "SGLCV"]
logger = logging.getLogger(__name__)


class SGL(SGLBaseEstimator, RegressorMixin, LinearModel):
    """An sklearn compatible sparse group lasso regressor.

    This solves the sparse group lasso [1]_ problem for a feature matrix
    partitioned into groups using the proximal gradient descent (PGD)
    algorithm.

    Parameters
    ----------
    l1_ratio : float, default=1.0
        Hyper-parameter : Combination between group lasso and lasso. l1_ratio=0
        gives the group lasso and l1_ratio=1 gives the lasso.

    alpha : float, default=1.0
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
        Stopping criterion. Convergence tolerance for the ``copt`` proximal gradient solver

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

    Attributes
    ----------
    coef_ : array of shape (n_features,)
        Estimated coefficients for the linear predictor (`X @ coef_ +
        intercept_`).

    intercept_ : float
        Intercept (a.k.a. bias) added to linear predictor.

    n_iter_ : int
        Actual number of iterations used in the solver.

    References
    ----------
    .. [1]  Noah Simon, Jerome Friedman, Trevor Hastie & Robert Tibshirani,
        "A Sparse-Group Lasso," Journal of Computational and Graphical
        Statistics, vol. 22:2, pp. 231-245, 2012
        DOI: 10.1080/10618600.2012.681250

    """

    def fit(self, X, y, loss="squared_loss"):
        """Fit a linear model using the sparse group lasso.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        loss : ["squared_loss", "huber"]
            The type of loss function to use in the PGD solver.

        Returns
        -------
        self : object
            Returns self.
        """
        return super().fit(X=X, y=y, loss=loss)

    def predict(self, X):
        """Predict targets for test vectors in ``X``.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")

        return safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_

    def _more_tags(self):  # pylint: disable=no-self-use
        return {"requires_y": True}


def _alpha_grid(
    X,
    y,
    Xy=None,
    groups=None,
    scale_l2_by="group_length",
    l1_ratio=1.0,
    fit_intercept=True,
    eps=1e-3,
    n_alphas=100,
    normalize=False,
    copy_X=True,
    model=SGL,
):
    """Compute the grid of alpha values for elastic net parameter search.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication

    y : ndarray of shape (n_samples,)
        Target values

    Xy : array-like of shape (n_features,), default=None
        Xy = np.dot(X.T, y) that can be precomputed. If supplying ``Xy``,
        prevent train/test leakage by ensuring the ``Xy`` is precomputed
        using only training data.

    groups : list of numpy.ndarray
        list of arrays of non-overlapping indices for each group. For
        example, if nine features are grouped into equal contiguous groups of
        three, then groups would be ``[array([0, 1, 2]), array([3, 4, 5]),
        array([6, 7, 8])]``. If the feature matrix contains a bias or
        intercept feature, do not include it as a group. If None, all
        features will belong to one group.

    scale_l2_by : ["group_length", None], default="group_length"
        Scaling technique for the group-wise L2 penalty.
        By default, ``scale_l2_by="group_length`` and the L2 penalty is
        scaled by the square root of the group length so that each variable
        has the same effect on the penalty. This may not be appropriate for
        one-hot encoded features and ``scale_l2_by=None`` would be more
        appropriate for that case. ``scale_l2_by=None`` will also reproduce
        ElasticNet results when all features belong to one group.

    l1_ratio : float, default=1.0
        The elastic net mixing parameter, with ``0 < l1_ratio <= 1``.
        For ``l1_ratio = 0`` the penalty is an L2 penalty. (currently not
        supported) ``For l1_ratio = 1`` it is an L1 penalty. For
        ``0 < l1_ratio <1``, the penalty is a combination of L1 and L2.

    eps : float, default=1e-3
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    n_alphas : int, default=100
        Number of alphas along the regularization path

    fit_intercept : bool, default=True
        Whether to fit an intercept or not

    normalize : bool, default=False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    model : class, default=SGL
        The estimator class that will be used to confirm that alpha_max sets
        all coef values to zero. The default value of ``model=SGL`` is
        appropriate for regression while ``model=LogisticSGL`` is appropriate
        for classification.
    """
    if l1_ratio == 1.0:
        return _lasso_alpha_grid(
            X=X,
            y=y,
            Xy=Xy,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            eps=eps,
            n_alphas=n_alphas,
            normalize=normalize,
            copy_X=copy_X,
        )

    n_samples = len(y)
    if Xy is None:
        X = check_array(X, accept_sparse=False, copy=(copy_X and fit_intercept))
        X, y, _, _, _ = _preprocess_data(X, y, fit_intercept, normalize, copy=False)
        Xy = safe_sparse_dot(X.T, y, dense_output=True)

    if Xy.ndim == 1:
        Xy = Xy[:, np.newaxis]

    groups = check_groups(groups, X, allow_overlap=False, fit_intercept=False)

    if scale_l2_by not in ["group_length", None]:
        raise ValueError(
            "scale_l2_by must be 'group_length' or None; " "got {0}".format(scale_l2_by)
        )

    # When l1_ratio < 1 (i.e. not the lasso), then for each group, the
    # smallest alpha for which coef_ = 0 minimizes the objective will be
    # achieved when
    #
    # || S(Xy / n_samples, l1_ratio * alpha) ||_2 == sqrt(p_l) * (1 - l1_ratio) * alpha
    #
    # where S() is the element-wise soft-thresholding operator and p_l is
    # the group size (or 1 if ``scale_l2_by is None``)
    def beta_zero_root(alpha, group):
        soft = _soft_threshold(Xy[group] / n_samples, l1_ratio * alpha)
        scale = np.sqrt(group.size) if scale_l2_by == "group_length" else 1
        return np.linalg.norm(soft) - (1 - l1_ratio) * alpha * scale

    # We use the brentq method to find the root, which requires a bracket
    # within which to find the root. We know that ``beta_zero_root`` will
    # be positive when alpha=0. In order to ensure that the upper limit
    # brackets the root, we increase the upper limit until
    # ``beta_zero_root`` returns a negative number for all groups
    def bracket_too_low(alpha):
        return any([beta_zero_root(alpha, group=grp) > 0 for grp in groups])

    upper_bracket_lim = 1e1
    while bracket_too_low(upper_bracket_lim):
        upper_bracket_lim *= 10

    min_alphas = np.array(
        [
            root_scalar(
                partial(beta_zero_root, group=grp),
                bracket=[0, upper_bracket_lim],
                method="brentq",
            ).root
            for grp in groups
        ]
    )

    alpha_max = np.max(min_alphas) * 1.2

    # Test feature sparsity just to make sure we're on the right side of the root
    while (  # pragma: no cover
        model(
            groups=groups,
            alpha=alpha_max,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            scale_l2_by=scale_l2_by,
        )
        .fit(X, y)
        .chosen_features_.size
        > 0
    ):
        alpha_max *= 1.2  # pragma: no cover

    if alpha_max <= np.finfo(float).resolution:
        alphas = np.empty(n_alphas)
        alphas.fill(np.finfo(float).resolution)
        return alphas

    return np.logspace(np.log10(alpha_max * eps), np.log10(alpha_max), num=n_alphas)[
        ::-1
    ]


def sgl_path(
    X,
    y,
    l1_ratio=0.5,
    groups=None,
    scale_l2_by="group_length",
    eps=1e-3,
    n_alphas=100,
    alphas=None,
    Xy=None,
    normalize=False,
    copy_X=True,
    verbose=False,
    return_n_iter=False,
    check_input=True,
    **params,
):
    """
    Compute sparse group lasso path.

    We use the previous solution as the initial guess for subsequent alpha values

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication.

    y : {array-like, sparse matrix} of shape (n_samples,)
        Target values.

    l1_ratio : float, default=0.5
        Number between 0 and 1 passed to SGL estimator (scaling between the
        group lasso and lasso penalties). ``l1_ratio=1`` corresponds to the
        Lasso.

    groups : list of numpy.ndarray
        list of arrays of non-overlapping indices for each group. For
        example, if nine features are grouped into equal contiguous groups of
        three, then groups would be ``[array([0, 1, 2]), array([3, 4, 5]),
        array([6, 7, 8])]``. If the feature matrix contains a bias or
        intercept feature, do not include it as a group. If None, all
        features will belong to one group.

    scale_l2_by : ["group_length", None], default="group_length"
        Scaling technique for the group-wise L2 penalty.
        By default, ``scale_l2_by="group_length`` and the L2 penalty is
        scaled by the square root of the group length so that each variable
        has the same effect on the penalty. This may not be appropriate for
        one-hot encoded features and ``scale_l2_by=None`` would be more
        appropriate for that case. ``scale_l2_by=None`` will also reproduce
        ElasticNet results when all features belong to one group.

    eps : float, default=1e-3
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, default=100
        Number of alphas along the regularization path.

    alphas : ndarray, default=None
        List of alphas where to compute the models.
        If None alphas are set automatically.

    Xy : array-like of shape (n_features,), default=None
        Xy = np.dot(X.T, y) that can be precomputed. If supplying ``Xy``,
        prevent train/test leakage by ensuring the ``Xy`` is precomputed
        using only training data.

    normalize : bool, default=False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    verbose : bool or int, default=False
        Amount of verbosity.

    check_input : bool, default=True
        Skip input validation checks, assuming there are handled by the
        caller when check_input=False.

    **params : kwargs
        Keyword arguments passed to the SGL estimator

    Returns
    -------
    coefs : ndarray of shape (n_features, n_alphas) or (n_features + 1, n_alphas)
        List of coefficients for the Logistic Regression model. If
        fit_intercept is set to True then the first dimension will be
        n_features + 1, where the last item represents the intercept.

    alphas : ndarray of shape (n_alphas,)
        The alphas along the path where models are computed.

    n_iters : array of shape (n_alphas,)
        Actual number of iteration for each alpha.

    See Also
    --------
    SGL
    SGLCV
    """
    # We expect X and y to be already Fortran ordered when bypassing
    # checks
    if check_input:
        X = check_array(
            X,
            accept_sparse=False,
            dtype=[np.float64, np.float32],
            order="F",
            copy=copy_X,
        )
        y = check_array(
            y,
            accept_sparse=False,
            dtype=X.dtype.type,
            order="F",
            copy=False,
            ensure_2d=False,
        )
        if Xy is not None:
            # Xy should be a 1d contiguous array
            Xy = check_array(
                Xy, dtype=X.dtype.type, order="C", copy=False, ensure_2d=False
            )
        groups = check_groups(groups, X, allow_overlap=False, fit_intercept=False)

    _, n_features = X.shape

    fit_intercept = params.get("fit_intercept", True)

    if alphas is None:
        alphas = _alpha_grid(
            X=X,
            y=y,
            Xy=Xy,
            groups=groups,
            scale_l2_by=scale_l2_by,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            eps=eps,
            n_alphas=n_alphas,
            normalize=normalize,
            copy_X=copy_X,
        )
    else:
        alphas = np.sort(alphas)[::-1]  # make sure alphas are properly ordered

    n_alphas = len(alphas)
    tol = params.get("tol", 1e-7)
    max_iter = params.get("max_iter", 1000)
    loss = params.get("loss", "squared_loss")
    n_iters = np.empty((n_alphas,), dtype=int)

    if fit_intercept:
        coefs = np.empty((n_features + 1, n_alphas), dtype=X.dtype)
    else:
        coefs = np.empty((n_features, n_alphas), dtype=X.dtype)

    model = SGL(
        l1_ratio=l1_ratio,
        alpha=alphas[0],
        groups=groups,
        scale_l2_by=scale_l2_by,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        tol=tol,
        warm_start=True,
        verbose=False,
        suppress_solver_warnings=True,
        include_solver_trace=False,
    )

    if verbose and verbose == 1:
        alpha_sequence = tqdm(alphas, desc="Reg path", total=n_alphas)
    else:
        alpha_sequence = alphas

    for i, alpha in enumerate(alpha_sequence):
        model.set_params(alpha=alpha)
        model.fit(X, y, loss=loss)

        if fit_intercept:
            coefs[..., i] = np.concatenate([model.coef_, [model.intercept_]])
        else:
            coefs[..., i] = model.coef_

        n_iters[i] = model.n_iter_

        if verbose:
            if verbose > 2:
                print(model)
            elif verbose > 1:
                print("Path: %03i out of %03i" % (i, n_alphas))

    return coefs, alphas, n_iters


def sgl_scoring_path(
    X,
    y,
    train,
    test,
    l1_ratio=0.5,
    groups=None,
    scale_l2_by="group_length",
    eps=1e-3,
    n_alphas=100,
    alphas=None,
    Xy=None,
    normalize=False,
    copy_X=True,
    verbose=False,
    check_input=True,
    scoring=None,
    **params,
):
    """Return the scores for the models computed by 'path'.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.

    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.

    train : list of indices
        The indices of the train set.

    test : list of indices
        The indices of the test set.

    l1_ratio : float, default=0.5
        Number between 0 and 1 passed to SGL estimator (scaling between the
        group lasso and lasso penalties). ``l1_ratio=1`` corresponds to the
        Lasso.

    groups : list of numpy.ndarray
        list of arrays of non-overlapping indices for each group. For
        example, if nine features are grouped into equal contiguous groups of
        three, then groups would be ``[array([0, 1, 2]), array([3, 4, 5]),
        array([6, 7, 8])]``. If the feature matrix contains a bias or
        intercept feature, do not include it as a group. If None, all
        features will belong to one group.

    scale_l2_by : ["group_length", None], default="group_length"
        Scaling technique for the group-wise L2 penalty.
        By default, ``scale_l2_by="group_length`` and the L2 penalty is
        scaled by the square root of the group length so that each variable
        has the same effect on the penalty. This may not be appropriate for
        one-hot encoded features and ``scale_l2_by=None`` would be more
        appropriate for that case. ``scale_l2_by=None`` will also reproduce
        ElasticNet results when all features belong to one group.

    eps : float, default=1e-3
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, default=100
        Number of alphas along the regularization path.

    alphas : ndarray, default=None
        List of alphas where to compute the models.
        If None alphas are set automatically.

    Xy : array-like of shape (n_features,), default=None
        Xy = np.dot(X.T, y) that can be precomputed. If supplying ``Xy``,
        prevent train/test leakage by ensuring the ``Xy`` is precomputed
        using only training data.

    normalize : bool, default=False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    verbose : bool or int, default=False
        Amount of verbosity.

    check_input : bool, default=True
        Skip input validation checks, assuming there are handled by the
        caller when check_input=False.

    scoring : callable, default=None
        A string (see sklearn model evaluation documentation) or a scorer
        callable object / function with signature ``scorer(estimator, X, y)``.
        For a list of scoring functions that can be used, look at
        `sklearn.metrics`. The default scoring option used is accuracy_score.

    **params : kwargs
        Keyword arguments passed to the SGL estimator

    Returns
    -------
    coefs : ndarray of shape (n_features, n_alphas) or (n_features + 1, n_alphas)
        List of coefficients for the SGL model. If fit_intercept is set to
        True then the first dimension will be n_features + 1, where the last
        item represents the intercept.

    alphas : ndarray
        Grid of alphas used for cross-validation.

    scores : ndarray of shape (n_alphas,)
        Scores obtained for each alpha.

    n_iter : ndarray of shape(n_alphas,)
        Actual number of iteration for each alpha.
    """
    if Xy is not None:
        logger.warning(
            "You supplied the `Xy` parameter. Remember to ensure "
            "that Xy is computed from the training data alone."
        )

    X_train = X[train]
    y_train = y[train]
    X_test = X[test]
    y_test = y[test]

    coefs, alphas, n_iter = sgl_path(
        X_train,
        y_train,
        l1_ratio=l1_ratio,
        groups=groups,
        scale_l2_by=scale_l2_by,
        eps=eps,
        n_alphas=n_alphas,
        alphas=alphas,
        Xy=Xy,
        normalize=normalize,
        copy_X=copy_X,
        verbose=verbose,
        check_input=False,
        **params,
    )
    del X_train, y_train

    fit_intercept = params.get("fit_intercept", True)
    max_iter = params.get("max_iter", 1000)
    tol = params.get("tol", 1e-7)

    model = SGL(
        l1_ratio=l1_ratio,
        groups=groups,
        scale_l2_by=scale_l2_by,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        tol=tol,
        warm_start=True,
        verbose=False,
        suppress_solver_warnings=True,
        include_solver_trace=False,
    )

    if scoring is None:
        scoring = "neg_mean_squared_error"

    model.is_fitted_ = True

    scores = list()
    scoring = get_scorer(scoring)
    for w in coefs.T:
        if fit_intercept:
            model.coef_ = w[:-1]
            model.intercept_ = w[-1]
        else:
            model.coef_ = w
            model.intercept_ = 0.0

        scores.append(scoring(model, X_test, y_test))

    return coefs, alphas, np.array(scores), n_iter


class SGLCV(LinearModel, RegressorMixin, TransformerMixin):
    """Iterative SGL model fitting along a regularization path.

    See the scikit-learn glossary entry for `cross-validation estimator
    <https://scikit-learn.org/stable/glossary.html#term-cross-validation-estimator>`_

    Parameters
    ----------
    l1_ratio : float or list of float, default=1.0
        float between 0 and 1 passed to SGL (scaling between group lasso and
        lasso penalties). For ``l1_ratio = 0`` the penalty is the group lasso
        penalty. For ``l1_ratio = 1`` it is the lasso penalty. For ``0 <
        l1_ratio < 1``, the penalty is a combination of group lasso and
        lasso. This parameter can be a list, in which case the different
        values are tested by cross-validation and the one giving the best
        prediction score is used. Note that a good choice of list of values
        will depend on the problem. For problems where we expect strong
        overall sparsity and would like to encourage grouping, put more
        values close to 1 (i.e. Lasso). In contrast, if we expect strong
        group-wise sparsity, but only mild sparsity within groups, put more
        values close to 0 (i.e. group lasso).

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

    eps : float, default=1e-3
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, default=100
        Number of alphas along the regularization path, used for each l1_ratio.

    alphas : ndarray, default=None
        List of alphas where to compute the models.
        If None alphas are set automatically

    fit_intercept : bool, default=True
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    normalize : bool, default=False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`sklearn:sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    max_iter : int, default=1000
        The maximum number of iterations

    tol : float, default=1e-7
        Stopping criterion. Convergence tolerance for the ``copt`` proximal gradient solver

    cv : int, cross-validation generator or iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - int, to specify the number of folds.
        - an sklearn `CV splitter <https://scikit-learn.org/stable/glossary.html#term-cv-splitter>`_,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, :class:`sklearn:sklearn.model_selection.KFold` is used.

        Refer to the :ref:`scikit-learn User Guide
        <sklearn:cross_validation>` for the various cross-validation
        strategies that can be used here.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    verbose : bool or int, default=0
        Amount of verbosity.

    n_jobs : int, default=None
        Number of CPUs to use during the cross validation.
        ``None`` means 1 unless in a :obj:`joblib:joblib.parallel_backend` context.
        ``-1`` means using all processors.

    tuning_strategy : ["grid", "bayes"], default="grid"
        Hyperparameter tuning strategy to use. If ``tuning_strategy == "grid"``,
        then evaluate all parameter points on the ``l1_ratio`` and ``alphas`` grid,
        using warm start to evaluate different ``alpha`` values along the
        regularization path. If ``tuning_strategy == "bayes"``, then a fixed
        number of parameter settings is sampled using ``skopt.BayesSearchCV``.
        The fixed number of settings is set by ``n_bayes_iter``. The
        ``l1_ratio`` setting is sampled uniformly from the minimum and maximum
        of the input ``l1_ratio`` parameter. The ``alpha`` setting is sampled
        log-uniformly either from the maximum and minumum of the input ``alphas``
        parameter, if provided or from ``eps`` * max_alpha to max_alpha where
        max_alpha is a conservative estimate of the maximum alpha for which the
        solution coefficients are non-trivial.

    n_bayes_iter : int, default=50
        Number of parameter settings that are sampled if using Bayes search
        for hyperparameter optimization. ``n_bayes_iter`` trades off runtime
        vs quality of the solution. Consider increasing ``n_bayes_points`` if
        you want to try more parameter settings in parallel.

    n_bayes_points : int, default=1
        Number of parameter settings to sample in parallel if using Bayes
        search for hyperparameter optimization. If this does not align with
        ``n_bayes_iter``, the last iteration will sample fewer points.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    suppress_solver_warnings : bool, default=True
        If True, suppress warnings from BayesSearchCV when the objective is
        evaluated at the same point multiple times. Setting this to False,
        may be useful for debugging.

    Attributes
    ----------
    alpha_ : float
        The amount of penalization chosen by cross validation

    l1_ratio_ : float
        The compromise between l1 and l2 penalization chosen by
        cross validation

    coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
        Parameter vector (w in the cost function formula),

    intercept_ : float or ndarray of shape (n_targets, n_features)
        Independent term in the decision function.

    scoring_path_ : ndarray of shape (n_l1_ratio, n_alpha, n_folds)
        Mean square error for the test set on each fold, varying l1_ratio and
        alpha.

    alphas_ : ndarray of shape (n_alphas,) or (n_l1_ratio, n_alphas)
        The grid of alphas used for fitting, for each l1_ratio.

    n_iter_ : int
        number of iterations run by the proximal gradient descent solver to
        reach the specified tolerance for the optimal alpha.

    bayes_optimizer_ : skopt.BayesSearchCV instance or None
        The BayesSearchCV instance used for hyperparameter optimization if
        ``tuning_strategy == "bayes"``. If ``tuning_strategy == "grid"``,
        then this attribute is None.

    See Also
    --------
    sgl_path
    SGL
    """

    def __init__(
        self,
        l1_ratio=1.0,
        groups=None,
        scale_l2_by="group_length",
        eps=1e-3,
        n_alphas=100,
        alphas=None,
        fit_intercept=True,
        normalize=False,
        max_iter=1000,
        tol=1e-7,
        copy_X=True,
        cv=None,
        verbose=False,
        n_jobs=None,
        tuning_strategy="grid",
        n_bayes_iter=50,
        n_bayes_points=1,
        random_state=None,
        suppress_solver_warnings=True,
    ):
        self.l1_ratio = l1_ratio
        self.groups = groups
        self.scale_l2_by = scale_l2_by
        self.eps = eps
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol
        self.copy_X = copy_X
        self.cv = cv
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.tuning_strategy = tuning_strategy
        self.n_bayes_iter = n_bayes_iter
        self.n_bayes_points = n_bayes_points
        self.random_state = random_state
        self.suppress_solver_warnings = suppress_solver_warnings

    def fit(self, X, y):
        """Fit sparse group lasso linear model.

        Fit is on grid of alphas and best alpha estimated by cross-validation.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data. Pass directly as Fortran-contiguous data
            to avoid unnecessary memory duplication. If y is mono-output,
            X can be sparse.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values
        """
        if self.tuning_strategy.lower() not in ["grid", "bayes"]:
            raise ValueError(
                "`tuning_strategy` must be either 'grid' or 'bayes'; got "
                "{0} instead.".format(self.tuning_strategy)
            )

        # This makes sure that there is no duplication in memory.
        # Dealing right with copy_X is important in the following:
        # Multiple functions touch X and subsamples of X and can induce a
        # lot of duplication of memory
        copy_X = self.copy_X and self.fit_intercept

        check_y_params = dict(
            copy=False, dtype=[np.float64, np.float32], ensure_2d=False
        )

        if isinstance(X, np.ndarray) or sparse.isspmatrix(X):
            # Keep a reference to X
            reference_to_old_X = X
            # Let us not impose fortran ordering so far: it is
            # not useful for the cross-validation loop and will be done
            # by the model fitting itself

            # Need to validate separately here.
            # We can't pass multi_ouput=True because that would allow y to be
            # csr. We also want to allow y to be 64 or 32 but check_X_y only
            # allows to convert for 64.
            check_X_params = dict(
                accept_sparse=False, dtype=[np.float64, np.float32], copy=False
            )
            X, y = self._validate_data(
                X, y, validate_separately=(check_X_params, check_y_params)
            )
            if not np.may_share_memory(reference_to_old_X, X):
                # X has been copied
                copy_X = False
            del reference_to_old_X
        else:
            # Need to validate separately here.
            # We can't pass multi_ouput=True because that would allow y to be
            # csr. We also want to allow y to be 64 or 32 but check_X_y only
            # allows to convert for 64.
            check_X_params = dict(
                accept_sparse=False,
                dtype=[np.float64, np.float32],
                order="F",
                copy=copy_X,
            )
            X, y = self._validate_data(
                X, y, validate_separately=(check_X_params, check_y_params)
            )
            copy_X = False

        model = SGL()
        y = column_or_1d(y, warn=True)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X and y have inconsistent dimensions (%d != %d)"
                % (X.shape[0], y.shape[0])
            )

        groups = check_groups(self.groups, X, allow_overlap=False, fit_intercept=False)

        # All SGLCV parameters except "cv" and "n_jobs" are acceptable
        path_params = self.get_params()
        path_params.pop("cv", None)
        path_params.pop("n_jobs", None)
        path_params.pop("tuning_strategy", None)
        path_params.pop("n_bayes_iter", None)
        path_params.pop("n_bayes_points", None)
        path_params.pop("random_state", None)

        l1_ratios = np.atleast_1d(path_params["l1_ratio"])
        # For the first path, we need to set l1_ratio
        path_params["l1_ratio"] = l1_ratios[0]

        alphas = self.alphas
        n_l1_ratio = len(l1_ratios)
        if alphas is None:
            alphas = [
                _alpha_grid(
                    X=X,
                    y=y,
                    groups=groups,
                    scale_l2_by=self.scale_l2_by,
                    l1_ratio=l1_ratio,
                    fit_intercept=self.fit_intercept,
                    eps=self.eps,
                    n_alphas=self.n_alphas,
                    normalize=self.normalize,
                    copy_X=self.copy_X,
                )
                for l1_ratio in l1_ratios
            ]
        else:
            # Making sure alphas is properly ordered.
            alphas = np.tile(np.sort(alphas)[::-1], (n_l1_ratio, 1))

        # We want n_alphas to be the number of alphas used for each l1_ratio.
        n_alphas = len(alphas[0])
        path_params.update({"n_alphas": n_alphas})

        path_params["copy_X"] = copy_X
        # We are not computing in parallel, we can modify X
        # inplace in the folds
        if effective_n_jobs(self.n_jobs) > 1:
            path_params["copy_X"] = False

        if isinstance(self.verbose, int):  # pragma: no cover
            path_params["verbose"] = self.verbose - 1

        # init cross-validation generator
        cv = check_cv(self.cv)

        if self.tuning_strategy == "grid":
            # Compute path for all folds and compute MSE to get the best alpha
            folds = list(cv.split(X, y))
            best_score = -np.inf

            path_params.pop("cv", None)
            path_params.pop("n_jobs", None)
            path_params.pop("alphas", None)
            path_params.pop("l1_ratio", None)
            path_params.update({"groups": groups})

            # We do a double for loop folded in one, in order to be able to
            # iterate in parallel on l1_ratio and folds
            jobs = (
                delayed(sgl_scoring_path)(
                    X=X,
                    y=y,
                    train=train,
                    test=test,
                    l1_ratio=this_l1_ratio,
                    alphas=this_alphas,
                    Xy=None,
                    **path_params,
                )
                for this_l1_ratio, this_alphas in zip(l1_ratios, alphas)
                for train, test in folds
            )

            if isinstance(self.verbose, int):  # pragma: no cover
                parallel_verbosity = self.verbose - 2
                if parallel_verbosity < 0:
                    parallel_verbosity = 0
            else:  # pragma: no cover
                parallel_verbosity = self.verbose

            score_paths = Parallel(
                n_jobs=self.n_jobs,
                verbose=parallel_verbosity,
                **_joblib_parallel_args(prefer="threads"),
            )(jobs)

            coefs_paths, alphas_paths, scores, n_iters = zip(*score_paths)
            scores = np.reshape(scores, (n_l1_ratio, len(folds), -1))
            alphas_paths = np.reshape(alphas_paths, (n_l1_ratio, len(folds), -1))
            n_iters = np.reshape(n_iters, (n_l1_ratio, len(folds), -1))
            coefs_paths = np.reshape(
                coefs_paths, (n_l1_ratio, len(folds), -1, n_alphas)
            )

            mean_score = np.mean(scores, axis=1)
            self.scoring_path_ = np.squeeze(np.moveaxis(scores, 2, 1))

            for l1_ratio, l1_alphas, score_alphas in zip(l1_ratios, alphas, mean_score):
                i_best_alpha = np.argmax(score_alphas)
                this_best_score = score_alphas[i_best_alpha]
                if this_best_score > best_score:
                    best_alpha = l1_alphas[i_best_alpha]
                    best_l1_ratio = l1_ratio
                    best_score = this_best_score

            self.l1_ratio_ = best_l1_ratio
            self.alpha_ = best_alpha

            if self.alphas is None:
                self.alphas_ = np.asarray(alphas)
                if n_l1_ratio == 1:
                    self.alphas_ = self.alphas_[0]
            # Remove duplicate alphas in case alphas is provided.
            else:
                self.alphas_ = np.asarray(alphas[0])

            # Refit the model with the parameters selected
            common_params = {
                name: value
                for name, value in self.get_params().items()
                if name in model.get_params()
            }

            model.set_params(**common_params)
            model.alpha = best_alpha
            model.l1_ratio = best_l1_ratio
            model.copy_X = copy_X

            model.fit(X, y)

            self.coef_ = model.coef_
            self.intercept_ = model.intercept_
            self.n_iter_ = model.n_iter_
            self.bayes_optimizer_ = None
            self.is_fitted_ = True
        else:
            # Set the model with the common input params
            common_params = {
                name: value
                for name, value in self.get_params().items()
                if name in model.get_params()
            }
            model.set_params(**common_params)
            model.copy_X = copy_X

            search_spaces = {"alpha": (np.min(alphas), np.max(alphas), "log-uniform")}

            l1_min = np.min(self.l1_ratio)
            l1_max = np.max(self.l1_ratio)

            if l1_min < l1_max:
                search_spaces["l1_ratio"] = (
                    np.min(self.l1_ratio),
                    np.max(self.l1_ratio),
                    "uniform",
                )

            self.bayes_optimizer_ = BayesSearchCV(
                model,
                search_spaces,
                n_iter=self.n_bayes_iter,
                cv=cv,
                n_jobs=self.n_jobs,
                n_points=self.n_bayes_points,
                verbose=self.verbose,
                random_state=self.random_state,
                return_train_score=True,
                scoring="neg_mean_squared_error",
                error_score=-np.inf,
            )

            if self.suppress_solver_warnings:
                ctx_mgr = warnings.catch_warnings()
            else:
                ctx_mgr = contextlib.suppress()

            with ctx_mgr:
                # If n_bayes_points > 1 the objective may be evaluated at the
                # same point multiple times. This is okay and we give the user
                # the choice as to whether or not to see these warnings. The
                # default is to suppress them.
                if self.suppress_solver_warnings:
                    warnings.filterwarnings("ignore", category=UserWarning)

                self.bayes_optimizer_.fit(X, y)

            self.alpha_ = self.bayes_optimizer_.best_estimator_.alpha
            self.l1_ratio_ = self.bayes_optimizer_.best_estimator_.l1_ratio
            self.coef_ = self.bayes_optimizer_.best_estimator_.coef_
            self.intercept_ = self.bayes_optimizer_.best_estimator_.intercept_
            self.n_iter_ = self.bayes_optimizer_.best_estimator_.n_iter_
            self.is_fitted_ = True
            self.scoring_path_ = None
            param_alpha = self.bayes_optimizer_.cv_results_["param__alpha"]
            self.alphas_ = np.sort(param_alpha)[::-1]

        return self

    @property
    def chosen_features_(self):
        """Return index array of chosen features."""
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
        return {"multioutput": False, "requires_y": True}
