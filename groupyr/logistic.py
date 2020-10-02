"""
This module contains logistic estimators based on the sparse group lasso
"""
import numpy as np

from joblib import delayed, effective_n_jobs
from scipy import sparse
from tqdm.auto import tqdm

from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.metrics import get_scorer
from sklearn.model_selection import check_cv
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, column_or_1d

from ._base import SGLBaseEstimator
from .sgl import _alpha_grid
from .utils import check_groups, ProgressParallel

__all__ = []


def registered(fn):
    __all__.append(fn.__name__)
    return fn


@registered
class LogisticSGL(SGLBaseEstimator, LinearClassifierMixin):
    """
    An sklearn compatible sparse group lasso classifier.

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

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes, )
        A list of class labels known to the classifier.

    coef_ : array of shape (n_features,)
        Estimated coefficients for the linear predictor (`X @ coef_ +
        intercept_`).

    intercept_ : float
        Intercept (a.k.a. bias) added to linear predictor.

    n_iter_ : int
        Actual number of iterations used in the solver.

    Examples
    --------

    References
    ----------
    .. [1]  Noah Simon, Jerome Friedman, Trevor Hastie & Robert Tibshirani,
        "A Sparse-Group Lasso," Journal of Computational and Graphical
        Statistics, vol. 22:2, pp. 231-245, 2012
        DOI: 10.1080/10618600.2012.681250

    """

    def fit(self, X, y):  # pylint: disable=arguments-differ
        """Fit a linear model using the sparse group lasso

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        return super().fit(X=X, y=y, loss="log")

    def decision_function(self, X):
        """
        Predict confidence scores for samples.

        The confidence score for a sample is the signed distance of that
        sample to the hyperplane.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
            Confidence scores per (sample, class) combination. In the binary
            case, confidence score for self.classes_[1] where >0 means this
            class would be predicted.
        """
        check_is_fitted(self)

        X = check_array(X, accept_sparse="csr")

        n_features = self.coef_.size
        if X.shape[1] != n_features:
            raise ValueError(
                "X has %d features per sample; expecting %d" % (X.shape[1], n_features)
            )

        scores = safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
        return scores

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape [n_samples]
            Predicted class label per sample.
        """
        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(np.int)
        else:
            indices = scores.argmax(axis=1)

        return self.classes_[indices]

    def predict_proba(self, X):
        """
        Probability estimates.

        The returned estimates for all classes are ordered by the label of classes.

        Else use a one-vs-rest approach, i.e calculate the probability of
        each class assuming it to be positive using the logistic function.
        and normalize these values across all the classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        check_is_fitted(self)
        return super()._predict_proba_lr(X)

    def predict_log_proba(self, X):
        """
        Predict logarithm of probability estimates.

        The returned estimates for all classes are ordered by the label of classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.
        """
        return np.log(self.predict_proba(X))

    def _more_tags(self):  # pylint: disable=no-self-use
        return {"binary_only": True, "requires_y": True}


def logistic_sgl_path(
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
    check_input=True,
    **params,
):
    """Compute a Logistic SGL model for a list of regularization parameters.

    This is an implementation that uses the result of the previous model
    to speed up computations along the regularization path, making it faster
    than calling LogisticSGL for the different parameters without warm start.

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
        Xy = np.dot(X.T, y) that can be precomputed.

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
        Keyword arguments passed to the LogisticSGL estimator

    Returns
    -------
    coefs : ndarray of shape (n_features, n_alphas) or (n_features + 1, n_alphas)
        List of coefficients for the Logistic Regression model. If
        fit_intercept is set to True then the second dimension will be
        n_features + 1, where the last item represents the intercept.

    alphas : ndarray
        Grid of alphas used for cross-validation.

    n_iter : array of shape (n_alphas,)
        Actual number of iteration for each alpha.
    """
    # Preprocessing.
    if check_input:
        X = check_array(
            X,
            accept_sparse="csc",
            dtype=[np.float64, np.float32],
            order="F",
            copy=copy_X,
        )
        y = check_array(
            y,
            accept_sparse="csc",
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

    classes = np.unique(y)
    if classes.size > 2:
        raise NotImplementedError(
            "Multiclass classification is not currently implemented. We suggest "
            "using the `sklearn.multiclass.OneVsRestClassifier` to wrap the "
            "`LogisticSGL` or `LogisticSGLCV` estimators."
        )

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
            model=LogisticSGL,
        )
    else:
        alphas = np.sort(alphas)[::-1]  # make sure alphas are properly ordered

    n_alphas = len(alphas)
    tol = params.get("tol", 1e-7)
    max_iter = params.get("max_iter", 1000)
    n_iters = np.empty(n_alphas, dtype=int)

    if fit_intercept:
        coefs = np.empty((n_features + 1, n_alphas), dtype=X.dtype)
    else:
        coefs = np.empty((n_features, n_alphas), dtype=X.dtype)

    model = LogisticSGL(
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
        model.fit(X, y)

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


# helper function for LogisticSGLCV
def logistic_sgl_scoring_path(
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
    """Computes scores across logistic SGL path

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication.

    y : {array-like, sparse matrix} of shape (n_samples,)
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
        Xy = np.dot(X.T, y) that can be precomputed.

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
        List of coefficients for the Logistic Regression model. If
        fit_intercept is set to True then the second dimension will be
        n_features + 1, where the last item represents the intercept.

    alphas : ndarray
        Grid of alphas used for cross-validation.

    scores : ndarray of shape (n_alphas,)
        Scores obtained for each alpha.

    n_iter : ndarray of shape(n_alphas,)
        Actual number of iteration for each Cs.
    """
    X_train = X[train]
    X_test = X[test]
    y_train = y[train]
    y_test = y[test]

    coefs, alphas, n_iter = logistic_sgl_path(
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

    fit_intercept = params.get("fit_intercept", True)
    max_iter = params.get("max_iter", 1000)
    tol = params.get("tol", 1e-7)

    model = LogisticSGL(
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

    # The score method of LogisticSGL has a classes_ attribute
    # that is assigned during fit(). We don't call fit here so
    # we must assign it first
    model.classes_ = np.unique(y_train)

    scores = list()
    scoring = get_scorer(scoring)
    for w in coefs.T:
        if fit_intercept:
            model.coef_ = w[:-1]
            model.intercept_ = w[-1]
        else:
            model.coef_ = w
            model.intercept_ = 0.0

        if scoring is None:
            scores.append(model.score(X_test, y_test))
        else:
            scores.append(scoring(model, X_test, y_test))

    return coefs, alphas, np.array(scores), n_iter


# TODO: Test alpha_grid with X^T dot logit(y) to see if it works
@registered
class LogisticSGLCV(LogisticSGL):
    """Class for iterative Logistic SGL model fitting along a regularization path

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
        :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    max_iter : int, default=1000
        The maximum number of iterations

    tol : float, default=1e-7
        The tolerance for the SGL solver

    scoring : callable, default=None
        A string (see sklearn model evaluation documentation) or a scorer
        callable object / function with signature ``scorer(estimator, X, y)``.
        For a list of scoring functions that can be used, look at
        `sklearn.metrics`. The default scoring option used is accuracy_score.

    cv : int, cross-validation generator or iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - int, to specify the number of folds.
        - an sklearn `CV splitter <https://scikit-learn.org/stable/glossary.html#term-cv-splitter>`_,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, :class:`KFold` is used.

        Refer to the scikit-learn User Guide for the various
        cross-validation strategies that can be used here.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    verbose : bool or int, default=False
        Amount of verbosity.

    n_jobs : int, default=None
        Number of CPUs to use during the cross validation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    Attributes
    ----------
    alpha_ : float
        The amount of penalization chosen by cross validation

    l1_ratio_ : float
        The compromise between l1 and l2 penalization chosen by
        cross validation

    classes_ : ndarray of shape (n_classes, )
        A list of class labels known to the classifier.

    coef_ : array of shape (n_features,)
        Estimated coefficients for the linear predictor (`X @ coef_ +
        intercept_`).

    intercept_ : float
        Intercept (a.k.a. bias) added to linear predictor.

    scoring_path_ : ndarray of shape (n_l1_ratio, n_alpha, n_folds)
        Classification score for the test set on each fold, varying l1_ratio and
        alpha.

    alphas_ : ndarray of shape (n_alphas,) or (n_l1_ratio, n_alphas)
        The grid of alphas used for fitting, for each l1_ratio.

    n_iter_ : int
        number of iterations run by the proximal gradient descent solver to
        reach the specified tolerance for the optimal alpha.

    See also
    --------
    logistic_sgl_path
    LogisticSGL
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
        scoring=None,
        cv=None,
        copy_X=True,
        verbose=False,
        n_jobs=None,
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
        self.scoring = scoring
        self.cv = cv
        self.copy_X = copy_X
        self.verbose = verbose
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit logistic sparse group lasso linear model

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
        # This makes sure that there is no duplication in memory.
        # Dealing right with copy_X is important in the following:
        # Multiple functions touch X and subsamples of X and can induce a
        # lot of duplication of memory
        copy_X = self.copy_X and self.fit_intercept

        check_y_params = dict(copy=False, ensure_2d=False, dtype=None)

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
                accept_sparse="csc", dtype=[np.float64, np.float32], copy=False
            )
            X, y = self._validate_data(
                X, y, validate_separately=(check_X_params, check_y_params)
            )
            if sparse.isspmatrix(X):
                if hasattr(reference_to_old_X, "data") and not np.may_share_memory(
                    reference_to_old_X.data, X.data
                ):
                    # X is a sparse matrix and has been copied
                    copy_X = False
            elif not np.may_share_memory(reference_to_old_X, X):
                # X has been copied
                copy_X = False
            del reference_to_old_X
        else:
            # Need to validate separately here.
            # We can't pass multi_ouput=True because that would allow y to be
            # csr. We also want to allow y to be 64 or 32 but check_X_y only
            # allows to convert for 64.
            check_X_params = dict(
                accept_sparse="csc",
                dtype=[np.float64, np.float32],
                order="F",
                copy=copy_X,
            )
            X, y = self._validate_data(
                X, y, validate_separately=(check_X_params, check_y_params)
            )
            copy_X = False

        _, self.n_features_in_ = X.shape

        if y.shape[0] == 0:
            raise ValueError("y has 0 samples: %r" % y)

        check_classification_targets(y)

        # Encode for string labels
        label_encoder = LabelEncoder().fit(y)
        y = label_encoder.transform(y)

        # The original class labels
        self.classes_ = label_encoder.classes_

        model = LogisticSGL()
        y = column_or_1d(y, warn=True)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X and y have inconsistent dimensions (%d != %d)"
                % (X.shape[0], y.shape[0])
            )

        groups = check_groups(self.groups, X, allow_overlap=False, fit_intercept=False)

        # All LogisticSGLCV parameters except "cv" and "n_jobs" are acceptable
        path_params = self.get_params()

        l1_ratios = np.atleast_1d(path_params["l1_ratio"])
        alphas = self.alphas
        n_l1_ratio = len(l1_ratios)
        if alphas is None:
            alphas = [
                _alpha_grid(
                    X=X,
                    y=y,
                    Xy=X.T.dot(y),
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

        # We are not computing in parallel, we can modify X
        # inplace in the folds
        if effective_n_jobs(self.n_jobs) > 1:
            path_params["copy_X"] = False

        if isinstance(self.verbose, int):
            path_params["verbose"] = self.verbose - 1

        # init cross-validation generator
        cv = check_cv(self.cv)

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
            delayed(logistic_sgl_scoring_path)(
                X=X,
                y=y,
                train=train,
                test=test,
                l1_ratio=this_l1_ratio,
                alphas=this_alphas,
                Xy=None,
                return_n_iter=False,
                check_input=True,
                **path_params,
            )
            for this_l1_ratio, this_alphas in zip(l1_ratios, alphas)
            for train, test in folds
        )

        if isinstance(self.verbose, int):
            parallel_verbosity = self.verbose - 2
            if parallel_verbosity < 0:
                parallel_verbosity = 0
        else:
            parallel_verbosity = self.verbose

        score_paths = ProgressParallel(
            n_jobs=self.n_jobs,
            verbose=parallel_verbosity,
            use_tqdm=bool(self.verbose),
            desc="L1 ratios * CV folds",
            total=n_l1_ratio * len(folds),
            **_joblib_parallel_args(prefer="threads"),
        )(jobs)

        coefs_paths, alphas_paths, scores, n_iters = zip(*score_paths)

        scores = np.reshape(scores, (n_l1_ratio, len(folds), -1))
        alphas_paths = np.reshape(alphas_paths, (n_l1_ratio, len(folds), -1))
        n_iters = np.reshape(n_iters, (n_l1_ratio, len(folds), -1))
        coefs_paths = np.reshape(coefs_paths, (n_l1_ratio, len(folds), -1, n_alphas))

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
        self.is_fitted_ = True
        return self

    def score(self, X, y):
        """Returns the score using the `scoring` option on the given
        test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            Score of self.predict(X) wrt. y.
        """
        scoring = self.scoring or "accuracy"
        scoring = get_scorer(scoring)

        return scoring(self, X, y)

    @property
    def chosen_features_(self):
        """An index array of chosen features"""
        return np.nonzero(self.coef_)[0]

    @property
    def sparsity_mask_(self):
        """A boolean array indicating which features survived regularization"""
        return self.coef_ != 0

    def like_nonzero_mask_(self, rtol=1e-8):
        """A boolean array indicating which features are zero or close to zero

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
        """A set of the group IDs that survived regularization"""
        if self.groups is not None:
            group_mask = [
                bool(set(grp).intersection(set(self.chosen_features_)))
                for grp in self.groups
            ]
            return np.nonzero(group_mask)[0]
        else:
            return self.chosen_features_

    def transform(self, X):
        """Remove columns corresponding to zeroed-out coefficients"""
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
        return {"binary_only": True, "requires_y": True}
