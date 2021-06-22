#############
API Reference
#############

*Groupyr* contains estimator classes that are fully compliant
with the `scikit-learn <https://scikit-learn.org>`_ ecosystem. Consequently,
their initialization, ``fit``, ``predict``, ``transform``, and ``score``
methods will be familiar to ``sklearn`` users.

.. currentmodule:: groupyr

Sparse Groups Lasso Estimators
==============================

These are *groupyr*'s canonical estimators. ``SGL`` is intended for regression
problems while ``LogisticSGL`` is intended for classification problems.

.. autoclass:: SGL

.. autoclass:: LogisticSGL

Cross-validation Estimators
===========================

These estimators have built-in `cross-validation
<https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-evaluating-estimator-performance>`_
capabilities to find the best values of the hyperparameters ``alpha`` and
``l1_ratio``. These are more efficient than using the canonical estimators
with grid search because they make use of warm-starting. Alternatively, you
can specify ``tuning_strategy = "bayes"`` to use `Bayesian optimization over
the hyperparameters
<https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html>`_
instead of a grid search.

.. autoclass:: SGLCV

.. autoclass:: LogisticSGLCV

Dataset Generation
==================

Use these functions to generate synthetic sparse grouped data.

.. currentmodule:: groupyr.datasets

.. autofunction:: make_group_classification

.. autofunction:: make_group_regression

Regularization Paths
====================

Use these functions to compute regression coefficients along a regularization path.

.. currentmodule:: groupyr

.. autofunction:: sgl_path

.. currentmodule:: groupyr.logistic

.. autofunction:: logistic_sgl_path