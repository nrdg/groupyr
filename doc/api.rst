#############
API Reference
#############

.. currentmodule:: groupyr

Sparse Groups Lasso Estimators
==============================

These are groupyr's canonical estimators. ``SGL`` is intended for regression
problems while ``LogisticSGL`` is intended for classification problems.

.. autoclass:: SGL

.. autoclass:: LogisticSGL

Cross-validation Estimators
===========================

These estimators have built-in cross-validation capabilities to find the best
values of the hyperparameters ``alpha`` and ``l1_ratio``. These are more
efficient than using the canonical estimators with grid search because they
make use of warm-starting.

.. autoclass:: SGLCV

.. autoclass:: LogisticSGLCV

Dataset Generation
==================

Use these functions to generate synthetic sparse grouped data.

.. currentmodule:: groupyr.datasets

.. autofunction:: make_group_classification

.. autofunction:: make_group_regression
