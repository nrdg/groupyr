.. _faq-label:

Frequently Asked Questions
==========================

Here we'll maintain a list of frequently asked questions about *groupyr*. Do
you have a question that isn't addressed here? If so, please see our `getting
help page <getting_help.html>`_ for information about how to file a new
issue.

.. dropdown:: Why did we create *groupyr* and how does it compare to other similar packages?

    We created *groupyr* to be useful in our own research and we hope it is
    useful in yours. There are other packages for penalized regression in
    python. The `lightning <http://contrib.scikit-learn.org/lightning/>`_
    package has a lasso penalty but does not allow the user to specify
    groups. The `group_lasso
    <https://group-lasso.readthedocs.io/en/latest/#>`_ package is well
    designed and documented, but we found that *groupyr*'s execution time was
    faster for most problems. We also wanted estimators with built-in
    cross-validation using both grid search and the ``BayesSearchCV``
    sequential model based optimization.

    In the future, we hope that *groupyr* will include other methods for
    statistical learning with grouped covariates (e.g. unsupervised learning
    methods). These would also be out of scope for the aforementioned
    libraries. However, we encourage you to try many tools. If you find that
    another one is better suited to your problem, please `leave us some
    feedback <https://github.com/richford/groupyr/issues/new/choose>`_, go
    forth, and do good work. Happy coding!
