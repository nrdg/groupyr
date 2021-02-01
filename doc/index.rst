*Groupyr*: Sparse Group Lasso in Python
=======================================

*Groupyr* is a scikit-learn compatible implementation of the sparse group lasso
linear model. It is intended for high-dimensional supervised learning
problems where related covariates can be assigned to predefined groups.

The Sparse Group Lasso
----------------------

The sparse group lasso [1]_ is a penalized regression approach that combines the
group lasso with the normal lasso penalty to promote both global sparsity and
group-wise sparsity. It estimates a target variable :math:`\hat{y}` from a
feature matrix :math:`\mathbf{X}`, using

.. math::

    \hat{y} = \mathbf{X} \hat{\beta},

where the coefficients in :math:`\hat{\beta}` characterize the relationship
between the features and the target and must satisfy [1]_

.. math::

    \hat{\beta} = \min_{\beta} \frac{1}{2}
    || y - \sum_{\ell = 1}^{G} \mathbf{X}^{(\ell)} \beta^{(\ell)} ||_2^2
    + (1 - \alpha) \lambda \sum_{\ell = 1}^{G} \sqrt{p_{\ell}} ||\beta^{(\ell)}||_2
    + \alpha \lambda ||\beta||_1,
   
where :math:`G` is the total number of groups, :math:`\mathbf{X}^{(\ell)}` is
the submatrix of :math:`\mathbf{X}` with columns belonging to group
:math:`\ell`, :math:`\beta^{(\ell)}` is the coefficient vector of group
:math:`\ell`, and :math:`p_{\ell}` is the length of :math:`\beta^{(\ell)}`.
The model hyperparameter :math:`\alpha` controls the combination of the
group-lasso and the lasso, with :math:`\alpha=0` giving the group lasso fit
and :math:`\alpha=1` yielding the lasso fit. The hyperparameter
:math:`\lambda` controls the strength of the regularization.

.. toctree::
   :hidden:
   :titlesonly:

   Home <self>


.. toctree::
   :maxdepth: 3
   :hidden:

   install
   auto_examples/index
   getting_help
   api
   FAQ <faq>
   contributing
   Groupyr on GitHub <https://github.com/richford/groupyr>

`Installation <install.html>`_
------------------------------

See the `installation guide <install.html>`_ for installation instructions.

Usage
-----

*Groupyr* is compatible with the scikit-learn API and its estimators offer the
same instantiate, ``fit``, ``predict`` workflow that will be familiar to
scikit-learn users. See the `API <api.html>`_ and `examples
<auto_examples/index.html>`_ for full details. Here, we describe only the key
differences necessary for scikit-learn users to get started with *groupyr*.

For syntactic parallelism with the scikit-learn ``ElasticNet`` estimator, we
use the keyword ``l1_ratio`` to refer to SGL's :math:`\alpha` hyperparameter
above that controls the mixture of group lasso and lasso penalties. In
addition to keyword parameters shared with scikit-learn's ``ElasticNet``,
``ElasticNetCV``, ``LogisticRegression``, and ``LogisticRegressionCV``
estimators, users must specify the group assignments for the columns of the
feature matrix ``X``. This is done during estimator instantiation using the
``groups`` parameter, which accepts a list of numpy arrays, where the
:math:`i`-th array specifies the feature indices of the :math:`i`-th group.
If no grouping information is provided, the default behavior assigns all
features to one group.

*Groupyr* also offers cross-validation estimators that automatically select
the best values of the hyperparameters :math:`\alpha` and :math:`\lambda`
using either an exhaustive grid search (with ``tuning_strategy="grid"``) or
sequential model based optimization (SMBO) using the scikit-optimize library
(with ``tuning_strategy="bayes"``). For the grid search strategy, our
implementation is more efficient than using the base estimator with
scikit-learn's ``GridSearchCV`` because it makes use of warm-starting, where
the model is fit along a pre-defined regularization path and the solution
from the previous fit is used as the initial guess for the current
hyperparameter value. The randomness associated with SMBO complicates the use
of a warm start strategy; it can be difficult to determine which of the
previously attempted hyperparameter combinations should provide the initial
guess for the current evaluation. However, even without warm-starting, we
find that the SMBO strategy usually outperforms grid search because far fewer
evaluations are needed to arrive at the optimal hyperparameters. We provide
`examples <auto_examples/index.html>`_ of both strategies.

`API Documentation <api.html>`_
-------------------------------

See the `API Documentation <api.html>`_ for detailed documentation of the API.

`Examples <auto_examples/index.html>`_
--------------------------------------

And look at the `example gallery <auto_examples/index.html>`_ for a set of introductory examples.

Citing groupyr
--------------

If you use *groupyr* in a scientific publication, we would appreciate
citations. Please see our `citation instructions
<https://github.com/richford/groupyr#citing-groupyr>`_ for the latest
reference and a bibtex entry.

Acknowledgements
----------------

*Groupyr* development is supported through a grant from the `Gordon and Betty
Moore Foundation <https://www.moore.org/>`_ and from the `Alfred P. Sloan
Foundation <https://sloan.org/>`_ to the `University of Washington eScience
Institute <http://escience.washington.edu/>`_, as well as `NIMH BRAIN
Initiative grant 1RF1MH121868-01
<https://projectreporter.nih.gov/project_info_details.cfm?aid=9886761&icde=46874320&ddparam=&ddvalue=&ddsub=&cr=2&csb=default&cs=ASC&pball=)>`_
to Ariel Rokem (University of Washington).

The API design of *groupyr* was facilitated by the `scikit-learn project
template`_ and it therefore borrows heavily from `scikit-learn`_ [2]_.
*Groupyr* relies on the copt optimization library [3]_ for its solver. The
*groupyr* logo is a flipped silhouette of an `image from J. E. Randall`_ and is
licensed `CC BY-SA`_.

.. _scikit-learn project template: https://github.com/scikit-learn-contrib/project-template
.. _scikit-learn: https://scikit-learn.org/stable/index.html
.. _image from J. E. Randall: https://commons.wikimedia.org/wiki/File:Epinephelus_amblycephalus,_banded_grouper.jpg
.. _CC BY-SA: https://creativecommons.org/licenses/by-sa/3.0

References
----------
.. [1] Simon, N., Friedman, J., Hastie, T., & Tibshirani, R. (2013).
    A sparse-group lasso. Journal of Computational and Graphical
    Statistics, 22(2), 231-245.
.. [2] Pedregosa et al. (2011). `Scikit-learn: Machine Learning in Python`_.
    Journal of Machine Learning Research, 12, 2825-2830;
    Buitnick et al. (2013). `API design for machine learning software:
    experiences from the scikit-learn project`_. ECML PKDD Workshop: Languages
    for Data Mining and Machine Learning, 108-122.
.. [3] Pedregosa et al. (2020). `copt: composite optimization in Python`__.
    DOI:10.5281/zenodo.1283339.
    
.. _Scikit-learn\: Machine Learning in Python: http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html
.. _API design for machine learning software\: experiences from the scikit-learn project: https://arxiv.org/abs/1309.0238
.. __: http://openopt.github.io/copt/