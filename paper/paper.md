---
title: 'Groupyr: Sparse Group Lasso in Python'
tags:
  - Python
  - group lasso
  - penalized regression
  - classification
authors:
  - name: Adam Richie-Halford
    orcid: 0000-0001-9276-9084
    affiliation: 1
  - name: Manjari Narayan
    orcid: 0000-0001-5348-270X
    affiliation: 2
  - name: Noah Simon
    orcid: 0000-0002-8985-2474
    affiliation: 4
  - name: Jason Yeatman
    orcid: 0000-0002-2686-1293
    affiliation: 5
  - name: Ariel Rokem
    orcid: 0000-0003-0679-1985
    affiliation: 3
affiliations:
  - name: eScience Institute, University of Washington
    index: 1
  - name: Department of Psychiatry and Behavioral Sciences, Stanford University
    index: 2
  - name: Department of Psychology, University of Washington
    index: 3
  - name: Department of Biostatistics, University of Washington
    index: 4
  - name: Graduate School of Education and Division of Developmental and Behavioral Pediatrics, Stanford University
    index: 5
date: 25 Dec 2021
bibliography: paper.bib
---

## Summary

For high-dimensional supervised learning, it is often beneficial to use
domain-specific knowledge to improve the performance of statistical learning
models. When the problem contains covariates which form groups, researchers
can include this grouping information to find parsimonious representations
of the relationship between covariates and targets. These groups may arise
artificially, as from the polynomial expansion of a smaller feature space, or
naturally, as from the anatomical grouping of different brain regions or the
geographical grouping of different cities. When the number of features is
large compared to the number of observations, one seeks a subset of the
features which is sparse at both the group and global level.

The sparse group lasso [@simon2013sparse] is a penalized regression technique
designed for exactly these situations. It combines the original lasso
[@tibshirani1996regression], which induces global sparsity, with the group
lasso [@yuan2006model], which induces group-level sparsity. It estimates a target variable $\hat{y}$ from a
feature matrix $\mathbf{X}$, using

$$
\hat{y} = \mathbf{X} \hat{\beta},
$$

as depicted in \autoref{fig:sgl_model}, with color encoding the group
structure of the covariates in $\mathbf{X}$. The coefficients in
$\hat{\beta}$ characterize the relationship between the features and the
target and must satisfy [@simon2013sparse]

$$
\hat{\beta} = \min_{\beta} \frac{1}{2}
|| y - \sum_{\ell = 1}^{G} \mathbf{X}^{(\ell)} \beta^{(\ell)} ||_2^2
+ (1 - \lambda) \alpha \sum_{\ell = 1}^{G} \sqrt{p_{\ell}} ||\beta^{(\ell)}||_2
+ \lambda \alpha ||\beta||_1,
$$
where $G$ is the total number of groups, $\mathbf{X}^{(\ell)}$ is the
submatrix of $\mathbf{X}$ with columns belonging to group $\ell$,
$\beta^{(\ell)}$ is the coefficient vector of group $\ell$, and $p_{\ell}$ is
the length of $\beta^{(\ell)}$. The model hyperparameter $\lambda$ controls
the combination of the group-lasso and the lasso, with $\lambda=0$ giving the
group lasso fit and $\lambda=1$ yielding the lasso fit. The hyperparameter
$\alpha$ controls the overall strength of the regularization.

![A linear model, $y = \mathbf{X} \cdot \beta$, with grouped covariates. The feature matrix $\mathbf{X}$ is color-coded to reveal a group structure. The coefficients in $\beta$ follow the same grouping. \label{fig:sgl_model}](groupyr_linear_model.pdf)

## Statement of need

*Groupyr* is a Python library that implements the sparse group lasso
as scikit-learn [@sklearn; @sklearn_api] compatible estimators.
It satisfies the need for grouped penalized regression models that
can be used interoperably in researcher's real-world scikit-learn
workflows. Some pre-existing Python libraries come close to satisfying
this need. [*Lightning*](http://contrib.scikit-learn.org/lightning/) [@lightning-2016]
is a Python library for large-scale linear classification and
regression. It supports many solvers with a combination of the
L1 and L2 penalties. However, it does not allow the user to
specify groups of covariates (see, for example, [this GitHub
issue](https://github.com/scikit-learn-contrib/lightning/issues/39)).
Another Python package,
[*group_lasso*](https://group-lasso.readthedocs.io/en/latest/#) [@group-lasso], is a
well-designed and well-documented implementation of the sparse group lasso.
It meets the basic API requirements of scikit-learn compatible estimators.
However, we found that our implementation in *groupyr*, which relies on the
*copt* optimization library [@copt], was between two and ten times faster
for the problem sizes that we encounter in our research (see the
repository's examples directory for a performance comparison).
Additionally, we needed estimators with built-in cross-validation
support using both grid search and sequential model based optimization
strategies. For example, the speed and cross-validation enhancements
were crucial to using *groupyr* in *AFQ-Insight*, a neuroinformatics
research library [@richiehalford2019multidimensional].

## Usage

*Groupyr* is available on the Python Package Index (PyPI) and can be installed
with

```shell
pip install groupyr
```

*Groupyr* is compatible with the scikit-learn API and its estimators offer the
same instantiate, ``fit``, ``predict`` workflow that will be familiar to
scikit-learn users. See the online documentation for a detailed description of the
API and examples in both classification and regression settings. Here, we describe
only the key differences necessary for scikit-learn users to get started with *groupyr*.

For syntactic parallelism with the scikit-learn ``ElasticNet`` estimator, we use the
keyword ``l1_ratio`` to refer to SGL's $\lambda$ hyperparameter. In addition
to keyword parameters shared with scikit-learn's ``ElasticNet``,
``ElasticNetCV``, ``LogisticRegression``, and ``LogisticRegressionCV``
estimators, users must specify the group assignments for the columns of the
feature matrix ``X``. This is done during estimator instantiation using the
``groups`` parameter, which accepts a list of numpy arrays, where the $i$-th
array specifies the feature indices of the $i$-th group. If no grouping
information is provided, the default behavior assigns all features to one
group.

*Groupyr* also offers cross-validation estimators that automatically select
the best values of the hyperparameters $\alpha$ and $\lambda$ using either an
exhaustive grid search (with ``tuning_strategy="grid"``) or sequential model
based optimization (SMBO) using the scikit-optimize library (with
``tuning_strategy="bayes"``). For the grid search strategy, our
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
examples of both strategies (grid search for a classification example and
SMBO for a regression example) in the online documentation.

## Author statements and acknowledgments

The first author (referred to as A.R.H. below) is the lead and corresponding
author. The last author (referred to as A.R.) is the primary supervisor and
is responsible for funding acquisition. All other authors are listed in
alphabetical order by surname. We describe contributions to the paper using
the CRediT taxonomy [@credit].
Writing – Original Draft: A.R.H.;
Writing – Review & Editing: A.R.H., N.S., J.Y., and A.R.;
Conceptualization and methodology: A.R.H., N.S., and A.R.;
Software and data curation: A.R.H., M.N., and A.R.;
Validation: A.R.H. and M.N.;
Resources: A.R.H. and A.R;
Visualization: A.R.H.;
Supervision: N.S., J.Y., and A.R.;
Project Administration: A.R.H;
Funding Acquisition: A.R.;

Groupyr development was supported through a grant from the Gordon and
Betty Moore Foundation and from the Alfred P. Sloan Foundation to the
University of Washington eScience Institute, as well as NIMH BRAIN
Initiative grant 1RF1MH121868-01 to Ariel Rokem at the University of
Washington and through cloud credits from the Google Cloud Platform.

## References
