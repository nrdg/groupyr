v0.2.4 (June 22, 2021)
======================
  * Add sgl_path example to the documentation (#58)
  * Add GroupPCA, and supervised PCA variants (#55)

v0.2.3 (March 17, 2021)
=======================
  * ENH: Add GroupFPCA (#48)
  * ENH: Add group transformers (#51)
  * DOC: Update README.md with JOSS article info (#53)
  * DOC: One typo (#52)
  * CI: Only publish docs to GitHub pages for one Python version (#49)

v0.2.2 (February 24, 2021)
==========================
  * Micro release to accompany publication of JOSS paper

v0.2.1 (February 21, 2021)
==========================
  * DEP: Loosen dependency requirements (#46)
  * DOC: Add groupyr/group-lasso comparison example (#44)
  * Move matplotlib dependency to dev option (#45)
  * DOC: Add author contributions using the CRediT taxonomy (#42)
  * DOC: Add help target to makefile to make is self-documenting (#43)
  * ENH: Add lightning and group-lasso citations to paper (#41)
  * DEP: remove ipywidgets from setup.cfg dependencies (#40)
  * Remove redundant paper reference (#38)

v0.1.10 (December 10, 2020)
===========================
  * FIX: Assign error_score in BayesSearchCV (#34)

v0.1.9 (December 09, 2020)
==========================
  * ENH: Use sgl_scoring_path instead of sklearn's _path_residuals (#33)
  * FIX: Sets bayes_optimizer_ to None when "grid" strategy is used (#32)

v0.1.8 (December 05, 2020)
==========================
  * ENH: Add BayesSearchCV option to SGLCV and LogisticSGLCV (#31)

v0.1.7 (October 26, 2020)
=========================
  * ENH: Use joblib Parallel instead of custom _ProgressParallel wrapper (#28)


v0.1.6 (October 22, 2020)
=========================
  * DOC: Fix mathjax rendering in documentation (#27)


v0.1.5 (October 15, 2020)
=========================
  * ENH: Speed up `SparseGroupL1.prox()` (#23)


v0.1.4 (October 09, 2020)
=========================
  * DOC: Update citation instructions (#22)
  * STY: Prefer `python -m pip` over `pip`. Also use taxicab random_state instead of 42 (#21)
  * FIX: Use classifier=True in check_cv for LogisticSGLCV (#20)
  * MAINT: Automatically update zenodo file as part of release script (#17)
  * FIX: Shuffle groups make_group_regression. Use `generator.choice` in make_group_classification (#16)
  * TST: Add tests for `SGL`, `SGLCV`, etc. (#13)


v0.1.3 (October 05, 2020)
=========================
  * TST: Test sparsity masks and chosen groups/features in _base.py (#12)
  * TST: Test check_groups in utils.py (#11)
  * DOC: Add pull request examples to CONTRIBUTING.md (#10)
  * CI: Use pydocstyle in github actions (#8)
  * DOC: Replaces CRCNS grant with our BRAINI grant in README (#7)


v0.1.2 (October 03, 2020)
=========================

- Bump version to confirm GitHub action behavior.


v0.1.1 (October 03, 2020)
=========================

- Fix automatic documentation building.


v0.1.0 (October 03, 2020)
=========================

- Initial release

