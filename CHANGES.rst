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

