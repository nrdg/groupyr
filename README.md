![groupyr logo](https://raw.githubusercontent.com/richford/groupyr/main/doc/_static/groupyr-logo-large.svg)

# _Groupyr_: Sparse Group Lasso in Python

[![Build Status](https://github.com/richford/groupyr/workflows/Build/badge.svg)](https://github.com/richford/groupyr/workflows/Build/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/richford/groupyr/badge.svg?branch=main&service=github)](https://coveralls.io/github/richford/groupyr?branch=main&service=github)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
<br>
[![DOI](https://joss.theoj.org/papers/10.21105/joss.03024/status.svg)](https://doi.org/10.21105/joss.03024)
[![DOI](https://zenodo.org/badge/300933639.svg)](https://zenodo.org/badge/latestdoi/300933639)

_Groupyr_ is a Python library for penalized regression of grouped covariates.
This is the _groupyr_ development site. You can view the source code, file new issues, and contribute to _groupyr_'s development. If you just want to learn how to install and use _groupyr_, please look at the [_groupyr_ documentation][link_groupyr_docs].

## Contributing

We love contributions! _Groupyr_ is open source, built on open source,
and we'd love to have you hang out in our community.

We have developed some [guidelines](.github/CONTRIBUTING.md) for contributing to
_groupyr_.

## Citing _groupyr_

If you use _groupyr_ in a scientific publication, please see cite us:

Richie-Halford et al., (2021). Groupyr: Sparse Group Lasso in Python. Journal of Open Source Software, 6(58), 3024, https://doi.org/10.21105/joss.03024

```
@article{richie-halford-groupyr,
    doi = {10.21105/joss.03024},
    url = {https://doi.org/10.21105/joss.03024},
    year = {2021},
    publisher = {The Open Journal},
    volume = {6},
    number = {58},
    pages = {3024},
    author = {Adam {R}ichie-{H}alford and Manjari Narayan and Noah Simon and Jason Yeatman and Ariel Rokem},
    title = {{G}roupyr: {S}parse {G}roup {L}asso in {P}ython},
    journal = {Journal of Open Source Software}
}
```

## Acknowledgements

_Groupyr_ development is supported through a grant from the [Gordon
and Betty Moore Foundation](https://www.moore.org/) and from the
[Alfred P. Sloan Foundation](https://sloan.org/) to the [University of
Washington eScience Institute](http://escience.washington.edu/), as
well as
[NIMH BRAIN Initiative grant 1RF1MH121868-01](https://projectreporter.nih.gov/project_info_details.cfm?aid=9886761&icde=46874320&ddparam=&ddvalue=&ddsub=&cr=2&csb=default&cs=ASC&pball=)
to Ariel Rokem (University of Washington).

The API design of _groupyr_ was facilitated by the [scikit-learn project
template](https://github.com/scikit-learn-contrib/project-template) and it
therefore borrows heavily from
[scikit-learn](https://scikit-learn.org/stable/index.html). _Groupyr_ relies
on the [copt optimization library](http://openo.pt/copt/index.html) for its
solver. The _groupyr_ logo is a flipped silhouette of an [image from J. E.
Randall](https://commons.wikimedia.org/wiki/File:Epinephelus_amblycephalus,_banded_grouper.jpg)
and is licensed [CC BY-SA](https://creativecommons.org/licenses/by-sa/3.0).

[link_groupyr_docs]: https://richford.github.io/groupyr/
