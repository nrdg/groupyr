import numpy as np
import pytest

from groupyr.decomposition import GroupFPCA
from skfda import FDataGrid
from skfda.datasets import fetch_weather
from skfda.representation.basis import BSpline, Constant, Fourier
from sklearn.utils.estimator_checks import check_estimator


def test_group_fpca_matches_skfda_results():
    n_basis = 9
    n_components = 3

    grid_points = np.arange(0.5, 365, 1)
    fd_data = fetch_weather()["data"].coordinates[0]
    fd_data = FDataGrid(np.squeeze(fd_data.data_matrix), grid_points)
    X = fd_data.data_matrix.squeeze()
    X = np.hstack([X, X])
    groups = [np.arange(0, 365), np.arange(365, 730)]
    gfpca = GroupFPCA(
        n_components=n_components,
        n_basis=n_basis,
        basis_domain_range=(0, 365),
        basis="Fourier",
        groups=groups,
    )

    _ = gfpca.fit_transform(X, grid_points=np.hstack([grid_points, grid_points]))

    results = [
        [
            0.9231551,
            0.1364966,
            0.3569451,
            0.0092012,
            -0.0244525,
            -0.02923873,
            -0.003566887,
            -0.009654571,
            -0.0100063,
        ],
        [
            -0.3315211,
            -0.0508643,
            0.89218521,
            0.1669182,
            0.2453900,
            0.03548997,
            0.037938051,
            -0.025777507,
            0.008416904,
        ],
        [
            -0.1379108,
            0.9125089,
            0.00142045,
            0.2657423,
            -0.2146497,
            0.16833314,
            0.031509179,
            -0.006768189,
            0.047306718,
        ],
    ]
    results = np.array(results)

    for grp_idx in range(len(groups)):
        # The sign of the components is arbitrary so we change the sign if necessary
        for i in range(n_components):
            if np.sign(gfpca.components_[grp_idx].coefficients[i][0]) != np.sign(
                results[i][0]
            ):
                results[i, :] *= -1

        np.testing.assert_allclose(
            gfpca.components_[grp_idx].coefficients, results, atol=1e-7
        )


@pytest.mark.parametrize(
    "basis", ["constant", "bspline", None, BSpline, Constant, Fourier]
)
@pytest.mark.parametrize("use_grid_points", [True, False])
@pytest.mark.parametrize("exclude_groups", [None, 1, [1]])
def test_group_fpca_bases(basis, use_grid_points, exclude_groups):
    n_basis = 9
    n_components = 1

    grid_points = np.arange(0.5, 365, 1)
    fd_data = fetch_weather()["data"].coordinates[0]
    fd_data = FDataGrid(np.squeeze(fd_data.data_matrix), grid_points)
    X = fd_data.data_matrix.squeeze()
    X = np.hstack([X, X])
    groups = [np.arange(0, 365), np.arange(365, 730)]

    if use_grid_points:
        grid_points = np.hstack([grid_points, grid_points])
    else:
        grid_points = None

    gfpca = GroupFPCA(
        n_components=n_components,
        n_basis=n_basis,
        basis_domain_range=(0, 365),
        basis=basis,
        groups=groups,
        exclude_groups=exclude_groups,
    ).fit(X, grid_points=grid_points)

    _ = gfpca.transform(X, grid_points=grid_points)


def test_groupfpca_errors():
    n_basis = 9
    n_components = 1

    grid_points = np.arange(0.5, 365, 1)
    fd_data = fetch_weather()["data"].coordinates[0]
    fd_data = FDataGrid(np.squeeze(fd_data.data_matrix), grid_points)
    X = fd_data.data_matrix.squeeze()
    X = np.hstack([X, X])
    groups = [np.arange(0, 365), np.arange(365, 730)]

    with pytest.raises(ValueError):
        GroupFPCA(
            n_components=n_components, n_basis=n_basis, basis=object, groups=groups
        ).fit(X)

    with pytest.raises(ValueError):
        GroupFPCA(
            n_components=n_components, n_basis=n_basis, basis="error", groups=groups
        ).fit(X)

    with pytest.raises(TypeError):
        GroupFPCA(
            n_components=n_components,
            n_basis=n_basis,
            groups=groups,
            exclude_groups="error",
        ).fit(X)


@pytest.mark.parametrize("Transformer", [GroupFPCA])
def test_all_estimators(Transformer):
    return check_estimator(Transformer())
