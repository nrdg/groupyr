"""
=============================================================
Rudimentary performance comparison of groupyr and group-lasso
=============================================================

This example performs a rudimentary performance comparison of the groupyr and
group-lasso packages for various problem sizes. Speedup is defined as the
ratio of time to fit a `group-lasso.GroupLasso` model to the time to fit a
`groupyr.SGL` model.

In addition to `groupyr`, you will need to install the `group-lasso` package
using
```
pip install group-lasso
```
"""
import groupyr as gpr
import matplotlib.pyplot as plt
import numpy as np

from group_lasso import GroupLasso
from sklearn.model_selection import train_test_split
from timeit import timeit
from tqdm.auto import tqdm

n_features = [5, 10, 25, 50, 75, 100, 150, 200, 250]
speedup = []

for n_features_per_group in tqdm(n_features):
    X, y, groups, coef = gpr.datasets.make_group_regression(
        n_samples=400,
        n_groups=50,
        n_informative_groups=5,
        n_features_per_group=n_features_per_group,
        n_informative_per_group=int(0.8 * n_features_per_group),
        noise=200,
        coef=True,
        random_state=10,
        shuffle=True,
    )

    gl_groups = np.concatenate(
        [[idx] * len(grp) for idx, grp in enumerate(groups)]
    ).reshape([-1, 1])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=10
    )

    sgl = gpr.SGL(groups=groups, l1_ratio=0.5, alpha=1.0, max_iter=1000, tol=1e-3)

    gl = GroupLasso(
        groups=gl_groups,
        group_reg=0.5,
        l1_reg=0.5,
        frobenius_lipschitz=True,
        scale_reg="group_size",
        n_iter=1000,
        tol=1e-3,
        supress_warning=True,
    )

    gpr_time = timeit("sgl.fit(X_train, y_train)", globals=globals(), number=3)
    gl_time = timeit("gl.fit(X_train, y_train)", globals=globals(), number=3)

    speedup.append(gpr_time / gl_time)

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

_ = ax.plot(np.array(n_features) * 50, 1 / np.array(speedup))
_ = ax.set_xlabel(r"Number of features", fontsize=16)
_ = ax.set_ylabel(r"$t$(group-lasso) / $t$(groupyr)", fontsize=16)
_ = ax.set_title(
    "\n".join([r"Rudimentary performance comparison", r"of groupyr and group-lasso"]),
    fontsize=16,
)

fig.savefig("./groupyr_speedup.pdf", bbox_inches="tight")
