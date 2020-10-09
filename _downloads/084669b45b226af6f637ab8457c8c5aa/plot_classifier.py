"""
======================================================
Logistic Sparse Group Lasso for grouped sparse signals
======================================================

Estimates a Sparse Group Lasso logistic regression model on a simulated
sparse signal. The estimated important features are compared with the
ground-truth.

"""
import numpy as np
from matplotlib import pyplot as plt
from groupyr import LogisticSGLCV
from groupyr.datasets import make_group_classification

X, y, groups, idx = make_group_classification(
    n_samples=100,
    n_groups=20,
    n_informative_groups=3,
    n_features_per_group=20,
    n_informative_per_group=10,
    n_redundant_per_group=0,
    n_repeated_per_group=0,
    n_classes=2,
    scale=100,
    useful_indices=True,
    random_state=1729,
)

_, n_features = X.shape

model = LogisticSGLCV(
    groups=groups, l1_ratio=[0.80, 0.90], n_alphas=40, tol=1e-3, eps=1e-2, cv=3
).fit(X, y)

plt.plot(
    np.arange(n_features),
    model.coef_,
    marker="o",
    mfc="black",
    mec="none",
    ms=3,
    mew=0,
    ls="",
    label="coefficients",
)

plt.plot(
    np.arange(n_features)[idx],
    model.coef_[idx],
    marker="o",
    mfc="none",
    mec="green",
    ms=5,
    mew=3,
    ls="",
    label="informative features",
)

plt.title("Estimated coefficients with ground truth imporant features highlighted")

plt.legend(loc="best")
plt.xlabel("Feature index")
plt.ylabel("Coefs")

plt.show()

print("Indices of ground-truth informative features:")
print(np.where(idx)[0])

print("Indices of non-zero estimated coefs:")
print(model.chosen_features_)
