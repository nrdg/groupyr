import numpy as np
import pytest

from groupyr._prox import SparseGroupL1

proximal_penalties = [
    (SparseGroupL1(0.5, 1.0, groups=[np.arange(16)]), 16),
    (SparseGroupL1(0.5, 1.0, groups=[np.arange(16)], scale_l2_by=None), 16),
    (SparseGroupL1(0.5, 1.0, groups=[np.arange(16)], bias_index=16), 17),
]


@pytest.mark.parametrize("penalty, n_features", proximal_penalties)
def test_three_inequality(penalty, n_features):
    """Test the prox using the three point inequality
    The three-point inequality is described e.g., in Lemma 1.4
    in "Gradient-Based Algorithms with Applications to Signal
    Recovery Problems", Amir Beck and Marc Teboulle
    """
    for _ in range(10):
        z = np.random.randn(n_features)
        u = np.random.randn(n_features)
        xi = penalty.prox(z, 1.0)

        lhs = 2 * (penalty(xi) - penalty(u))
        rhs = (
            np.linalg.norm(u - z) ** 2
            - np.linalg.norm(u - xi) ** 2
            - np.linalg.norm(xi - z) ** 2
        )

        assert lhs <= rhs, penalty  # nosec
