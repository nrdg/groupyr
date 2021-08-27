import numpy as np
import pytest

from groupyr.datasets import make_group_regression
from groupyr.utils import check_groups, _stringify_sequence
from sklearn.utils._testing import assert_array_almost_equal


@pytest.mark.parametrize(
    "_input, reference",
    [
        ("string", "string"),
        (["foo", "bar", "baz"], "foo_bar_baz"),
        (("foo", "bar", 1), "foo_bar_1"),
    ],
)
def test_stringify_sequence(_input, reference):
    output = _stringify_sequence(_input)
    assert output == reference  # nosec


def test_stringify_sequence_error():
    with pytest.raises(TypeError):
        _stringify_sequence(0)


def test_check_groups():
    X, y, groups_in = make_group_regression()

    _, n_features = X.shape

    groups_out = check_groups(groups=groups_in, X=X)
    assert groups_out == tuple(groups_in)

    # Test the groups=None defaults
    groups_out = check_groups(groups=None, X=X, fit_intercept=False)
    assert_array_almost_equal(groups_out, [np.arange(n_features)])

    groups_out = check_groups(groups=None, X=X, fit_intercept=True)
    assert_array_almost_equal(groups_out, [np.arange(n_features - 1)])

    # Test Value error on missing features in groups
    with pytest.raises(ValueError):
        check_groups(groups=groups_in[1:], X=X)

    # Test Value error on missing groups in features
    with pytest.raises(ValueError):
        check_groups(
            groups=groups_in + [np.arange(n_features + 1, n_features + 20)], X=X
        )

    # Test Value error on overlapping groups if allow_overlap is False
    groups_in += [np.arange(n_features)]
    with pytest.raises(ValueError):
        check_groups(groups=groups_in, X=X, allow_overlap=False)

    # And then test that it works if allow_overlap is True
    groups_out = check_groups(groups=groups_in, X=X, allow_overlap=True)
    assert groups_out == tuple(groups_in)
