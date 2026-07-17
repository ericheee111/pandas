import numpy as np
import pytest

from pandas._libs import ops


@pytest.mark.parametrize(
    "left,right",
    [
        (np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
        (
            np.array([0, 1, -1, np.iinfo(np.int64).min, np.iinfo(np.int64).max]),
            np.array([0, 0, 0, -1, 2]),
        ),
        (
            np.array([4, -9, 7, -8], dtype=np.int64),
            np.array([2, 3, -2, -4], dtype=np.int64),
        ),
    ],
)
def test_int64_true_divide(left, right):
    with np.errstate(all="ignore"):
        expected = np.true_divide(left, right)

    result = ops.int64_true_divide(left, right)
