import operator

import numpy as np
import pytest

from pandas.core.dtypes.missing import isna

import pandas._testing as tm
import pandas.core.ops.array_ops as array_ops
from pandas.core.ops.array_ops import (
    arithmetic_op,
    comparison_op,
    na_logical_op,
)


def test_na_logical_op_2d():
    left = np.arange(8).reshape(4, 2)
    right = left.astype(object)
    right[0, 0] = np.nan

    # Check that we fall back to the vec_binop branch
    with pytest.raises(TypeError, match="unsupported operand type"):
        operator.or_(left, right)

    result = na_logical_op(left, right, operator.or_)
    expected = right
    tm.assert_numpy_array_equal(result, expected)


def test_object_comparison_2d():
    left = np.arange(9).reshape(3, 3).astype(object)
    right = left.T

    result = comparison_op(left, right, operator.eq)
    expected = np.eye(3).astype(bool)
    tm.assert_numpy_array_equal(result, expected)

    # Ensure that cython doesn't raise on non-writeable arg, which
    #  we can get from np.broadcast_to
    right.flags.writeable = False
    result = comparison_op(left, right, operator.ne)
    tm.assert_numpy_array_equal(result, ~expected)


@pytest.mark.parametrize("rvalues", [1, [1, 1, 1], np.nan, None])
@pytest.mark.parametrize(
    "op", [operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge]
)
def test_comparison_for_subclasses(rvalues, op):
    # GH#63205 Ensure subclasses of ndarray are correctly handled in comparison_op
    # Define a custom ndarray subclass
    class TestArray(np.ndarray):
        def __new__(cls, input_array):
            return np.asarray(input_array).view(cls)

        def __array_finalize__(self, obj) -> None:
            self._is_test_array = True

    def expected_with_na_handling(lvalues, rvalues, op):
        # Similar to comparison_op, handle zerodim arrays with na value separately
        if (rvalues.ndim == 0) and isna(rvalues.item()):
            # numpy does not like comparisons vs None
            if op is operator.ne:
                return np.ones(lvalues.shape, dtype=bool)
            else:
                return np.zeros(lvalues.shape, dtype=bool)
        return op(lvalues, rvalues)

    # Define test data
    lvalues = [1, 2, 3]

    # Test with both ndarray and TestArray
    result = comparison_op(np.array(lvalues), np.array(rvalues), op)
    expected = expected_with_na_handling(np.array(lvalues), np.array(rvalues), op)
    tm.assert_numpy_array_equal(result, expected)

    result = comparison_op(TestArray(lvalues), TestArray(rvalues), op)
    expected = expected_with_na_handling(TestArray(lvalues), TestArray(rvalues), op)
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("op", [operator.eq, operator.ne])
@pytest.mark.parametrize(
    "scalar",
    [
        np.int32(4),
        np.uint64(5),
        np.uint64(2**63),
        3.0,
        np.float64(5.0),
        3.5,
        np.nan,
        np.inf,
    ],
)
def test_comparison_op_aarch64_int64_scalar_fastpath(monkeypatch, op, scalar):
    left = np.array([np.iinfo(np.int64).min, -4, -3, 0, 3, 5], dtype=np.int64)

    monkeypatch.setattr(array_ops, "_USE_AARCH64_COMPARISON_FASTPATH", True)
    result = comparison_op(left, scalar, op)

    monkeypatch.setattr(array_ops, "_USE_AARCH64_COMPARISON_FASTPATH", False)
    expected = comparison_op(left, scalar, op)
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("op", [operator.eq, operator.ne])
@pytest.mark.parametrize(
    "scalar",
    [
        np.float64(2**53 - 1),
        np.float64(-(2**53 - 1)),
        np.float64(2**53),
        np.float64(-(2**53)),
    ],
)
def test_comparison_op_aarch64_int64_large_float_scalar(monkeypatch, op, scalar):
    left = np.array(
        [-(2**53) - 1, -(2**53 - 1), 2**53 - 1, 2**53 + 1],
        dtype=np.int64,
    )

    monkeypatch.setattr(array_ops, "_USE_AARCH64_COMPARISON_FASTPATH", True)
    result = comparison_op(left, scalar, op)

    monkeypatch.setattr(array_ops, "_USE_AARCH64_COMPARISON_FASTPATH", False)
    expected = comparison_op(left, scalar, op)
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "op",
    [
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
    ],
)
@pytest.mark.parametrize("scalar", [0, 2, np.int32(4), 2**54 + 1])
def test_arithmetic_op_aarch64_float64_int_scalar_fastpath(monkeypatch, op, scalar):
    left = np.array([-np.inf, -3.5, -0.0, 0.0, 2.5, np.inf, np.nan])

    monkeypatch.setattr(array_ops, "_USE_AARCH64_FLOAT64_SCALAR_FASTPATH", True)
    with np.errstate(all="ignore"):
        result = arithmetic_op(left, scalar, op)

    monkeypatch.setattr(array_ops, "_USE_AARCH64_FLOAT64_SCALAR_FASTPATH", False)
    with np.errstate(all="ignore"):
        expected = arithmetic_op(left, scalar, op)
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("op", [operator.eq, operator.ne])
@pytest.mark.parametrize("scalar", [2, np.int32(4), 2**54 + 1])
def test_comparison_op_aarch64_float64_int_scalar_fastpath(monkeypatch, op, scalar):
    left = np.array([-np.inf, -3.5, -0.0, 0.0, 2.5, np.inf, np.nan])

    monkeypatch.setattr(array_ops, "_USE_AARCH64_FLOAT64_SCALAR_FASTPATH", True)
    result = comparison_op(left, scalar, op)

    monkeypatch.setattr(array_ops, "_USE_AARCH64_FLOAT64_SCALAR_FASTPATH", False)
    expected = comparison_op(left, scalar, op)
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("scalar", [True, np.bool_(True), np.bool_(False)])
def test_float64_scalar_fastpath_excludes_bool(monkeypatch, scalar):
    left = np.array([0.0, 1.0], dtype=np.float64)

    monkeypatch.setattr(array_ops, "_USE_AARCH64_FLOAT64_SCALAR_FASTPATH", True)
    result = array_ops._maybe_cast_int_scalar_for_float64_op_aarch64(
        left, scalar, operator.eq
    )
    assert result is scalar


@pytest.mark.parametrize(
    "dtype, scalar, op, expected",
    [
        (np.float64, 2, operator.add, True),
        (np.float64, np.float64(5.0), operator.ne, True),
        (np.float64, True, operator.add, False),
        (np.float64, np.bool_(True), operator.eq, False),
        (np.int64, 2, operator.add, True),
        (np.int64, np.int32(4), operator.mul, False),
        (np.int64, 3.0, operator.mul, True),
        (np.int64, np.float64(5.0), operator.ne, True),
        (np.int64, "5", operator.eq, False),
        (np.int32, 2, operator.add, False),
    ],
)
def test_aarch64_numexpr_bypass(monkeypatch, dtype, scalar, op, expected):
    left = np.array([1, 2, 3], dtype=dtype)

    monkeypatch.setattr(array_ops, "_USE_AARCH64_NUMEXPR_BYPASS", True)
    result = array_ops._should_bypass_numexpr_aarch64(left, scalar, op)
    assert result is expected

    monkeypatch.setattr(array_ops, "_USE_AARCH64_NUMEXPR_BYPASS", False)
    result = array_ops._should_bypass_numexpr_aarch64(left, scalar, op)
    assert result is False
