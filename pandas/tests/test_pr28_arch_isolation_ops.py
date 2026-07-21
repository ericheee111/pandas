import operator

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import masked
from pandas.core.arrays.base import ExtensionArray
from pandas.core.ops import array_ops
from pandas.core import series


def test_int64_divide_non_arm_uses_na_arithmetic(monkeypatch):
    monkeypatch.setattr(array_ops, "IS_ARM", False, raising=False)
    monkeypatch.setattr(
        array_ops.libops,
        "int64_true_divide",
        lambda *args: pytest.fail("ARM helper called on non-ARM"),
    )

    result = array_ops.arithmetic_op(
        np.array([4, 9], dtype=np.int64),
        np.array([2, 3], dtype=np.int64),
        operator.truediv,
    )

    tm.assert_numpy_array_equal(result, np.array([2.0, 3.0]))


def test_series_arithmetic_manager_non_arm_uses_constructor(monkeypatch):
    left = pd.Series([1, 2])
    right = pd.Series([3, 4])
    expected = pd.Series([4, 6])
    monkeypatch.setattr(series, "IS_ARM", False, raising=False)
    monkeypatch.setattr(
        series.Series,
        "_constructor_from_mgr",
        lambda *args: pytest.fail("ARM manager construction called on non-ARM"),
    )

    result = left + right

    tm.assert_series_equal(result, expected)


def test_masked_where_non_arm_uses_base_implementation(monkeypatch):
    arr = pd.array([1.0, pd.NA, 3.0], dtype="Float64")
    called = False
    original_where = ExtensionArray._where

    def where(self, mask, value):
        nonlocal called
        called = True
        return original_where(self, mask, value)

    monkeypatch.setattr(masked, "IS_ARM", False, raising=False)
    monkeypatch.setattr(ExtensionArray, "_where", where)

    result = arr._where(np.array([True, True, False]), 0.0)

    expected = pd.array([1.0, pd.NA, 0.0], dtype="Float64")
    tm.assert_extension_array_equal(result, expected)
    assert called


def test_masked_putmask_non_arm_uses_base_implementation(monkeypatch):
    arr = pd.array([1.0, pd.NA, 3.0], dtype="Float64")
    monkeypatch.setattr(masked, "IS_ARM", False, raising=False)
    monkeypatch.setattr(
        masked.libalgos,
        "putmask_masked_float64",
        lambda *args: pytest.fail("ARM putmask helper called on non-ARM"),
    )

    arr._putmask(np.array([False, True, False]), 2.0)

    expected = pd.array([1.0, 2.0, 3.0], dtype="Float64")
    tm.assert_extension_array_equal(arr, expected)


def test_masked_boolean_factorize_non_arm_uses_generic_implementation(monkeypatch):
    arr = pd.array([True, False] * 50_001, dtype="boolean")
    monkeypatch.setattr(masked, "IS_ARM", False, raising=False)
    monkeypatch.setattr(
        masked.libalgos,
        "factorize_bool_masked",
        lambda *args: pytest.fail("ARM factorize helper called on non-ARM"),
    )
    original_factorize_array = masked.factorize_array

    def factorize_array(values, *args, **kwargs):
        assert values is arr._data
        assert kwargs["mask"] is arr._mask
        return original_factorize_array(values, *args, **kwargs)

    monkeypatch.setattr(masked, "factorize_array", factorize_array)

    codes, uniques = arr.factorize()

    tm.assert_numpy_array_equal(codes, np.tile([0, 1], 50_001))
    tm.assert_extension_array_equal(uniques, pd.array([True, False], dtype="boolean"))


@pytest.mark.parametrize("method", ["_where", "_putmask"])
def test_arrow_scalar_selection_non_arm_uses_base_implementation(monkeypatch, method):
    pytest.importorskip("pyarrow")
    from pandas.core.arrays.arrow import array as arrow_array

    arr = pd.array([1, 2, 3], dtype="int64[pyarrow]")
    monkeypatch.setattr(arrow_array, "IS_ARM", False, raising=False)
    monkeypatch.setattr(
        arrow_array.ArrowExtensionArray,
        "_if_else",
        lambda *args: pytest.fail("ARM Arrow selection called on non-ARM"),
    )

    if method == "_where":
        result = arr._where(np.array([True, False, True]), 0)
    else:
        arr._putmask(np.array([False, True, False]), 0)
        result = arr

    expected = pd.array([1, 0, 3], dtype="int64[pyarrow]")
    tm.assert_extension_array_equal(result, expected)
