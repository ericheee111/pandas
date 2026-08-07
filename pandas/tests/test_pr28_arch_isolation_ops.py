import operator
import re

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import masked
from pandas.core.arrays.base import ExtensionArray
from pandas.core.indexes import multi
from pandas.core.ops import array_ops
from pandas.core.reshape import merge
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


@pytest.mark.parametrize("op", [operator.eq, operator.ne])
def test_series_scalar_extension_result_arm_uses_constructor(monkeypatch, op):
    from pandas.tests.extension.json.array import JSONArray, make_data

    obj = pd.Series(JSONArray(make_data(3)))
    monkeypatch.setattr(series, "IS_ARM", False, raising=False)
    expected = op(obj, 0)
    monkeypatch.setattr(series, "IS_ARM", True, raising=False)
    result = op(obj, 0)

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


@pytest.mark.parametrize(
    "dtype,value",
    [
        ("Int64", 1.0),
        ("UInt64", np.int8(1)),
        ("Int8", 1.0),
    ],
)
def test_masked_where_arm_preserves_storage_dtype(monkeypatch, dtype, value):
    mask = np.array([True, True, False])

    monkeypatch.setattr(masked, "IS_ARM", False, raising=False)
    expected = pd.array([1, None, 3], dtype=dtype)._where(mask, value)
    monkeypatch.setattr(masked, "IS_ARM", True, raising=False)
    result = pd.array([1, None, 3], dtype=dtype)._where(mask, value)

    tm.assert_extension_array_equal(result, expected)


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


@pytest.mark.parametrize("method", ["_where", "_putmask"])
@pytest.mark.parametrize(
    "data,dtype,value",
    [
        ([1, 2, 3], "int64[pyarrow]", pd.NA),
        ([1.0, 2.0, 3.0], "float64[pyarrow]", np.nan),
        (
            pd.date_range("2020-01-01", periods=3),
            "timestamp[ns][pyarrow]",
            pd.Timestamp("2020-02-03"),
        ),
        (
            pd.timedelta_range("1 day", periods=3),
            "duration[ns][pyarrow]",
            pd.Timedelta("2 days"),
        ),
    ],
)
def test_arrow_scalar_selection_arm_matches_base(
    monkeypatch, method, data, dtype, value
):
    pytest.importorskip("pyarrow")
    from pandas.core.arrays.arrow import array as arrow_array

    mask = np.array([True, False, True])

    def apply(arr):
        if method == "_where":
            return arr._where(mask, value)
        arr._putmask(mask, value)
        return arr

    monkeypatch.setattr(arrow_array, "IS_ARM", False, raising=False)
    expected = apply(pd.array(data, dtype=dtype))
    monkeypatch.setattr(arrow_array, "IS_ARM", True, raising=False)
    result = apply(pd.array(data, dtype=dtype))

    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize("method", ["_where", "_putmask"])
def test_arrow_scalar_selection_arm_preserves_invalid_value_error(
    monkeypatch, method
):
    pa = pytest.importorskip("pyarrow")
    from pandas.core.arrays.arrow import array as arrow_array

    mask = np.array([True, False, True])

    def apply():
        arr = pd.array([1, 2, 3], dtype="int64[pyarrow]")
        if method == "_where":
            arr._where(mask, "bad")
        else:
            arr._putmask(mask, "bad")

    monkeypatch.setattr(arrow_array, "IS_ARM", False, raising=False)
    with pytest.raises(pa.ArrowInvalid) as expected:
        apply()
    monkeypatch.setattr(arrow_array, "IS_ARM", True, raising=False)
    with pytest.raises(type(expected.value), match=re.escape(str(expected.value))):
        apply()


def test_merge_masked_ea_non_arm_avoids_hash_fastpath(monkeypatch):
    monkeypatch.setattr(merge, "IS_ARM", False)
    monkeypatch.setattr(
        merge,
        "_masked_hash_inner_join_fastpath",
        lambda *args: pytest.fail("non-ARM merge used masked EA hash fast path"),
    )
    left = pd.DataFrame(
        {"key": pd.Series([1, None, 2, 1], dtype="Int64"), "left": range(4)}
    )
    right = pd.DataFrame(
        {"key": pd.Series([1, None, 3], dtype="Int64"), "right": range(3)}
    )

    result = pd.merge(left, right, on="key", how="inner", sort=False)

    expected = pd.DataFrame(
        {
            "key": pd.Series([1, None, 1], dtype="Int64"),
            "left": [0, 1, 3],
            "right": [0, 1, 0],
        }
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", ["Int64", "Float64"])
def test_merge_masked_ea_arm_matches_legacy(monkeypatch, dtype):
    left = pd.DataFrame(
        {"key": pd.Series([1, None, 2, 1], dtype=dtype), "left": range(4)}
    )
    right = pd.DataFrame(
        {"key": pd.Series([1, None, 3], dtype=dtype), "right": range(3)}
    )

    monkeypatch.setattr(merge, "IS_ARM", False)
    expected = pd.merge(left, right, on="key", how="inner", sort=False)
    calls = []
    original = merge._masked_hash_inner_join_fastpath

    def tracked(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(merge, "_masked_hash_inner_join_fastpath", tracked)
    monkeypatch.setattr(merge, "IS_ARM", True)
    result = pd.merge(left, right, on="key", how="inner", sort=False)

    assert calls == [None]
    tm.assert_frame_equal(result, expected)


def test_merge_masked_float_monotonic_arm_uses_ordered_join(monkeypatch):
    left = pd.DataFrame(
        {"key": pd.Series([1.0, 1.0, 2.0], dtype="Float64"), "left": range(3)}
    )
    right = pd.DataFrame(
        {"key": pd.Series([1.0, 2.0], dtype="Float64"), "right": range(2)}
    )

    monkeypatch.setattr(merge, "IS_ARM", True)
    monkeypatch.setattr(
        merge,
        "_masked_hash_inner_join_fastpath",
        lambda *args: pytest.fail("ordered Float64 merge used masked hash join"),
    )

    result = pd.merge(left, right, on="key", how="inner", sort=False)

    expected = pd.DataFrame(
        {
            "key": pd.Series([1.0, 1.0, 2.0], dtype="Float64"),
            "left": [0, 1, 2],
            "right": [0, 0, 1],
        }
    )
    tm.assert_frame_equal(result, expected)


def test_multiindex_unique_non_arm_avoids_packed_codes(monkeypatch):
    monkeypatch.setattr(multi, "IS_ARM", False)
    monkeypatch.setattr(
        multi.np,
        "unique",
        lambda *args, **kwargs: pytest.fail(
            "non-ARM MultiIndex.unique used packed-code fast path"
        ),
    )
    index = pd.MultiIndex.from_arrays(
        [pd.array([1, None, 1, 2, None], dtype="Int64"), ["a", "b", "a", "c", "b"]]
    )

    result = index.unique()

    expected = index[[0, 1, 3]]
    tm.assert_index_equal(result, expected)


def test_multiindex_unique_arm_matches_legacy(monkeypatch):
    index = pd.MultiIndex.from_arrays(
        [pd.array([1, None, 1, 2, None], dtype="Int64"), ["a", "b", "a", "c", "b"]],
        names=["number", "label"],
    )

    monkeypatch.setattr(multi, "IS_ARM", False)
    expected = index.unique()
    calls = []
    original = multi.np.unique

    def tracked(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(multi.np, "unique", tracked)
    monkeypatch.setattr(multi, "IS_ARM", True)
    result = index.unique()

    assert calls == [None]
    tm.assert_index_equal(result, expected)
