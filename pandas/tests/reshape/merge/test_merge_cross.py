from unittest import mock

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm
from pandas.core.reshape.merge import (
    MergeError,
    merge,
)


@pytest.mark.parametrize(
    ("input_col", "output_cols"), [("b", ["a", "b"]), ("a", ["a_x", "a_y"])]
)
def test_merge_cross(input_col, output_cols):
    # GH#5401
    left = DataFrame({"a": [1, 3]})
    right = DataFrame({input_col: [3, 4]})
    left_copy = left.copy()
    right_copy = right.copy()
    result = merge(left, right, how="cross")
    expected = DataFrame({output_cols[0]: [1, 1, 3, 3], output_cols[1]: [3, 4, 3, 4]})
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(left, left_copy)
    tm.assert_frame_equal(right, right_copy)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"left_index": True},
        {"right_index": True},
        {"on": "a"},
        {"left_on": "a"},
        {"right_on": "b"},
    ],
)
def test_merge_cross_error_reporting(kwargs):
    # GH#5401
    left = DataFrame({"a": [1, 3]})
    right = DataFrame({"b": [3, 4]})
    msg = (
        "Can not pass on, right_on, left_on or set right_index=True or left_index=True"
    )
    with pytest.raises(MergeError, match=msg):
        merge(left, right, how="cross", **kwargs)


def test_merge_cross_mixed_dtypes():
    # GH#5401
    left = DataFrame(["a", "b", "c"], columns=["A"])
    right = DataFrame(range(2), columns=["B"])
    result = merge(left, right, how="cross")
    expected = DataFrame({"A": ["a", "a", "b", "b", "c", "c"], "B": [0, 1, 0, 1, 0, 1]})
    tm.assert_frame_equal(result, expected)


def test_merge_cross_more_than_one_column():
    # GH#5401
    left = DataFrame({"A": list("ab"), "B": [2, 1]})
    right = DataFrame({"C": range(2), "D": range(4, 6)})
    result = merge(left, right, how="cross")
    expected = DataFrame(
        {
            "A": ["a", "a", "b", "b"],
            "B": [2, 2, 1, 1],
            "C": [0, 1, 0, 1],
            "D": [4, 5, 4, 5],
        }
    )
    tm.assert_frame_equal(result, expected)


def test_merge_cross_null_values(nulls_fixture):
    # GH#5401
    left = DataFrame({"a": [1, nulls_fixture]})
    right = DataFrame({"b": ["a", "b"], "c": [1.0, 2.0]})
    result = merge(left, right, how="cross")
    expected = DataFrame(
        {
            "a": [1, 1, nulls_fixture, nulls_fixture],
            "b": ["a", "b", "a", "b"],
            "c": [1.0, 2.0, 1.0, 2.0],
        }
    )
    tm.assert_frame_equal(result, expected)


def test_join_cross_error_reporting():
    # GH#5401
    left = DataFrame({"a": [1, 3]})
    right = DataFrame({"a": [3, 4]})
    msg = (
        "Can not pass on, right_on, left_on or set right_index=True or left_index=True"
    )
    with pytest.raises(MergeError, match=msg):
        left.join(right, how="cross", on="a")


def test_merge_cross_series():
    # GH#54055
    ls = Series([1, 2, 3, 4], index=[1, 2, 3, 4], name="left")
    rs = Series([3, 4, 5, 6], index=[3, 4, 5, 6], name="right")
    res = merge(ls, rs, how="cross")

    expected = merge(ls.to_frame(), rs.to_frame(), how="cross")
    tm.assert_frame_equal(res, expected)


def test_merge_cross_pyarrow_columns():
    pytest.importorskip("pyarrow")
    left = DataFrame({"a": pd.array(["x", "y"], dtype="string[pyarrow]")})
    right = DataFrame({"b": pd.array(["m", "n"], dtype="string[pyarrow]")})
    result = merge(left, right, how="cross")
    expected = DataFrame(
        {
            "a": pd.array(["x", "x", "y", "y"], dtype="string[pyarrow]"),
            "b": pd.array(["m", "n", "m", "n"], dtype="string[pyarrow]"),
        }
    )
    tm.assert_frame_equal(result, expected)


def test_merge_cross_masked_array_fallback():
    left = DataFrame({"a": pd.array([1, 2], dtype="Int64")})
    right = DataFrame({"b": [3, 4]})
    result = merge(left, right, how="cross")
    expected = DataFrame(
        {
            "a": pd.array([1, 1, 2, 2], dtype="Int64"),
            "b": [3, 4, 3, 4],
        }
    )
    tm.assert_frame_equal(result, expected)


def test_merge_cross_arm_exception_fallback():
    # Verify that when the ARM fast path raises an exception, the code
    # falls back to the generic synthetic-column path. We mock IS_ARM=True
    # to ensure the fast path is actually entered (otherwise on x86 the
    # mock would never be called and the test would be a false positive).
    left = DataFrame({"a": [1, 2]})
    right = DataFrame({"b": [3, 4]})
    with (
        mock.patch("pandas.core.reshape.merge.IS_ARM", True),
        mock.patch(
            "pandas.core.reshape.merge._cross_merge_arm",
            side_effect=RuntimeError("forced"),
        ),
    ):
        with tm.assert_produces_warning(RuntimeWarning, match="ARM cross-merge"):
            result = merge(left, right, how="cross")
    expected = DataFrame({"a": [1, 1, 2, 2], "b": [3, 4, 3, 4]})
    tm.assert_frame_equal(result, expected)


def test_cross_merge_arm_pyarrow_direct():
    pa = pytest.importorskip("pyarrow")
    from pandas.core.reshape.merge import _cross_merge_arm

    left = DataFrame({"a": pd.array(["x", "y"], dtype="string[pyarrow]")})
    right = DataFrame({"b": pd.array(["m", "n"], dtype="string[pyarrow]")})
    result = _cross_merge_arm(left, right, ("_x", "_y"))
    expected = DataFrame(
        {
            "a": pd.array(["x", "x", "y", "y"], dtype="string[pyarrow]"),
            "b": pd.array(["m", "n", "m", "n"], dtype="string[pyarrow]"),
        }
    )
    tm.assert_frame_equal(result, expected)


def test_cross_merge_arm_pyarrow_multi_chunk():
    pa = pytest.importorskip("pyarrow")
    from pandas.core.reshape.merge import _cross_merge_arm

    left_arr = pa.chunked_array([["x"], ["y"]])
    right_arr = pa.chunked_array([["m"], ["n"]])
    left = DataFrame({"a": pd.arrays.ArrowExtensionArray(left_arr)})
    right = DataFrame({"b": pd.arrays.ArrowExtensionArray(right_arr)})
    result = _cross_merge_arm(left, right, ("_x", "_y"))
    expected = DataFrame(
        {
            "a": pd.arrays.ArrowExtensionArray(
                pa.array(["x", "x", "y", "y"])
            ),
            "b": pd.arrays.ArrowExtensionArray(
                pa.array(["m", "n", "m", "n"])
            ),
        }
    )
    tm.assert_frame_equal(result, expected)


def test_cross_merge_arm_masked_array_fallback_direct():
    from pandas.core.reshape.merge import _cross_merge_arm

    left = DataFrame({"a": pd.array([1, 2], dtype="Int64")})
    right = DataFrame({"b": [3, 4]})
    result = _cross_merge_arm(left, right, ("_x", "_y"))
    expected = DataFrame(
        {
            "a": pd.array([1, 1, 2, 2], dtype="Int64"),
            "b": [3, 4, 3, 4],
        }
    )
    tm.assert_frame_equal(result, expected)


def test_cross_merge_arm_datetime_tz_aware():
    # GH#XXXXX: verify tz-aware datetime columns are preserved correctly
    # in the ARM cross-merge fast path (regression test for _simple_new
    # positional argument bug that silently dropped timezone info).
    from pandas.core.reshape.merge import _cross_merge_arm

    left = DataFrame(
        {"a": pd.to_datetime(["2020-01-01", "2020-01-02"]).tz_localize("UTC")}
    )
    right = DataFrame({"b": [3, 4]})
    result = _cross_merge_arm(left, right, ("_x", "_y"))
    expected = DataFrame(
        {
            "a": pd.to_datetime(
                ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]
            ).tz_localize("UTC"),
            "b": [3, 4, 3, 4],
        }
    )
    tm.assert_frame_equal(result, expected)


def test_cross_merge_arm_timedelta():
    # Verify timedelta columns work correctly in the ARM cross-merge fast path.
    from pandas.core.reshape.merge import _cross_merge_arm

    left = DataFrame({"a": pd.to_timedelta(["1 day", "2 days"])})
    right = DataFrame({"b": [3, 4]})
    result = _cross_merge_arm(left, right, ("_x", "_y"))
    expected = DataFrame(
        {
            "a": pd.to_timedelta(["1 day", "1 day", "2 days", "2 days"]),
            "b": [3, 4, 3, 4],
        }
    )
    tm.assert_frame_equal(result, expected)
