import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    Period,
    RangeIndex,
    Series,
    Timestamp,
    array,
)
import pandas._testing as tm
from pandas.core import (
    frame,
    generic,
)
from pandas.core.internals import (
    blocks,
    construction,
    managers,
)


def test_astype_extension_non_arm_avoids_direct_manager(monkeypatch):
    monkeypatch.setattr(generic, "IS_ARM", False, raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("non-ARM astype used direct manager construction")

    monkeypatch.setattr(managers, "create_block_manager_from_column_arrays", forbidden)

    df = DataFrame({"a": [1, None], "b": [2, 3]}, dtype=object)
    result = df.astype("Int64")
    expected = DataFrame(
        {
            "a": Series([1, None], dtype="Int64"),
            "b": Series([2, 3], dtype="Int64"),
        }
    )
    tm.assert_frame_equal(result, expected)


def test_astype_non_arm_leaves_dtype_normalization_to_manager(monkeypatch):
    monkeypatch.setattr(generic, "IS_ARM", False, raising=False)
    original = managers.BaseBlockManager.astype
    seen = []

    def wrapped(self, dtype, errors="raise"):
        seen.append(dtype)
        return original(self, dtype=dtype, errors=errors)

    monkeypatch.setattr(managers.BaseBlockManager, "astype", wrapped)

    DataFrame({"a": [1, 2]}).astype("float64")

    assert len(seen) == 1
    assert isinstance(seen[0], str)


def test_shallow_copy_non_arm_uses_manager_apply(monkeypatch):
    monkeypatch.setattr(managers, "IS_ARM", False, raising=False)
    original_apply = managers.BaseBlockManager.apply
    original_copy = blocks.Block.copy
    apply_calls = []
    block_copy_calls = []

    def apply(self, f, align_keys=None, **kwargs):
        apply_calls.append((f, kwargs.get("deep")))
        return original_apply(self, f, align_keys=align_keys, **kwargs)

    def block_copy(self, deep=True):
        block_copy_calls.append(deep)
        return original_copy(self, deep=deep)

    monkeypatch.setattr(managers.BaseBlockManager, "apply", apply)
    monkeypatch.setattr(blocks.Block, "copy", block_copy)

    DataFrame({"a": [1, 2], "b": [3.0, 4.0]}).copy(deep=False)

    assert ("copy", False) in apply_calls
    assert block_copy_calls
    assert not any(block_copy_calls)


def test_arrays_to_mgr_runs_validation_path(monkeypatch):
    original = construction.ensure_index
    seen = []

    def wrapped(index_like, *args, **kwargs):
        seen.append(index_like)
        return original(index_like, *args, **kwargs)

    monkeypatch.setattr(construction, "ensure_index", wrapped)
    index = Index([0, 1])
    columns = Index(["a", "b"])

    construction.arrays_to_mgr(
        [np.array([1, 2]), np.array([3, 4])],
        columns,
        index,
        verify_integrity=False,
    )

    assert seen[0] is index
    assert any(obj is columns for obj in seen)


def test_arrays_to_mgr_rejects_2d_array_without_integrity_verification():
    with pytest.raises(ValueError, match="Arrays must be 1-dimensional"):
        construction.arrays_to_mgr(
            [np.array([[1, 2]])],
            Index(["a"]),
            Index(range(2)),
            verify_integrity=False,
        )


def test_arrays_to_mgr_rejects_length_mismatch_without_integrity_verification():
    with pytest.raises(ValueError, match="length matching len\\(index\\)"):
        construction.arrays_to_mgr(
            [np.array([1])],
            Index(["a"]),
            Index(range(2)),
            verify_integrity=False,
        )


def test_arrays_to_mgr_rejects_columns_mismatch_without_integrity_verification():
    with pytest.raises(ValueError, match=r"len\(arrays\) must match len\(columns\)"):
        construction.arrays_to_mgr(
            [np.array([1, 2])],
            Index(["a", "b"]),
            Index(range(2)),
            verify_integrity=False,
        )


def test_arrays_to_mgr_normalizes_range_index():

    result = DataFrame._from_arrays(
        [np.array([1, 2])],
        columns=Index(["a"]),
        index=range(2),
        verify_integrity=False,
    )

    assert isinstance(result.index, RangeIndex)
    tm.assert_index_equal(result.index, RangeIndex(2))
    tm.assert_frame_equal(result.copy(), DataFrame({"a": [1, 2]}))


def test_arrays_to_mgr_unwraps_numpy_extension_array():
    values = array([1, 2], dtype=np.dtype("int64"))

    result = DataFrame._from_arrays(
        [values],
        columns=Index(["a"]),
        index=range(2),
        verify_integrity=False,
    )

    expected = DataFrame({"a": [1, 2]})
    assert isinstance(result._mgr.iget_values(0), np.ndarray)
    tm.assert_frame_equal(result, expected)


def test_arrays_to_mgr_unwraps_unconsolidated_numpy_extension_array():
    values = array([1, 2], dtype=np.dtype("int64"))

    result = construction.arrays_to_mgr(
        [values],
        Index(["a"]),
        Index(range(2)),
        verify_integrity=False,
        consolidate=False,
    )

    assert isinstance(result.iget_values(0), np.ndarray)


def test_stack_arrays_non_arm_avoids_array_constructor(monkeypatch):
    monkeypatch.setattr(managers, "IS_ARM", False, raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("non-ARM stacking used np.array")

    monkeypatch.setattr(managers.np, "array", forbidden)
    tuples = [(0, np.arange(3)), (1, np.arange(3, 6))]

    result, placement = managers._stack_arrays(tuples, np.dtype("int64"))

    expected = np.arange(6).reshape(2, 3)
    tm.assert_numpy_array_equal(result, expected)
    assert placement == (0, 1)


@pytest.mark.parametrize("axis", [0, 1])
def test_dropna_non_arm_avoids_nanvalidity(monkeypatch, axis):
    monkeypatch.setattr(frame, "IS_ARM", False)

    def forbidden(*args, **kwargs):
        pytest.fail("non-ARM dropna used a Cython float-block helper")

    monkeypatch.setattr(frame.libalgos, "nanvalidity_2d", forbidden)
    monkeypatch.setattr(frame.libalgos, "nancount_2d", forbidden)
    monkeypatch.setattr(frame.DataFrame, "_nancount_float_block", forbidden)
    df = DataFrame([[1.0, np.nan], [2.0, 3.0]])

    result = df.dropna(how="any", axis=axis)

    if axis == 0:
        tm.assert_frame_equal(result, df.iloc[[1]])
    else:
        tm.assert_frame_equal(result, df.iloc[:, [0]])


def test_fillna_extension_blocks_non_arm_avoids_batch_helper(monkeypatch):
    monkeypatch.setattr(generic, "IS_ARM", False, raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("non-ARM fillna used fillna_by_column")

    monkeypatch.setattr(managers.BlockManager, "fillna_by_column", forbidden)
    df = DataFrame(
        {
            "a": Series([1, None], dtype="Int64"),
            "b": Series([None, 2.5], dtype="Float64"),
        }
    )

    result = df.fillna({"a": 0, "b": 1.5})
    expected = DataFrame(
        {
            "a": Series([1, 0], dtype="Int64"),
            "b": Series([1.5, 2.5], dtype="Float64"),
        }
    )
    tm.assert_frame_equal(result, expected)


def test_fillna_non_1d_ea_block_non_arm_avoids_batch_fastpath(monkeypatch):
    monkeypatch.setattr(generic, "IS_ARM", False, raising=False)
    original = managers.BlockManager.fillna

    def reject_2d_fill_values(self, value, limit, inplace):
        if isinstance(value, np.ndarray) and value.ndim == 2:
            pytest.fail("non-ARM EA fillna used the single-block batch fast path")
        return original(self, value=value, limit=limit, inplace=inplace)

    monkeypatch.setattr(managers.BlockManager, "fillna", reject_2d_fill_values)
    df = DataFrame(
        {
            "a": Series(["2020-01-01", None], dtype="datetime64[ns]"),
            "b": Series([None, "2020-01-02"], dtype="datetime64[ns]"),
        }
    )
    assert len(df._mgr.blocks) == 1
    block = df._mgr.blocks[0]
    assert not isinstance(block.values, np.ndarray)
    assert block.values.ndim == 2

    result = df.fillna(
        {"a": np.datetime64("2020-01-03"), "b": np.datetime64("2020-01-04")}
    )
    expected = DataFrame(
        {
            "a": Series(["2020-01-01", "2020-01-03"], dtype="datetime64[ns]"),
            "b": Series(["2020-01-04", "2020-01-02"], dtype="datetime64[ns]"),
        }
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "values,fill_value",
    [
        (
            Series([None, "2020-01-01"], dtype="datetime64[ns, UTC]"),
            Timestamp("2021-01-01", tz="UTC"),
        ),
        (
            Series([None, "2020-01"], dtype="period[M]"),
            Period("2021-01", freq="M"),
        ),
    ],
)
def test_fillna_single_ea_block_arm_matches_base(monkeypatch, values, fill_value):
    df = DataFrame({"a": values})

    monkeypatch.setattr(generic, "IS_ARM", False, raising=False)
    expected = df.fillna({"a": fill_value})
    monkeypatch.setattr(generic, "IS_ARM", True, raising=False)
    result = df.fillna({"a": fill_value})

    tm.assert_frame_equal(result, expected)
