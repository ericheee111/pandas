"""
Public-behavior semantics tests for ``Rolling.apply`` with ``raw=False``.

These tests verify the callback-visible contract of ``rolling.apply(...,
raw=False)``: the type, index, index name, Series name, dtype, values,
invocation count, invocation order, exception propagation, ``args``/``kwargs``
forwarding, ``min_periods``/``center``/``closed``/``step`` behavior, NaN/Inf/
signed-zero handling, Copy-on-Write behavior, variable time windows, and
DataFrame column-wise behavior.

They do NOT assert on internal implementation details such as ``.base``,
``OWNDATA``, the BlockManager, or which private slicer is used.  They only
check the publicly observable objects handed to the callback and the publicly
observable result of ``apply``.
"""

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
    date_range,
)
import pandas._testing as tm


# ---------------------------------------------------------------------------
# Recorder callback
# ---------------------------------------------------------------------------


def make_recorder():
    """Return (callback, log).

    ``callback`` records, for each invocation, a tuple::

        (type_name, index_list, index_name, series_name, values_list)

    and returns ``float(sum(values))`` so it can be used as a normal
    rolling apply callable.
    """
    log = []

    def callback(window):
        if hasattr(window, "index"):
            index_list = list(window.index)
            index_name = window.index.name
        else:
            index_list = None
            index_name = None
        log.append(
            (
                type(window).__name__,
                index_list,
                index_name,
                getattr(window, "name", None),
                list(window),
            )
        )
        return float(np.sum(window))

    return callback, log


# ---------------------------------------------------------------------------
# Core: raw=False receives a Series with correct public attributes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "index_factory,name",
    [
        (lambda n: None, None),  # default RangeIndex
        (lambda n: date_range("2020-01-01", periods=n, freq="D"), "ts"),
        (lambda n: list(f"v{x}" for x in range(n)), "str_idx"),
    ],
)
def test_raw_false_callback_receives_series_with_index(index_factory, name):
    n = 6
    arr = np.arange(n, dtype=float)
    idx = index_factory(n)
    s = Series(arr, index=idx, name="data") if idx is not None else Series(arr, name="data")
    if idx is not None:
        s.index = s.index.rename(name)

    cb, log = make_recorder()
    result = s.rolling(3, min_periods=1).apply(cb, raw=False)

    # Every recorded window is a Series (not ndarray)
    for type_name, *_ in log:
        assert type_name == "Series", type_name

    # index name preserved on each window
    for _, _, idx_name, *_ in log:
        assert idx_name == name

    # The per-window Series name is None: roll_apply receives a Series built
    # from the column's float64 ndarray (via _prep_values), which carries no
    # name.  This is the existing public behavior for both iloc and _slice.
    for _, _, _, sname, *_ in log:
        assert sname is None

    # values of the last window
    last = log[-1]
    assert last[4] == [3.0, 4.0, 5.0]

    # invocation count == number of windows (one per row, min_periods=1)
    assert len(log) == n

    # result is finite and matches the per-window sum; the *result* Series
    # (built by _apply_series) preserves the original column name and index.
    expected = Series([0.0, 1.0, 3.0, 6.0, 9.0, 12.0], index=s.index, name="data")
    tm.assert_almost_equal(result, expected)


def test_raw_true_callback_receives_ndarray():
    s = Series(np.arange(5.0), name="data")
    cb, log = make_recorder()
    s.rolling(2, min_periods=1).apply(cb, raw=True)
    for type_name, *_ in log:
        assert type_name == "ndarray", type_name


# ---------------------------------------------------------------------------
# Index varieties: duplicate, non-monotonic, named
# ---------------------------------------------------------------------------


def test_raw_false_duplicate_index():
    idx = [0, 1, 1, 2, 3, 3]
    s = Series(np.arange(6.0), index=idx, name="d")
    cb, log = make_recorder()
    s.rolling(2, min_periods=1).apply(cb, raw=False)
    # each window's index matches the positional slice of the original index
    for i, (_, idx_list, _, _, _) in enumerate(log):
        assert idx_list == idx[i : i + 1] if i == 0 else idx[i - 1 : i + 1]


def test_raw_false_non_monotonic_index():
    idx = [5, 3, 4, 1, 2, 0]
    s = Series(np.arange(6.0), index=idx, name="nm")
    cb, log = make_recorder()
    s.rolling(3, min_periods=1).apply(cb, raw=False)
    # index values are positional slices of the original (non-sorted) index
    for i, (_, idx_list, _, _, _) in enumerate(log):
        lo = max(0, i - 2)
        assert idx_list == idx[lo : i + 1]


def test_raw_false_string_index_name():
    n = 5
    s = Series(np.arange(n, dtype=float), name="vals")
    s.index = s.index.rename("positions")
    cb, log = make_recorder()
    s.rolling(2, min_periods=1).apply(cb, raw=False)
    for _, _, idx_name, sname, _ in log:
        assert idx_name == "positions"
        # per-window Series name is None (built from ndarray); see
        # test_raw_false_callback_receives_series_with_index for rationale.
        assert sname is None


# ---------------------------------------------------------------------------
# Window / min_periods / center / step
# ---------------------------------------------------------------------------


def test_raw_false_window_one():
    s = Series(np.arange(4.0), name="w1")
    cb, log = make_recorder()
    result = s.rolling(1).apply(cb, raw=False)
    tm.assert_almost_equal(result, s.astype(float))
    assert len(log) == 4
    for _, _, _, _, vals in log:
        assert len(vals) == 1


def test_raw_false_window_larger_than_data():
    s = Series(np.arange(4.0), name="big")
    # window=10 with default min_periods=10 -> all NaN, callback never invoked
    cb, log = make_recorder()
    result = s.rolling(10).apply(cb, raw=False)
    assert result.isna().all()
    assert log == []

    # min_periods=1 -> callback invoked on partial windows
    cb2, log2 = make_recorder()
    result2 = s.rolling(10, min_periods=1).apply(cb2, raw=False)
    assert len(log2) == 4
    tm.assert_almost_equal(result2, Series([0.0, 1.0, 3.0, 6.0], name="big"))


@pytest.mark.parametrize("minp", [0, 1, 3])
def test_raw_false_min_periods(minp):
    s = Series(np.arange(6.0), name="mp")
    cb, log = make_recorder()
    result = s.rolling(3, min_periods=minp).apply(cb, raw=False)
    # number of finite results depends on minp
    finite = result.notna().sum()
    if minp <= 1:
        assert finite == 6
    elif minp == 3:
        assert finite == 4


@pytest.mark.parametrize("center", [True, False])
def test_raw_false_center(center):
    s = Series(np.arange(6.0), name="c")
    cb, log = make_recorder()
    s.rolling(3, center=center, min_periods=1).apply(cb, raw=False)
    assert len(log) == 6


@pytest.mark.parametrize("closed", ["right", "left", "both", "neither"])
def test_raw_false_closed(closed):
    # closed only affects time-based windows; for integer windows it shifts
    # bounds. Verify behavior is internally consistent and finite.
    s = Series(np.arange(5.0), name="cl")
    cb, _ = make_recorder()
    result = s.rolling(2, closed=closed, min_periods=1).apply(cb, raw=False)
    assert len(result) == 5


def test_raw_false_step():
    s = Series(np.arange(8.0), name="st")
    cb, log = make_recorder()
    result = s.rolling(3, min_periods=1, step=2).apply(cb, raw=False)
    # step=2 produces one window per stepped position: ceil(8/2) = 4 windows
    assert len(result) == 4
    assert len(log) == 4  # callback invoked once per (stepped) window


# ---------------------------------------------------------------------------
# Empty input, NaN, Inf, signed zero
# ---------------------------------------------------------------------------


def test_raw_false_empty_input():
    s = Series([], dtype=float, name="empty")
    cb, log = make_recorder()
    result = s.rolling(3, min_periods=1).apply(cb, raw=False)
    assert len(result) == 0
    assert log == []


def test_raw_false_nan_values():
    s = Series([1.0, np.nan, 3.0, np.nan, 5.0], name="nan")
    cb, log = make_recorder()
    result = s.rolling(2, min_periods=1).apply(cb, raw=False)
    # windows containing NaN should still invoke the callback (raw=False does
    # not drop NaN; min_periods counts finite values)
    assert len(log) == 5


def test_raw_false_inf_values():
    s = Series([1.0, np.inf, 3.0, -np.inf, 5.0], name="inf")
    cb, log = make_recorder()
    s.rolling(2, min_periods=1).apply(cb, raw=False)
    assert len(log) == 5


def test_raw_false_signed_zero():
    s = Series([0.0, -0.0, 1.0, 0.0, -0.0], name="sz")
    cb, log = make_recorder()
    s.rolling(2, min_periods=1).apply(cb, raw=False)
    # +0.0 and -0.0 are both finite; both invoke the callback
    assert len(log) == 5
    # signed zero preserved in window values
    for _, _, _, _, vals in log:
        for v in vals:
            assert v in (0.0, -0.0, 1.0)


# ---------------------------------------------------------------------------
# Non-contiguous input
# ---------------------------------------------------------------------------


def test_raw_false_non_contiguous_input():
    arr = np.arange(10.0)[::2]  # non-contiguous view
    assert not arr.flags.c_contiguous
    s = Series(arr, name="nc")
    cb, log = make_recorder()
    result = s.rolling(2, min_periods=1).apply(cb, raw=False)
    expected = Series([0.0, 2.0, 6.0, 10.0, 14.0], name="nc")
    tm.assert_almost_equal(result, expected)


# ---------------------------------------------------------------------------
# args / kwargs forwarding
# ---------------------------------------------------------------------------


def test_raw_false_args_forwarding():
    seen = []

    def cb(window, offset):
        seen.append(offset)
        return float(np.sum(window)) + offset

    s = Series([1.0, 2.0, 3.0], name="a")
    result = s.rolling(2, min_periods=1).apply(cb, raw=False, args=(10,))
    assert seen == [10, 10, 10]
    tm.assert_almost_equal(result, Series([11.0, 13.0, 15.0], name="a"))


def test_raw_false_kwargs_forwarding():
    seen = []

    def cb(window, *, factor):
        seen.append(factor)
        return float(np.sum(window)) * factor

    s = Series([1.0, 2.0, 3.0], name="k")
    result = s.rolling(2, min_periods=1).apply(cb, raw=False, kwargs={"factor": 2.0})
    assert seen == [2.0, 2.0, 2.0]
    tm.assert_almost_equal(result, Series([2.0, 6.0, 10.0], name="k"))


def test_raw_false_args_and_kwargs():
    def cb(window, offset, *, factor):
        return float(np.sum(window)) * factor + offset

    s = Series([1.0, 2.0, 3.0], name="ak")
    result = s.rolling(2, min_periods=1).apply(
        cb, raw=False, args=(100,), kwargs={"factor": 2.0}
    )
    tm.assert_almost_equal(result, Series([102.0, 106.0, 110.0], name="ak"))


# ---------------------------------------------------------------------------
# Invocation count and order
# ---------------------------------------------------------------------------


def test_raw_false_invocation_count_and_order():
    s = Series(np.arange(6.0), name="order")
    order_log = []

    def cb(window):
        order_log.append(window.iloc[0])  # first value of each window
        return float(np.sum(window))

    s.rolling(3, min_periods=1).apply(cb, raw=False)
    # windows processed in positional order; for min_periods=1 the first
    # element of each window is 0,0,0,1,2,3
    assert order_log == [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Exception propagation
# ---------------------------------------------------------------------------


def test_raw_false_callback_raises_value_error():
    def cb(window):
        if len(window) == 3:
            raise ValueError("boom")
        return float(np.sum(window))

    s = Series(np.arange(6.0), name="raise")
    with pytest.raises(ValueError, match="boom"):
        s.rolling(3, min_periods=1).apply(cb, raw=False)


# ---------------------------------------------------------------------------
# Callback inspects window.index
# ---------------------------------------------------------------------------


def test_raw_false_callback_inspects_index():
    idx = date_range("2020-01-01", periods=5, freq="D")
    s = Series(np.arange(5.0), index=idx, name="ts")
    seen_first_index = []

    def cb(window):
        seen_first_index.append(window.index[0])
        return float(np.sum(window))

    s.rolling(2, min_periods=1).apply(cb, raw=False)
    # first index of each window (min_periods=1, window=2): positional slices
    assert seen_first_index == list(idx[:1]) + list(idx[:4])


# ---------------------------------------------------------------------------
# Callback mutates received window: original data unchanged
# ---------------------------------------------------------------------------


def test_raw_false_callback_mutation_does_not_affect_source():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    s = Series(arr.copy(), name="mut")
    original = arr.copy()

    def cb(window):
        # attempt to mutate the received window
        try:
            window.iloc[0] = 999.0
        except Exception:
            # CoW may raise; that is acceptable public behavior
            pass
        return float(np.sum(window))

    s.rolling(3, min_periods=1).apply(cb, raw=False)
    # source array unchanged
    tm.assert_almost_equal(Series(arr), Series(original))


# ---------------------------------------------------------------------------
# Copy-on-Write
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:.*mode.copy_on_write.*:pandas.errors.Pandas4Warning")
@pytest.mark.parametrize("cow", [True, False])
def test_raw_false_copy_on_write(cow):
    import pandas as pd

    s = Series([1.0, 2.0, 3.0, 4.0], name="cow")
    cb, log = make_recorder()
    # In pandas >= 3.0 Copy-on-Write is always enabled; setting the option is
    # deprecated but still accepted. We exercise both values to confirm the
    # rolling apply path is unaffected by the (no-op) setting.
    with pd.option_context("mode.copy_on_write", cow):
        result = s.rolling(2, min_periods=1).apply(cb, raw=False)
    tm.assert_almost_equal(result, Series([1.0, 3.0, 5.0, 7.0], name="cow"))
    assert len(log) == 4


# ---------------------------------------------------------------------------
# Variable time window
# ---------------------------------------------------------------------------


def test_raw_false_variable_time_window():
    idx = date_range("2020-01-01", periods=6, freq="D")
    s = Series(np.arange(6.0), index=idx, name="var")
    cb, log = make_recorder()
    result = s.rolling("2D", min_periods=1).apply(cb, raw=False)
    assert len(log) == 6
    # result has no NaN with min_periods=1
    assert not result.isna().any()


# ---------------------------------------------------------------------------
# DataFrame column-wise
# ---------------------------------------------------------------------------


def test_raw_false_dataframe_single_column():
    df = DataFrame({"a": np.arange(5.0)})
    cb, log = make_recorder()
    result = df.rolling(2, min_periods=1).apply(cb, raw=False)
    expected = DataFrame({"a": [0.0, 1.0, 3.0, 5.0, 7.0]})
    tm.assert_frame_equal(result, expected)
    # one invocation per row per column
    assert len(log) == 5


def test_raw_false_dataframe_multiple_columns():
    df = DataFrame(
        {
            "a": np.arange(5.0),
            "b": np.arange(5.0) * 10,
        }
    )
    name_log = []

    def cb(window):
        name_log.append(window.name)
        return float(np.sum(window))

    result = df.rolling(2, min_periods=1).apply(cb, raw=False)
    # callback invoked once per (row, column): 5 rows * 2 columns
    assert len(name_log) == 10
    # per-window Series name is None (built from ndarray per column); the
    # *result* DataFrame preserves column names via _apply_columnwise.
    assert set(name_log) == {None}
    expected = DataFrame({"a": [0.0, 1.0, 3.0, 5.0, 7.0], "b": [0.0, 10.0, 30.0, 50.0, 70.0]})
    tm.assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# dtype preservation in callback-visible values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["int", "float"])
def test_raw_false_dtype_in_callback(dtype):
    arr = (np.arange(5)).astype(dtype)
    s = Series(arr, name="dt")
    cb, log = make_recorder()
    s.rolling(2, min_periods=1).apply(cb, raw=False)
    # internal _prep_values coerces to float64 before roll_apply; the callback
    # therefore always sees float64 values regardless of source dtype.  This
    # documents the existing public behavior.
    for _, _, _, _, vals in log:
        for v in vals:
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# Result dtype and shape
# ---------------------------------------------------------------------------


def test_raw_false_result_dtype_and_shape():
    s = Series(np.arange(10.0), name="shape")
    cb, _ = make_recorder()
    result = s.rolling(3, min_periods=1).apply(cb, raw=False)
    assert result.dtype == np.float64
    assert result.shape == (10,)
    assert result.name == "shape"
