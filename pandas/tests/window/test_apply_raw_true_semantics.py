import warnings

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm


def sum_plus_five(values):
    return np.sum(values) + 5


@pytest.mark.parametrize("constructor", [Series, DataFrame])
@pytest.mark.parametrize("window", [3, 300])
@pytest.mark.parametrize("dtype", ["int64", "float64"])
@pytest.mark.parametrize(
    "function",
    [
        pytest.param(sum, id="builtins_sum"),
        pytest.param(np.sum, id="numpy_sum"),
        pytest.param(sum_plus_five, id="callable"),
    ],
)
def test_raw_true_owned_matrix_semantics(constructor, window, dtype, function):
    values = (np.arange(400) % 17).astype(dtype)
    obj = constructor(values)

    result = obj.rolling(window).apply(function, raw=True)

    prepared = values.astype(np.float64)
    expected_values = np.full(len(values), np.nan)
    for i in range(window - 1, len(values)):
        expected_values[i] = function(prepared[i - window + 1 : i + 1])
    expected = constructor(expected_values)
    tm.assert_equal(result, expected)


def test_raw_true_empty_args_kwargs_preserve_windows_and_order():
    windows = []

    class Recorder:
        def __call__(self, window):
            windows.append(window)
            return np.sum(window)

    result = Series(np.arange(6.0)).rolling(3, min_periods=1).apply(
        Recorder(), raw=True, args=(), kwargs={}
    )

    expected = Series([0.0, 1.0, 3.0, 6.0, 9.0, 12.0])
    tm.assert_series_equal(result, expected)
    assert len(windows) == 6
    assert len({id(window) for window in windows}) == len(windows)
    for window, expected_window in zip(
        windows,
        (
            [0.0],
            [0.0, 1.0],
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
        ),
        strict=True,
    ):
        assert isinstance(window, np.ndarray)
        tm.assert_numpy_array_equal(window, np.array(expected_window))


def test_raw_true_saved_overlapping_windows_share_values():
    windows = []

    def function(window):
        windows.append(window)
        return np.sum(window)

    Series(np.arange(4.0)).rolling(2, min_periods=1).apply(function, raw=True)

    windows[0][0] = 10.0
    assert windows[1][0] == 10.0
    assert windows[2][0] == 1.0


def test_raw_true_readonly_input_produces_readonly_windows():
    values = np.arange(4.0)
    values.flags.writeable = False
    windows = []

    def function(window):
        windows.append(window)
        return np.sum(window)

    Series(values, copy=False).rolling(2, min_periods=1).apply(function, raw=True)

    assert all(not window.flags.writeable for window in windows)
    with pytest.raises(ValueError, match="read-only"):
        windows[-1][0] = 10.0


def test_raw_true_empty_window_calls_callback():
    windows = []

    def function(window):
        windows.append(window)
        return len(window)

    result = Series([1.0, 2.0]).rolling(
        2, min_periods=0, closed="left"
    ).apply(function, raw=True)

    tm.assert_series_equal(result, Series([0.0, 1.0]))
    assert len(windows) == 2
    assert windows[0].shape == (0,)
    assert windows[0].dtype == np.dtype(np.float64)


def test_raw_true_nonempty_args_kwargs_fallback():
    seen = []

    def function(window, offset, *, scale):
        seen.append((window.copy(), offset, scale))
        return np.sum(window) * scale + offset

    result = Series([1.0, 2.0, 3.0]).rolling(2, min_periods=1).apply(
        function, raw=True, args=(10.0,), kwargs={"scale": 2.0}
    )

    expected = Series([12.0, 16.0, 20.0])
    tm.assert_series_equal(result, expected)
    assert [(offset, scale) for _, offset, scale in seen] == [
        (10.0, 2.0),
        (10.0, 2.0),
        (10.0, 2.0),
    ]


def test_raw_true_exception_stops_at_same_window():
    seen = []

    class MarkerError(Exception):
        pass

    def function(window):
        seen.append(window.copy())
        if len(seen) == 3:
            raise MarkerError("third raw window")
        return np.sum(window)

    with pytest.raises(MarkerError, match="third raw window"):
        Series(np.arange(6.0)).rolling(2, min_periods=1).apply(function, raw=True)

    assert len(seen) == 3
    tm.assert_numpy_array_equal(seen[-1], np.array([1.0, 2.0]))


def test_raw_true_warning_count_and_message():
    def function(window):
        warnings.warn("raw window callback", UserWarning)
        return np.sum(window)

    with pytest.warns(UserWarning, match="raw window callback") as recorded:
        result = Series([1.0, 2.0, 3.0]).rolling(2, min_periods=1).apply(
            function, raw=True
        )

    assert len(recorded) == 3
    tm.assert_series_equal(result, Series([1.0, 3.0, 5.0]))


def test_raw_true_nan_inf_and_signed_zero_callback_values():
    seen = []

    def function(window):
        seen.append(window.copy())
        return 0.0

    values = Series([0.0, -0.0, np.nan, np.inf, -np.inf, 1.0])
    values.rolling(3, min_periods=1).apply(function, raw=True)

    assert len(seen) == 5
    assert not np.signbit(seen[0][0])
    assert np.signbit(seen[1][1])
    assert np.isnan(seen[2][-1])
    assert np.isnan(seen[3][-1])
    assert np.isnan(seen[4][0])
    assert np.isnan(seen[4][1])
    assert seen[4][-1] == 1.0


def test_raw_true_noncontiguous_input():
    values = np.arange(12.0)[::2]
    assert not values.flags.c_contiguous

    seen = []

    def function(window):
        seen.append(window.copy())
        return np.sum(window)

    result = Series(values).rolling(2, min_periods=1).apply(function, raw=True)

    tm.assert_series_equal(result, Series([0.0, 2.0, 6.0, 10.0, 14.0, 18.0]))
    assert all(isinstance(window, np.ndarray) for window in seen)
