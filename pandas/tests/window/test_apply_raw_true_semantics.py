import builtins
from functools import partial
import sys
import warnings

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
)
from pandas.api.indexers import BaseIndexer
import pandas._testing as tm
from pandas.core import boostkit_fastpaths
from pandas._libs.window import aggregations as window_aggregations


def sum_plus_five(values):
    return np.sum(values) + 5


def builtin_sum_wrapper(values):
    return builtins.sum(values)


def _assert_rolling_result_bits_equal(left, right):
    tm.assert_equal(left, right)
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    tm.assert_numpy_array_equal(
        left_values.view(np.uint64), right_values.view(np.uint64)
    )


def _apply_builtin_sum_and_oracle(
    obj, window, *, min_periods=None, closed=None, raw=True, args=None, kwargs=None
):
    rolling = obj.rolling(window, min_periods=min_periods, closed=closed)
    with np.errstate(divide="warn", over="warn", under="ignore", invalid="warn"):
        result = rolling.apply(
            builtins.sum, raw=raw, args=args, kwargs=kwargs
        )
        expected = rolling.apply(
            lambda values: builtins.sum(values), raw=raw
        )
    _assert_rolling_result_bits_equal(result, expected)
    return result


@pytest.fixture
def enable_boostkit_fastpaths(monkeypatch):
    monkeypatch.setattr(boostkit_fastpaths, "USE_BOOSTKIT_FASTPATHS", True)


@pytest.mark.usefixtures("enable_boostkit_fastpaths")
@pytest.mark.parametrize("constructor", [Series, DataFrame])
@pytest.mark.parametrize("dtype", ["int64", "float64"])
@pytest.mark.parametrize(
    "window,min_periods",
    [(1, None), (3, 1), (300, 300), (500, 0)],
)
def test_raw_true_builtin_sum_matches_fallback_bits(
    constructor, dtype, window, min_periods
):
    values = (np.arange(400) % 17).astype(dtype)
    _apply_builtin_sum_and_oracle(
        constructor(values), window, min_periods=min_periods
    )


@pytest.mark.usefixtures("enable_boostkit_fastpaths")
@pytest.mark.parametrize("constructor", [Series, DataFrame])
@pytest.mark.parametrize(
    "values",
    [
        [1e16, 1.0, -1e16, 3.0, -3.0, 2.0],
        [0.0, -0.0, -0.0, 0.0, 1.0, -1.0],
    ],
)
def test_raw_true_builtin_sum_sensitive_values_match_fallback_bits(
    constructor, values
):
    _apply_builtin_sum_and_oracle(
        constructor(np.array(values, dtype=np.float64)), 3, min_periods=1
    )


@pytest.mark.usefixtures("enable_boostkit_fastpaths")
@pytest.mark.parametrize("constructor", [Series, DataFrame])
def test_raw_true_builtin_sum_empty_window_matches_fallback_bits(constructor):
    _apply_builtin_sum_and_oracle(
        constructor([1.0, 2.0]), 2, min_periods=0, closed="left"
    )


@pytest.mark.usefixtures("enable_boostkit_fastpaths")
@pytest.mark.parametrize("constructor", [Series, DataFrame])
def test_raw_true_builtin_sum_noncontiguous_matches_fallback_bits(constructor):
    values = np.arange(800.0)[::2]
    assert not values.flags.c_contiguous
    _apply_builtin_sum_and_oracle(
        constructor(values, copy=False), 300, min_periods=1
    )


def _fixed_bounds(length, window):
    end = np.arange(1, length + 1, dtype=np.int64)
    start = np.maximum(end - window, 0)
    return start, end


@pytest.mark.parametrize(
    "values,start,end",
    [
        (
            np.array([np.nan, 1.0, 2.0]),
            np.array([1], dtype=np.int64),
            np.array([3], dtype=np.int64),
        ),
        (np.array([1.0, np.inf]), *_fixed_bounds(2, 2)),
        (np.array([1.0, -np.inf]), *_fixed_bounds(2, 2)),
        (np.array([np.inf, -np.inf]), *_fixed_bounds(2, 2)),
        (
            np.array([np.finfo(np.float64).max] * 2),
            *_fixed_bounds(2, 2),
        ),
        (
            np.array([np.finfo(np.float64).max, -np.finfo(np.float64).max]),
            *_fixed_bounds(2, 2),
        ),
    ],
)
def test_raw_builtin_sum_helper_rejects_unsafe_values(values, start, end):
    result = window_aggregations.roll_apply_builtin_sum(values, start, end, 0)
    assert result is None


def test_raw_builtin_sum_helper_accepts_overflow_boundary():
    values = np.array([np.finfo(np.float64).max / 2] * 2)
    start, end = _fixed_bounds(2, 2)

    result = window_aggregations.roll_apply_builtin_sum(values, start, end, 1)
    expected = window_aggregations.roll_apply(
        values, start, end, 1, lambda x: builtins.sum(x), True, (), {}
    )

    assert result is not None
    tm.assert_numpy_array_equal(result.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize(
    "values",
    [
        np.array([1, 2], dtype=np.int64),
        np.array([[1.0], [2.0]], dtype=np.float64),
        np.array([1.0, 2.0], dtype=np.dtype(np.float64).newbyteorder()),
    ],
)
def test_raw_builtin_sum_helper_rejects_non_native_float64_1d(values):
    start, end = _fixed_bounds(len(values), 2)
    assert window_aggregations.roll_apply_builtin_sum(values, start, end, 1) is None


def test_raw_builtin_sum_helper_rejects_zero_dimensional_input():
    start, end = _fixed_bounds(1, 1)
    values = np.array(1.0)

    assert window_aggregations.roll_apply_builtin_sum(values, start, end, 1) is None


def test_raw_builtin_sum_helper_copies_noncontiguous_input():
    values = np.arange(12.0)[::2]
    assert not values.flags.c_contiguous
    start, end = _fixed_bounds(len(values), 3)

    result = window_aggregations.roll_apply_builtin_sum(values, start, end, 1)
    expected = window_aggregations.roll_apply(
        values, start, end, 1, lambda x: builtins.sum(x), True, (), {}
    )

    assert result is not None
    tm.assert_numpy_array_equal(result.view(np.uint64), expected.view(np.uint64))


def _capture_special_sum(values, function, error_mode):
    error_messages = []

    class ErrorLog:
        def write(self, message):
            error_messages.append(message)

    error_callback = (
        (lambda error, flag: error_messages.append((error, flag)))
        if error_mode == "call"
        else ErrorLog()
    )
    error_settings = (
        {
            "divide": "warn",
            "over": "warn",
            "under": "ignore",
            "invalid": "warn",
        }
        if error_mode == "default"
        else {"all": error_mode}
    )
    previous = np.geterrcall()
    np.seterrcall(error_callback)
    try:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            try:
                with np.errstate(**error_settings):
                    result = Series(values).rolling(2, min_periods=0).apply(
                        function, raw=True
                    )
            except Exception as err:
                outcome = ("error", type(err), str(err))
            else:
                result_values = result.to_numpy(dtype=np.float64)
                outcome = (
                    "result",
                    result_values.shape,
                    result_values.view(np.uint64).tobytes(),
                )
    finally:
        np.seterrcall(previous)

    warning_messages = tuple(
        (warning.category, str(warning.message)) for warning in recorded
    )
    return outcome, warning_messages, tuple(error_messages)


@pytest.mark.usefixtures("enable_boostkit_fastpaths")
@pytest.mark.parametrize(
    "values",
    [
        pytest.param([np.nan, 1.0, 2.0], id="nan"),
        pytest.param([np.inf, -np.inf, 1.0], id="mixed_inf"),
        pytest.param(
            [np.finfo(np.float64).max, np.finfo(np.float64).max, 1.0],
            id="overflow",
        ),
    ],
)
@pytest.mark.parametrize(
    "error_mode", ["default", "warn", "raise", "call", "log", "print"]
)
def test_raw_builtin_sum_special_values_preserve_fallback_behavior(
    values, error_mode, capsys
):
    actual = _capture_special_sum(values, builtins.sum, error_mode)
    actual_output = capsys.readouterr()
    expected = _capture_special_sum(
        values, lambda window: builtins.sum(window), error_mode
    )
    expected_output = capsys.readouterr()

    assert actual == expected
    assert actual_output == expected_output


def _fail_if_builtin_sum_helper_called(*args):
    raise AssertionError("raw builtin sum helper must not be called")


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(partial(builtins.sum), id="partial"),
        pytest.param(builtin_sum_wrapper, id="wrapper"),
        pytest.param(np.sum, id="numpy_sum"),
        pytest.param(lambda values: builtins.sum(values), id="lambda"),
    ],
)
def test_raw_builtin_sum_callable_identity_fallback(
    monkeypatch, enable_boostkit_fastpaths, function
):
    monkeypatch.setattr(
        window_aggregations,
        "roll_apply_builtin_sum",
        _fail_if_builtin_sum_helper_called,
    )

    Series([1.0, 2.0, 3.0]).rolling(2).apply(function, raw=True)


def test_raw_builtin_sum_rebound_builtin_falls_back(
    monkeypatch, enable_boostkit_fastpaths
):
    # ``builtins.sum`` is mutable at runtime.  If a user rebinds
    # ``builtins.sum`` (e.g. via monkeypatch) and passes the rebound object to
    # ``rolling.apply``, the fast path must NOT engage: the specialization
    # skips the user callable and runs a Cython reduction, so engaging it on a
    # rebound ``sum`` would silently ignore the user's replacement.  The gate
    # compares against ``_ORIGINAL_BUILTIN_SUM`` captured at module load and
    # also checks ``builtins.sum`` has not been replaced, so the rebound case
    # falls through to the generic callback path and the user callable runs.
    monkeypatch.setattr(
        window_aggregations,
        "roll_apply_builtin_sum",
        _fail_if_builtin_sum_helper_called,
    )

    def fake_sum(values):
        return 123.0

    monkeypatch.setattr(builtins, "sum", fake_sum)

    result = Series([1.0, 2.0, 3.0]).rolling(2).apply(fake_sum, raw=True)
    expected = Series([np.nan, 123.0, 123.0])
    tm.assert_series_equal(result, expected)


def test_raw_builtin_sum_rebound_builtin_disables_fastpath_even_for_real_sum(
    monkeypatch, enable_boostkit_fastpaths
):
    # When ``builtins.sum`` has been rebound, the fast path must stay off even
    # if the user passes the *original* builtin ``sum`` (captured before the
    # rebind).  This is the conservative side of the dual check: an instrumented
    # environment that replaced ``builtins.sum`` should not silently engage the
    # specialization.  The result is still correct because the generic path
    # calls the real ``sum``.
    monkeypatch.setattr(
        window_aggregations,
        "roll_apply_builtin_sum",
        _fail_if_builtin_sum_helper_called,
    )

    real_sum = builtins.sum
    monkeypatch.setattr(builtins, "sum", lambda values: 999.0)

    Series([1.0, 2.0, 3.0]).rolling(2).apply(real_sum, raw=True)


@pytest.mark.usefixtures("enable_boostkit_fastpaths")
@pytest.mark.parametrize("constructor", [Series, DataFrame])
def test_raw_true_builtin_sum_unaligned_falls_back(constructor):
    # NumPy supports C-contiguous-but-not-aligned arrays (e.g. a float64 view
    # at a 1-byte offset).  The direct-view and builtin-sum fast paths wrap the
    # source pointer with ``PyArray_SimpleNewFromData`` and dereference it via
    # typed pointers, so they must reject misaligned input and fall back to
    # the generic ``arr[s:e]`` path whose flags NumPy computes from the real
    # offset.  The result must still match the fallback bit-for-bit.
    n = 33
    raw = np.empty(n * 8 + 1, dtype=np.uint8)
    arr = np.ndarray(n, dtype=np.float64, buffer=raw, offset=1)
    arr[:] = np.arange(n, dtype=np.float64)
    assert arr.flags.c_contiguous
    assert not arr.flags.aligned

    _apply_builtin_sum_and_oracle(constructor(arr, copy=False), 3, min_periods=1)


@pytest.mark.usefixtures("enable_boostkit_fastpaths")
@pytest.mark.parametrize("constructor", [Series, DataFrame])
def test_raw_true_generic_callback_unaligned_falls_back(constructor):
    # The direct-view path (used for any raw callback when dtype is native
    # float64) also wraps the source pointer, so misaligned input must fall
    # back to ``arr[s:e]``.  The callback observes a window whose
    # ``flags.aligned`` is truthful (``False``), not the default ``True`` that
    # ``PyArray_SimpleNewFromData`` would have produced.
    n = 33
    raw = np.empty(n * 8 + 1, dtype=np.uint8)
    arr = np.ndarray(n, dtype=np.float64, buffer=raw, offset=1)
    arr[:] = np.arange(n, dtype=np.float64)
    assert not arr.flags.aligned

    observed_alignment = []

    def cb(window):
        observed_alignment.append(bool(window.flags.aligned))
        return float(np.sum(window))

    obj = constructor(arr, copy=False)
    result = obj.rolling(3, min_periods=1).apply(cb, raw=True)
    expected = obj.rolling(3, min_periods=1).apply(
        lambda w: float(np.sum(w)), raw=True
    )
    tm.assert_equal(result, expected)
    # Every observed window must report truthful (non-aligned) flags.
    assert observed_alignment and not any(observed_alignment)


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(False, id="raw_false"),
        pytest.param(np.bool_(True), id="numpy_true"),
    ],
)
def test_raw_builtin_sum_exact_raw_true_fallback(
    monkeypatch, enable_boostkit_fastpaths, raw
):
    monkeypatch.setattr(
        window_aggregations,
        "roll_apply_builtin_sum",
        _fail_if_builtin_sum_helper_called,
    )

    Series([1.0, 2.0, 3.0]).rolling(2).apply(builtins.sum, raw=raw)


@pytest.mark.parametrize(
    "args,kwargs",
    [
        pytest.param((10.0,), None, id="args"),
        pytest.param(None, {"start": 10.0}, id="kwargs"),
    ],
)
def test_raw_builtin_sum_arguments_fallback(
    monkeypatch, enable_boostkit_fastpaths, args, kwargs
):
    monkeypatch.setattr(
        window_aggregations,
        "roll_apply_builtin_sum",
        _fail_if_builtin_sum_helper_called,
    )

    Series([1.0, 2.0, 3.0]).rolling(2).apply(
        builtins.sum, raw=True, args=args, kwargs=kwargs
    )


@pytest.mark.parametrize("mode", ["warn", "raise", "call", "log", "print"])
def test_raw_builtin_sum_nondefault_numpy_error_state_fallback(
    monkeypatch, enable_boostkit_fastpaths, mode
):
    monkeypatch.setattr(
        window_aggregations,
        "roll_apply_builtin_sum",
        _fail_if_builtin_sum_helper_called,
    )
    previous = np.geterrcall()
    error_log = []

    class Log:
        def write(self, message):
            error_log.append(message)

    error_callback = (
        (lambda error, flag: error_log.append((error, flag)))
        if mode == "call"
        else Log()
    )
    np.seterrcall(error_callback)
    try:
        with np.errstate(all=mode):
            Series([1.0, 2.0, 3.0]).rolling(2).apply(builtins.sum, raw=True)
    finally:
        np.seterrcall(previous)


@pytest.mark.parametrize("hook", ["trace", "profile"])
def test_raw_builtin_sum_active_hook_fallback(
    monkeypatch, enable_boostkit_fastpaths, hook
):
    monkeypatch.setattr(
        window_aggregations,
        "roll_apply_builtin_sum",
        _fail_if_builtin_sum_helper_called,
    )

    def trace(frame, event, arg):
        return trace

    def profile(frame, event, arg):
        return None

    if hook == "trace":
        previous = sys.gettrace()
        sys.settrace(trace)
        restore = lambda: sys.settrace(previous)
    else:
        previous = sys.getprofile()
        sys.setprofile(profile)
        restore = lambda: sys.setprofile(previous)
    try:
        Series([1.0, 2.0, 3.0]).rolling(2).apply(builtins.sum, raw=True)
    finally:
        restore()


def test_raw_builtin_sum_disabled_boostkit_fallback(monkeypatch):
    monkeypatch.setattr(boostkit_fastpaths, "USE_BOOSTKIT_FASTPATHS", False)
    monkeypatch.setattr(
        window_aggregations,
        "roll_apply_builtin_sum",
        _fail_if_builtin_sum_helper_called,
    )

    Series([1.0, 2.0, 3.0]).rolling(2).apply(builtins.sum, raw=True)


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


def test_raw_true_reversed_bounds_fall_back_to_numpy_slice():
    class ReversedBoundsIndexer(BaseIndexer):
        def get_window_bounds(
            self, num_values, min_periods, center, closed, step
        ):
            start = np.array([0, 1, 2], dtype=np.int64)
            end = np.array([1, 0, 3], dtype=np.int64)
            return start, end

    windows = []

    def function(window):
        windows.append(window.copy())
        return len(window)

    result = (
        Series([1.0, 2.0, 3.0])
        .rolling(ReversedBoundsIndexer(), min_periods=0)
        .apply(function, raw=True)
    )

    expected = Series([1.0, 0.0, 1.0])
    tm.assert_series_equal(result, expected)
    assert [window.tolist() for window in windows] == [[1.0], [], [3.0]]


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
