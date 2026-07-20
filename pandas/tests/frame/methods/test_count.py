import numpy as np
import pytest

from pandas._libs import algos

from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm


class TestDataFrameCount:
    def test_count(self):
        # corner case
        frame = DataFrame()
        ct1 = frame.count(1)
        assert isinstance(ct1, Series)

        ct2 = frame.count(0)
        assert isinstance(ct2, Series)

        # GH#423
        df = DataFrame(index=range(10))
        result = df.count(1)
        expected = Series(0, index=df.index)
        tm.assert_series_equal(result, expected)

        df = DataFrame(columns=range(10))
        result = df.count(0)
        expected = Series(0, index=df.columns)
        tm.assert_series_equal(result, expected)

        df = DataFrame()
        result = df.count()
        expected = Series(dtype="int64")
        tm.assert_series_equal(result, expected)

    def test_count_objects(self, float_string_frame):
        dm = DataFrame(float_string_frame._series)
        df = DataFrame(float_string_frame._series)

        tm.assert_series_equal(dm.count(), df.count())
        tm.assert_series_equal(dm.count(1), df.count(1))


@pytest.mark.parametrize("axis", [0, 1])
def test_count_float_block_uses_nancount(monkeypatch, axis):
    df = DataFrame([[1.0, np.nan, 3.0], [np.nan, 2.0, 4.0]])
    original = algos.nancount_2d
    called = False

    def wrapped(values, op_axis):
        nonlocal called
        called = True
        assert op_axis == axis
        return original(values, op_axis)

    monkeypatch.setattr(algos, "nancount_2d", wrapped)
    result = df.count(axis=axis)
    assert called
    expected = (
        Series([1, 1, 2], index=df.columns, dtype="int64")
        if axis == 0
        else Series([2, 2], index=df.index, dtype="int64")
    )
    tm.assert_series_equal(result, expected)


def test_count_nullable_float_does_not_use_nancount(monkeypatch):
    df = DataFrame({"a": Series([1, None], dtype="Float64")})

    def fail_if_called(*args, **kwargs):
        pytest.fail("nancount_2d must not receive an ExtensionBlock")

    monkeypatch.setattr(algos, "nancount_2d", fail_if_called)
    tm.assert_series_equal(df.count(), Series([1], index=["a"]))
