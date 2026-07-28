from functools import partial
import sys

import numpy as np
import pytest

import pandas._libs.window.aggregations as window_aggregations

from pandas import Series
import pandas._testing as tm


def _get_rolling_aggregations():
    # list pairs of name and function
    # each function has this signature:
    # (const float64_t[:] values, ndarray[int64_t] start,
    #  ndarray[int64_t] end, int64_t minp) -> np.ndarray
    named_roll_aggs = (
        [
            ("roll_sum", window_aggregations.roll_sum),
            ("roll_mean", window_aggregations.roll_mean),
        ]
        + [
            (f"roll_var({ddof})", partial(window_aggregations.roll_var, ddof=ddof))
            for ddof in [0, 1]
        ]
        + [
            ("roll_skew", window_aggregations.roll_skew),
            ("roll_kurt", window_aggregations.roll_kurt),
            ("roll_median_c", window_aggregations.roll_median_c),
            ("roll_max", window_aggregations.roll_max),
            ("roll_min", window_aggregations.roll_min),
            ("roll_first", window_aggregations.roll_first),
            ("roll_last", window_aggregations.roll_last),
            ("roll_nunique", window_aggregations.roll_nunique),
        ]
        + [
            (
                f"roll_quantile({quantile},{interpolation})",
                partial(
                    window_aggregations.roll_quantile,
                    quantile=quantile,
                    interpolation=interpolation,
                ),
            )
            for quantile in [0.0001, 0.5, 0.9999]
            for interpolation in window_aggregations.interpolation_types
        ]
        + [
            (
                f"roll_rank({percentile},{method},{ascending})",
                partial(
                    window_aggregations.roll_rank,
                    percentile=percentile,
                    method=method,
                    ascending=ascending,
                ),
            )
            for percentile in [True, False]
            for method in window_aggregations.rolling_rank_tiebreakers.keys()
            for ascending in [True, False]
        ]
    )
    # unzip to a list of 2 tuples, names and functions
    unzipped = list(zip(*named_roll_aggs, strict=True))
    return {"ids": unzipped[0], "params": unzipped[1]}


_rolling_aggregations = _get_rolling_aggregations()


@pytest.fixture(
    params=_rolling_aggregations["params"], ids=_rolling_aggregations["ids"]
)
def rolling_aggregation(request):
    """Make a rolling aggregation function as fixture."""
    return request.param


def test_rolling_aggregation_boundary_consistency(rolling_aggregation):
    # GH-45647
    minp, step, width, size, selection = 0, 1, 3, 11, [2, 7]
    values = np.arange(1, 1 + size, dtype=np.float64)
    end = np.arange(width, size, step, dtype=np.int64)
    start = end - width
    selarr = np.array(selection, dtype=np.int32)
    result = Series(rolling_aggregation(values, start[selarr], end[selarr], minp))
    expected = Series(rolling_aggregation(values, start, end, minp)[selarr])
    tm.assert_equal(expected, result)


def test_rolling_aggregation_with_unused_elements(rolling_aggregation):
    # GH-45647
    minp, width = 0, 5  # width at least 4 for kurt
    size = 2 * width + 5
    values = np.arange(1, size + 1, dtype=np.float64)
    values[width : width + 2] = sys.float_info.min
    values[width + 2] = np.nan
    values[width + 3 : width + 5] = sys.float_info.max
    start = np.array([0, size - width], dtype=np.int64)
    end = np.array([width, size], dtype=np.int64)
    loc = np.array(
        [j for i in range(len(start)) for j in range(start[i], end[i])],
        dtype=np.int32,
    )
    result = Series(rolling_aggregation(values, start, end, minp))
    compact_values = np.array(values[loc], dtype=np.float64)
    compact_start = np.arange(0, len(start) * width, width, dtype=np.int64)
    compact_end = compact_start + width
    expected = Series(
        rolling_aggregation(compact_values, compact_start, compact_end, minp)
    )
    assert np.isfinite(expected.values).all(), "Not all expected values are finite"
    tm.assert_equal(expected, result)


@pytest.mark.parametrize(
    "values, expected",
    [
        ([], True),
        ([0.0, -1.0, np.finfo(np.float64).max], True),
        ([0.0, np.nan], False),
        ([0.0, np.inf], False),
        ([0.0, -np.inf], False),
    ],
)
def test_roll_all_finite(values, expected):
    result = window_aggregations.roll_all_finite(np.array(values, dtype=np.float64))

    assert result is expected


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("position", [0, 7, 8])
def test_roll_all_finite_vectorized_boundaries(nonfinite, position):
    values = np.arange(9, dtype=np.float64)
    assert window_aggregations.roll_all_finite(values)

    values[position] = nonfinite

    assert not window_aggregations.roll_all_finite(values)


def test_roll_all_finite_strided():
    values = np.array([0.0, np.nan, 1.0, np.inf, 2.0])

    assert window_aggregations.roll_all_finite(values[::2])
    assert window_aggregations.roll_all_finite(values[::-2])
    assert not window_aggregations.roll_all_finite(values[1::2])


@pytest.mark.parametrize("dtype", [np.float64, np.int64])
@pytest.mark.parametrize("minp", [0, 1, 3])
@pytest.mark.parametrize("window", [1, 3, 10])
@pytest.mark.parametrize("method,kernel", [
    ("sum", "roll_sum_fixed_no_nan"),
    ("sum", "roll_sum_fixed_no_nan_int64"),
    ("max", "roll_max_fixed_no_nan"),
    ("max", "roll_max_fixed_no_nan_int64"),
    ("min", "roll_min_fixed_no_nan"),
    ("min", "roll_min_fixed_no_nan_int64"),
    ("mean", "roll_mean_fixed_no_nan"),
    ("mean", "roll_mean_fixed_no_nan_int64"),
])
def test_fixed_no_nan_matches_general(dtype, minp, window, method, kernel):
    if dtype == np.int64 and not kernel.endswith("_int64"):
        pytest.skip("dtype/kernel mismatch")
    if dtype == np.float64 and kernel.endswith("_int64"):
        pytest.skip("dtype/kernel mismatch")

    rng = np.random.RandomState(42)
    values = rng.randint(0, 100, 50).astype(dtype)
    N = len(values)
    start = np.maximum(0, np.arange(N) + 1 - window).astype(np.int64)
    end = (np.arange(N) + 1).astype(np.int64)

    f64_values = values.astype(np.float64)
    if method == "sum":
        expected = window_aggregations.roll_sum(f64_values, start, end, minp)
    elif method == "max":
        expected = window_aggregations.roll_max(f64_values, start, end, minp)
    elif method == "min":
        expected = window_aggregations.roll_min(f64_values, start, end, minp)
    elif method == "mean":
        expected = window_aggregations.roll_mean(f64_values, start, end, minp)

    fn = getattr(window_aggregations, kernel)
    result = fn(values, window, minp)

    if method in ("mean",):
        np.testing.assert_allclose(result, expected, rtol=1e-10, equal_nan=True)
    else:
        tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "dtype,kernel,method",
    [
        (np.float64, "roll_var_fixed_no_nan", "var"),
        (np.float64, "roll_std_fixed_no_nan", "std"),
        (np.int64, "roll_std_fixed_no_nan_int64", "std"),
    ],
)
@pytest.mark.parametrize("window", [1, 3, 10])
@pytest.mark.parametrize("minp,ddof", [(1, 0), (1, 1), (3, 1), (1, 10)])
def test_fixed_no_nan_variance_numerical_stability(
    dtype, kernel, method, window, minp, ddof
):
    values = np.array(
        [0, 10**12 + 1] + [10**12 + 2] * 12 + [3, 4, 5],
        dtype=dtype,
    )
    start = np.maximum(0, np.arange(len(values)) + 1 - window).astype(np.int64)
    end = np.arange(1, len(values) + 1, dtype=np.int64)
    if method == "var":
        expected = window_aggregations.roll_var(
            values, start, end, minp, ddof
        )
    else:
        expected = np.full(len(values), np.nan)
        for i, (s, e) in enumerate(zip(start, end, strict=True)):
            if e - s >= minp and e - s > ddof:
                current = values[s:e].astype(np.float64)
                expected[i] = np.std(current - current[0], ddof=ddof)

    result = getattr(window_aggregations, kernel)(values, window, minp, ddof)

    np.testing.assert_allclose(
        result, expected, rtol=1e-10, atol=1e-15, equal_nan=True
    )
    if method == "std" and window >= minp and window > ddof:
        assert result[11] == 0


@pytest.mark.parametrize("dtype", [np.float64, np.int64])
@pytest.mark.parametrize("minp", [0, 1, 3])
@pytest.mark.parametrize("method,kernel", [
    ("sum", "roll_sum_expanding_no_nan"),
    ("sum", "roll_sum_expanding_no_nan_int64"),
    ("max", "roll_max_expanding_no_nan"),
    ("max", "roll_max_expanding_no_nan_int64"),
    ("min", "roll_min_expanding_no_nan"),
    ("min", "roll_min_expanding_no_nan_int64"),
    ("mean", "roll_mean_expanding_no_nan"),
    ("mean", "roll_mean_expanding_no_nan_int64"),
    ("std", "roll_std_expanding_no_nan"),
    ("std", "roll_std_expanding_no_nan_int64"),
])
def test_expanding_no_nan_matches_general(dtype, minp, method, kernel):
    if dtype == np.int64 and not kernel.endswith("_int64"):
        pytest.skip("dtype/kernel mismatch")
    if dtype == np.float64 and kernel.endswith("_int64"):
        pytest.skip("dtype/kernel mismatch")

    rng = np.random.RandomState(42)
    values = rng.randint(0, 100, 50).astype(dtype)
    N = len(values)
    start = np.zeros(N, dtype=np.int64)
    end = np.arange(1, N + 1, dtype=np.int64)

    f64_values = values.astype(np.float64)
    if method == "sum":
        expected = window_aggregations.roll_sum(f64_values, start, end, minp)
    elif method == "max":
        expected = window_aggregations.roll_max(f64_values, start, end, minp)
    elif method == "min":
        expected = window_aggregations.roll_min(f64_values, start, end, minp)
    elif method == "mean":
        expected = window_aggregations.roll_mean(f64_values, start, end, minp)
    elif method == "std":
        expected = np.sqrt(
            window_aggregations.roll_var(f64_values, start, end, minp, ddof=1)
        )

    fn = getattr(window_aggregations, kernel)
    if method == "std":
        result = fn(values, minp, 1)
    else:
        result = fn(values, minp)

    if method in ("mean", "std"):
        np.testing.assert_allclose(result, expected, rtol=1e-10, equal_nan=True)
    else:
        tm.assert_numpy_array_equal(result, expected)
