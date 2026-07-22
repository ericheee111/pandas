from __future__ import annotations

import importlib

import numpy as np
import pytest

from pandas._libs import hashtable as htable

import pandas as pd
import pandas._testing as tm
from pandas.core import algorithms
from pandas.core.arrays.masked import BaseMaskedArray


def _set_fastpaths(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    fastpaths = importlib.import_module("pandas.core.boostkit_fastpaths")
    monkeypatch.setattr(fastpaths, "USE_BOOSTKIT_FASTPATHS", enabled)
    htable.set_use_boostkit_fastpaths(enabled)


@pytest.fixture(autouse=True)
def _restore_cython_fastpath_state(monkeypatch: pytest.MonkeyPatch):
    fastpaths = importlib.import_module("pandas.core.boostkit_fastpaths")
    original = fastpaths.USE_BOOSTKIT_FASTPATHS
    yield
    htable.set_use_boostkit_fastpaths(original)


@pytest.mark.parametrize(
    "machine, setting, expected",
    [
        ("aarch64", None, True),
        ("arm64", "auto", True),
        ("aarch64", "0", False),
        ("aarch64", "false", False),
        ("aarch64", "no", False),
        ("aarch64", "off", False),
        ("x86_64", None, False),
        ("AMD64", "1", False),
    ],
)
def test_boostkit_fastpaths_environment(
    monkeypatch: pytest.MonkeyPatch,
    machine: str,
    setting: str | None,
    expected: bool,
) -> None:
    fastpaths = importlib.import_module("pandas.core.boostkit_fastpaths")

    try:
        with monkeypatch.context() as context:
            calls: list[bool] = []
            context.setattr(fastpaths.platform, "machine", lambda: machine)
            context.setattr(
                fastpaths.htable, "set_use_boostkit_fastpaths", calls.append
            )
            if setting is None:
                context.delenv("PANDAS_BOOSTKIT_FASTPATHS", raising=False)
            else:
                context.setenv("PANDAS_BOOSTKIT_FASTPATHS", setting)

            importlib.reload(fastpaths)
            assert fastpaths.USE_BOOSTKIT_FASTPATHS is expected
            assert calls == [expected]
    finally:
        importlib.reload(fastpaths)


def test_isin_zero_range_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_fastpaths(monkeypatch, False)
    comps = np.arange(1_000_000, dtype=np.int64)
    values = np.arange(1_000, dtype=np.int64)

    result = algorithms._isin_zero_range(comps, values)

    assert result is None


@pytest.mark.parametrize("enabled, expected_calls", [(False, 1), (True, 0)])
def test_isin_dtype_normalization_dispatch(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, expected_calls: int
) -> None:
    _set_fastpaths(monkeypatch, enabled)
    original = algorithms.np_find_common_type
    calls = 0

    def wrapped(left: np.dtype, right: np.dtype) -> np.dtype:
        nonlocal calls
        calls += 1
        return original(left, right)

    monkeypatch.setattr(algorithms, "np_find_common_type", wrapped)
    comps = np.arange(1_000, dtype=np.int64)
    values = np.arange(100, dtype=np.int64)

    result = algorithms.isin(comps, values)

    expected = np.zeros(1_000, dtype=bool)
    expected[:100] = True
    tm.assert_numpy_array_equal(result, expected)
    assert calls == expected_calls


@pytest.mark.parametrize("enabled", [False, True])
def test_isin_zero_range_public_dispatch(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    _set_fastpaths(monkeypatch, enabled)
    calls = 0

    def sentinel(comps: np.ndarray, values: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.ones(len(comps), dtype=bool)

    monkeypatch.setattr(algorithms, "_isin_zero_range", sentinel)
    comps = np.array([0, 2], dtype=np.int64)
    values = np.array([0], dtype=np.int64)

    result = algorithms.isin(comps, values)

    expected = np.ones(2, dtype=bool) if enabled else np.array([True, False])
    tm.assert_numpy_array_equal(result, expected)
    assert calls == int(enabled)


def test_float64_monotonic_helpers_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_fastpaths(monkeypatch, False)
    values = np.repeat(np.arange(1_000, dtype=np.float64), 100)

    assert algorithms._unique_float64_monotonic_runs(values) is None
    assert algorithms._factorize_float64_monotonic_runs(values) is None


def test_float64_monotonic_helpers_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_fastpaths(monkeypatch, True)
    values = np.repeat(np.arange(1_000, dtype=np.float64), 100)

    uniques = algorithms._unique_float64_monotonic_runs(values)
    factorized = algorithms._factorize_float64_monotonic_runs(values)

    assert uniques is not None
    tm.assert_numpy_array_equal(uniques, np.arange(1_000, dtype=np.float64))
    assert factorized is not None
    codes, factorized_uniques = factorized
    tm.assert_numpy_array_equal(codes, np.repeat(np.arange(1_000), 100))
    tm.assert_numpy_array_equal(factorized_uniques, uniques)


@pytest.mark.parametrize("enabled", [False, True])
def test_float64_monotonic_public_dispatch(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    _set_fastpaths(monkeypatch, enabled)
    values = np.array([2.0, 1.0, 2.0])
    unique_calls = 0
    factorize_calls = 0

    def unique_sentinel(values: np.ndarray) -> np.ndarray:
        nonlocal unique_calls
        unique_calls += 1
        return np.array([9.0])

    def factorize_sentinel(
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        nonlocal factorize_calls
        factorize_calls += 1
        return np.array([0, 0, 0]), np.array([9.0])

    monkeypatch.setattr(algorithms, "_unique_float64_monotonic_runs", unique_sentinel)
    monkeypatch.setattr(
        algorithms, "_factorize_float64_monotonic_runs", factorize_sentinel
    )

    uniques = algorithms.unique(values)
    codes, factorized_uniques = algorithms.factorize_array(values)

    if enabled:
        tm.assert_numpy_array_equal(uniques, np.array([9.0]))
        tm.assert_numpy_array_equal(codes, np.array([0, 0, 0]))
        tm.assert_numpy_array_equal(factorized_uniques, np.array([9.0]))
    else:
        tm.assert_numpy_array_equal(uniques, np.array([2.0, 1.0]))
        tm.assert_numpy_array_equal(codes, np.array([0, 1, 0]))
        tm.assert_numpy_array_equal(factorized_uniques, np.array([2.0, 1.0]))
    assert unique_calls == int(enabled)
    assert factorize_calls == int(enabled)


@pytest.mark.parametrize("enabled", [False, True])
def test_float64_hashtable_paths_preserve_semantics(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    _set_fastpaths(monkeypatch, enabled)
    values = np.array([0.0, -0.0, np.nan, np.nan, 1.0, 1.0])

    uniques, inverse = htable.Float64HashTable().unique(values, return_inverse=True)
    tm.assert_numpy_array_equal(uniques, np.array([0.0, np.nan, 1.0]))
    tm.assert_numpy_array_equal(inverse, np.array([0, 0, 1, 1, 2, 2]))
    assert not np.signbit(uniques[0])

    factorized_uniques, labels = htable.Float64HashTable().factorize(values)
    tm.assert_numpy_array_equal(factorized_uniques, np.array([0.0, 1.0]))
    tm.assert_numpy_array_equal(labels, np.array([0, 0, -1, -1, 1, 1]))

    factorizer = htable.Float64Factorizer(len(values))
    factorizer_labels = factorizer.factorize(values)
    tm.assert_numpy_array_equal(factorizer_labels, labels)


def test_nullable_unique_helpers_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_fastpaths(monkeypatch, False)

    def unexpected(self: BaseMaskedArray) -> None:
        raise AssertionError("BoostKit nullable unique helper was called")

    monkeypatch.setattr(BaseMaskedArray, "_unique_if_monotonic", unexpected)
    monkeypatch.setattr(BaseMaskedArray, "_unique_if_repeated_chunks", unexpected)

    result = pd.array([1, 1, pd.NA, 2], dtype="Int64").unique()

    expected = pd.array([1, pd.NA, 2], dtype="Int64")
    tm.assert_extension_array_equal(result, expected)


def test_nullable_unique_helper_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_fastpaths(monkeypatch, True)
    expected = pd.array([9, pd.NA], dtype="Int64")
    calls = 0

    def sentinel(self: BaseMaskedArray) -> BaseMaskedArray:
        nonlocal calls
        calls += 1
        return expected

    def unexpected(self: BaseMaskedArray) -> None:
        raise AssertionError("Repeated-chunk helper should not be called")

    monkeypatch.setattr(BaseMaskedArray, "_unique_if_monotonic", sentinel)
    monkeypatch.setattr(BaseMaskedArray, "_unique_if_repeated_chunks", unexpected)

    result = pd.array([1, 1, pd.NA, 2], dtype="Int64").unique()

    assert result is expected
    assert calls == 1


@pytest.mark.parametrize("enabled, expected_calls", [(False, 1), (True, 0)])
def test_sorted_factorize_safe_sort_dispatch(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, expected_calls: int
) -> None:
    _set_fastpaths(monkeypatch, enabled)
    original = algorithms.safe_sort
    calls = 0

    def wrapped(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(algorithms, "safe_sort", wrapped)

    codes, uniques = algorithms.factorize(
        np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float64), sort=True
    )

    tm.assert_numpy_array_equal(codes, np.array([0, 0, 1, 1]))
    tm.assert_numpy_array_equal(uniques, np.array([1.0, 2.0]))
    assert calls == expected_calls
