from __future__ import annotations

import importlib

import numpy as np
import pytest

from pandas._libs import hashtable as htable

import pandas._testing as tm
from pandas.core import algorithms


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


@pytest.mark.parametrize("enabled", [False, True])
def test_hash_inner_join_paths_preserve_semantics(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    _set_fastpaths(monkeypatch, enabled)
    right = np.array([1, 2, 4, 8], dtype=np.int64)
    left = np.array([8, 3, 2, 1, 5], dtype=np.int64)
    factorizer = htable.Int64Factorizer(len(right))
    factorizer.factorize(right)

    right_indexer, left_indexer = factorizer.hash_inner_join(left)

    tm.assert_numpy_array_equal(right_indexer, np.array([3, 1, 0], dtype=np.intp))
    tm.assert_numpy_array_equal(left_indexer, np.array([0, 2, 3], dtype=np.intp))


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
