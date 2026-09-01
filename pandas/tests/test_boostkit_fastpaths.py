from __future__ import annotations

import importlib

import numpy as np
import pytest

from pandas._libs import hashtable as htable

from pandas import (
    Categorical,
    Series,
    array,
)
import pandas._testing as tm
from pandas.core import algorithms

_INTEGER_DTYPES = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
]


def _set_fastpaths(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    fastpaths = importlib.import_module("pandas.core.boostkit_fastpaths")
    monkeypatch.setattr(fastpaths, "USE_BOOSTKIT_FASTPATHS", enabled)


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
            context.setattr(fastpaths.platform, "machine", lambda: machine)
            if setting is None:
                context.delenv("PANDAS_BOOSTKIT_FASTPATHS", raising=False)
            else:
                context.setenv("PANDAS_BOOSTKIT_FASTPATHS", setting)

            importlib.reload(fastpaths)
            assert fastpaths.USE_BOOSTKIT_FASTPATHS is expected
    finally:
        importlib.reload(fastpaths)


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


def test_hash_inner_join_dispatch_matches_legacy():
    right = np.array([1, 2, 4, 8], dtype=np.int64)
    left = np.array([8, 3, 2, 1, 5], dtype=np.int64)
    factorizer = htable.Int64Factorizer(len(right))
    factorizer.factorize(right)

    expected = factorizer.table.hash_inner_join_legacy(left)
    result = factorizer.hash_inner_join(left)

    tm.assert_numpy_array_equal(result[0], expected[0])
    tm.assert_numpy_array_equal(result[1], expected[1])
    tm.assert_numpy_array_equal(result[0], np.array([3, 1, 0], dtype=np.intp))
    tm.assert_numpy_array_equal(result[1], np.array([0, 2, 3], dtype=np.intp))


def test_hash_inner_join_legacy_preserves_masked_na():
    right = np.array([1, 0, 4], dtype=np.int64)
    right_mask = np.array([False, True, False])
    left = np.array([0, 4, 2, 0], dtype=np.int64)
    left_mask = np.array([True, False, False, True])
    factorizer = htable.Int64Factorizer(len(right), uses_mask=True)
    factorizer.table.map_locations(right, mask=right_mask)

    result = factorizer.table.hash_inner_join_legacy(left, mask=left_mask)

    tm.assert_numpy_array_equal(result[0], np.array([1, 2, 1], dtype=np.intp))
    tm.assert_numpy_array_equal(result[1], np.array([0, 1, 3], dtype=np.intp))


@pytest.mark.parametrize("dtype", _INTEGER_DTYPES)
@pytest.mark.parametrize("keep", ["first", "last", False])
def test_duplicated_integer_small_range_dispatch(
    monkeypatch: pytest.MonkeyPatch, dtype: str, keep: str | bool
) -> None:
    values = (np.arange(20_000) % 257).astype(dtype)
    np_dtype = np.dtype(dtype)
    direct = algorithms._duplicated_int_direct[np_dtype]
    portable = htable.duplicated
    calls = {"direct": 0, "portable": 0}

    def wrapped_direct(*args, **kwargs):
        calls["direct"] += 1
        return direct(*args, **kwargs)

    def wrapped_portable(*args, **kwargs):
        calls["portable"] += 1
        return portable(*args, **kwargs)

    monkeypatch.setattr(algorithms, "get_use_swisstable", lambda: True)
    monkeypatch.setitem(algorithms._duplicated_int_direct, np_dtype, wrapped_direct)
    monkeypatch.setattr(htable, "duplicated", wrapped_portable)

    result = algorithms.duplicated(values, keep=keep)
    public_result = Series(values).duplicated(keep=keep).to_numpy()
    expected = portable(values, keep=keep)

    tm.assert_numpy_array_equal(result, expected)
    tm.assert_numpy_array_equal(public_result, expected)
    assert calls == {"direct": 2, "portable": 0}


@pytest.mark.parametrize("layout", ["high_range", "noncontiguous"])
def test_duplicated_integer_direct_falls_back(
    monkeypatch: pytest.MonkeyPatch, layout: str
) -> None:
    if layout == "high_range":
        values = np.arange(20_000, dtype=np.int64)
    else:
        values = (np.arange(40_000) % 257).astype(np.int64)[::2]

    dtype = np.dtype("int64")
    direct = algorithms._duplicated_int_direct[dtype]
    portable = htable.duplicated
    calls = {"direct": 0, "portable": 0}

    def wrapped_direct(*args, **kwargs):
        calls["direct"] += 1
        return direct(*args, **kwargs)

    def wrapped_portable(*args, **kwargs):
        calls["portable"] += 1
        return portable(*args, **kwargs)

    monkeypatch.setattr(algorithms, "get_use_swisstable", lambda: True)
    monkeypatch.setitem(algorithms._duplicated_int_direct, dtype, wrapped_direct)
    monkeypatch.setattr(htable, "duplicated", wrapped_portable)

    result = algorithms.duplicated(values)
    expected = portable(values)

    tm.assert_numpy_array_equal(result, expected)
    assert calls == {"direct": 1, "portable": 1}


@pytest.mark.parametrize("dtype", _INTEGER_DTYPES)
def test_duplicated_integer_direct_small_falls_back_without_allocating(
    monkeypatch: pytest.MonkeyPatch, dtype: str
) -> None:
    values = np.arange(1000).astype(dtype)
    empty = np.empty
    calls = 0

    def wrapped_empty(*args, **kwargs):
        nonlocal calls
        calls += 1
        return empty(*args, **kwargs)

    monkeypatch.setattr(np, "empty", wrapped_empty)

    result = algorithms._duplicated_int_direct[np.dtype(dtype)](values)

    assert result is None
    assert calls == 0


@pytest.mark.parametrize("mask_kind", ["all_false", "partial", "multiple_na"])
@pytest.mark.parametrize("keep", ["first", "last", False])
def test_duplicated_masked_integer_uses_portable(
    monkeypatch: pytest.MonkeyPatch, mask_kind: str, keep: str | bool
) -> None:
    values = (np.arange(20_000) % 257).astype(np.int64)
    mask = np.zeros(len(values), dtype=np.bool_)
    if mask_kind == "partial":
        mask[1] = True
    elif mask_kind == "multiple_na":
        mask[[1, 5, 101]] = True

    direct = algorithms._duplicated_int_direct[np.dtype("int64")]
    portable = htable.duplicated
    calls = {"direct": 0, "portable": 0}

    def wrapped_direct(*args, **kwargs):
        calls["direct"] += 1
        return direct(*args, **kwargs)

    def wrapped_portable(*args, **kwargs):
        calls["portable"] += 1
        return portable(*args, **kwargs)

    monkeypatch.setattr(algorithms, "get_use_swisstable", lambda: True)
    monkeypatch.setitem(
        algorithms._duplicated_int_direct, np.dtype("int64"), wrapped_direct
    )
    monkeypatch.setattr(htable, "duplicated", wrapped_portable)

    result = algorithms.duplicated(values, keep=keep, mask=mask)
    expected = portable(values, keep=keep, mask=mask)

    tm.assert_numpy_array_equal(result, expected)
    assert calls == {"direct": 0, "portable": 1}


@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
@pytest.mark.parametrize("masked", [False, True])
def test_duplicated_float_complex_keeps_swisstable_dispatch(
    monkeypatch: pytest.MonkeyPatch, dtype: str, masked: bool
) -> None:
    values = np.array([1.5, 2.5, 1.5, np.nan], dtype=dtype)
    mask = np.array([False, True, False, False]) if masked else None
    np_dtype = np.dtype(dtype)
    swiss = algorithms._duplicated_swisstable[np_dtype]
    portable = htable.duplicated
    calls = {"swiss": 0, "portable": 0}

    def wrapped_swiss(*args, **kwargs):
        calls["swiss"] += 1
        return swiss(*args, **kwargs)

    def wrapped_portable(*args, **kwargs):
        calls["portable"] += 1
        return portable(*args, **kwargs)

    monkeypatch.setattr(algorithms, "get_use_swisstable", lambda: True)
    monkeypatch.setitem(algorithms._duplicated_swisstable, np_dtype, wrapped_swiss)
    monkeypatch.setattr(htable, "duplicated", wrapped_portable)

    result = algorithms.duplicated(values, mask=mask)
    expected = portable(values, mask=mask)

    tm.assert_numpy_array_equal(result, expected)
    assert calls == {"swiss": 1, "portable": 0}


@pytest.mark.parametrize(
    "dtype", [*_INTEGER_DTYPES, "float32", "float64", "complex64", "complex128"]
)
def test_duplicated_disabled_uses_portable(
    monkeypatch: pytest.MonkeyPatch, dtype: str
) -> None:
    values = np.array([1, 2, 1], dtype=dtype)
    portable = htable.duplicated
    calls = 0

    def wrapped_portable(*args, **kwargs):
        nonlocal calls
        calls += 1
        return portable(*args, **kwargs)

    monkeypatch.setattr(algorithms, "get_use_swisstable", lambda: False)
    monkeypatch.setattr(htable, "duplicated", wrapped_portable)

    result = algorithms.duplicated(values)
    expected = portable(values)

    tm.assert_numpy_array_equal(result, expected)
    assert calls == 1


@pytest.mark.parametrize("dtype", _INTEGER_DTYPES)
@pytest.mark.parametrize("keep", ["first", "last", False])
def test_duplicated_integer_shapes_match_portable(
    monkeypatch: pytest.MonkeyPatch, dtype: str, keep: str | bool
) -> None:
    info = np.iinfo(dtype)
    boundary = [info.min, info.max, info.min, info.max]
    if dtype.startswith("int"):
        boundary.extend([-2, -1, -2])
    values = [
        np.array([], dtype=dtype),
        np.arange(min(200, info.max), dtype=dtype),
        np.full(1000, 7, dtype=dtype),
        (np.arange(1000) % 17).astype(dtype),
        np.array(boundary, dtype=dtype),
        (np.arange(2000) % 17).astype(dtype)[::2],
    ]

    for case in values:
        monkeypatch.setattr(algorithms, "get_use_swisstable", lambda: False)
        expected = algorithms.duplicated(case, keep=keep)
        monkeypatch.setattr(algorithms, "get_use_swisstable", lambda: True)
        result = algorithms.duplicated(case, keep=keep)
        tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("keep", ["first", "last", False])
def test_duplicated_ensure_data_dtypes(
    monkeypatch: pytest.MonkeyPatch, keep: str | bool
) -> None:
    values = [
        np.array([True, False, True], dtype=np.bool_),
        np.array(["2020-01-01", "NaT", "2020-01-01"], dtype="datetime64[ns]"),
        np.array([1, "NaT", 1], dtype="m8[ns]"),
    ]

    for case in values:
        monkeypatch.setattr(algorithms, "get_use_swisstable", lambda: False)
        expected = algorithms.duplicated(case, keep=keep)
        monkeypatch.setattr(algorithms, "get_use_swisstable", lambda: True)
        result = algorithms.duplicated(case, keep=keep)
        tm.assert_numpy_array_equal(result, expected)


def test_duplicated_categorical_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    values = Categorical(["a", "b", "a", None, None])
    monkeypatch.setattr(algorithms, "get_use_swisstable", lambda: False)
    expected = algorithms.duplicated(values)
    monkeypatch.setattr(algorithms, "get_use_swisstable", lambda: True)
    result = algorithms.duplicated(values)
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("values", [[1, 2, 3], [1, None, 2], [1, None, None]])
@pytest.mark.parametrize("keep", ["first", "last", False])
def test_duplicated_nullable_integer_keeps_portable_semantics(
    monkeypatch: pytest.MonkeyPatch,
    values: list[int | None],
    keep: str | bool,
) -> None:
    masked = array(values, dtype="Int64")
    monkeypatch.setattr(algorithms, "get_use_swisstable", lambda: False)
    expected = masked.duplicated(keep=keep)
    monkeypatch.setattr(algorithms, "get_use_swisstable", lambda: True)
    result = masked.duplicated(keep=keep)
    tm.assert_numpy_array_equal(result, expected)
