from __future__ import annotations

import importlib

import numpy as np
import pytest

import pandas._testing as tm
from pandas.core import algorithms


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
