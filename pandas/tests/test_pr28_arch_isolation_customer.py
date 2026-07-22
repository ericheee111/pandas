import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas._libs import hashtable as libhashtable
from pandas.core.arrays import (
    categorical,
    string_,
)
from pandas.core.arrays.base import ExtensionArray


def test_categorical_value_counts_non_arm_uses_numpy(monkeypatch):
    monkeypatch.setattr(categorical, "IS_ARM", False)
    monkeypatch.setattr(
        categorical.libalgos,
        "count_categorical_codes",
        lambda *args: pytest.fail("ARM helper called on non-ARM"),
    )

    result = pd.Categorical(["a", "b", "a"]).value_counts()

    tm.assert_numpy_array_equal(result.to_numpy(), np.array([2, 1]))


def test_string_factorize_non_arm_delegates(monkeypatch):
    monkeypatch.setattr(string_, "IS_ARM", False)
    calls = 0
    original = ExtensionArray.factorize

    def wrapped(self, use_na_sentinel=True):
        nonlocal calls
        calls += 1
        return original(self, use_na_sentinel=use_na_sentinel)

    monkeypatch.setattr(ExtensionArray, "factorize", wrapped)

    dtype = pd.StringDtype(storage="python")
    pd.array(["a", "b", "a"], dtype=dtype).factorize()

    assert calls == 1


def test_string_factorize_arm_noncanonical_na_matches_base(monkeypatch):
    dtype = pd.StringDtype(storage="python", na_value=np.nan)
    values = np.array(["a", None, "b", "a"], dtype=object)

    monkeypatch.setattr(string_, "IS_ARM", False)
    expected_codes, expected_uniques = pd.arrays.StringArray(
        values.copy(), dtype=dtype
    ).factorize()

    def forbidden(*args, **kwargs):
        pytest.fail("ARM np.nan StringArray used generic factorize")

    monkeypatch.setattr(ExtensionArray, "factorize", forbidden)
    monkeypatch.setattr(string_, "IS_ARM", True)
    result_codes, result_uniques = pd.arrays.StringArray(
        values.copy(), dtype=dtype
    ).factorize()

    tm.assert_numpy_array_equal(result_codes, expected_codes)
    tm.assert_extension_array_equal(result_uniques, expected_uniques)


@pytest.mark.parametrize("source_na,target_na", [(pd.NA, np.nan), (np.nan, pd.NA)])
def test_string_factorize_arm_changed_na_value_matches_base(
    monkeypatch, source_na, target_na
):
    source_dtype = pd.StringDtype(storage="python", na_value=source_na)
    target_dtype = pd.StringDtype(storage="python", na_value=target_na)
    source = pd.arrays.StringArray(
        np.array(["a", source_na, "b", "a"], dtype=object), dtype=source_dtype
    )

    monkeypatch.setattr(string_, "IS_ARM", False)
    expected_codes, expected_uniques = pd.arrays.StringArray(
        source, dtype=target_dtype
    ).factorize()
    monkeypatch.setattr(string_, "IS_ARM", True)
    result_codes, result_uniques = pd.arrays.StringArray(
        source, dtype=target_dtype
    ).factorize()

    tm.assert_numpy_array_equal(result_codes, expected_codes)
    tm.assert_extension_array_equal(result_uniques, expected_uniques)
    assert source[1] is source_na


def test_string_factorize_arm_lone_surrogate_matches_base(monkeypatch):
    values = ["a", "\ud800", "b", "\ud800"]
    dtype = pd.StringDtype(storage="python")

    monkeypatch.setattr(string_, "IS_ARM", False)
    expected_codes, expected_uniques = pd.array(values, dtype=dtype).factorize()
    monkeypatch.setattr(string_, "IS_ARM", True)
    result_codes, result_uniques = pd.array(values, dtype=dtype).factorize()

    tm.assert_numpy_array_equal(result_codes, expected_codes)
    tm.assert_extension_array_equal(result_uniques, expected_uniques)


def test_string_hashtable_specialization_requires_ignore_na():
    values = np.array(["a", None, "a"], dtype=object)

    with pytest.raises(
        ValueError, match="string_array=True requires ignore_na=True"
    ):
        libhashtable.StringHashTable().factorize(
            values, ignore_na=False, string_array=True
        )
