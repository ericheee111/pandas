import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas._libs import hashtable as libhashtable
from pandas.core import (
    apply,
    series,
)
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


def test_apply_rowwise_non_arm_avoids_label_cache(monkeypatch):
    monkeypatch.setattr(apply, "IS_ARM", False)
    monkeypatch.setattr(series, "IS_ARM", False)
    monkeypatch.setattr(
        pd.Series,
        "_get_row_apply_cached_value",
        lambda *args: pytest.fail("non-ARM row-wise apply used the label cache"),
    )
    df = pd.DataFrame(
        {"amount": [10.0, 20.0], "event_type": ["view", "purchase"], "rating": [1, 2]}
    )

    result = df.apply(lambda row: row["amount"] + row["rating"], axis=1)

    expected = pd.Series([11.0, 22.0])
    tm.assert_series_equal(result, expected)


def test_apply_rowwise_arm_matches_legacy(monkeypatch):
    df = pd.DataFrame(
        {"amount": [10.0, 20.0], "event_type": ["view", "purchase"], "rating": [1, 2]},
        index=["first", "second"],
    )

    def score(row):
        result = row.copy(deep=False)
        result["score"] = row["amount"] + row["rating"]
        result["row_name"] = row.name
        return result

    monkeypatch.setattr(apply, "IS_ARM", False)
    monkeypatch.setattr(series, "IS_ARM", False)
    expected = df.apply(score, axis=1)
    calls = []
    original = pd.Series._get_row_apply_cached_value

    def tracked(self, *args, **kwargs):
        calls.append(None)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "_get_row_apply_cached_value", tracked)
    monkeypatch.setattr(apply, "IS_ARM", True)
    monkeypatch.setattr(series, "IS_ARM", True)
    result = df.apply(score, axis=1)

    assert calls
    tm.assert_frame_equal(result, expected)


def test_apply_rowwise_arm_preserves_exception_order(monkeypatch):
    df = pd.DataFrame({"value": [1, 2, 3]}, index=["a", "b", "c"])

    def run(is_arm):
        seen = []

        def func(row):
            seen.append(row.name)
            if row.name == "b":
                raise RuntimeError("row failure")
            return row["value"]

        monkeypatch.setattr(apply, "IS_ARM", is_arm)
        monkeypatch.setattr(series, "IS_ARM", is_arm)
        with pytest.raises(RuntimeError, match="row failure"):
            df.apply(func, axis=1)
        return seen

    assert run(False) == ["a", "b"]
    assert run(True) == ["a", "b"]
