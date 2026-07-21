import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
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

    pd.array(["a", "b", "a"], dtype="str").factorize()

    assert calls == 1
