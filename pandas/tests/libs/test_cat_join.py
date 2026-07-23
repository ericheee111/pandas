import numpy as np
import pytest

from pandas._libs import lib


class TestCatJoin:
    def test_all_ascii(self):
        arr = np.array(["a", "b", "c"], dtype=object)
        result = lib.cat_join(arr, ",")
        assert result == "a,b,c"

    def test_all_ascii_no_sep(self):
        arr = np.array(["a", "b", "c"], dtype=object)
        result = lib.cat_join(arr, "")
        assert result == "abc"

    def test_non_ascii_fallback(self):
        arr = np.array(["é", "ü"], dtype=object)
        result = lib.cat_join(arr, "-")
        assert result == "é-ü"

    def test_mixed_ascii_non_ascii(self):
        arr = np.array(["hello", "世界"], dtype=object)
        result = lib.cat_join(arr, " ")
        assert result == "hello 世界"

    def test_empty_array(self):
        arr = np.array([], dtype=object)
        result = lib.cat_join(arr, ",")
        assert result == ""

    def test_single_element(self):
        arr = np.array(["hello"], dtype=object)
        result = lib.cat_join(arr, ",")
        assert result == "hello"

    def test_single_element_no_sep(self):
        arr = np.array(["hello"], dtype=object)
        result = lib.cat_join(arr, "")
        assert result == "hello"

    def test_non_string_raises(self):
        arr = np.array(["a", 1], dtype=object)
        with pytest.raises(TypeError, match="expected str instance"):
            lib.cat_join(arr, ",")

    def test_all_non_string_raises(self):
        arr = np.array([1, 2, 3], dtype=object)
        with pytest.raises(TypeError, match="expected str instance"):
            lib.cat_join(arr, ",")

    def test_str_subclass(self):
        class MyStr(str):
            pass

        arr = np.array([MyStr("a"), MyStr("b")], dtype=object)
        result = lib.cat_join(arr, "-")
        assert result == "a-b"

    def test_empty_sep_multi_element(self):
        arr = np.array(["x", "y", "z"], dtype=object)
        result = lib.cat_join(arr, "")
        assert result == "xyz"

    def test_none_in_array_raises(self):
        arr = np.array(["a", None, "b"], dtype=object)
        with pytest.raises(TypeError, match="expected str instance"):
            lib.cat_join(arr, ",")

    def test_bool_in_array_raises(self):
        arr = np.array(["a", True, "b"], dtype=object)
        with pytest.raises(TypeError, match="expected str instance"):
            lib.cat_join(arr, "-")


class TestCatJoinMulti:
    def test_all_ascii(self):
        cols = [np.array(["a", "b"]), np.array(["x", "y"])]
        result = lib.cat_join_multi(cols, ",")
        expected = np.array(["a,x", "b,y"], dtype=object)
        np.testing.assert_array_equal(result, expected)

    def test_all_ascii_no_sep(self):
        cols = [np.array(["a", "b"]), np.array(["x", "y"])]
        result = lib.cat_join_multi(cols, "")
        expected = np.array(["ax", "by"], dtype=object)
        np.testing.assert_array_equal(result, expected)

    def test_non_ascii_fallback(self):
        cols = [np.array(["é", "ü"]), np.array(["o", "a"])]
        result = lib.cat_join_multi(cols, "-")
        expected = np.array(["é-o", "ü-a"], dtype=object)
        np.testing.assert_array_equal(result, expected)

    def test_three_columns(self):
        cols = [
            np.array(["a", "b"]),
            np.array(["x", "y"]),
            np.array(["1", "2"]),
        ]
        result = lib.cat_join_multi(cols, "-")
        expected = np.array(["a-x-1", "b-y-2"], dtype=object)
        np.testing.assert_array_equal(result, expected)

    def test_empty_columns(self):
        cols = [np.array([], dtype=object)]
        result = lib.cat_join_multi(cols, ",")
        assert len(result) == 0

    def test_non_string_raises(self):
        cols = [np.array(["a", 1], dtype=object)]
        with pytest.raises(TypeError, match="expected str instance"):
            lib.cat_join_multi(cols, ",")

    def test_all_non_string_raises(self):
        cols = [np.array([1, 2], dtype=object)]
        with pytest.raises(TypeError, match="expected str instance"):
            lib.cat_join_multi(cols, ",")

    def test_single_column(self):
        cols = [np.array(["a", "b", "c"])]
        result = lib.cat_join_multi(cols, ",")
        expected = np.array(["a", "b", "c"], dtype=object)
        np.testing.assert_array_equal(result, expected)

    def test_str_subclass(self):
        class MyStr(str):
            pass

        cols = [np.array([MyStr("a"), MyStr("b")])]
        result = lib.cat_join_multi(cols, ",")
        expected = np.array(["a", "b"], dtype=object)
        np.testing.assert_array_equal(result, expected)
