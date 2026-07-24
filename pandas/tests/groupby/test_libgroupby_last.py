"""
Direct libgroupby tests for ``group_last``.

These exercise the AArch64 K==1 reverse-scan float fastpath as well as the
generic fallback path.  On AArch64 the single-column native-float skipna cases
hit the fastpath; multi-column, masked, non-float, skipna=False, min_count>1
and datetimelike cases hit the fallback.  Every case is cross-checked against an
independent NumPy reference and, where both paths are reachable, against the
fallback path itself.
"""

import numpy as np
import pytest

from pandas._libs import groupby as libgroupby
from pandas._libs.groupby import group_last

import pandas._testing as tm


def _reference_last(values, labels, ngroups, skipna=True, min_count=1):
    """Independent NumPy reference for group_last semantics."""
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    K = values.shape[1]
    out = np.empty((ngroups, K), dtype=values.dtype)
    counts = np.zeros(ngroups, dtype=np.int64)
    nobs = np.zeros((ngroups, K), dtype=np.int64)
    for i, lab in enumerate(labels):
        if lab < 0:
            continue
        counts[lab] += 1
        for j in range(K):
            v = values[i, j]
            is_na = v != v  # NaN test for floats
            if skipna and is_na:
                continue
            nobs[lab, j] += 1
            out[lab, j] = v
    for g in range(ngroups):
        for j in range(K):
            if nobs[g, j] < min_count:
                out[g, j] = np.nan
    return out, counts


def _run_group_last(values, labels, ngroups, skipna=True, min_count=-1,
                    mask=None, is_datetimelike=False):
    """Call libgroupby.group_last with fresh output/counts buffers."""
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    dtype = values.dtype
    out = np.empty((ngroups, values.shape[1]), dtype=dtype)
    counts = np.zeros(ngroups, dtype=np.int64)
    result_mask = None
    if mask is not None:
        result_mask = np.zeros((ngroups, values.shape[1]), dtype=np.uint8)
    group_last(
        out, counts, values, labels, mask,
        result_mask=result_mask,
        min_count=min_count,
        is_datetimelike=is_datetimelike,
        skipna=skipna,
    )
    return out, counts, result_mask


# ---------------------------------------------------------------------------
# Basic correctness on the fastpath-eligible shape (1-D native float, skipna)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_last_basic_forward_equivalence(dtype):
    # Each group's last value is the final row in forward order.
    rng = np.random.default_rng(0)
    values = rng.standard_normal(50).astype(dtype)
    labels = np.tile(np.arange(5, dtype=np.intp), 10)
    out, counts, _ = _run_group_last(values, labels, 5)
    expected_out, expected_counts = _reference_last(values, labels, 5)
    tm.assert_almost_equal(out, expected_out, rtol=1e-6)
    tm.assert_numpy_array_equal(counts, expected_counts)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_last_random_labels(dtype):
    rng = np.random.default_rng(7)
    n, ngroups = 200, 13
    values = rng.standard_normal(n).astype(dtype)
    labels = rng.integers(0, ngroups, size=n).astype(np.intp)
    out, counts, _ = _run_group_last(values, labels, ngroups)
    expected_out, expected_counts = _reference_last(values, labels, ngroups)
    tm.assert_almost_equal(out, expected_out, rtol=1e-6)
    tm.assert_numpy_array_equal(counts, expected_counts)


def test_last_negative_labels():
    # Negative labels (NaN-group sentinels) must be skipped.
    rng = np.random.default_rng(3)
    n, ngroups = 60, 5
    values = rng.standard_normal(n).astype(np.float64)
    labels = rng.integers(-1, ngroups, size=n).astype(np.intp)
    out, counts, _ = _run_group_last(values, labels, ngroups)
    expected_out, expected_counts = _reference_last(values, labels, ngroups)
    tm.assert_almost_equal(out, expected_out, rtol=1e-6)
    tm.assert_numpy_array_equal(counts, expected_counts)


def test_last_counts_contract_with_nan():
    # counts must be the full group size including NaN rows.
    values = np.array([np.nan, 1.0, np.nan, 2.0, 3.0, np.nan], dtype=np.float64)
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.intp)
    out, counts, _ = _run_group_last(values, labels, 2)
    assert counts.tolist() == [3, 3]
    # last valid value per group
    assert out[0, 0] == 1.0
    assert out[1, 0] == 3.0


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

def test_last_no_nan():
    values = np.arange(1, 11, dtype=np.float64)
    labels = np.tile(np.arange(5, dtype=np.intp), 2)
    out, counts, _ = _run_group_last(values, labels, 5)
    expected_out, expected_counts = _reference_last(values, labels, 5)
    tm.assert_almost_equal(out, expected_out, rtol=1e-6)
    tm.assert_numpy_array_equal(counts, expected_counts)


def test_last_sparse_nan():
    values = np.array([1.0, np.nan, 2.0, np.nan, np.nan, 3.0], dtype=np.float64)
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.intp)
    out, counts, _ = _run_group_last(values, labels, 2)
    expected_out, expected_counts = _reference_last(values, labels, 2)
    tm.assert_almost_equal(out, expected_out, rtol=1e-6)
    tm.assert_numpy_array_equal(counts, expected_counts)


def test_last_all_nan_group():
    # A group whose values are all NaN must yield NaN with nobs==0.
    values = np.array([np.nan, np.nan, np.nan, 5.0, 6.0], dtype=np.float64)
    labels = np.array([0, 0, 0, 1, 1], dtype=np.intp)
    out, counts, _ = _run_group_last(values, labels, 2)
    assert counts.tolist() == [3, 2]
    assert np.isnan(out[0, 0])
    assert out[1, 0] == 6.0


def test_last_infinity_and_signed_zero():
    values = np.array(
        [np.inf, -np.inf, 0.0, -0.0, np.nan, np.inf], dtype=np.float64
    )
    labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.intp)
    out, counts, _ = _run_group_last(values, labels, 3)
    expected_out, expected_counts = _reference_last(values, labels, 3)
    tm.assert_almost_equal(out, expected_out, rtol=1e-6)
    tm.assert_numpy_array_equal(counts, expected_counts)
    # group 0: {inf, -inf} -> last is -inf
    assert np.isinf(out[0, 0]) and out[0, 0] < 0
    # group 1: {0.0, -0.0} -> last is -0.0
    assert out[1, 0] == 0.0 and np.signbit(out[1, 0])
    # group 2: {nan, inf} -> last valid is inf
    assert np.isinf(out[2, 0]) and out[2, 0] > 0


def test_last_all_nan_then_nan_group():
    # A group whose values are all NaN yields NaN.
    values = np.array([np.nan, np.nan, 1.0], dtype=np.float64)
    labels = np.array([0, 0, 1], dtype=np.intp)
    out, counts, _ = _run_group_last(values, labels, 2)
    assert np.isnan(out[0, 0])
    assert out[1, 0] == 1.0
    assert counts.tolist() == [2, 1]


# ---------------------------------------------------------------------------
# Empty / edge inputs
# ---------------------------------------------------------------------------

def test_last_empty_input():
    values = np.array([], dtype=np.float64)
    labels = np.array([], dtype=np.intp)
    out, counts, _ = _run_group_last(values, labels, 3)
    assert counts.tolist() == [0, 0, 0]
    assert np.all(np.isnan(out))


def test_last_empty_groups():
    # Groups present in counts but with no rows in labels.
    values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    labels = np.array([0, 0, 0], dtype=np.intp)
    out, counts, _ = _run_group_last(values, labels, 4)
    assert counts.tolist() == [3, 0, 0, 0]
    assert out[0, 0] == 3.0
    assert np.isnan(out[1, 0])
    assert np.isnan(out[2, 0])
    assert np.isnan(out[3, 0])


# ---------------------------------------------------------------------------
# min_count behavior
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mc", [-1, 0, 1, 2])
def test_last_min_count(mc):
    rng = np.random.default_rng(11)
    n, ngroups = 40, 4
    values = rng.standard_normal(n).astype(np.float64)
    # Sprinkle NaN so some groups may fall below min_count.
    values[::5] = np.nan
    labels = rng.integers(0, ngroups, size=n).astype(np.intp)
    # libgroupby clamps min_count to >= 1 internally.
    expected_mc = max(mc, 1)
    out, counts, _ = _run_group_last(values, labels, ngroups, min_count=mc)
    expected_out, expected_counts = _reference_last(
        values, labels, ngroups, min_count=expected_mc
    )
    tm.assert_almost_equal(out, expected_out, rtol=1e-6)
    tm.assert_numpy_array_equal(counts, expected_counts)


def test_last_min_count_2_blocks_sparse_group():
    # Group with only one valid value and min_count=2 must be NaN.
    values = np.array([1.0, np.nan, np.nan, 2.0], dtype=np.float64)
    labels = np.array([0, 0, 1, 1], dtype=np.intp)
    out, counts, _ = _run_group_last(values, labels, 2, min_count=2)
    # group 0: one valid (1.0) but min_count=2 -> NaN
    assert np.isnan(out[0, 0])
    # group 1: one valid (2.0) but min_count=2 -> NaN
    assert np.isnan(out[1, 0])
    assert counts.tolist() == [2, 2]


# ---------------------------------------------------------------------------
# skipna=False forces fallback (fastpath requires skipna=True)
# ---------------------------------------------------------------------------

def test_last_skipna_false():
    # skipna=False: last value is taken verbatim, including NaN.
    values = np.array([1.0, np.nan, 2.0, 3.0, np.nan, 4.0], dtype=np.float64)
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.intp)
    out, counts, _ = _run_group_last(values, labels, 2, skipna=False)
    # group 0: rows 1.0, NaN, 2.0 -> last is 2.0
    assert out[0, 0] == 2.0
    # group 1: rows 3.0, NaN, 4.0 -> last is 4.0
    assert out[1, 0] == 4.0
    assert counts.tolist() == [3, 3]


def test_last_skipna_false_trailing_nan():
    # skipna=False with a trailing NaN: last is NaN.
    values = np.array([1.0, 2.0, np.nan, 3.0, 4.0, np.nan], dtype=np.float64)
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.intp)
    out, counts, _ = _run_group_last(values, labels, 2, skipna=False)
    assert np.isnan(out[0, 0])
    assert np.isnan(out[1, 0])
    assert counts.tolist() == [3, 3]


# ---------------------------------------------------------------------------
# Multi-column forces fallback (fastpath requires K==1)
# ---------------------------------------------------------------------------

def test_last_multi_column():
    rng = np.random.default_rng(5)
    values = rng.standard_normal((30, 2)).astype(np.float64)
    labels = np.tile(np.arange(6, dtype=np.intp), 5)
    out, counts, _ = _run_group_last(values, labels, 6)
    expected_out, expected_counts = _reference_last(values, labels, 6)
    tm.assert_almost_equal(out, expected_out, rtol=1e-6)
    tm.assert_numpy_array_equal(counts, expected_counts)


def test_last_multi_column_with_nan():
    values = np.array(
        [[1.0, np.nan], [2.0, 2.0], [np.nan, 3.0], [4.0, 4.0]],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.intp)
    out, counts, _ = _run_group_last(values, labels, 2)
    expected_out, expected_counts = _reference_last(values, labels, 2)
    tm.assert_almost_equal(out, expected_out, rtol=1e-6)
    tm.assert_numpy_array_equal(counts, expected_counts)


# ---------------------------------------------------------------------------
# Mask path forces fallback (fastpath requires not uses_mask)
# ---------------------------------------------------------------------------

def test_last_with_mask():
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    mask = np.array([[False], [True], [False], [False]], dtype=np.bool_)
    labels = np.array([0, 0, 1, 1], dtype=np.intp)
    out, counts, result_mask = _run_group_last(
        values, labels, 2, mask=mask, skipna=True
    )
    # group 0: rows 1.0 (valid), 2.0 (masked) -> last valid is 1.0
    assert out[0, 0] == 1.0
    assert result_mask[0, 0] == 0
    # group 1: rows 3.0 (valid), 4.0 (valid) -> last valid is 4.0
    assert out[1, 0] == 4.0
    assert result_mask[1, 0] == 0
    assert counts.tolist() == [2, 2]


def test_last_with_mask_all_masked_group():
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    mask = np.array([[True], [True], [False], [False]], dtype=np.bool_)
    labels = np.array([0, 0, 1, 1], dtype=np.intp)
    out, counts, result_mask = _run_group_last(
        values, labels, 2, mask=mask, skipna=True
    )
    # group 0 fully masked -> result masked
    assert result_mask[0, 0] == 1
    assert out[0, 0] == 0
    # group 1 -> 4.0
    assert out[1, 0] == 4.0
    assert counts.tolist() == [2, 2]


# ---------------------------------------------------------------------------
# Fastpath vs fallback public-result equality on ARM
# ---------------------------------------------------------------------------

def test_last_fastpath_equals_fallback():
    """On AArch64 the 1-D float skipna path hits the reverse-scan fastpath;
    the same input reshaped to 2 columns forces the generic fallback.  Both
    must produce identical per-group last values for the shared column."""
    rng = np.random.default_rng(13)
    n, ngroups = 500, 17
    col = rng.standard_normal(n).astype(np.float64)
    col[::7] = np.nan
    labels = rng.integers(0, ngroups, size=n).astype(np.intp)

    # Fastpath-eligible: single column.
    out1, counts1, _ = _run_group_last(col, labels, ngroups)

    # Fallback: duplicate column -> K==2 forces generic path.
    two_col = np.column_stack([col, col])
    out2, counts2, _ = _run_group_last(two_col, labels, ngroups)

    tm.assert_numpy_array_equal(counts1, counts2)
    tm.assert_almost_equal(out1[:, 0], out2[:, 0], rtol=1e-6)
    tm.assert_almost_equal(out2[:, 0], out2[:, 1], rtol=1e-6)
