"""
Direct libgroupby tests for ``group_prod``.

These exercise the AArch64 native-float fastpaths (min_count <= 0) as well as
the generic fallback path. On AArch64, contiguous native-float arrays with
skipna and min_count <= 0 hit a scalar or multi-column NEON fastpath; masked,
non-float, skipna=False, min_count>0 cases hit the fallback.  Every case is
cross-checked against an independent NumPy reference and, where both paths are
reachable, against the fallback path itself.
"""

import numpy as np
import pytest

from pandas._libs.groupby import group_prod

import pandas._testing as tm


def _reference_prod(values, labels, ngroups, skipna=True, min_count=0):
    """Independent NumPy reference for group_prod semantics."""
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    K = values.shape[1]
    out = np.empty((ngroups, K), dtype=values.dtype)
    counts = np.zeros(ngroups, dtype=np.int64)
    nobs = np.zeros((ngroups, K), dtype=np.int64)
    # Initialize to multiplicative identity.
    out[:] = 1
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
            out[lab, j] *= v
    for g in range(ngroups):
        for j in range(K):
            if nobs[g, j] < min_count:
                out[g, j] = np.nan
    return out, counts


def _run_group_prod(values, labels, ngroups, skipna=True, min_count=0, mask=None):
    """Call libgroupby.group_prod with fresh output/counts buffers."""
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    dtype = values.dtype
    out = np.empty((ngroups, values.shape[1]), dtype=dtype)
    counts = np.zeros(ngroups, dtype=np.int64)
    result_mask = None
    if mask is not None:
        result_mask = np.zeros((ngroups, values.shape[1]), dtype=np.uint8)
    group_prod(
        out,
        counts,
        values,
        labels,
        mask,
        result_mask=result_mask,
        min_count=min_count,
        skipna=skipna,
    )
    return out, counts, result_mask


# ---------------------------------------------------------------------------
# Basic correctness on the fastpath-eligible shape (1-D native float, skipna,
# min_count <= 0)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_prod_basic(dtype):
    rng = np.random.default_rng(0)
    values = (rng.standard_normal(50) + 1.5).astype(dtype)  # avoid zero products
    labels = np.tile(np.arange(5, dtype=np.intp), 10)
    out, counts, _ = _run_group_prod(values, labels, 5, min_count=0)
    expected_out, expected_counts = _reference_prod(values, labels, 5, min_count=0)
    tm.assert_almost_equal(out, expected_out, rtol=1e-5)
    tm.assert_numpy_array_equal(counts, expected_counts)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_prod_random_labels(dtype):
    rng = np.random.default_rng(7)
    n, ngroups = 200, 13
    values = (rng.standard_normal(n) + 2.0).astype(dtype)
    labels = rng.integers(0, ngroups, size=n).astype(np.intp)
    out, counts, _ = _run_group_prod(values, labels, ngroups, min_count=0)
    expected_out, expected_counts = _reference_prod(
        values, labels, ngroups, min_count=0
    )
    tm.assert_almost_equal(out, expected_out, rtol=1e-5)
    tm.assert_numpy_array_equal(counts, expected_counts)


def test_prod_negative_labels():
    rng = np.random.default_rng(3)
    n, ngroups = 60, 5
    values = (rng.standard_normal(n) + 2.0).astype(np.float64)
    labels = rng.integers(-1, ngroups, size=n).astype(np.intp)
    out, counts, _ = _run_group_prod(values, labels, ngroups, min_count=0)
    expected_out, expected_counts = _reference_prod(
        values, labels, ngroups, min_count=0
    )
    tm.assert_almost_equal(out, expected_out, rtol=1e-6)
    tm.assert_numpy_array_equal(counts, expected_counts)


def test_prod_counts_contract_with_nan():
    # counts must be the full group size including NaN rows.
    values = np.array([np.nan, 2.0, np.nan, 3.0, 4.0, np.nan], dtype=np.float64)
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.intp)
    out, counts, _ = _run_group_prod(values, labels, 2, min_count=0)
    assert counts.tolist() == [3, 3]
    # group 0: product of {2.0} = 2.0 (NaN skipped)
    assert out[0, 0] == 2.0
    # group 1: product of {3.0, 4.0} = 12.0
    assert out[1, 0] == 12.0


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


def test_prod_no_nan():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.intp)
    out, counts, _ = _run_group_prod(values, labels, 2, min_count=0)
    expected_out, expected_counts = _reference_prod(values, labels, 2, min_count=0)
    tm.assert_almost_equal(out, expected_out, rtol=1e-6)
    tm.assert_numpy_array_equal(counts, expected_counts)


def test_prod_sparse_nan():
    values = np.array([2.0, np.nan, 3.0, np.nan, np.nan, 5.0], dtype=np.float64)
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.intp)
    out, counts, _ = _run_group_prod(values, labels, 2, min_count=0)
    expected_out, expected_counts = _reference_prod(values, labels, 2, min_count=0)
    tm.assert_almost_equal(out, expected_out, rtol=1e-6)
    tm.assert_numpy_array_equal(counts, expected_counts)


def test_prod_all_nan_group():
    # A group whose values are all NaN: product of empty = 1 (init), and with
    # min_count <= 0 the group is not masked -> result is 1.0.
    values = np.array([np.nan, np.nan, np.nan, 5.0, 6.0], dtype=np.float64)
    labels = np.array([0, 0, 0, 1, 1], dtype=np.intp)
    out, counts, _ = _run_group_prod(values, labels, 2, min_count=0)
    assert counts.tolist() == [3, 2]
    # group 0 all NaN, min_count=0 -> product identity 1.0
    assert out[0, 0] == 1.0
    # group 1 -> 5.0 * 6.0 = 30.0
    assert out[1, 0] == 30.0


# ---------------------------------------------------------------------------
# Infinity and signed zero
# ---------------------------------------------------------------------------


def test_prod_infinity():
    values = np.array([np.inf, 2.0, -np.inf, 3.0, 0.0, 4.0], dtype=np.float64)
    labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.intp)
    out, counts, _ = _run_group_prod(values, labels, 3, min_count=0)
    expected_out, expected_counts = _reference_prod(values, labels, 3, min_count=0)
    tm.assert_almost_equal(out, expected_out, rtol=1e-6)
    tm.assert_numpy_array_equal(counts, expected_counts)
    # group 0: inf * 2.0 = inf
    assert np.isinf(out[0, 0]) and out[0, 0] > 0
    # group 1: -inf * 3.0 = -inf
    assert np.isinf(out[1, 0]) and out[1, 0] < 0
    # group 2: 0.0 * 4.0 = 0.0
    assert out[2, 0] == 0.0


def test_prod_signed_zero():
    # -0.0 * positive = -0.0
    values = np.array([-0.0, 5.0], dtype=np.float64)
    labels = np.array([0, 0], dtype=np.intp)
    out, counts, _ = _run_group_prod(values, labels, 1, min_count=0)
    assert out[0, 0] == 0.0
    assert np.signbit(out[0, 0])  # -0.0 preserved


# ---------------------------------------------------------------------------
# Empty / edge inputs
# ---------------------------------------------------------------------------


def test_prod_empty_input():
    values = np.array([], dtype=np.float64)
    labels = np.array([], dtype=np.intp)
    out, counts, _ = _run_group_prod(values, labels, 3, min_count=0)
    assert counts.tolist() == [0, 0, 0]
    # empty groups -> product identity 1.0 (min_count=0)
    assert np.all(out == 1.0)


def test_prod_empty_groups():
    # Groups present in counts but with no rows in labels.
    values = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    labels = np.array([0, 0, 0], dtype=np.intp)
    out, counts, _ = _run_group_prod(values, labels, 4, min_count=0)
    assert counts.tolist() == [3, 0, 0, 0]
    assert out[0, 0] == 24.0  # 2*3*4
    assert out[1, 0] == 1.0  # empty -> identity
    assert out[2, 0] == 1.0
    assert out[3, 0] == 1.0


# ---------------------------------------------------------------------------
# min_count behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mc", [-1, 0, 1, 2])
def test_prod_min_count(mc):
    rng = np.random.default_rng(11)
    n, ngroups = 40, 4
    values = (rng.standard_normal(n) + 2.0).astype(np.float64)
    values[::5] = np.nan
    labels = rng.integers(0, ngroups, size=n).astype(np.intp)
    out, counts, _ = _run_group_prod(values, labels, ngroups, min_count=mc)
    expected_out, expected_counts = _reference_prod(
        values, labels, ngroups, min_count=mc
    )
    tm.assert_almost_equal(out, expected_out, rtol=1e-6)
    tm.assert_numpy_array_equal(counts, expected_counts)


def test_prod_min_count_2_blocks_sparse_group():
    # Group with only one valid value and min_count=2 must be NaN.
    values = np.array([2.0, np.nan, np.nan, 3.0], dtype=np.float64)
    labels = np.array([0, 0, 1, 1], dtype=np.intp)
    out, counts, _ = _run_group_prod(values, labels, 2, min_count=2)
    assert np.isnan(out[0, 0])
    assert np.isnan(out[1, 0])
    assert counts.tolist() == [2, 2]


def test_prod_min_count_neg1_treated_as_le0():
    # min_count=-1 should behave like min_count=0 for the fastpath (le0).
    values = np.array([np.nan, np.nan, 2.0, 3.0], dtype=np.float64)
    labels = np.array([0, 0, 1, 1], dtype=np.intp)
    out, counts, _ = _run_group_prod(values, labels, 2, min_count=-1)
    expected_out, expected_counts = _reference_prod(values, labels, 2, min_count=-1)
    tm.assert_almost_equal(out, expected_out, rtol=1e-6)
    tm.assert_numpy_array_equal(counts, expected_counts)
    # group 0 all NaN, min_count=-1 -> product identity 1.0
    assert out[0, 0] == 1.0


# ---------------------------------------------------------------------------
# skipna=False forces fallback (fastpath requires skipna=True)
# ---------------------------------------------------------------------------


def test_prod_skipna_false():
    # skipna=False: NaN propagates to the product for that group.
    values = np.array([2.0, np.nan, 3.0, 4.0, np.nan, 5.0], dtype=np.float64)
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.intp)
    out, counts, _ = _run_group_prod(values, labels, 2, skipna=False, min_count=0)
    # group 0: 2.0, NaN -> once NA, stays NA
    assert np.isnan(out[0, 0])
    assert np.isnan(out[1, 0])
    assert counts.tolist() == [3, 3]


# ---------------------------------------------------------------------------
# Multi-column native floats use NEON on AArch64
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("ncols", [2, 3, 4, 7, 10])
@pytest.mark.parametrize("order", ["C", "F"])
def test_prod_multi_column(dtype, ncols, order):
    rng = np.random.default_rng(5)
    values = np.array(rng.standard_normal((30, ncols)) + 2.0, dtype=dtype, order=order)
    labels = np.tile(np.arange(6, dtype=np.intp), 5)
    out, counts, _ = _run_group_prod(values, labels, 6, min_count=0)
    expected_out, expected_counts = _reference_prod(values, labels, 6, min_count=0)
    tm.assert_almost_equal(out, expected_out, rtol=1e-4)
    tm.assert_numpy_array_equal(counts, expected_counts)


def test_prod_multi_column_with_nan():
    values = np.array(
        [[2.0, np.nan], [3.0, 3.0], [np.nan, 4.0], [5.0, 5.0]],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.intp)
    out, counts, _ = _run_group_prod(values, labels, 2, min_count=0)
    expected_out, expected_counts = _reference_prod(values, labels, 2, min_count=0)
    tm.assert_almost_equal(out, expected_out, rtol=1e-6)
    tm.assert_numpy_array_equal(counts, expected_counts)


def test_prod_multi_column_noncontiguous_fallback():
    rng = np.random.default_rng(15)
    base = (rng.standard_normal((40, 12)) + 2.0).astype(np.float64)
    values = base[:, ::2]
    assert not values.flags.c_contiguous
    labels = np.tile(np.arange(8, dtype=np.intp), 5)
    out, counts, _ = _run_group_prod(values, labels, 8, min_count=0)
    expected_out, expected_counts = _reference_prod(values, labels, 8)
    tm.assert_almost_equal(out, expected_out, rtol=1e-5)
    tm.assert_numpy_array_equal(counts, expected_counts)


def test_prod_multi_column_signed_zero_and_infinity():
    values = np.array([[-0.0, np.inf, np.nan], [2.0, 3.0, 4.0]], dtype=np.float64)
    labels = np.array([0, 0], dtype=np.intp)
    out, counts, _ = _run_group_prod(values, labels, 1, min_count=0)
    assert np.signbit(out[0, 0])
    assert out[0, 1] == np.inf
    assert out[0, 2] == 4.0
    assert counts.tolist() == [2]


# ---------------------------------------------------------------------------
# Mask path forces fallback (fastpath requires not uses_mask)
# ---------------------------------------------------------------------------


def test_prod_with_mask():
    values = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    mask = np.array([[False], [True], [False], [False]], dtype=np.bool_)
    labels = np.array([0, 0, 1, 1], dtype=np.intp)
    out, counts, result_mask = _run_group_prod(
        values, labels, 2, mask=mask, skipna=True, min_count=0
    )
    # group 0: 2.0 (valid), 3.0 (masked) -> 2.0
    assert result_mask[0, 0] == 0
    assert out[0, 0] == 2.0
    # group 1: 4.0 * 5.0 = 20.0
    assert out[1, 0] == 20.0
    assert counts.tolist() == [2, 2]


def test_prod_with_mask_all_masked_group():
    values = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    mask = np.array([[True], [True], [False], [False]], dtype=np.bool_)
    labels = np.array([0, 0, 1, 1], dtype=np.intp)
    out, counts, result_mask = _run_group_prod(
        values, labels, 2, mask=mask, skipna=True, min_count=0
    )
    # group 0 fully masked, min_count=0: nobs=0 >= 0 -> not masked, product
    # identity 1.0.
    assert result_mask[0, 0] == 0
    assert out[0, 0] == 1.0
    # group 1 -> 4.0 * 5.0 = 20.0
    assert out[1, 0] == 20.0
    assert counts.tolist() == [2, 2]


# ---------------------------------------------------------------------------
# int64 path forces fallback (fastpath requires float32/float64)
# ---------------------------------------------------------------------------


def test_prod_int64_fallback():
    values = np.array([2, 3, 4, 5], dtype=np.int64).reshape(-1, 1)
    labels = np.array([0, 0, 1, 1], dtype=np.intp)
    out, counts, _ = _run_group_prod(values, labels, 2, min_count=0)
    assert out[0, 0] == 6  # 2*3
    assert out[1, 0] == 20  # 4*5
    assert counts.tolist() == [2, 2]


# ---------------------------------------------------------------------------
# Fastpath vs fallback public-result equality on ARM
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ncols", [1, 2, 3, 8])
def test_prod_fastpath_equals_fallback(ncols):
    """Compare eligible native-float fastpaths with min_count fallback."""
    rng = np.random.default_rng(13)
    n, ngroups = 500, 17
    values = (rng.standard_normal((n, ncols)) + 2.0).astype(np.float64)
    values[::7] = np.nan
    labels = rng.integers(0, ngroups, size=n).astype(np.intp)

    fast, fast_counts, _ = _run_group_prod(values, labels, ngroups, min_count=0)

    # Positive min_count forces the generic loop while preserving results:
    # every group/column has substantially more than one valid observation.
    fallback, fallback_counts, _ = _run_group_prod(values, labels, ngroups, min_count=1)

    tm.assert_numpy_array_equal(fast_counts, fallback_counts)
    tm.assert_almost_equal(fast, fallback, rtol=1e-6)
