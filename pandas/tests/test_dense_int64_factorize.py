"""Tests for the dense-range int64 factorization fast path.

These tests verify the *public* result of ``pd.factorize`` (and the
``Grouping`` factorization path) for dense int64 inputs, and that all
unsupported cases fall back to the original implementation with identical
results.

They deliberately do **not** mandate that a specific internal backend be
used for ordinary inputs — fallback cases assert correctness via
cross-checks, not by probing internal state.
"""

from __future__ import annotations

import importlib
import threading

import numpy as np
import pytest

import pandas as pd
from pandas._libs import hashtable as ht
import pandas._testing as tm
from pandas.core import algorithms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_fastpaths(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    """Toggle both the Python-level and Cython-level BoostKit gate."""
    fastpaths = importlib.import_module("pandas.core._boostkit_fastpaths")
    monkeypatch.setattr(fastpaths, "USE_BOOSTKIT_FASTPATHS", enabled)
    ht._set_use_boostkit_fastpaths(enabled)


@pytest.fixture(autouse=True)
def _restore_fastpath_state(monkeypatch: pytest.MonkeyPatch):
    """Restore the Cython-level BoostKit gate after each test.

    ``monkeypatch`` auto-reverts the Python-level ``USE_BOOSTKIT_FASTPATHS``
    attribute, but it cannot restore the Cython module global
    ``_use_boostkit_fastpaths``.  Capture the value the test inherited and
    restore it on teardown so a test that disables the gate does not leak that
    state into subsequent tests (the previous ``finally`` block hardcoded
    ``True``, which would silently re-enable the gate even if the surrounding
    suite had intentionally disabled it).  Mirrors the fixture in
    ``pandas/tests/test_boostkit_fastpaths.py``.
    """
    fastpaths = importlib.import_module("pandas.core._boostkit_fastpaths")
    original = fastpaths.USE_BOOSTKIT_FASTPATHS
    yield
    ht._set_use_boostkit_fastpaths(original)


def _make_dense_int64(n: int, ngroups: int, start: int = 0) -> np.ndarray:
    """A length-``n`` int64 array whose values form a dense range of
    ``ngroups`` distinct values starting at ``start``, shuffled."""
    rng = np.random.RandomState(42)
    keys = rng.randint(start, start + ngroups, size=n).astype(np.int64)
    return np.ascontiguousarray(keys)


def _reference_factorize(values: np.ndarray, sort: bool) -> tuple[np.ndarray, np.ndarray]:
    """Factorize via the non-BoostKit path by disabling fast paths.

    Returns ``(codes, uniques)`` identical to the regular hashtable path.
    """
    # Use the hashtable directly to get a reference result independent of
    # the dense fast path.  For sort=True we apply safe_sort manually.
    codes, uniques = algorithms.factorize_array(
        values, use_na_sentinel=True
    )
    if sort and len(uniques) > 0:
        uniques, codes = algorithms.safe_sort(
            uniques, codes, use_na_sentinel=True, assume_unique=True, verify=False
        )
    return codes, uniques


# ---------------------------------------------------------------------------
# Kernel-level tests (direct cpdef call)
# ---------------------------------------------------------------------------

class TestFactorizeInt64DenseRangeKernel:
    """Direct tests for ``ht.factorize_int64_dense_range``."""

    def test_dense_zero_based(self):
        n = 200_000
        keys = _make_dense_int64(n, ngroups=500, start=0)
        result = ht.factorize_int64_dense_range(keys)
        assert result is not None
        codes, uniques = result
        # Cross-check against the hashtable path.
        ref_codes, ref_uniques = _reference_factorize(keys, sort=True)
        tm.assert_numpy_array_equal(codes, ref_codes)
        tm.assert_numpy_array_equal(uniques, ref_uniques)

    def test_dense_nonzero_start(self):
        n = 200_000
        keys = _make_dense_int64(n, ngroups=500, start=1000)
        result = ht.factorize_int64_dense_range(keys)
        assert result is not None
        codes, uniques = result
        ref_codes, ref_uniques = _reference_factorize(keys, sort=True)
        tm.assert_numpy_array_equal(codes, ref_codes)
        tm.assert_numpy_array_equal(uniques, ref_uniques)

    def test_dense_negative(self):
        n = 200_000
        keys = _make_dense_int64(n, ngroups=500, start=-500)
        result = ht.factorize_int64_dense_range(keys)
        assert result is not None
        codes, uniques = result
        ref_codes, ref_uniques = _reference_factorize(keys, sort=True)
        tm.assert_numpy_array_equal(codes, ref_codes)
        tm.assert_numpy_array_equal(uniques, ref_uniques)
        # uniques must be sorted ascending
        assert (uniques[1:] >= uniques[:-1]).all()

    def test_dense_with_gaps(self):
        """Small span but some intermediate values are missing."""
        n = 200_000
        rng = np.random.RandomState(7)
        # Use only even numbers in [0, 1000) → gaps at odd values.
        keys = (rng.randint(0, 500, size=n) * 2).astype(np.int64)
        keys = np.ascontiguousarray(keys)
        result = ht.factorize_int64_dense_range(keys)
        assert result is not None
        codes, uniques = result
        ref_codes, ref_uniques = _reference_factorize(keys, sort=True)
        tm.assert_numpy_array_equal(codes, ref_codes)
        tm.assert_numpy_array_equal(uniques, ref_uniques)

    def test_shuffled_and_duplicates(self):
        n = 200_000
        keys = _make_dense_int64(n, ngroups=300, start=0)
        result = ht.factorize_int64_dense_range(keys)
        assert result is not None
        codes, uniques = result
        ref_codes, ref_uniques = _reference_factorize(keys, sort=True)
        tm.assert_numpy_array_equal(codes, ref_codes)
        tm.assert_numpy_array_equal(uniques, ref_uniques)

    def test_all_same_value(self):
        n = 200_000
        keys = np.full(n, 42, dtype=np.int64)
        result = ht.factorize_int64_dense_range(keys)
        assert result is not None
        codes, uniques = result
        expected_codes = np.zeros(n, dtype=np.intp)
        expected_uniques = np.array([42], dtype=np.int64)
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    def test_uniques_sorted_ascending(self):
        n = 200_000
        keys = _make_dense_int64(n, ngroups=1000, start=-200)
        result = ht.factorize_int64_dense_range(keys)
        assert result is not None
        _, uniques = result
        assert (uniques[1:] > uniques[:-1]).all()

    def test_codes_dtype_intp(self):
        n = 200_000
        keys = _make_dense_int64(n, ngroups=100, start=0)
        result = ht.factorize_int64_dense_range(keys)
        assert result is not None
        codes, _ = result
        assert codes.dtype == np.dtype(np.intp)

    def test_uniques_dtype_int64(self):
        n = 200_000
        keys = _make_dense_int64(n, ngroups=100, start=0)
        result = ht.factorize_int64_dense_range(keys)
        assert result is not None
        _, uniques = result
        assert uniques.dtype == np.dtype(np.int64)

    def test_input_not_modified(self):
        n = 200_000
        keys = _make_dense_int64(n, ngroups=100, start=0)
        keys_copy = keys.copy()
        result = ht.factorize_int64_dense_range(keys)
        assert result is not None
        tm.assert_numpy_array_equal(keys, keys_copy)

    # --- fallback cases: kernel returns None ---------------------------

    def test_short_input_returns_none(self):
        keys = np.arange(1000, dtype=np.int64)
        assert ht.factorize_int64_dense_range(keys) is None

    def test_large_span_sparse_returns_none(self):
        # Values span almost the full int64 range → span far exceeds cap.
        n = 200_000
        rng = np.random.RandomState(1)
        keys = rng.randint(
            -(2**62), 2**62, size=n
        ).astype(np.int64)
        keys = np.ascontiguousarray(keys)
        assert ht.factorize_int64_dense_range(keys) is None

    def test_int64_extremes_span_overflow_safe(self):
        # vmin=INT64_MIN, vmax=INT64_MAX → span = UINT64_MAX → None (not crash)
        n = 200_000
        keys = np.empty(n, dtype=np.int64)
        keys[: n // 2] = np.iinfo(np.int64).min
        keys[n // 2 :] = np.iinfo(np.int64).max
        result = ht.factorize_int64_dense_range(keys)
        assert result is None  # overflow-safe: returns None, does not crash

    def test_non_contiguous_returns_none(self):
        n = 200_000
        keys = np.arange(n, dtype=np.int64)[::2]
        # The typed memoryview ``int64_t[::1]`` requires C-contiguity; a
        # non-contiguous array will raise ValueError when passed directly
        # to the kernel.  The public ``factorize`` dispatch guards on
        # ``c_contiguous`` before calling, so non-contiguous inputs never
        # reach the kernel in practice.  We accept ValueError or None.
        try:
            result = ht.factorize_int64_dense_range(keys)
        except (TypeError, ValueError):
            return
        assert result is None

    def test_non_native_byteorder_returns_none(self):
        n = 200_000
        keys = np.arange(n, dtype=np.int64)
        swapped = keys.view(keys.dtype.newbyteorder())
        # Non-native byte order: the typed memoryview will reject it.
        try:
            result = ht.factorize_int64_dense_range(swapped)
        except (TypeError, ValueError):
            return
        assert result is None

    def test_uint64_not_applicable(self):
        # The kernel signature is int64-only; uint64 should never reach it
        # via the public path.  Direct call with uint64 should either
        # raise or return None — we accept either.
        n = 200_000
        keys = np.arange(n, dtype=np.uint64)
        try:
            result = ht.factorize_int64_dense_range(keys)
        except (TypeError, ValueError):
            return
        assert result is None


# ---------------------------------------------------------------------------
# Public API integration tests (pd.factorize)
# ---------------------------------------------------------------------------

class TestFactorizeDenseInt64Public:
    """Integration via ``pd.factorize`` and ``algorithms.factorize``."""

    def test_sort_true_matches_reference(self):
        n = 200_000
        keys = _make_dense_int64(n, ngroups=800, start=0)
        codes, uniques = pd.factorize(keys, sort=True)
        ref_codes, ref_uniques = _reference_factorize(keys, sort=True)
        tm.assert_numpy_array_equal(codes, ref_codes)
        tm.assert_numpy_array_equal(uniques, ref_uniques)

    def test_sort_false_unchanged(self):
        """sort=False must not trigger the dense path; result unchanged."""
        n = 200_000
        keys = _make_dense_int64(n, ngroups=800, start=0)
        codes, uniques = pd.factorize(keys, sort=False)
        ref_codes, ref_uniques = _reference_factorize(keys, sort=False)
        tm.assert_numpy_array_equal(codes, ref_codes)
        tm.assert_numpy_array_equal(uniques, ref_uniques)

    @pytest.mark.parametrize("start", [0, 1000, -500])
    def test_various_starts(self, start):
        n = 200_000
        keys = _make_dense_int64(n, ngroups=500, start=start)
        codes, uniques = pd.factorize(keys, sort=True)
        ref_codes, ref_uniques = _reference_factorize(keys, sort=True)
        tm.assert_numpy_array_equal(codes, ref_codes)
        tm.assert_numpy_array_equal(uniques, ref_uniques)

    def test_use_na_sentinel_false_fallback(self):
        """use_na_sentinel=False must not trigger the dense path."""
        n = 200_000
        keys = _make_dense_int64(n, ngroups=500, start=0)
        codes, uniques = pd.factorize(keys, sort=True, use_na_sentinel=False)
        # Reference: same call with fast paths disabled is identical since
        # the dense path is gated on use_na_sentinel=True.  Cross-check
        # against the hashtable path directly.
        ref_codes, ref_uniques = algorithms.factorize_array(
            keys, use_na_sentinel=False
        )
        ref_uniques, ref_codes = algorithms.safe_sort(
            ref_uniques, ref_codes, use_na_sentinel=False,
            assume_unique=True, verify=False,
        )
        tm.assert_numpy_array_equal(codes, ref_codes)
        tm.assert_numpy_array_equal(uniques, ref_uniques)

    def test_fast_path_disabled(self, monkeypatch):
        """With BoostKit fast paths off, result is still correct."""
        n = 200_000
        keys = _make_dense_int64(n, ngroups=500, start=0)
        codes_enabled, uniques_enabled = pd.factorize(keys, sort=True)
        _set_fastpaths(monkeypatch, False)
        codes_disabled, uniques_disabled = pd.factorize(keys, sort=True)
        tm.assert_numpy_array_equal(codes_enabled, codes_disabled)
        tm.assert_numpy_array_equal(uniques_enabled, uniques_disabled)

    def test_small_input_fallback(self):
        """Below the length threshold the regular path is used."""
        keys = np.array([5, 3, 5, 1, 3], dtype=np.int64)
        codes, uniques = pd.factorize(keys, sort=True)
        expected_codes = np.array([2, 1, 2, 0, 1], dtype=np.intp)
        expected_uniques = np.array([1, 3, 5], dtype=np.int64)
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    def test_large_sparse_fallback(self):
        """Sparse large-range int64 falls back and stays correct."""
        n = 200_000
        rng = np.random.RandomState(9)
        keys = rng.randint(
            -(2**40), 2**40, size=n
        ).astype(np.int64)
        keys = np.ascontiguousarray(keys)
        codes, uniques = pd.factorize(keys, sort=True)
        ref_codes, ref_uniques = _reference_factorize(keys, sort=True)
        tm.assert_numpy_array_equal(codes, ref_codes)
        tm.assert_numpy_array_equal(uniques, ref_uniques)

    def test_unaligned_dense_int64_falls_back(self, monkeypatch):
        """A C-contiguous-but-misaligned int64 array must fall back to the
        hashtable path.  The Cython dense kernel dereferences the buffer via
        a typed memoryview under ``nogil``; requiring aligned memory keeps the
        access safe on strict-alignment architectures.  The result must still
        match the reference factorization.
        """
        n = 200_000
        keys = _make_dense_int64(n, ngroups=500, start=0)
        # Build a C-contiguous, native, 1-D int64 array that is NOT aligned:
        # allocate a uint8 buffer one byte too long and view it as int64 at
        # offset 1.  numpy permits this (the view is contiguous but unaligned).
        raw = np.empty(n * 8 + 1, dtype=np.uint8)
        unaligned = np.ndarray(
            n, dtype=np.int64, buffer=raw, offset=1
        )
        unaligned[:] = keys
        assert unaligned.flags.c_contiguous
        assert unaligned.dtype.isnative
        assert not unaligned.flags.aligned

        _set_fastpaths(monkeypatch, True)
        codes, uniques = pd.factorize(unaligned, sort=True)
        ref_codes, ref_uniques = _reference_factorize(unaligned, sort=True)
        tm.assert_numpy_array_equal(codes, ref_codes)
        tm.assert_numpy_array_equal(uniques, ref_uniques)

    def test_uint64_fallback(self):
        """uint64 input must not trigger the int64 path."""
        n = 200_000
        rng = np.random.RandomState(3)
        keys = rng.randint(0, 1000, size=n).astype(np.uint64)
        codes, uniques = pd.factorize(keys, sort=True)
        # Cross-check against the reference (which handles uint64 via
        # the hashtable).  We rebuild the reference with the same dtype.
        ref_codes, ref_uniques = algorithms.factorize_array(
            keys, use_na_sentinel=True
        )
        ref_uniques, ref_codes = algorithms.safe_sort(
            ref_uniques, ref_codes, use_na_sentinel=True,
            assume_unique=True, verify=False,
        )
        tm.assert_numpy_array_equal(codes, ref_codes)
        tm.assert_numpy_array_equal(uniques, ref_uniques)

    def test_non_contiguous_fallback(self):
        """Non-contiguous int64 input falls back correctly."""
        base = np.arange(400_000, dtype=np.int64)
        keys = base[::2]  # non-contiguous
        codes, uniques = pd.factorize(keys, sort=True)
        ref_codes, ref_uniques = _reference_factorize(
            np.ascontiguousarray(keys), sort=True
        )
        tm.assert_numpy_array_equal(codes, ref_codes)
        tm.assert_numpy_array_equal(uniques, ref_uniques)

    def test_non_native_byteorder_candidate_rejected(self):
        """Non-native byte order int64 is rejected by the candidate check,
        so the dense fast path is never attempted."""
        n = 200_000
        keys = np.arange(n, dtype=np.int64)
        swapped = keys.view(keys.dtype.newbyteorder()).copy()
        # The candidate pre-screen must reject non-native byte order.
        assert not algorithms._is_int64_dense_range_candidate(swapped)
        # The fast-path wrapper returns None (falls back).
        assert algorithms._factorize_int64_dense_range(swapped) is None


# ---------------------------------------------------------------------------
# GroupBy integration
# ---------------------------------------------------------------------------

class TestGroupByDenseInt64:
    """Verify groupby with dense int64 keys produces correct results."""

    @pytest.mark.parametrize(
        "method", ["count", "sum", "mean", "var", "min", "max", "prod", "last"]
    )
    def test_groupby_dense_int64_all_methods(self, method):
        n = 200_000
        ngroups = 500
        rng = np.random.RandomState(11)
        keys = rng.randint(0, ngroups, size=n).astype(np.int64)
        data = rng.randn(n)
        df = pd.DataFrame({"key": keys, "data": data})

        result = getattr(df.groupby("key", sort=True)["data"], method)()

        # Reference: groupby with sort=False then manually sort.
        ref = getattr(df.groupby("key", sort=False)["data"], method)()
        ref = ref.sort_index()

        tm.assert_series_equal(result, ref)

    def test_groupby_result_index_sorted(self):
        n = 200_000
        ngroups = 500
        rng = np.random.RandomState(13)
        keys = rng.randint(0, ngroups, size=n).astype(np.int64)
        data = rng.randn(n)
        df = pd.DataFrame({"key": keys, "data": data})
        result = df.groupby("key", sort=True)["data"].sum()
        assert result.index.is_monotonic_increasing

    def test_groupby_dense_negative_keys(self):
        n = 200_000
        ngroups = 300
        rng = np.random.RandomState(17)
        keys = rng.randint(-ngroups, ngroups, size=n).astype(np.int64)
        data = rng.randn(n)
        df = pd.DataFrame({"key": keys, "data": data})

        result = df.groupby("key", sort=True)["data"].sum()
        ref = df.groupby("key", sort=False)["data"].sum().sort_index()
        tm.assert_series_equal(result, ref)


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestDenseInt64ThreadSafety:
    """Concurrent calls must not corrupt results (no shared mutable state)."""

    def test_concurrent_factorize(self):
        n = 200_000
        results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        errors: list[Exception] = []

        def worker(idx: int) -> None:
            try:
                rng = np.random.RandomState(idx)
                keys = rng.randint(0, 500, size=n).astype(np.int64)
                keys = np.ascontiguousarray(keys)
                codes, uniques = pd.factorize(keys, sort=True)
                ref_codes, ref_uniques = _reference_factorize(keys, sort=True)
                # Validate immediately to catch corruption.
                np.testing.assert_array_equal(codes, ref_codes)
                np.testing.assert_array_equal(uniques, ref_uniques)
                results[idx] = (codes, uniques)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"errors in threads: {errors}"
        assert len(results) == 8
