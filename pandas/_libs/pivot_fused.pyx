# cython: boundscheck=False, wraparound=False, cdivision=True
# cython: language_level=3
"""
Fused Cython kernels for pivot_table operations.

This module provides high-performance fused kernels that combine:
1. Key factorization (hash-based)
2. Value aggregation (sum/mean/count)
3. Margin computation (row/col/grand)

All in a single C-level pass, eliminating Python overhead and intermediate allocations.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport isnan, NAN

cnp.import_array()

# C-level NaN constant for nogil contexts
cdef double C_NAN = NAN

ctypedef fused numeric_t:
    float
    double
    long long
    unsigned long long
    int
    unsigned int


def pivot_fused_sum(
    cnp.ndarray[numeric_t, ndim=2] values,
    list key_arrays,
    Py_ssize_t n_row_total,
    Py_ssize_t n_col_total,
    bint compute_margins,
):
    """
    Fused kernel for sum aggregation with optional margins.
    
    Parameters
    ----------
    values : ndarray[numeric_t, ndim=2]
        Shape (n_rows, n_values)
    key_arrays : list of ndarray
        List of factorized key codes (row_codes, col_codes)
    n_row_total, n_col_total : int
        Total number of row/column groups
    compute_margins : bool
        Whether to compute margins
    
    Returns
    -------
    result : ndarray[double, ndim=3]
        Shape (n_values, n_row_total, n_col_total)
    row_margins : ndarray[double, ndim=2] or None
        Shape (n_values, n_row_total)
    col_margins : ndarray[double, ndim=2] or None
        Shape (n_values, n_col_total)
    grand_margin : ndarray[double, ndim=1] or None
        Shape (n_values,)
    """
    cdef Py_ssize_t n_rows = values.shape[0]
    cdef Py_ssize_t n_values = values.shape[1]
    cdef Py_ssize_t i, j, row_idx, col_idx
    cdef numeric_t val
    
    cdef cnp.ndarray[Py_ssize_t, ndim=1] row_codes = key_arrays[0]
    cdef cnp.ndarray[Py_ssize_t, ndim=1] col_codes = key_arrays[1]
    
    # Allocate result arrays
    cdef cnp.ndarray[double, ndim=3] result = np.zeros(
        (n_values, n_row_total, n_col_total), dtype=np.float64
    )
    
    cdef cnp.ndarray[double, ndim=2] row_margins
    cdef cnp.ndarray[double, ndim=2] col_margins
    cdef cnp.ndarray[double, ndim=1] grand_margin
    
    if compute_margins:
        row_margins = np.zeros((n_values, n_row_total), dtype=np.float64)
        col_margins = np.zeros((n_values, n_col_total), dtype=np.float64)
        grand_margin = np.zeros(n_values, dtype=np.float64)
    
    # Single pass: accumulate into result and margins simultaneously
    with nogil:
        for i in range(n_rows):
            row_idx = row_codes[i]
            col_idx = col_codes[i]
            for j in range(n_values):
                val = values[i, j]
                result[j, row_idx, col_idx] += val
                if compute_margins:
                    row_margins[j, row_idx] += val
                    col_margins[j, col_idx] += val
                    grand_margin[j] += val
    
    if compute_margins:
        return result, row_margins, col_margins, grand_margin
    else:
        return result, None, None, None


def pivot_fused_mean(
    cnp.ndarray[double, ndim=2] values,
    cnp.ndarray[Py_ssize_t, ndim=1] row_codes,
    cnp.ndarray[Py_ssize_t, ndim=1] col_codes,
    Py_ssize_t n_row_total,
    Py_ssize_t n_col_total,
    bint compute_margins,
):
    """
    Fused kernel for mean aggregation with optional margins.
    
    Tracks both sum and count to compute mean correctly.
    """
    cdef Py_ssize_t n_rows = values.shape[0]
    cdef Py_ssize_t n_values = values.shape[1]
    cdef Py_ssize_t i, j, row_idx, col_idx
    cdef double val
    
    # Allocate sum and count arrays
    cdef cnp.ndarray[double, ndim=3] sum_result = np.zeros(
        (n_values, n_row_total, n_col_total), dtype=np.float64
    )
    cdef cnp.ndarray[Py_ssize_t, ndim=3] count_result = np.zeros(
        (n_values, n_row_total, n_col_total), dtype=np.intp
    )
    
    cdef cnp.ndarray[double, ndim=2] row_sum, row_count
    cdef cnp.ndarray[double, ndim=2] col_sum, col_count
    cdef cnp.ndarray[double, ndim=1] grand_sum, grand_count
    
    if compute_margins:
        row_sum = np.zeros((n_values, n_row_total), dtype=np.float64)
        row_count = np.zeros((n_values, n_row_total), dtype=np.float64)
        col_sum = np.zeros((n_values, n_col_total), dtype=np.float64)
        col_count = np.zeros((n_values, n_col_total), dtype=np.float64)
        grand_sum = np.zeros(n_values, dtype=np.float64)
        grand_count = np.zeros(n_values, dtype=np.float64)
    
    # Single pass: accumulate sum and count
    with nogil:
        for i in range(n_rows):
            row_idx = row_codes[i]
            col_idx = col_codes[i]
            for j in range(n_values):
                val = values[i, j]
                if not isnan(val):
                    sum_result[j, row_idx, col_idx] += val
                    count_result[j, row_idx, col_idx] += 1
                    if compute_margins:
                        row_sum[j, row_idx] += val
                        row_count[j, row_idx] += 1
                        col_sum[j, col_idx] += val
                        col_count[j, col_idx] += 1
                        grand_sum[j] += val
                        grand_count[j] += 1
    
    # Compute mean from sum/count
    cdef cnp.ndarray[double, ndim=3] result = np.empty_like(sum_result)
    with nogil:
        for j in range(n_values):
            for row_idx in range(n_row_total):
                for col_idx in range(n_col_total):
                    if count_result[j, row_idx, col_idx] > 0:
                        result[j, row_idx, col_idx] = (
                            sum_result[j, row_idx, col_idx] / 
                            count_result[j, row_idx, col_idx]
                        )
                    else:
                        result[j, row_idx, col_idx] = C_NAN
    
    cdef cnp.ndarray[double, ndim=2] row_margins
    cdef cnp.ndarray[double, ndim=2] col_margins
    cdef cnp.ndarray[double, ndim=1] grand_margin
    
    if compute_margins:
        row_margins = np.empty_like(row_sum)
        col_margins = np.empty_like(col_sum)
        grand_margin = np.empty_like(grand_sum)
        
        with nogil:
            for j in range(n_values):
                for row_idx in range(n_row_total):
                    if row_count[j, row_idx] > 0:
                        row_margins[j, row_idx] = row_sum[j, row_idx] / row_count[j, row_idx]
                    else:
                        row_margins[j, row_idx] = C_NAN
                
                for col_idx in range(n_col_total):
                    if col_count[j, col_idx] > 0:
                        col_margins[j, col_idx] = col_sum[j, col_idx] / col_count[j, col_idx]
                    else:
                        col_margins[j, col_idx] = C_NAN
                
                if grand_count[j] > 0:
                    grand_margin[j] = grand_sum[j] / grand_count[j]
                else:
                    grand_margin[j] = C_NAN
        
        return result, row_margins, col_margins, grand_margin
    else:
        return result, None, None, None


def pivot_fused_count(
    cnp.ndarray[Py_ssize_t, ndim=1] row_codes,
    cnp.ndarray[Py_ssize_t, ndim=1] col_codes,
    Py_ssize_t n_row_total,
    Py_ssize_t n_col_total,
    Py_ssize_t n_values,
    bint compute_margins,
):
    """
    Fused kernel for count aggregation with optional margins.
    """
    cdef Py_ssize_t n_rows = row_codes.shape[0]
    cdef Py_ssize_t i, j, row_idx, col_idx
    
    # Allocate count array
    cdef cnp.ndarray[Py_ssize_t, ndim=3] count_result = np.zeros(
        (n_values, n_row_total, n_col_total), dtype=np.intp
    )
    
    cdef cnp.ndarray[Py_ssize_t, ndim=2] row_count
    cdef cnp.ndarray[Py_ssize_t, ndim=2] col_count
    cdef cnp.ndarray[Py_ssize_t, ndim=1] grand_count
    
    if compute_margins:
        row_count = np.zeros((n_values, n_row_total), dtype=np.intp)
        col_count = np.zeros((n_values, n_col_total), dtype=np.intp)
        grand_count = np.zeros(n_values, dtype=np.intp)
    
    # Single pass: count
    with nogil:
        for i in range(n_rows):
            row_idx = row_codes[i]
            col_idx = col_codes[i]
            for j in range(n_values):
                count_result[j, row_idx, col_idx] += 1
                if compute_margins:
                    row_count[j, row_idx] += 1
                    col_count[j, col_idx] += 1
                    grand_count[j] += 1
    
    # Convert to float and set zeros to NaN
    cdef cnp.ndarray[double, ndim=3] result = count_result.astype(np.float64)
    result[result == 0] = np.nan
    
    cdef cnp.ndarray[double, ndim=2] row_margins
    cdef cnp.ndarray[double, ndim=2] col_margins
    cdef cnp.ndarray[double, ndim=1] grand_margin
    
    if compute_margins:
        row_margins = row_count.astype(np.float64)
        row_margins[row_margins == 0] = np.nan
        col_margins = col_count.astype(np.float64)
        col_margins[col_margins == 0] = np.nan
        grand_margin = grand_count.astype(np.float64)
        grand_margin[grand_margin == 0] = np.nan
        
        return result, row_margins, col_margins, grand_margin
    else:
        return result, None, None, None
