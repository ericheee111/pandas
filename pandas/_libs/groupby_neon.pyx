"""
AArch64 ``prod`` specialization entry for native float arrays.

Kept in a translation unit of its own: a dispatch branch inside the
fused ``group_prod`` (see pandas/_libs/groupby.pyx) shifts the generated
code of every dtype specialization of the generic loop -- measured on
Kunpeng 920B, both directions -- while a separately compiled module
leaves that layout identical to v3.0.1.
"""

import numpy as np

cimport numpy as cnp

cimport cython
from cython cimport (
    Py_ssize_t,
    floating,
)

from numpy cimport (
    float32_t,
    float64_t,
    int64_t,
    intp_t,
    ndarray,
)

cnp.import_array()

cdef extern from "pandas/portable.h":
    bint pandas_is_aarch64() noexcept nogil


cdef extern from "pandas/groupby_neon.h":
    void pandas_group_prod_float64_neon(
        float64_t*, const float64_t*, const intp_t*, int64_t*,
        Py_ssize_t, Py_ssize_t,
    ) noexcept nogil
    void pandas_group_prod_float32_neon(
        float32_t*, const float32_t*, const intp_t*, int64_t*,
        Py_ssize_t, Py_ssize_t,
    ) noexcept nogil
    void pandas_group_prod_float64_neon_colmajor(
        float64_t*, const float64_t*, const intp_t*, int64_t*,
        Py_ssize_t, Py_ssize_t,
    ) noexcept nogil
    void pandas_group_prod_float32_neon_colmajor(
        float32_t*, const float32_t*, const intp_t*, int64_t*,
        Py_ssize_t, Py_ssize_t,
    ) noexcept nogil


@cython.wraparound(False)
@cython.boundscheck(False)
def group_prod_native_float(
    floating[:, ::1] out,
    int64_t[::1] counts,
    ndarray[floating, ndim=2] values,
    const intp_t[::1] labels,
    Py_ssize_t min_count,
    bint skipna,
) -> bool:
    """
    Handle an eligible ``prod`` aggregation with the AArch64 NEON kernels
    (multi-column contiguous values) or a branch-free scalar loop (single
    column) that skips the ``nobs`` bookkeeping.

    Eligibility: skipna, ``min_count <= 0``, no mask, non-empty. Returns
    True when the call was handled; the caller falls back to the fused
    ``group_prod`` otherwise.
    """
    cdef:
        Py_ssize_t i, lab, N, K
        bint is_c_contiguous, is_f_contiguous
        floating val

    if not pandas_is_aarch64() or not skipna or min_count > 0:
        return False

    N, K = (<object>values).shape
    if N == 0 or K == 0 or len(counts) == 0:
        return False

    if K == 1:
        # min_count <= 0 keeps below-min-count groups at the product
        # identity, so no nobs bookkeeping is needed
        out[:, :] = 1
        with nogil:
            for i in range(N):
                lab = labels[i]
                if lab < 0:
                    continue
                counts[lab] += 1
                val = values[i, 0]
                if val == val:
                    out[lab, 0] *= val
        return True

    is_c_contiguous = (
        values.strides[1] == sizeof(floating)
        and values.strides[0] == K * sizeof(floating)
    )
    is_f_contiguous = (
        values.strides[0] == sizeof(floating)
        and values.strides[1] == N * sizeof(floating)
    )
    if not (is_c_contiguous or is_f_contiguous):
        return False

    out[:, :] = 1
    with nogil:
        if floating is float64_t:
            if is_c_contiguous:
                pandas_group_prod_float64_neon(
                    &out[0, 0], &values[0, 0], &labels[0], &counts[0], N, K
                )
            else:
                pandas_group_prod_float64_neon_colmajor(
                    &out[0, 0], &values[0, 0], &labels[0], &counts[0], N, K
                )
        else:
            if is_c_contiguous:
                pandas_group_prod_float32_neon(
                    &out[0, 0], &values[0, 0], &labels[0], &counts[0], N, K
                )
            else:
                pandas_group_prod_float32_neon_colmajor(
                    &out[0, 0], &values[0, 0], &labels[0], &counts[0], N, K
                )
    return True
