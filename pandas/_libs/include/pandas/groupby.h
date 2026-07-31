#ifndef PANDAS_GROUPBY_H
#define PANDAS_GROUPBY_H

#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <stdint.h>

void pandas_group_idx_float32(
    npy_intp *out,
    const float *values,
    const npy_intp *labels,
    float *best,
    uint8_t *seen,
    const uint8_t *mask,
    int uses_mask,
    int skipna,
    int compute_max,
    Py_ssize_t nrows,
    Py_ssize_t ncols,
    Py_ssize_t value_stride0,
    Py_ssize_t value_stride1,
    Py_ssize_t mask_stride0,
    Py_ssize_t mask_stride1);

void pandas_group_idx_float64(
    npy_intp *out,
    const double *values,
    const npy_intp *labels,
    double *best,
    uint8_t *seen,
    const uint8_t *mask,
    int uses_mask,
    int skipna,
    int compute_max,
    Py_ssize_t nrows,
    Py_ssize_t ncols,
    Py_ssize_t value_stride0,
    Py_ssize_t value_stride1,
    Py_ssize_t mask_stride0,
    Py_ssize_t mask_stride1);

#endif
