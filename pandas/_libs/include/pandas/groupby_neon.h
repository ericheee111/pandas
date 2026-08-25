#pragma once

#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <stdint.h>

void pandas_group_prod_float64_neon(double *prodx, const double *values,
                                    const npy_intp *labels, int64_t *counts,
                                    Py_ssize_t nrows, Py_ssize_t ncols);

void pandas_group_prod_float32_neon(float *prodx, const float *values,
                                    const npy_intp *labels, int64_t *counts,
                                    Py_ssize_t nrows, Py_ssize_t ncols);

void pandas_group_prod_float64_neon_colmajor(double *prodx,
                                             const double *values,
                                             const npy_intp *labels,
                                             int64_t *counts, Py_ssize_t nrows,
                                             Py_ssize_t ncols);

void pandas_group_prod_float32_neon_colmajor(float *prodx, const float *values,
                                             const npy_intp *labels,
                                             int64_t *counts, Py_ssize_t nrows,
                                             Py_ssize_t ncols);
