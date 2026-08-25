#include "pandas/groupby.h"

static void
pandas_group_idx_min_float32(
    npy_intp *out,
    const float *values,
    const npy_intp *labels,
    float *best,
    uint8_t *seen,
    const uint8_t *mask,
    int uses_mask,
    int skipna,
    Py_ssize_t nrows,
    Py_ssize_t ncols,
    Py_ssize_t value_stride0,
    Py_ssize_t value_stride1,
    Py_ssize_t mask_stride0,
    Py_ssize_t mask_stride1)
{
    const char *value_bytes = (const char *)values;

    for (Py_ssize_t i = 0; i < nrows; ++i) {
        const npy_intp lab = labels[i];
        if (lab < 0) {
            continue;
        }
        for (Py_ssize_t j = 0; j < ncols; ++j) {
            const Py_ssize_t pos = lab * ncols + j;
            const float val = *(const float *)(
                value_bytes + i * value_stride0 + j * value_stride1);
            const int isna = uses_mask
                ? *(mask + i * mask_stride0 + j * mask_stride1)
                : val != val;

            if (!skipna && out[pos] == -1) {
                continue;
            }
            if (isna) {
                if (!skipna || !seen[pos]) {
                    out[pos] = -1;
                }
            }
            else if (!seen[pos]) {
                seen[pos] = 1;
                best[pos] = val;
                out[pos] = i;
            }
            else if (val < best[pos]) {
                best[pos] = val;
                out[pos] = i;
            }
        }
    }
}

static void
pandas_group_idx_max_float32(
    npy_intp *out,
    const float *values,
    const npy_intp *labels,
    float *best,
    uint8_t *seen,
    const uint8_t *mask,
    int uses_mask,
    int skipna,
    Py_ssize_t nrows,
    Py_ssize_t ncols,
    Py_ssize_t value_stride0,
    Py_ssize_t value_stride1,
    Py_ssize_t mask_stride0,
    Py_ssize_t mask_stride1)
{
    const char *value_bytes = (const char *)values;

    for (Py_ssize_t i = 0; i < nrows; ++i) {
        const npy_intp lab = labels[i];
        if (lab < 0) {
            continue;
        }
        for (Py_ssize_t j = 0; j < ncols; ++j) {
            const Py_ssize_t pos = lab * ncols + j;
            const float val = *(const float *)(
                value_bytes + i * value_stride0 + j * value_stride1);
            const int isna = uses_mask
                ? *(mask + i * mask_stride0 + j * mask_stride1)
                : val != val;

            if (!skipna && out[pos] == -1) {
                continue;
            }
            if (isna) {
                if (!skipna || !seen[pos]) {
                    out[pos] = -1;
                }
            }
            else if (!seen[pos]) {
                seen[pos] = 1;
                best[pos] = val;
                out[pos] = i;
            }
            else if (val > best[pos]) {
                best[pos] = val;
                out[pos] = i;
            }
        }
    }
}

static void
pandas_group_idx_min_float64(
    npy_intp *out,
    const double *values,
    const npy_intp *labels,
    double *best,
    uint8_t *seen,
    const uint8_t *mask,
    int uses_mask,
    int skipna,
    Py_ssize_t nrows,
    Py_ssize_t ncols,
    Py_ssize_t value_stride0,
    Py_ssize_t value_stride1,
    Py_ssize_t mask_stride0,
    Py_ssize_t mask_stride1)
{
    const char *value_bytes = (const char *)values;

    for (Py_ssize_t i = 0; i < nrows; ++i) {
        const npy_intp lab = labels[i];
        if (lab < 0) {
            continue;
        }
        for (Py_ssize_t j = 0; j < ncols; ++j) {
            const Py_ssize_t pos = lab * ncols + j;
            const double val = *(const double *)(
                value_bytes + i * value_stride0 + j * value_stride1);
            const int isna = uses_mask
                ? *(mask + i * mask_stride0 + j * mask_stride1)
                : val != val;

            if (!skipna && out[pos] == -1) {
                continue;
            }
            if (isna) {
                if (!skipna || !seen[pos]) {
                    out[pos] = -1;
                }
            }
            else if (!seen[pos]) {
                seen[pos] = 1;
                best[pos] = val;
                out[pos] = i;
            }
            else if (val < best[pos]) {
                best[pos] = val;
                out[pos] = i;
            }
        }
    }
}

static void
pandas_group_idx_max_float64(
    npy_intp *out,
    const double *values,
    const npy_intp *labels,
    double *best,
    uint8_t *seen,
    const uint8_t *mask,
    int uses_mask,
    int skipna,
    Py_ssize_t nrows,
    Py_ssize_t ncols,
    Py_ssize_t value_stride0,
    Py_ssize_t value_stride1,
    Py_ssize_t mask_stride0,
    Py_ssize_t mask_stride1)
{
    const char *value_bytes = (const char *)values;

    for (Py_ssize_t i = 0; i < nrows; ++i) {
        const npy_intp lab = labels[i];
        if (lab < 0) {
            continue;
        }
        for (Py_ssize_t j = 0; j < ncols; ++j) {
            const Py_ssize_t pos = lab * ncols + j;
            const double val = *(const double *)(
                value_bytes + i * value_stride0 + j * value_stride1);
            const int isna = uses_mask
                ? *(mask + i * mask_stride0 + j * mask_stride1)
                : val != val;

            if (!skipna && out[pos] == -1) {
                continue;
            }
            if (isna) {
                if (!skipna || !seen[pos]) {
                    out[pos] = -1;
                }
            }
            else if (!seen[pos]) {
                seen[pos] = 1;
                best[pos] = val;
                out[pos] = i;
            }
            else if (val > best[pos]) {
                best[pos] = val;
                out[pos] = i;
            }
        }
    }
}

void
pandas_group_idx_float32(
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
    Py_ssize_t mask_stride1)
{
    if (compute_max) {
        pandas_group_idx_max_float32(
            out, values, labels, best, seen, mask, uses_mask, skipna,
            nrows, ncols, value_stride0, value_stride1,
            mask_stride0, mask_stride1);
    }
    else {
        pandas_group_idx_min_float32(
            out, values, labels, best, seen, mask, uses_mask, skipna,
            nrows, ncols, value_stride0, value_stride1,
            mask_stride0, mask_stride1);
    }
}

void
pandas_group_idx_float64(
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
    Py_ssize_t mask_stride1)
{
    if (compute_max) {
        pandas_group_idx_max_float64(
            out, values, labels, best, seen, mask, uses_mask, skipna,
            nrows, ncols, value_stride0, value_stride1,
            mask_stride0, mask_stride1);
    }
    else {
        pandas_group_idx_min_float64(
            out, values, labels, best, seen, mask, uses_mask, skipna,
            nrows, ncols, value_stride0, value_stride1,
            mask_stride0, mask_stride1);
    }
}
