#include "pandas/groupby_neon.h"

#if defined(__aarch64__)
#  include <arm_neon.h>
#endif

void pandas_group_prod_float64_neon(double *prodx, const double *values,
                                    const npy_intp *labels, int64_t *counts,
                                    Py_ssize_t nrows, Py_ssize_t ncols) {
#if defined(__aarch64__)
  const float64x2_t ones = vdupq_n_f64(1.0);
  for (Py_ssize_t i = 0; i < nrows; ++i) {
    const npy_intp lab = labels[i];
    if (lab < 0) {
      continue;
    }
    counts[lab] += 1;
    const double *row = values + i * ncols;
    double *acc = prodx + lab * ncols;
    Py_ssize_t j = 0;
    for (; j + 2 <= ncols; j += 2) {
      const float64x2_t val = vld1q_f64(row + j);
      const uint64x2_t valid = vceqq_f64(val, val);
      const float64x2_t factor = vbslq_f64(valid, val, ones);
      vst1q_f64(acc + j, vmulq_f64(vld1q_f64(acc + j), factor));
    }
    for (; j < ncols; ++j) {
      const double val = row[j];
      if (val == val) {
        acc[j] *= val;
      }
    }
  }
#endif
}

void pandas_group_prod_float32_neon(float *prodx, const float *values,
                                    const npy_intp *labels, int64_t *counts,
                                    Py_ssize_t nrows, Py_ssize_t ncols) {
#if defined(__aarch64__)
  const float32x4_t ones = vdupq_n_f32(1.0f);
  for (Py_ssize_t i = 0; i < nrows; ++i) {
    const npy_intp lab = labels[i];
    if (lab < 0) {
      continue;
    }
    counts[lab] += 1;
    const float *row = values + i * ncols;
    float *acc = prodx + lab * ncols;
    Py_ssize_t j = 0;
    for (; j + 4 <= ncols; j += 4) {
      const float32x4_t val = vld1q_f32(row + j);
      const uint32x4_t valid = vceqq_f32(val, val);
      const float32x4_t factor = vbslq_f32(valid, val, ones);
      vst1q_f32(acc + j, vmulq_f32(vld1q_f32(acc + j), factor));
    }
    for (; j < ncols; ++j) {
      const float val = row[j];
      if (val == val) {
        acc[j] *= val;
      }
    }
  }
#endif
}

void pandas_group_prod_float64_neon_colmajor(double *prodx,
                                             const double *values,
                                             const npy_intp *labels,
                                             int64_t *counts, Py_ssize_t nrows,
                                             Py_ssize_t ncols) {
#if defined(__aarch64__)
  const float64x2_t ones = vdupq_n_f64(1.0);
  for (Py_ssize_t i = 0; i < nrows; ++i) {
    const npy_intp lab = labels[i];
    if (lab < 0) {
      continue;
    }
    counts[lab] += 1;
    double *acc = prodx + lab * ncols;
    Py_ssize_t j = 0;
    for (; j + 2 <= ncols; j += 2) {
      float64x2_t val = vdupq_n_f64(values[i + j * nrows]);
      val = vsetq_lane_f64(values[i + (j + 1) * nrows], val, 1);
      const uint64x2_t valid = vceqq_f64(val, val);
      const float64x2_t factor = vbslq_f64(valid, val, ones);
      vst1q_f64(acc + j, vmulq_f64(vld1q_f64(acc + j), factor));
    }
    for (; j < ncols; ++j) {
      const double val = values[i + j * nrows];
      if (val == val) {
        acc[j] *= val;
      }
    }
  }
#endif
}

void pandas_group_prod_float32_neon_colmajor(float *prodx, const float *values,
                                             const npy_intp *labels,
                                             int64_t *counts, Py_ssize_t nrows,
                                             Py_ssize_t ncols) {
#if defined(__aarch64__)
  const float32x4_t ones = vdupq_n_f32(1.0f);
  for (Py_ssize_t i = 0; i < nrows; ++i) {
    const npy_intp lab = labels[i];
    if (lab < 0) {
      continue;
    }
    counts[lab] += 1;
    float *acc = prodx + lab * ncols;
    Py_ssize_t j = 0;
    for (; j + 4 <= ncols; j += 4) {
      float32x4_t val = vdupq_n_f32(values[i + j * nrows]);
      val = vsetq_lane_f32(values[i + (j + 1) * nrows], val, 1);
      val = vsetq_lane_f32(values[i + (j + 2) * nrows], val, 2);
      val = vsetq_lane_f32(values[i + (j + 3) * nrows], val, 3);
      const uint32x4_t valid = vceqq_f32(val, val);
      const float32x4_t factor = vbslq_f32(valid, val, ones);
      vst1q_f32(acc + j, vmulq_f32(vld1q_f32(acc + j), factor));
    }
    for (; j < ncols; ++j) {
      const float val = values[i + j * nrows];
      if (val == val) {
        acc[j] *= val;
      }
    }
  }
#endif
}
