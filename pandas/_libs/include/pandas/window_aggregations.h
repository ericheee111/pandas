/*
Copyright (c) 2026, PyData Development Team
All rights reserved.

Distributed under the terms of the BSD Simplified License.

The full license is in the LICENSE file, distributed with this software.
*/

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace pandas {

#if defined(__aarch64__)
namespace detail {

static inline uint64x2_t find_nonfinite(const double *values,
                                        std::size_t n) noexcept {
  constexpr std::uint64_t exponent_mask = UINT64_C(0x7ff0000000000000);
  const uint64x2_t exponent_mask_vec = vdupq_n_u64(exponent_mask);
  uint64x2_t nonfinite0 = vdupq_n_u64(0);
  uint64x2_t nonfinite1 = vdupq_n_u64(0);
  uint64x2_t nonfinite2 = vdupq_n_u64(0);
  uint64x2_t nonfinite3 = vdupq_n_u64(0);

  for (std::size_t i = 0; i < n; i += 8) {
    const uint64x2_t bits0 =
        vreinterpretq_u64_f64(vld1q_f64(values + i));
    const uint64x2_t bits1 =
        vreinterpretq_u64_f64(vld1q_f64(values + i + 2));
    const uint64x2_t bits2 =
        vreinterpretq_u64_f64(vld1q_f64(values + i + 4));
    const uint64x2_t bits3 =
        vreinterpretq_u64_f64(vld1q_f64(values + i + 6));

    nonfinite0 = vorrq_u64(
        nonfinite0,
        vceqq_u64(vandq_u64(bits0, exponent_mask_vec), exponent_mask_vec));
    nonfinite1 = vorrq_u64(
        nonfinite1,
        vceqq_u64(vandq_u64(bits1, exponent_mask_vec), exponent_mask_vec));
    nonfinite2 = vorrq_u64(
        nonfinite2,
        vceqq_u64(vandq_u64(bits2, exponent_mask_vec), exponent_mask_vec));
    nonfinite3 = vorrq_u64(
        nonfinite3,
        vceqq_u64(vandq_u64(bits3, exponent_mask_vec), exponent_mask_vec));
  }

  return vorrq_u64(vorrq_u64(nonfinite0, nonfinite1),
                   vorrq_u64(nonfinite2, nonfinite3));
}

static inline bool any_nonfinite(uint64x2_t values) noexcept {
  return (vgetq_lane_u64(values, 0) | vgetq_lane_u64(values, 1)) != 0;
}

} // namespace detail
#endif

static inline bool all_finite(const double *values, std::size_t n,
                              std::ptrdiff_t stride) noexcept {
#if defined(__aarch64__)
  if (stride == 1) {
    constexpr std::size_t block_size = 256;
    std::size_t i = 0;

    for (; i + block_size <= n; i += block_size) {
      if (detail::any_nonfinite(
              detail::find_nonfinite(values + i, block_size))) {
        return false;
      }
    }

    const std::size_t vectorized_size = (n - i) / 8 * 8;
    if (vectorized_size != 0) {
      if (detail::any_nonfinite(
              detail::find_nonfinite(values + i, vectorized_size))) {
        return false;
      }
      i += vectorized_size;
    }

    for (; i < n; ++i) {
      if (!std::isfinite(values[i])) {
        return false;
      }
    }
    return true;
  }
#endif

  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(values[i * stride])) {
      return false;
    }
  }
  return true;
}

} // namespace pandas
