/*
Copyright (c) 2016, PyData Development Team
All rights reserved.

Distributed under the terms of the BSD Simplified License.

The full license is in the LICENSE file, distributed with this software.
*/

#pragma once

#include <string.h>
#include <stdint.h>

#if defined(_MSC_VER)
#  define strcasecmp(s1, s2) _stricmp(s1, s2)
#endif

// GH-23516 - works around locale perf issues
// from MUSL libc, licence at LICENSES/MUSL_LICENSE
#define isdigit_ascii(c) (((unsigned)(c) - '0') < 10u)
#define getdigit_ascii(c, default)                                             \
  (isdigit_ascii(c) ? ((int)((c) - '0')) : default)
#define isspace_ascii(c) (((c) == ' ') || (((unsigned)(c) - '\t') < 5))
#define toupper_ascii(c) ((((unsigned)(c) - 'a') < 26) ? ((c) & 0x5f) : (c))
#define tolower_ascii(c) ((((unsigned)(c) - 'A') < 26) ? ((c) | 0x20) : (c))

static inline int pandas_is_aarch64(void) {
#if defined(__aarch64__) || defined(_M_ARM64)
  return 1;
#else
  return 0;
#endif
}

#if defined(_WIN32)
#  define PD_FALLTHROUGH                                                       \
    do {                                                                       \
    } while (0) /* fallthrough */
#elif __has_attribute(__fallthrough__)
#  define PD_FALLTHROUGH __attribute__((__fallthrough__))
#else
#  define PD_FALLTHROUGH                                                       \
    do {                                                                       \
    } while (0) /* fallthrough */
#endif

#if defined(_WIN32)
#  ifndef ENABLE_INTSAFE_SIGNED_FUNCTIONS
#    define ENABLE_INTSAFE_SIGNED_FUNCTIONS
#  endif
#  include <intsafe.h>
#  define checked_add(a, b, res)                                               \
    _Generic((res),                                                            \
        int *: IntAdd,                                                         \
        unsigned int *: UIntAdd,                                               \
        long *: LongAdd,                                                       \
        unsigned long *: ULongAdd,                                             \
        long long *: LongLongAdd,                                              \
        unsigned long long *: ULongLongAdd,                                    \
        short *: ShortAdd,                                                     \
        unsigned short *: UShortAdd)(a, b, res)

#  define checked_sub(a, b, res)                                               \
    _Generic((res),                                                            \
        int *: IntSub,                                                         \
        unsigned int *: UIntSub,                                               \
        long *: LongSub,                                                       \
        unsigned long *: ULongSub,                                             \
        long long *: LongLongSub,                                              \
        unsigned long long *: ULongLongSub,                                    \
        short *: ShortSub,                                                     \
        unsigned short *: UShortSub)(a, b, res)

#  define checked_mul(a, b, res)                                               \
    _Generic((res),                                                            \
        int *: IntMult,                                                        \
        unsigned int *: UIntMult,                                              \
        long *: LongMult,                                                      \
        unsigned long *: ULongMult,                                            \
        long long *: LongLongMult,                                             \
        unsigned long long *: ULongLongMult,                                   \
        short *: ShortMult,                                                    \
        unsigned short *: UShortMult)(a, b, res)

#elif (defined(__has_builtin) && __has_builtin(__builtin_add_overflow)) ||     \
    __GNUC__ > 7
#  define checked_add(a, b, res) __builtin_add_overflow(a, b, res)
#  define checked_sub(a, b, res) __builtin_sub_overflow(a, b, res)
#  define checked_mul(a, b, res) __builtin_mul_overflow(a, b, res)
#else
_Static_assert(0,
               "Overflow checking not detected; please try a newer compiler");
#endif

#if defined(_MSC_VER)
#  include <intrin.h>
static inline int pandas_ctz(uint32_t x) {
  if (x == 0) return 32;
  unsigned long index;
  _BitScanForward(&index, x);
  return (int)index;
}
#elif (defined(__has_builtin) && __has_builtin(__builtin_ctz)) || defined(__GNUC__)
static inline int pandas_ctz(uint32_t x) {
  return x ? __builtin_ctz(x) : 32;
}
#else
static inline int pandas_ctz(uint32_t x) {
  if (x == 0) return 32;
  int n = 0;
  while (!(x & 1)) { x >>= 1; ++n; }
  return n;
}
#endif
