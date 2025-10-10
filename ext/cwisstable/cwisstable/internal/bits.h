// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CWISSTABLE_INTERNAL_BITS_H_
#define CWISSTABLE_INTERNAL_BITS_H_

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "cwisstable/internal/base.h"

/// Bit manipulation utilities.

CWISS_BEGIN
CWISS_BEGIN_EXTERN

/// Counts the number of trailing zeroes in the binary representation of `x`.
CWISS_INLINE_ALWAYS
static inline uint32_t CWISS_TrailingZeroes64(uint64_t x) {
#if CWISS_HAVE_CLANG_BUILTIN(__builtin_ctzll) || CWISS_IS_GCC
  static_assert(sizeof(unsigned long long) == sizeof(x),
                "__builtin_ctzll does not take 64-bit arg");
  return __builtin_ctzll(x);
#elif CWISS_IS_MSVC
  unsigned long result = 0;
  #if defined(_M_X64) || defined(_M_ARM64)
  _BitScanForward64(&result, x);
  #else
  if (((uint32_t)x) == 0) {
    _BitScanForward(&result, (unsigned long)(x >> 32));
    return result + 32;
  }
  _BitScanForward(&result, (unsigned long)(x));
  #endif
  return result;
#else
  uint32_t c = 63;
  x &= ~x + 1;
  if (x & 0x00000000FFFFFFFF) c -= 32;
  if (x & 0x0000FFFF0000FFFF) c -= 16;
  if (x & 0x00FF00FF00FF00FF) c -= 8;
  if (x & 0x0F0F0F0F0F0F0F0F) c -= 4;
  if (x & 0x3333333333333333) c -= 2;
  if (x & 0x5555555555555555) c -= 1;
  return c;
#endif
}

/// Counts the number of leading zeroes in the binary representation of `x`.
CWISS_INLINE_ALWAYS
static inline uint32_t CWISS_LeadingZeroes64(uint64_t x) {
#if CWISS_HAVE_CLANG_BUILTIN(__builtin_clzll) || CWISS_IS_GCC
  static_assert(sizeof(unsigned long long) == sizeof(x),
                "__builtin_clzll does not take 64-bit arg");
  // Handle 0 as a special case because __builtin_clzll(0) is undefined.
  return x == 0 ? 64 : __builtin_clzll(x);
#elif CWISS_IS_MSVC
  unsigned long result = 0;
  #if defined(_M_X64) || defined(_M_ARM64)
  if (_BitScanReverse64(&result, x)) {
    return 63 - result;
  }
  #else
  unsigned long result = 0;
  if ((x >> 32) && _BitScanReverse(&result, (unsigned long)(x >> 32))) {
    return 31 - result;
  }
  if (_BitScanReverse(&result, static_cast<unsigned long>(x))) {
    return 63 - result;
  }
  #endif
  return 64;
#else
  uint32_t zeroes = 60;
  if (x >> 32) {
    zeroes -= 32;
    x >>= 32;
  }
  if (x >> 16) {
    zeroes -= 16;
    x >>= 16;
  }
  if (x >> 8) {
    zeroes -= 8;
    x >>= 8;
  }
  if (x >> 4) {
    zeroes -= 4;
    x >>= 4;
  }
  return "\4\3\2\2\1\1\1\1\0\0\0\0\0\0\0"[x] + zeroes;
#endif
}

/// Counts the number of trailing zeroes in the binary representation of `x_` in
/// a type-generic fashion.
#define CWISS_TrailingZeros(x_) (CWISS_TrailingZeroes64(x_))

/// Counts the number of leading zeroes in the binary representation of `x_` in
/// a type-generic fashion.
#define CWISS_LeadingZeros(x_) \
  (CWISS_LeadingZeroes64(x_) - \
   (uint32_t)((sizeof(unsigned long long) - sizeof(x_)) * 8))

/// Computes the number of bits necessary to represent `x_`, i.e., the bit index
/// of the most significant one.
#define CWISS_BitWidth(x_) \
  (((uint32_t)(sizeof(x_) * 8)) - CWISS_LeadingZeros(x_))

#define CWISS_RotateLeft(x_, bits_) \
  (((x_) << bits_) | ((x_) >> (sizeof(x_) * 8 - bits_)))

/// The return type of `CWISS_Mul128`.
typedef struct {
  uint64_t lo, hi;
} CWISS_U128;

/// Computes a double-width multiplication operation.
static inline CWISS_U128 CWISS_Mul128(uint64_t a, uint64_t b) {
  // TODO: de-intrinsics-ize this.
  __uint128_t p = a;
  p *= b;
  return (CWISS_U128){(uint64_t)p, (uint64_t)(p >> 64)};
}

/// Loads an unaligned u32.
static inline uint32_t CWISS_Load32(const void* p) {
  uint32_t v;
  memcpy(&v, p, sizeof(v));
  return v;
}

/// Loads an unaligned u64.
static inline uint64_t CWISS_Load64(const void* p) {
  uint64_t v;
  memcpy(&v, p, sizeof(v));
  return v;
}

/// Reads 9 to 16 bytes from p.
static inline CWISS_U128 CWISS_Load9To16(const void* p, size_t len) {
  const unsigned char* p8 = (const unsigned char*)p;
  uint64_t lo = CWISS_Load64(p8);
  uint64_t hi = CWISS_Load64(p8 + len - 8);
  return (CWISS_U128){lo, hi >> (128 - len * 8)};
}

/// Reads 4 to 8 bytes from p.
static inline uint64_t CWISS_Load4To8(const void* p, size_t len) {
  const unsigned char* p8 = (const unsigned char*)p;
  uint64_t lo = CWISS_Load32(p8);
  uint64_t hi = CWISS_Load32(p8 + len - 4);
  return lo | (hi << (len - 4) * 8);
}

/// Reads 1 to 3 bytes from p.
static inline uint32_t CWISS_Load1To3(const void* p, size_t len) {
  const unsigned char* p8 = (const unsigned char*)p;
  uint32_t mem0 = p8[0];
  uint32_t mem1 = p8[len / 2];
  uint32_t mem2 = p8[len - 1];
  return (mem0 | (mem1 << (len / 2 * 8)) | (mem2 << ((len - 1) * 8)));
}

/// A abstract bitmask, such as that emitted by a SIMD instruction.
///
/// Specifically, this type implements a simple bitset whose representation is
/// controlled by `width` and `shift`. `width` is the number of abstract bits in
/// the bitset, while `shift` is the log-base-two of the width of an abstract
/// bit in the representation.
///
/// For example, when `width` is 16 and `shift` is zero, this is just an
/// ordinary 16-bit bitset occupying the low 16 bits of `mask`. When `width` is
/// 8 and `shift` is 3, abstract bits are represented as the bytes `0x00` and
/// `0x80`, and it occupies all 64 bits of the bitmask.
typedef struct {
  /// The mask, in the representation specified by `width` and `shift`.
  uint64_t mask;
  /// The number of abstract bits in the mask.
  uint32_t width;
  /// The log-base-two width of an abstract bit.
  uint32_t shift;
} CWISS_BitMask;

/// Returns the index of the lowest abstract bit set in `self`.
static inline uint32_t CWISS_BitMask_LowestBitSet(const CWISS_BitMask* self) {
  return CWISS_TrailingZeros(self->mask) >> self->shift;
}

/// Returns the index of the highest abstract bit set in `self`.
static inline uint32_t CWISS_BitMask_HighestBitSet(const CWISS_BitMask* self) {
  return (uint32_t)(CWISS_BitWidth(self->mask) - 1) >> self->shift;
}

/// Return the number of trailing zero abstract bits.
static inline uint32_t CWISS_BitMask_TrailingZeros(const CWISS_BitMask* self) {
  return CWISS_TrailingZeros(self->mask) >> self->shift;
}

/// Return the number of leading zero abstract bits.
static inline uint32_t CWISS_BitMask_LeadingZeros(const CWISS_BitMask* self) {
  uint32_t total_significant_bits = self->width << self->shift;
  uint32_t extra_bits = sizeof(self->mask) * 8 - total_significant_bits;
  return (uint32_t)(CWISS_LeadingZeros(self->mask << extra_bits)) >>
         self->shift;
}

/// Iterates over the one bits in the mask.
///
/// If the mask is empty, returns `false`; otherwise, returns the index of the
/// lowest one bit in the mask, and removes it from the set.
static inline bool CWISS_BitMask_next(CWISS_BitMask* self, uint32_t* bit) {
  if (self->mask == 0) {
    return false;
  }

  *bit = CWISS_BitMask_LowestBitSet(self);
  self->mask &= (self->mask - 1);
  return true;
}

CWISS_END_EXTERN
CWISS_END

#endif  // CWISSTABLE_INTERNAL_BITS_H_