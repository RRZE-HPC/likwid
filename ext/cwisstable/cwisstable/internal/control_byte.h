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

#ifndef CWISSTABLE_INTERNAL_CTRL_H_
#define CWISSTABLE_INTERNAL_CTRL_H_

#include <assert.h>
#include <limits.h>
#include <stdalign.h>
#include <stdint.h>
#include <string.h>

#include "cwisstable/internal/base.h"
#include "cwisstable/internal/bits.h"

CWISS_BEGIN
CWISS_BEGIN_EXTERN

/// Control bytes and groups: the core of SwissTable optimization.
///
/// Control bytes are bytes (collected into groups of a platform-specific size)
/// that define the state of the corresponding slot in the slot array. Group
/// manipulation is tightly optimized to be as efficient as possible.

/// A `CWISS_ControlByte` is a single control byte, which can have one of four
/// states: empty, deleted, full (which has an associated seven-bit hash) and
/// the sentinel. They have the following bit patterns:
///
/// ```
///    empty: 1 0 0 0 0 0 0 0
///  deleted: 1 1 1 1 1 1 1 0
///     full: 0 h h h h h h h  // h represents the hash bits.
/// sentinel: 1 1 1 1 1 1 1 1
/// ```
///
/// These values are specifically tuned for SSE-flavored SIMD; future ports to
/// other SIMD platforms may require choosing new values. The static_asserts
/// below detail the source of these choices.
typedef int8_t CWISS_ControlByte;
#define CWISS_kEmpty (INT8_C(-128))
#define CWISS_kDeleted (INT8_C(-2))
#define CWISS_kSentinel (INT8_C(-1))
// TODO: Wrap CWISS_ControlByte in a single-field struct to get strict-aliasing
// benefits.

static_assert(
    (CWISS_kEmpty & CWISS_kDeleted & CWISS_kSentinel & 0x80) != 0,
    "Special markers need to have the MSB to make checking for them efficient");
static_assert(
    CWISS_kEmpty < CWISS_kSentinel && CWISS_kDeleted < CWISS_kSentinel,
    "CWISS_kEmpty and CWISS_kDeleted must be smaller than "
    "CWISS_kSentinel to make the SIMD test of IsEmptyOrDeleted() efficient");
static_assert(
    CWISS_kSentinel == -1,
    "CWISS_kSentinel must be -1 to elide loading it from memory into SIMD "
    "registers (pcmpeqd xmm, xmm)");
static_assert(CWISS_kEmpty == -128,
              "CWISS_kEmpty must be -128 to make the SIMD check for its "
              "existence efficient (psignb xmm, xmm)");
static_assert(
    (~CWISS_kEmpty & ~CWISS_kDeleted & CWISS_kSentinel & 0x7F) != 0,
    "CWISS_kEmpty and CWISS_kDeleted must share an unset bit that is not "
    "shared by CWISS_kSentinel to make the scalar test for "
    "MatchEmptyOrDeleted() efficient");
static_assert(CWISS_kDeleted == -2,
              "CWISS_kDeleted must be -2 to make the implementation of "
              "ConvertSpecialToEmptyAndFullToDeleted efficient");

/// Returns a pointer to a control byte group that can be used by empty tables.
static inline CWISS_ControlByte* CWISS_EmptyGroup() {
  // A single block of empty control bytes for tables without any slots
  // allocated. This enables removing a branch in the hot path of find().
  alignas(16) static const CWISS_ControlByte kEmptyGroup[16] = {
      CWISS_kSentinel, CWISS_kEmpty, CWISS_kEmpty, CWISS_kEmpty,
      CWISS_kEmpty,    CWISS_kEmpty, CWISS_kEmpty, CWISS_kEmpty,
      CWISS_kEmpty,    CWISS_kEmpty, CWISS_kEmpty, CWISS_kEmpty,
      CWISS_kEmpty,    CWISS_kEmpty, CWISS_kEmpty, CWISS_kEmpty,
  };

  // Const must be cast away here; no uses of this function will actually write
  // to it, because it is only used for empty tables.
  return (CWISS_ControlByte*)&kEmptyGroup;
}

/// Returns a hash seed.
///
/// The seed consists of the ctrl_ pointer, which adds enough entropy to ensure
/// non-determinism of iteration order in most cases.
static inline size_t CWISS_HashSeed(const CWISS_ControlByte* ctrl) {
  // The low bits of the pointer have little or no entropy because of
  // alignment. We shift the pointer to try to use higher entropy bits. A
  // good number seems to be 12 bits, because that aligns with page size.
  return ((uintptr_t)ctrl) >> 12;
}

/// Extracts the H1 portion of a hash: the high 57 bits mixed with a per-table
/// salt.
static inline size_t CWISS_H1(size_t hash, const CWISS_ControlByte* ctrl) {
  return (hash >> 7) ^ CWISS_HashSeed(ctrl);
}

/// Extracts the H2 portion of a hash: the low 7 bits, which can be used as
/// control byte.
typedef uint8_t CWISS_h2_t;
static inline CWISS_h2_t CWISS_H2(size_t hash) { return hash & 0x7F; }

/// Returns whether `c` is empty.
static inline bool CWISS_IsEmpty(CWISS_ControlByte c) {
  return c == CWISS_kEmpty;
}

/// Returns whether `c` is full.
static inline bool CWISS_IsFull(CWISS_ControlByte c) { return c >= 0; }

/// Returns whether `c` is deleted.
static inline bool CWISS_IsDeleted(CWISS_ControlByte c) {
  return c == CWISS_kDeleted;
}

/// Returns whether `c` is empty or deleted.
static inline bool CWISS_IsEmptyOrDeleted(CWISS_ControlByte c) {
  return c < CWISS_kSentinel;
}

/// Asserts that `ctrl` points to a full control byte.
#define CWISS_AssertIsFull(ctrl)                                               \
  CWISS_CHECK((ctrl) != NULL && CWISS_IsFull(*(ctrl)),                         \
              "Invalid operation on iterator (%p/%d). The element might have " \
              "been erased, or the table might have rehashed.",                \
              (ctrl), (ctrl) ? *(ctrl) : -1)

/// Asserts that `ctrl` is either null OR points to a full control byte.
#define CWISS_AssertIsValid(ctrl)                                              \
  CWISS_CHECK((ctrl) == NULL || CWISS_IsFull(*(ctrl)),                         \
              "Invalid operation on iterator (%p/%d). The element might have " \
              "been erased, or the table might have rehashed.",                \
              (ctrl), (ctrl) ? *(ctrl) : -1)

/// Constructs a `BitMask` with the correct parameters for whichever
/// implementation of `CWISS_Group` is in use.
#define CWISS_Group_BitMask(x) \
  (CWISS_BitMask){(uint64_t)(x), CWISS_Group_kWidth, CWISS_Group_kShift};

// TODO(#4): Port this to NEON.
#if CWISS_HAVE_SSE2
// Reference guide for intrinsics used below:
//
// * __m128i: An XMM (128-bit) word.
//
// * _mm_setzero_si128: Returns a zero vector.
// * _mm_set1_epi8:     Returns a vector with the same i8 in each lane.
//
// * _mm_subs_epi8:    Saturating-subtracts two i8 vectors.
// * _mm_and_si128:    Ands two i128s together.
// * _mm_or_si128:     Ors two i128s together.
// * _mm_andnot_si128: And-nots two i128s together.
//
// * _mm_cmpeq_epi8: Component-wise compares two i8 vectors for equality,
//                   filling each lane with 0x00 or 0xff.
// * _mm_cmpgt_epi8: Same as above, but using > rather than ==.
//
// * _mm_loadu_si128:  Performs an unaligned load of an i128.
// * _mm_storeu_si128: Performs an unaligned store of a u128.
//
// * _mm_sign_epi8:     Retains, negates, or zeroes each i8 lane of the first
//                      argument if the corresponding lane of the second
//                      argument is positive, negative, or zero, respectively.
// * _mm_movemask_epi8: Selects the sign bit out of each i8 lane and produces a
//                      bitmask consisting of those bits.
// * _mm_shuffle_epi8:  Selects i8s from the first argument, using the low
//                      four bits of each i8 lane in the second argument as
//                      indices.
typedef __m128i CWISS_Group;
  #define CWISS_Group_kWidth ((size_t)16)
  #define CWISS_Group_kShift 0

// https://github.com/abseil/abseil-cpp/issues/209
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=87853
// _mm_cmpgt_epi8 is broken under GCC with -funsigned-char
// Work around this by using the portable implementation of Group
// when using -funsigned-char under GCC.
static inline CWISS_Group CWISS_mm_cmpgt_epi8_fixed(CWISS_Group a,
                                                    CWISS_Group b) {
  if (CWISS_IS_GCC && CHAR_MIN == 0) {  // std::is_unsigned_v<char>
    const CWISS_Group mask = _mm_set1_epi8(0x80);
    const CWISS_Group diff = _mm_subs_epi8(b, a);
    return _mm_cmpeq_epi8(_mm_and_si128(diff, mask), mask);
  }
  return _mm_cmpgt_epi8(a, b);
}

static inline CWISS_Group CWISS_Group_new(const CWISS_ControlByte* pos) {
  return _mm_loadu_si128((const CWISS_Group*)pos);
}

// Returns a bitmask representing the positions of slots that match hash.
static inline CWISS_BitMask CWISS_Group_Match(const CWISS_Group* self,
                                              CWISS_h2_t hash) {
  return CWISS_Group_BitMask(
      _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_set1_epi8(hash), *self)))
}

// Returns a bitmask representing the positions of empty slots.
static inline CWISS_BitMask CWISS_Group_MatchEmpty(const CWISS_Group* self) {
  #if CWISS_HAVE_SSSE3
  // This only works because ctrl_t::kEmpty is -128.
  return CWISS_Group_BitMask(_mm_movemask_epi8(_mm_sign_epi8(*self, *self)));
  #else
  return CWISS_Group_Match(self, CWISS_kEmpty);
  #endif
}

// Returns a bitmask representing the positions of empty or deleted slots.
static inline CWISS_BitMask CWISS_Group_MatchEmptyOrDeleted(
    const CWISS_Group* self) {
  CWISS_Group special = _mm_set1_epi8((uint8_t)CWISS_kSentinel);
  return CWISS_Group_BitMask(
      _mm_movemask_epi8(CWISS_mm_cmpgt_epi8_fixed(special, *self)));
}

// Returns the number of trailing empty or deleted elements in the group.
static inline uint32_t CWISS_Group_CountLeadingEmptyOrDeleted(
    const CWISS_Group* self) {
  CWISS_Group special = _mm_set1_epi8((uint8_t)CWISS_kSentinel);
  return CWISS_TrailingZeros((uint32_t)(
      _mm_movemask_epi8(CWISS_mm_cmpgt_epi8_fixed(special, *self)) + 1));
}

static inline void CWISS_Group_ConvertSpecialToEmptyAndFullToDeleted(
    const CWISS_Group* self, CWISS_ControlByte* dst) {
  CWISS_Group msbs = _mm_set1_epi8((char)-128);
  CWISS_Group x126 = _mm_set1_epi8(126);
  #if CWISS_HAVE_SSSE3
  CWISS_Group res = _mm_or_si128(_mm_shuffle_epi8(x126, *self), msbs);
  #else
  CWISS_Group zero = _mm_setzero_si128();
  CWISS_Group special_mask = CWISS_mm_cmpgt_epi8_fixed(zero, *self);
  CWISS_Group res = _mm_or_si128(msbs, _mm_andnot_si128(special_mask, x126));
  #endif
  _mm_storeu_si128((CWISS_Group*)dst, res);
}
#else  // CWISS_HAVE_SSE2
typedef uint64_t CWISS_Group;
  #define CWISS_Group_kWidth ((size_t)8)
  #define CWISS_Group_kShift 3

// NOTE: Endian-hostile.
static inline CWISS_Group CWISS_Group_new(const CWISS_ControlByte* pos) {
  CWISS_Group val;
  memcpy(&val, pos, sizeof(val));
  return val;
}

static inline CWISS_BitMask CWISS_Group_Match(const CWISS_Group* self,
                                              CWISS_h2_t hash) {
  // For the technique, see:
  // http://graphics.stanford.edu/~seander/bithacks.html##ValueInWord
  // (Determine if a word has a byte equal to n).
  //
  // Caveat: there are false positives but:
  // - they only occur if there is a real match
  // - they never occur on ctrl_t::kEmpty, ctrl_t::kDeleted, ctrl_t::kSentinel
  // - they will be handled gracefully by subsequent checks in code
  //
  // Example:
  //   v = 0x1716151413121110
  //   hash = 0x12
  //   retval = (v - lsbs) & ~v & msbs = 0x0000000080800000
  uint64_t msbs = 0x8080808080808080ULL;
  uint64_t lsbs = 0x0101010101010101ULL;
  uint64_t x = *self ^ (lsbs * hash);
  return CWISS_Group_BitMask((x - lsbs) & ~x & msbs);
}

static inline CWISS_BitMask CWISS_Group_MatchEmpty(const CWISS_Group* self) {
  uint64_t msbs = 0x8080808080808080ULL;
  return CWISS_Group_BitMask((*self & (~*self << 6)) & msbs);
}

static inline CWISS_BitMask CWISS_Group_MatchEmptyOrDeleted(
    const CWISS_Group* self) {
  uint64_t msbs = 0x8080808080808080ULL;
  return CWISS_Group_BitMask((*self & (~*self << 7)) & msbs);
}

static inline uint32_t CWISS_Group_CountLeadingEmptyOrDeleted(
    const CWISS_Group* self) {
  uint64_t gaps = 0x00FEFEFEFEFEFEFEULL;
  return (CWISS_TrailingZeros(((~*self & (*self >> 7)) | gaps) + 1) + 7) >> 3;
}

static inline void CWISS_Group_ConvertSpecialToEmptyAndFullToDeleted(
    const CWISS_Group* self, CWISS_ControlByte* dst) {
  uint64_t msbs = 0x8080808080808080ULL;
  uint64_t lsbs = 0x0101010101010101ULL;
  uint64_t x = *self & msbs;
  uint64_t res = (~x + (x >> 7)) & ~lsbs;
  memcpy(dst, &res, sizeof(res));
}
#endif  // CWISS_HAVE_SSE2

CWISS_END_EXTERN
CWISS_END

#endif  // CWISSTABLE_INTERNAL_CTRL_H_
