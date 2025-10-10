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

#ifndef CWISSTABLE_INTERNAL_CAPACITY_H_
#define CWISSTABLE_INTERNAL_CAPACITY_H_

#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cwisstable/internal/base.h"
#include "cwisstable/internal/control_byte.h"

/// Capacity, load factor, and allocation size computations for a SwissTable.
///
/// A SwissTable's backing array consists of control bytes followed by slots
/// that may or may not contain objects.
///
/// The layout of the backing array, for `capacity` slots, is thus, as a
/// pseudo-struct:
/// ```
/// struct CWISS_BackingArray {
///   // Control bytes for the "real" slots.
///   CWISS_ControlByte ctrl[capacity];
///   // Always `CWISS_kSentinel`. This is used by iterators to find when to
///   // stop and serves no other purpose.
///   CWISS_ControlByte sentinel;
///   // A copy of the first `kWidth - 1` elements of `ctrl`. This is used so
///   // that if a probe sequence picks a value near the end of `ctrl`,
///   // `CWISS_Group` will have valid control bytes to look at.
///   //
///   // As an interesting special-case, such probe windows will never choose
///   // the zeroth slot as a candidate, because they will see `kSentinel`
///   // instead of the correct H2 value.
///   CWISS_ControlByte clones[kWidth - 1];
///   // Alignment padding equal to `alignof(slot_type)`.
///   char padding_;
///   // The actual slot data.
///   char slots[capacity * sizeof(slot_type)];
/// };
/// ```
///
/// The length of this array is computed by `CWISS_AllocSize()`.

CWISS_BEGIN
CWISS_BEGIN_EXTERN

/// Returns he number of "cloned control bytes".
///
/// This is the number of control bytes that are present both at the beginning
/// of the control byte array and at the end, such that we can create a
/// `CWISS_Group_kWidth`-width probe window starting from any control byte.
static inline size_t CWISS_NumClonedBytes(void) {
  return CWISS_Group_kWidth - 1;
}

/// Returns whether `n` is a valid capacity (i.e., number of slots).
///
/// A valid capacity is a non-zero integer `2^m - 1`.
static inline bool CWISS_IsValidCapacity(size_t n) {
  return ((n + 1) & n) == 0 && n > 0;
}

/// Returns some per-call entropy.
///
/// Currently, the entropy is produced by XOR'ing the address of a (preferably
/// thread-local) value with a perpetually-incrementing value.
static inline size_t RandomSeed(void) {
#ifdef CWISS_THREAD_LOCAL
  static CWISS_THREAD_LOCAL size_t counter;
  size_t value = ++counter;
#else
  static volatile CWISS_ATOMIC_T(size_t) counter;
  size_t value = CWISS_ATOMIC_INC(counter);
#endif
  return value ^ ((size_t)&counter);
}

/// Mixes a randomly generated per-process seed with `hash` and `ctrl` to
/// randomize insertion order within groups.
CWISS_INLINE_NEVER static bool CWISS_ShouldInsertBackwards(
    size_t hash, const CWISS_ControlByte* ctrl) {
  // To avoid problems with weak hashes and single bit tests, we use % 13.
  // TODO(kfm,sbenza): revisit after we do unconditional mixing
  return (CWISS_H1(hash, ctrl) ^ RandomSeed()) % 13 > 6;
}

/// Applies the following mapping to every byte in the control array:
///   * kDeleted -> kEmpty
///   * kEmpty -> kEmpty
///   * _ -> kDeleted
///
/// Preconditions: `CWISS_IsValidCapacity(capacity)`,
/// `ctrl[capacity]` == `kSentinel`, `ctrl[i] != kSentinel for i < capacity`.
CWISS_INLINE_NEVER static void CWISS_ConvertDeletedToEmptyAndFullToDeleted(
    CWISS_ControlByte* ctrl, size_t capacity) {
  CWISS_DCHECK(ctrl[capacity] == CWISS_kSentinel, "bad ctrl value at %zu: %02x",
               capacity, ctrl[capacity]);
  CWISS_DCHECK(CWISS_IsValidCapacity(capacity), "invalid capacity: %zu",
               capacity);

  for (CWISS_ControlByte* pos = ctrl; pos < ctrl + capacity;
       pos += CWISS_Group_kWidth) {
    CWISS_Group g = CWISS_Group_new(pos);
    CWISS_Group_ConvertSpecialToEmptyAndFullToDeleted(&g, pos);
  }
  // Copy the cloned ctrl bytes.
  memcpy(ctrl + capacity + 1, ctrl, CWISS_NumClonedBytes());
  ctrl[capacity] = CWISS_kSentinel;
}

/// Sets `ctrl` to `{kEmpty, ..., kEmpty, kSentinel}`, marking the entire
/// array as deleted.
static inline void CWISS_ResetCtrl(size_t capacity, CWISS_ControlByte* ctrl,
                                   const void* slots, size_t slot_size) {
  memset(ctrl, CWISS_kEmpty, capacity + 1 + CWISS_NumClonedBytes());
  ctrl[capacity] = CWISS_kSentinel;
  CWISS_PoisonMemory(slots, slot_size * capacity);
}

/// Sets `ctrl[i]` to `h`.
///
/// Unlike setting it directly, this function will perform bounds checks and
/// mirror the value to the cloned tail if necessary.
static inline void CWISS_SetCtrl(size_t i, CWISS_ControlByte h, size_t capacity,
                                 CWISS_ControlByte* ctrl, const void* slots,
                                 size_t slot_size) {
  CWISS_DCHECK(i < capacity, "CWISS_SetCtrl out-of-bounds: %zu >= %zu", i,
               capacity);

  const char* slot = ((const char*)slots) + i * slot_size;
  if (CWISS_IsFull(h)) {
    CWISS_UnpoisonMemory(slot, slot_size);
  } else {
    CWISS_PoisonMemory(slot, slot_size);
  }

  // This is intentionally branchless. If `i < kWidth`, it will write to the
  // cloned bytes as well as the "real" byte; otherwise, it will store `h`
  // twice.
  size_t mirrored_i = ((i - CWISS_NumClonedBytes()) & capacity) +
                      (CWISS_NumClonedBytes() & capacity);
  ctrl[i] = h;
  ctrl[mirrored_i] = h;
}

/// Converts `n` into the next valid capacity, per `CWISS_IsValidCapacity`.
static inline size_t CWISS_NormalizeCapacity(size_t n) {
  return n ? SIZE_MAX >> CWISS_LeadingZeros(n) : 1;
}

// General notes on capacity/growth methods below:
// - We use 7/8th as maximum load factor. For 16-wide groups, that gives an
//   average of two empty slots per group.
// - For (capacity+1) >= Group::kWidth, growth is 7/8*capacity.
// - For (capacity+1) < Group::kWidth, growth == capacity. In this case, we
//   never need to probe (the whole table fits in one group) so we don't need a
//   load factor less than 1.

/// Given `capacity`, applies the load factor; i.e., it returns the maximum
/// number of values we should put into the table before a rehash.
static inline size_t CWISS_CapacityToGrowth(size_t capacity) {
  CWISS_DCHECK(CWISS_IsValidCapacity(capacity), "invalid capacity: %zu",
               capacity);
  // `capacity*7/8`
  if (CWISS_Group_kWidth == 8 && capacity == 7) {
    // x-x/8 does not work when x==7.
    return 6;
  }
  return capacity - capacity / 8;
}

/// Given `growth`, "unapplies" the load factor to find how large the capacity
/// should be to stay within the load factor.
///
/// This might not be a valid capacity and `CWISS_NormalizeCapacity()` may be
/// necessary.
static inline size_t CWISS_GrowthToLowerboundCapacity(size_t growth) {
  // `growth*8/7`
  if (CWISS_Group_kWidth == 8 && growth == 7) {
    // x+(x-1)/7 does not work when x==7.
    return 8;
  }
  return growth + (size_t)((((int64_t)growth) - 1) / 7);
}

// The allocated block consists of `capacity + 1 + NumClonedBytes()` control
// bytes followed by `capacity` slots, which must be aligned to `slot_align`.
// SlotOffset returns the offset of the slots into the allocated block.

/// Given the capacity of a table, computes the offset (from the start of the
/// backing allocation) at which the slots begin.
static inline size_t CWISS_SlotOffset(size_t capacity, size_t slot_align) {
  CWISS_DCHECK(CWISS_IsValidCapacity(capacity), "invalid capacity: %zu",
               capacity);
  const size_t num_control_bytes = capacity + 1 + CWISS_NumClonedBytes();
  return (num_control_bytes + slot_align - 1) & (~slot_align + 1);
}

/// Given the capacity of a table, computes the total size of the backing
/// array.
static inline size_t CWISS_AllocSize(size_t capacity, size_t slot_size,
                                     size_t slot_align) {
  return CWISS_SlotOffset(capacity, slot_align) + capacity * slot_size;
}

/// Whether a table is "small". A small table fits entirely into a probing
/// group, i.e., has a capacity equal to the size of a `CWISS_Group`.
///
/// In small mode we are able to use the whole capacity. The extra control
/// bytes give us at least one "empty" control byte to stop the iteration.
/// This is important to make 1 a valid capacity.
///
/// In small mode only the first `capacity` control bytes after the sentinel
/// are valid. The rest contain dummy ctrl_t::kEmpty values that do not
/// represent a real slot. This is important to take into account on
/// `CWISS_FindFirstNonFull()`, where we never try
/// `CWISS_ShouldInsertBackwards()` for small tables.
static inline bool CWISS_IsSmall(size_t capacity) {
  return capacity < CWISS_Group_kWidth - 1;
}

CWISS_END_EXTERN
CWISS_END

#endif  // CWISSTABLE_INTERNAL_CAPACITY_H_