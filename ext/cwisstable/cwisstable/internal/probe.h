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

#ifndef CWISSTABLE_INTERNAL_PROBE_H_
#define CWISSTABLE_INTERNAL_PROBE_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "cwisstable/internal/base.h"
#include "cwisstable/internal/bits.h"
#include "cwisstable/internal/capacity.h"
#include "cwisstable/internal/control_byte.h"

/// Table probing functions.
///
/// "Probing" refers to the process of trying to find the matching entry for a
/// given lookup by repeatedly searching for values throughout the table.

CWISS_BEGIN
CWISS_BEGIN_EXTERN

/// The state for a probe sequence.
///
/// Currently, the sequence is a triangular progression of the form
/// ```
/// p(i) := kWidth/2 * (i^2 - i) + hash (mod mask + 1)
/// ```
///
/// The use of `kWidth` ensures that each probe step does not overlap groups;
/// the sequence effectively outputs the addresses of *groups* (although not
/// necessarily aligned to any boundary). The `CWISS_Group` machinery allows us
/// to check an entire group with minimal branching.
///
/// Wrapping around at `mask + 1` is important, but not for the obvious reason.
/// As described in capacity.h, the first few entries of the control byte array
/// is mirrored at the end of the array, which `CWISS_Group` will find and use
/// for selecting candidates. However, when those candidates' slots are
/// actually inspected, there are no corresponding slots for the cloned bytes,
/// so we need to make sure we've treated those offsets as "wrapping around".
typedef struct {
  size_t mask_;
  size_t offset_;
  size_t index_;
} CWISS_ProbeSeq;

/// Creates a new probe sequence using `hash` as the initial value of the
/// sequence and `mask` (usually the capacity of the table) as the mask to
/// apply to each value in the progression.
static inline CWISS_ProbeSeq CWISS_ProbeSeq_new(size_t hash, size_t mask) {
  return (CWISS_ProbeSeq){
      .mask_ = mask,
      .offset_ = hash & mask,
  };
}

/// Returns the slot `i` indices ahead of `self` within the bounds expressed by
/// `mask`.
static inline size_t CWISS_ProbeSeq_offset(const CWISS_ProbeSeq* self,
                                           size_t i) {
  return (self->offset_ + i) & self->mask_;
}

/// Advances the sequence; the value can be obtained by calling
/// `CWISS_ProbeSeq_offset()` or inspecting `offset_`.
static inline void CWISS_ProbeSeq_next(CWISS_ProbeSeq* self) {
  self->index_ += CWISS_Group_kWidth;
  self->offset_ += self->index_;
  self->offset_ &= self->mask_;
}

/// Begins a probing operation on `ctrl`, using `hash`.
static inline CWISS_ProbeSeq CWISS_ProbeSeq_Start(const CWISS_ControlByte* ctrl,
                                                  size_t hash,
                                                  size_t capacity) {
  return CWISS_ProbeSeq_new(CWISS_H1(hash, ctrl), capacity);
}

// The return value of `CWISS_FindFirstNonFull()`.
typedef struct {
  size_t offset;
  size_t probe_length;
} CWISS_FindInfo;

/// Probes an array of control bits using a probe sequence derived from `hash`,
/// and returns the offset corresponding to the first deleted or empty slot.
///
/// Behavior when the entire table is full is undefined.
///
/// NOTE: this function must work with tables having both empty and deleted
/// slots in the same group. Such tables appear during
/// `CWISS_RawTable_DropDeletesWithoutResize()`.
static inline CWISS_FindInfo CWISS_FindFirstNonFull(
    const CWISS_ControlByte* ctrl, size_t hash, size_t capacity) {
  CWISS_ProbeSeq seq = CWISS_ProbeSeq_Start(ctrl, hash, capacity);
  while (true) {
    CWISS_Group g = CWISS_Group_new(ctrl + seq.offset_);
    CWISS_BitMask mask = CWISS_Group_MatchEmptyOrDeleted(&g);
    if (mask.mask) {
#ifndef NDEBUG
      // We want to add entropy even when ASLR is not enabled.
      // In debug build we will randomly insert in either the front or back of
      // the group.
      // TODO(kfm,sbenza): revisit after we do unconditional mixing
      if (!CWISS_IsSmall(capacity) && CWISS_ShouldInsertBackwards(hash, ctrl)) {
        return (CWISS_FindInfo){
            CWISS_ProbeSeq_offset(&seq, CWISS_BitMask_HighestBitSet(&mask)),
            seq.index_};
      }
#endif
      return (CWISS_FindInfo){
          CWISS_ProbeSeq_offset(&seq, CWISS_BitMask_TrailingZeros(&mask)),
          seq.index_};
    }
    CWISS_ProbeSeq_next(&seq);
    CWISS_DCHECK(seq.index_ <= capacity, "full table!");
  }
}

CWISS_END_EXTERN
CWISS_END

#endif  // CWISSTABLE_INTERNAL_PROBE_H_