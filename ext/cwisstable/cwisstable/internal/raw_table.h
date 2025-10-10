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

#ifndef CWISSTABLE_INTERNAL_RAW_TABLE_H_
#define CWISSTABLE_INTERNAL_RAW_TABLE_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "cwisstable/internal/base.h"
#include "cwisstable/internal/bits.h"
#include "cwisstable/internal/capacity.h"
#include "cwisstable/internal/control_byte.h"
#include "cwisstable/internal/probe.h"
#include "cwisstable/policy.h"

/// The SwissTable implementation.
///
/// `CWISS_RawTable` is the core data structure that all SwissTables wrap.
///
/// All functions in this header take a `const CWISS_Policy*`, which describes
/// how to manipulate the elements in a table. The same pointer (i.e., same
/// address and provenance) passed to the function that created the
/// `CWISS_RawTable` MUST be passed to all subsequent function calls, and it
/// must not be mutated at any point between those calls. Failure to adhere to
/// these requirements is UB.
///
/// It is STRONGLY recommended that this pointer point to a const global.

CWISS_BEGIN
CWISS_BEGIN_EXTERN

/// A SwissTable.
///
/// This is absl::container_internal::raw_hash_set in Abseil.
typedef struct {
  /// The control bytes (and, also, a pointer to the base of the backing array).
  ///
  /// This contains `capacity_ + 1 + CWISS_NumClonedBytes()` entries.
  CWISS_ControlByte* ctrl_;
  /// The beginning of the slots, located at `CWISS_SlotOffset()` bytes after
  /// `ctrl_`. May be null for empty tables.
  char* slots_;
  /// The number of filled slots.
  size_t size_;
  /// The total number of available slots.
  size_t capacity_;
  /// The number of slots we can still fill before a rehash. See
  /// `CWISS_CapacityToGrowth()`.
  size_t growth_left_;
} CWISS_RawTable;

/// Prints full details about the internal state of `self` to `stderr`.
static inline void CWISS_RawTable_dump(const CWISS_Policy* policy,
                                       const CWISS_RawTable* self) {
  fprintf(stderr, "ptr: %p, len: %zu, cap: %zu, growth: %zu\n", self->ctrl_,
          self->size_, self->capacity_, self->growth_left_);
  if (self->capacity_ == 0) {
    return;
  }

  size_t ctrl_bytes = self->capacity_ + CWISS_NumClonedBytes();
  for (size_t i = 0; i <= ctrl_bytes; ++i) {
    fprintf(stderr, "[%4zu] %p / ", i, &self->ctrl_[i]);
    switch (self->ctrl_[i]) {
      case CWISS_kSentinel:
        fprintf(stderr, "kSentinel: //\n");
        continue;
      case CWISS_kEmpty:
        fprintf(stderr, "   kEmpty");
        break;
      case CWISS_kDeleted:
        fprintf(stderr, " kDeleted");
        break;
      default:
        fprintf(stderr, " H2(0x%02x)", self->ctrl_[i]);
        break;
    }

    if (i >= self->capacity_) {
      fprintf(stderr, ": <mirrored>\n");
      continue;
    }

    char* slot = self->slots_ + i * policy->slot->size;
    fprintf(stderr, ": %p /", slot);
    for (size_t j = 0; j < policy->slot->size; ++j) {
      fprintf(stderr, " %02x", (unsigned char)slot[j]);
    }
    char* elem = (char*)policy->slot->get(slot);
    if (elem != slot && CWISS_IsFull(self->ctrl_[i])) {
      fprintf(stderr, " ->");
      for (size_t j = 0; j < policy->obj->size; ++j) {
        fprintf(stderr, " %02x", (unsigned char)elem[j]);
      }
    }
    fprintf(stderr, "\n");
  }
}

/// An iterator into a SwissTable.
///
/// Unlike a C++ iterator, there is no "end" to compare to. Instead,
/// `CWISS_RawIter_get()` will yield a null pointer once the iterator is
/// exhausted.
///
/// Invariants:
/// - `ctrl_` and `slot_` are always in sync (i.e., the pointed to control byte
///   corresponds to the pointed to slot), or both are null. `set_` may be null
///   in the latter case.
/// - `ctrl_` always points to a full slot.
typedef struct {
  CWISS_RawTable* set_;
  CWISS_ControlByte* ctrl_;
  char* slot_;
} CWISS_RawIter;

/// Fixes up `ctrl_` to point to a full by advancing it and `slot_` until they
/// reach one.
///
/// If a sentinel is reached, we null both of them out instead.
static inline void CWISS_RawIter_SkipEmptyOrDeleted(const CWISS_Policy* policy,
                                                    CWISS_RawIter* self) {
  while (CWISS_IsEmptyOrDeleted(*self->ctrl_)) {
    CWISS_Group g = CWISS_Group_new(self->ctrl_);
    uint32_t shift = CWISS_Group_CountLeadingEmptyOrDeleted(&g);
    self->ctrl_ += shift;
    self->slot_ += shift * policy->slot->size;
  }

  // Not sure why this is a branch rather than a cmov; Abseil uses a branch.
  if (CWISS_UNLIKELY(*self->ctrl_ == CWISS_kSentinel)) {
    self->ctrl_ = NULL;
    self->slot_ = NULL;
  }
}

/// Creates a valid iterator starting at the `index`th slot.
static inline CWISS_RawIter CWISS_RawTable_iter_at(const CWISS_Policy* policy,
                                                   CWISS_RawTable* self,
                                                   size_t index) {
  CWISS_RawIter iter = {
      self,
      self->ctrl_ + index,
      self->slots_ + index * policy->slot->size,
  };
  CWISS_RawIter_SkipEmptyOrDeleted(policy, &iter);
  CWISS_AssertIsValid(iter.ctrl_);
  return iter;
}

/// Creates an iterator for `self`.
static inline CWISS_RawIter CWISS_RawTable_iter(const CWISS_Policy* policy,
                                                CWISS_RawTable* self) {
  return CWISS_RawTable_iter_at(policy, self, 0);
}

/// Creates a valid iterator starting at the `index`th slot, accepting a `const`
/// pointer instead.
static inline CWISS_RawIter CWISS_RawTable_citer_at(const CWISS_Policy* policy,
                                                    const CWISS_RawTable* self,
                                                    size_t index) {
  return CWISS_RawTable_iter_at(policy, (CWISS_RawTable*)self, index);
}

/// Creates an iterator for `self`, accepting a `const` pointer instead.
static inline CWISS_RawIter CWISS_RawTable_citer(const CWISS_Policy* policy,
                                                 const CWISS_RawTable* self) {
  return CWISS_RawTable_iter(policy, (CWISS_RawTable*)self);
}

/// Returns a pointer into the currently pointed-to slot (*not* to the slot
/// itself, but rather its contents).
///
/// Returns null if the iterator has been exhausted.
static inline void* CWISS_RawIter_get(const CWISS_Policy* policy,
                                      const CWISS_RawIter* self) {
  CWISS_AssertIsValid(self->ctrl_);
  if (self->slot_ == NULL) {
    return NULL;
  }

  return policy->slot->get(self->slot_);
}

/// Advances the iterator and returns the result of `CWISS_RawIter_get()`.
///
/// Calling on an empty iterator is UB.
static inline void* CWISS_RawIter_next(const CWISS_Policy* policy,
                                       CWISS_RawIter* self) {
  CWISS_AssertIsFull(self->ctrl_);
  ++self->ctrl_;
  self->slot_ += policy->slot->size;

  CWISS_RawIter_SkipEmptyOrDeleted(policy, self);
  return CWISS_RawIter_get(policy, self);
}

/// Erases, but does not destroy, the value pointed to by `it`.
static inline void CWISS_RawTable_EraseMetaOnly(const CWISS_Policy* policy,
                                                CWISS_RawIter it) {
  CWISS_DCHECK(CWISS_IsFull(*it.ctrl_), "erasing a dangling iterator");
  --it.set_->size_;
  const size_t index = (size_t)(it.ctrl_ - it.set_->ctrl_);
  const size_t index_before = (index - CWISS_Group_kWidth) & it.set_->capacity_;
  CWISS_Group g_after = CWISS_Group_new(it.ctrl_);
  CWISS_BitMask empty_after = CWISS_Group_MatchEmpty(&g_after);
  CWISS_Group g_before = CWISS_Group_new(it.set_->ctrl_ + index_before);
  CWISS_BitMask empty_before = CWISS_Group_MatchEmpty(&g_before);

  // We count how many consecutive non empties we have to the right and to the
  // left of `it`. If the sum is >= kWidth then there is at least one probe
  // window that might have seen a full group.
  bool was_never_full =
      empty_before.mask && empty_after.mask &&
      (size_t)(CWISS_BitMask_TrailingZeros(&empty_after) +
               CWISS_BitMask_LeadingZeros(&empty_before)) < CWISS_Group_kWidth;

  CWISS_SetCtrl(index, was_never_full ? CWISS_kEmpty : CWISS_kDeleted,
                it.set_->capacity_, it.set_->ctrl_, it.set_->slots_,
                policy->slot->size);
  it.set_->growth_left_ += was_never_full;
  // infoz().RecordErase();
}

/// Computes a lower bound for the expected available growth and applies it to
/// `self_`.
static inline void CWISS_RawTable_ResetGrowthLeft(const CWISS_Policy* policy,
                                                  CWISS_RawTable* self) {
  self->growth_left_ = CWISS_CapacityToGrowth(self->capacity_) - self->size_;
}

/// Allocates a backing array for `self` and initializes its control bits. This
/// reads `capacity_` and updates all other fields based on the result of the
/// allocation.
///
/// This does not free the currently held array; `capacity_` must be nonzero.
static inline void CWISS_RawTable_InitializeSlots(const CWISS_Policy* policy,
                                                  CWISS_RawTable* self) {
  CWISS_DCHECK(self->capacity_, "capacity should be nonzero");
  // Folks with custom allocators often make unwarranted assumptions about the
  // behavior of their classes vis-a-vis trivial destructability and what
  // calls they will or wont make.  Avoid sampling for people with custom
  // allocators to get us out of this mess.  This is not a hard guarantee but
  // a workaround while we plan the exact guarantee we want to provide.
  //
  // People are often sloppy with the exact type of their allocator (sometimes
  // it has an extra const or is missing the pair, but rebinds made it work
  // anyway).  To avoid the ambiguity, we work off SlotAlloc which we have
  // bound more carefully.
  //
  // NOTE(mcyoung): Not relevant in C but kept in case we decide to do custom
  // alloc.
  /*if (std::is_same<SlotAlloc, std::allocator<slot_type>>::value &&
      slots_ == nullptr) {
    infoz() = Sample(sizeof(slot_type));
  }*/

  char* mem =
      (char*)  // Cast for C++.
      policy->alloc->alloc(CWISS_AllocSize(self->capacity_, policy->slot->size,
                                           policy->slot->align),
                           policy->slot->align);

  self->ctrl_ = (CWISS_ControlByte*)mem;
  self->slots_ = mem + CWISS_SlotOffset(self->capacity_, policy->slot->align);
  CWISS_ResetCtrl(self->capacity_, self->ctrl_, self->slots_,
                  policy->slot->size);
  CWISS_RawTable_ResetGrowthLeft(policy, self);

  // infoz().RecordStorageChanged(size_, capacity_);
}

/// Destroys all slots in the backing array, frees the backing array, and clears
/// all top-level book-keeping data.
static inline void CWISS_RawTable_DestroySlots(const CWISS_Policy* policy,
                                               CWISS_RawTable* self) {
  if (!self->capacity_) return;

  if (policy->slot->del != NULL) {
    for (size_t i = 0; i != self->capacity_; ++i) {
      if (CWISS_IsFull(self->ctrl_[i])) {
        policy->slot->del(self->slots_ + i * policy->slot->size);
      }
    }
  }

  policy->alloc->free(
      self->ctrl_,
      CWISS_AllocSize(self->capacity_, policy->slot->size, policy->slot->align),
      policy->slot->align);
  self->ctrl_ = CWISS_EmptyGroup();
  self->slots_ = NULL;
  self->size_ = 0;
  self->capacity_ = 0;
  self->growth_left_ = 0;
}

/// Grows the table to the given capacity, triggering a rehash.
static inline void CWISS_RawTable_Resize(const CWISS_Policy* policy,
                                         CWISS_RawTable* self,
                                         size_t new_capacity) {
  CWISS_DCHECK(CWISS_IsValidCapacity(new_capacity), "invalid capacity: %zu",
               new_capacity);

  CWISS_ControlByte* old_ctrl = self->ctrl_;
  char* old_slots = self->slots_;
  const size_t old_capacity = self->capacity_;
  self->capacity_ = new_capacity;
  CWISS_RawTable_InitializeSlots(policy, self);

  size_t total_probe_length = 0;
  for (size_t i = 0; i != old_capacity; ++i) {
    if (CWISS_IsFull(old_ctrl[i])) {
      size_t hash = policy->key->hash(
          policy->slot->get(old_slots + i * policy->slot->size));
      CWISS_FindInfo target =
          CWISS_FindFirstNonFull(self->ctrl_, hash, self->capacity_);
      size_t new_i = target.offset;
      total_probe_length += target.probe_length;
      CWISS_SetCtrl(new_i, CWISS_H2(hash), self->capacity_, self->ctrl_,
                    self->slots_, policy->slot->size);
      policy->slot->transfer(self->slots_ + new_i * policy->slot->size,
                             old_slots + i * policy->slot->size);
    }
  }
  if (old_capacity) {
    CWISS_UnpoisonMemory(old_slots, policy->slot->size * old_capacity);
    policy->alloc->free(
        old_ctrl,
        CWISS_AllocSize(old_capacity, policy->slot->size, policy->slot->align),
        policy->slot->align);
  }
  // infoz().RecordRehash(total_probe_length);
}

/// Prunes control bits to remove as many tombstones as possible.
///
/// See the comment on `CWISS_RawTable_rehash_and_grow_if_necessary()`.
CWISS_INLINE_NEVER
static void CWISS_RawTable_DropDeletesWithoutResize(const CWISS_Policy* policy,
                                                    CWISS_RawTable* self) {
  CWISS_DCHECK(CWISS_IsValidCapacity(self->capacity_), "invalid capacity: %zu",
               self->capacity_);
  CWISS_DCHECK(!CWISS_IsSmall(self->capacity_),
               "unexpected small capacity: %zu", self->capacity_);
  // Algorithm:
  // - mark all DELETED slots as EMPTY
  // - mark all FULL slots as DELETED
  // - for each slot marked as DELETED
  //     hash = Hash(element)
  //     target = find_first_non_full(hash)
  //     if target is in the same group
  //       mark slot as FULL
  //     else if target is EMPTY
  //       transfer element to target
  //       mark slot as EMPTY
  //       mark target as FULL
  //     else if target is DELETED
  //       swap current element with target element
  //       mark target as FULL
  //       repeat procedure for current slot with moved from element (target)
  CWISS_ConvertDeletedToEmptyAndFullToDeleted(self->ctrl_, self->capacity_);
  size_t total_probe_length = 0;
  // Unfortunately because we do not know this size statically, we need to take
  // a trip to the allocator. Alternatively we could use a variable length
  // alloca...
  void* slot = policy->alloc->alloc(policy->slot->size, policy->slot->align);

  for (size_t i = 0; i != self->capacity_; ++i) {
    if (!CWISS_IsDeleted(self->ctrl_[i])) continue;

    char* old_slot = self->slots_ + i * policy->slot->size;
    size_t hash = policy->key->hash(policy->slot->get(old_slot));

    const CWISS_FindInfo target =
        CWISS_FindFirstNonFull(self->ctrl_, hash, self->capacity_);
    const size_t new_i = target.offset;
    total_probe_length += target.probe_length;

    char* new_slot = self->slots_ + new_i * policy->slot->size;

    // Verify if the old and new i fall within the same group wrt the hash.
    // If they do, we don't need to move the object as it falls already in the
    // best probe we can.
    const size_t probe_offset =
        CWISS_ProbeSeq_Start(self->ctrl_, hash, self->capacity_).offset_;
#define CWISS_ProbeIndex(pos_) \
  (((pos_ - probe_offset) & self->capacity_) / CWISS_Group_kWidth)

    // Element doesn't move.
    if (CWISS_LIKELY(CWISS_ProbeIndex(new_i) == CWISS_ProbeIndex(i))) {
      CWISS_SetCtrl(i, CWISS_H2(hash), self->capacity_, self->ctrl_,
                    self->slots_, policy->slot->size);
      continue;
    }
    if (CWISS_IsEmpty(self->ctrl_[new_i])) {
      // Transfer element to the empty spot.
      // SetCtrl poisons/unpoisons the slots so we have to call it at the
      // right time.
      CWISS_SetCtrl(new_i, CWISS_H2(hash), self->capacity_, self->ctrl_,
                    self->slots_, policy->slot->size);
      policy->slot->transfer(new_slot, old_slot);
      CWISS_SetCtrl(i, CWISS_kEmpty, self->capacity_, self->ctrl_, self->slots_,
                    policy->slot->size);
    } else {
      CWISS_DCHECK(CWISS_IsDeleted(self->ctrl_[new_i]),
                   "bad ctrl value at %zu: %02x", new_i, self->ctrl_[new_i]);
      CWISS_SetCtrl(new_i, CWISS_H2(hash), self->capacity_, self->ctrl_,
                    self->slots_, policy->slot->size);
      // Until we are done rehashing, DELETED marks previously FULL slots.
      // Swap i and new_i elements.

      policy->slot->transfer(slot, old_slot);
      policy->slot->transfer(old_slot, new_slot);
      policy->slot->transfer(new_slot, slot);
      --i;  // repeat
    }
#undef CWISS_ProbeSeq_Start_index
  }
  CWISS_RawTable_ResetGrowthLeft(policy, self);
  policy->alloc->free(slot, policy->slot->size, policy->slot->align);
  // infoz().RecordRehash(total_probe_length);
}

/// Called whenever the table *might* need to conditionally grow.
///
/// This function is an optimization opportunity to perform a rehash even when
/// growth is unnecessary, because vacating tombstones is beneficial for
/// performance in the long-run.
static inline void CWISS_RawTable_rehash_and_grow_if_necessary(
    const CWISS_Policy* policy, CWISS_RawTable* self) {
  if (self->capacity_ == 0) {
    CWISS_RawTable_Resize(policy, self, 1);
  } else if (self->capacity_ > CWISS_Group_kWidth &&
             // Do these calculations in 64-bit to avoid overflow.
             self->size_ * UINT64_C(32) <= self->capacity_ * UINT64_C(25)) {
    // Squash DELETED without growing if there is enough capacity.
    //
    // Rehash in place if the current size is <= 25/32 of capacity_.
    // Rationale for such a high factor: 1) drop_deletes_without_resize() is
    // faster than resize, and 2) it takes quite a bit of work to add
    // tombstones.  In the worst case, seems to take approximately 4
    // insert/erase pairs to create a single tombstone and so if we are
    // rehashing because of tombstones, we can afford to rehash-in-place as
    // long as we are reclaiming at least 1/8 the capacity without doing more
    // than 2X the work.  (Where "work" is defined to be size() for rehashing
    // or rehashing in place, and 1 for an insert or erase.)  But rehashing in
    // place is faster per operation than inserting or even doubling the size
    // of the table, so we actually afford to reclaim even less space from a
    // resize-in-place.  The decision is to rehash in place if we can reclaim
    // at about 1/8th of the usable capacity (specifically 3/28 of the
    // capacity) which means that the total cost of rehashing will be a small
    // fraction of the total work.
    //
    // Here is output of an experiment using the BM_CacheInSteadyState
    // benchmark running the old case (where we rehash-in-place only if we can
    // reclaim at least 7/16*capacity_) vs. this code (which rehashes in place
    // if we can recover 3/32*capacity_).
    //
    // Note that although in the worst-case number of rehashes jumped up from
    // 15 to 190, but the number of operations per second is almost the same.
    //
    // Abridged output of running BM_CacheInSteadyState benchmark from
    // raw_hash_set_benchmark.   N is the number of insert/erase operations.
    //
    //      | OLD (recover >= 7/16        | NEW (recover >= 3/32)
    // size |    N/s LoadFactor NRehashes |    N/s LoadFactor NRehashes
    //  448 | 145284       0.44        18 | 140118       0.44        19
    //  493 | 152546       0.24        11 | 151417       0.48        28
    //  538 | 151439       0.26        11 | 151152       0.53        38
    //  583 | 151765       0.28        11 | 150572       0.57        50
    //  628 | 150241       0.31        11 | 150853       0.61        66
    //  672 | 149602       0.33        12 | 150110       0.66        90
    //  717 | 149998       0.35        12 | 149531       0.70       129
    //  762 | 149836       0.37        13 | 148559       0.74       190
    //  807 | 149736       0.39        14 | 151107       0.39        14
    //  852 | 150204       0.42        15 | 151019       0.42        15
    CWISS_RawTable_DropDeletesWithoutResize(policy, self);
  } else {
    // Otherwise grow the container.
    CWISS_RawTable_Resize(policy, self, self->capacity_ * 2 + 1);
  }
}

/// Prefetches the backing array to dodge potential TLB misses.
/// This is intended to overlap with execution of calculating the hash for a
/// key.
static inline void CWISS_RawTable_PrefetchHeapBlock(
    const CWISS_Policy* policy, const CWISS_RawTable* self) {
  CWISS_PREFETCH(self->ctrl_, 1);
}

/// Issues CPU prefetch instructions for the memory needed to find or insert
/// a key.
///
/// NOTE: This is a very low level operation and should not be used without
/// specific benchmarks indicating its importance.
static inline void CWISS_RawTable_Prefetch(const CWISS_Policy* policy,
                                           const CWISS_RawTable* self,
                                           const void* key) {
  (void)key;
#if CWISS_HAVE_PREFETCH
  CWISS_RawTable_PrefetchHeapBlock(policy, self);
  CWISS_ProbeSeq seq = CWISS_ProbeSeq_Start(self->ctrl_, policy->key->hash(key),
                                            self->capacity_);
  CWISS_PREFETCH(self->ctrl_ + seq.offset_, 3);
  CWISS_PREFETCH(self->ctrl_ + seq.offset_ * policy->slot->size, 3);
#endif
}

/// The return type of `CWISS_RawTable_PrepareInsert()`.
typedef struct {
  size_t index;
  bool inserted;
} CWISS_PrepareInsert;

/// Given the hash of a value not currently in the table, finds the next viable
/// slot index to insert it at.
///
/// If the table does not actually have space, UB.
CWISS_INLINE_NEVER
static size_t CWISS_RawTable_PrepareInsert(const CWISS_Policy* policy,
                                           CWISS_RawTable* self, size_t hash) {
  CWISS_FindInfo target =
      CWISS_FindFirstNonFull(self->ctrl_, hash, self->capacity_);
  if (CWISS_UNLIKELY(self->growth_left_ == 0 &&
                     !CWISS_IsDeleted(self->ctrl_[target.offset]))) {
    CWISS_RawTable_rehash_and_grow_if_necessary(policy, self);
    target = CWISS_FindFirstNonFull(self->ctrl_, hash, self->capacity_);
  }
  ++self->size_;
  self->growth_left_ -= CWISS_IsEmpty(self->ctrl_[target.offset]);
  CWISS_SetCtrl(target.offset, CWISS_H2(hash), self->capacity_, self->ctrl_,
                self->slots_, policy->slot->size);
  // infoz().RecordInsert(hash, target.probe_length);
  return target.offset;
}

/// Attempts to find `key` in the table; if it isn't found, returns where to
/// insert it, instead.
static inline CWISS_PrepareInsert CWISS_RawTable_FindOrPrepareInsert(
    const CWISS_Policy* policy, const CWISS_KeyPolicy* key_policy,
    CWISS_RawTable* self, const void* key) {
  CWISS_RawTable_PrefetchHeapBlock(policy, self);
  size_t hash = key_policy->hash(key);
  CWISS_ProbeSeq seq = CWISS_ProbeSeq_Start(self->ctrl_, hash, self->capacity_);
  while (true) {
    CWISS_Group g = CWISS_Group_new(self->ctrl_ + seq.offset_);
    CWISS_BitMask match = CWISS_Group_Match(&g, CWISS_H2(hash));
    uint32_t i;
    while (CWISS_BitMask_next(&match, &i)) {
      size_t idx = CWISS_ProbeSeq_offset(&seq, i);
      char* slot = self->slots_ + idx * policy->slot->size;
      if (CWISS_LIKELY(key_policy->eq(key, policy->slot->get(slot))))
        return (CWISS_PrepareInsert){idx, false};
    }
    if (CWISS_LIKELY(CWISS_Group_MatchEmpty(&g).mask)) break;
    CWISS_ProbeSeq_next(&seq);
    CWISS_DCHECK(seq.index_ <= self->capacity_, "full table!");
  }
  return (CWISS_PrepareInsert){CWISS_RawTable_PrepareInsert(policy, self, hash),
                               true};
}

/// Prepares a slot to insert an element into.
///
/// This function does all the work of calling the appropriate policy functions
/// to initialize the slot.
static inline void* CWISS_RawTable_PreInsert(const CWISS_Policy* policy,
                                             CWISS_RawTable* self, size_t i) {
  void* dst = self->slots_ + i * policy->slot->size;
  policy->slot->init(dst);
  return policy->slot->get(dst);
}

/// Creates a new empty table with the given capacity.
static inline CWISS_RawTable CWISS_RawTable_new(const CWISS_Policy* policy,
                                                size_t capacity) {
  CWISS_RawTable self = {
      .ctrl_ = CWISS_EmptyGroup(),
  };

  if (capacity != 0) {
    self.capacity_ = CWISS_NormalizeCapacity(capacity);
    CWISS_RawTable_InitializeSlots(policy, &self);
  }

  return self;
}

/// Ensures that at least `n` more elements can be inserted without a resize
/// (although this function my itself resize and rehash the table).
static inline void CWISS_RawTable_reserve(const CWISS_Policy* policy,
                                          CWISS_RawTable* self, size_t n) {
  if (n <= self->size_ + self->growth_left_) {
    return;
  }

  n = CWISS_NormalizeCapacity(CWISS_GrowthToLowerboundCapacity(n));
  CWISS_RawTable_Resize(policy, self, n);

  // This is after resize, to ensure that we have completed the allocation
  // and have potentially sampled the hashtable.
  // infoz().RecordReservation(n);
}

/// Creates a duplicate of this table.
static inline CWISS_RawTable CWISS_RawTable_dup(const CWISS_Policy* policy,
                                                const CWISS_RawTable* self) {
  CWISS_RawTable copy = CWISS_RawTable_new(policy, 0);

  CWISS_RawTable_reserve(policy, &copy, self->size_);
  // Because the table is guaranteed to be empty, we can do something faster
  // than a full `insert`. In particular we do not need to take a trip to
  // `CWISS_RawTable_rehash_and_grow_if_necessary()` because we are already
  // big enough (since `self` is a priori) and tombstones cannot be created
  // during this process.
  for (CWISS_RawIter iter = CWISS_RawTable_citer(policy, self);
       CWISS_RawIter_get(policy, &iter); CWISS_RawIter_next(policy, &iter)) {
    void* v = CWISS_RawIter_get(policy, &iter);
    size_t hash = policy->key->hash(v);

    CWISS_FindInfo target =
        CWISS_FindFirstNonFull(copy.ctrl_, hash, copy.capacity_);
    CWISS_SetCtrl(target.offset, CWISS_H2(hash), copy.capacity_, copy.ctrl_,
                  copy.slots_, policy->slot->size);
    void* slot = CWISS_RawTable_PreInsert(policy, &copy, target.offset);
    policy->obj->copy(slot, v);
    // infoz().RecordInsert(hash, target.probe_length);
  }
  copy.size_ = self->size_;
  copy.growth_left_ -= self->size_;
  return copy;
}

/// Destroys this table, destroying its elements and freeing the backing array.
static inline void CWISS_RawTable_destroy(const CWISS_Policy* policy,
                                          CWISS_RawTable* self) {
  CWISS_RawTable_DestroySlots(policy, self);
}

/// Returns whether the table is empty.
static inline bool CWISS_RawTable_empty(const CWISS_Policy* policy,
                                        const CWISS_RawTable* self) {
  return !self->size_;
}

/// Returns the number of elements in the table.
static inline size_t CWISS_RawTable_size(const CWISS_Policy* policy,
                                         const CWISS_RawTable* self) {
  return self->size_;
}

/// Returns the total capacity of the table, which is different from the number
/// of elements that would cause it to get resized.
static inline size_t CWISS_RawTable_capacity(const CWISS_Policy* policy,
                                             const CWISS_RawTable* self) {
  return self->capacity_;
}

/// Clears the table, erasing every element contained therein.
static inline void CWISS_RawTable_clear(const CWISS_Policy* policy,
                                        CWISS_RawTable* self) {
  // Iterating over this container is O(bucket_count()). When bucket_count()
  // is much greater than size(), iteration becomes prohibitively expensive.
  // For clear() it is more important to reuse the allocated array when the
  // container is small because allocation takes comparatively long time
  // compared to destruction of the elements of the container. So we pick the
  // largest bucket_count() threshold for which iteration is still fast and
  // past that we simply deallocate the array.
  if (self->capacity_ > 127) {
    CWISS_RawTable_DestroySlots(policy, self);

    // infoz().RecordClearedReservation();
  } else if (self->capacity_) {
    if (policy->slot->del != NULL) {
      for (size_t i = 0; i != self->capacity_; ++i) {
        if (CWISS_IsFull(self->ctrl_[i])) {
          policy->slot->del(self->slots_ + i * policy->slot->size);
        }
      }
    }

    self->size_ = 0;
    CWISS_ResetCtrl(self->capacity_, self->ctrl_, self->slots_,
                    policy->slot->size);
    CWISS_RawTable_ResetGrowthLeft(policy, self);
  }
  CWISS_DCHECK(!self->size_, "size was still nonzero");
  // infoz().RecordStorageChanged(0, capacity_);
}

/// The return type of `CWISS_RawTable_insert()`.
typedef struct {
  /// An iterator referring to the relevant element.
  CWISS_RawIter iter;
  /// True if insertion actually occurred; false if the element was already
  /// present.
  bool inserted;
} CWISS_Insert;

/// "Inserts" `val` into the table if it isn't already present.
///
/// This function does not perform insertion; it behaves exactly like
/// `CWISS_RawTable_insert()` up until it would copy-initialize the new
/// element, instead returning a valid iterator pointing to uninitialized data.
///
/// This allows, for example, lazily constructing the parts of the element that
/// do not figure into the hash or equality.
///
/// If this function returns `true` in `inserted`, the caller has *no choice*
/// but to insert, i.e., they may not change their minds at that point.
///
/// `key_policy` is a possibly heterogenous key policy for comparing `key`'s
/// type to types in the map. `key_policy` may be `&policy->key`.
static inline CWISS_Insert CWISS_RawTable_deferred_insert(
    const CWISS_Policy* policy, const CWISS_KeyPolicy* key_policy,
    CWISS_RawTable* self, const void* key) {
  CWISS_PrepareInsert res =
      CWISS_RawTable_FindOrPrepareInsert(policy, key_policy, self, key);

  if (res.inserted) {
    CWISS_RawTable_PreInsert(policy, self, res.index);
  }
  return (CWISS_Insert){CWISS_RawTable_citer_at(policy, self, res.index),
                        res.inserted};
}

/// Inserts `val` (by copy) into the table if it isn't already present.
///
/// Returns an iterator pointing to the element in the map and whether it was
/// just inserted or was already present.
static inline CWISS_Insert CWISS_RawTable_insert(const CWISS_Policy* policy,
                                                 CWISS_RawTable* self,
                                                 const void* val) {
  CWISS_PrepareInsert res =
      CWISS_RawTable_FindOrPrepareInsert(policy, policy->key, self, val);

  if (res.inserted) {
    void* slot = CWISS_RawTable_PreInsert(policy, self, res.index);
    policy->obj->copy(slot, val);
  }
  return (CWISS_Insert){CWISS_RawTable_citer_at(policy, self, res.index),
                        res.inserted};
}

/// Tries to find the corresponding entry for `key` using `hash` as a hint.
/// If not found, returns a null iterator.
///
/// `key_policy` is a possibly heterogenous key policy for comparing `key`'s
/// type to types in the map. `key_policy` may be `&policy->key`.
///
/// If `hash` is not actually the hash of `key`, UB.
static inline CWISS_RawIter CWISS_RawTable_find_hinted(
    const CWISS_Policy* policy, const CWISS_KeyPolicy* key_policy,
    const CWISS_RawTable* self, const void* key, size_t hash) {
  CWISS_ProbeSeq seq = CWISS_ProbeSeq_Start(self->ctrl_, hash, self->capacity_);
  while (true) {
    CWISS_Group g = CWISS_Group_new(self->ctrl_ + seq.offset_);
    CWISS_BitMask match = CWISS_Group_Match(&g, CWISS_H2(hash));
    uint32_t i;
    while (CWISS_BitMask_next(&match, &i)) {
      char* slot =
          self->slots_ + CWISS_ProbeSeq_offset(&seq, i) * policy->slot->size;
      if (CWISS_LIKELY(key_policy->eq(key, policy->slot->get(slot))))
        return CWISS_RawTable_citer_at(policy, self,
                                       CWISS_ProbeSeq_offset(&seq, i));
    }
    if (CWISS_LIKELY(CWISS_Group_MatchEmpty(&g).mask))
      return (CWISS_RawIter){0};
    CWISS_ProbeSeq_next(&seq);
    CWISS_DCHECK(seq.index_ <= self->capacity_, "full table!");
  }
}

/// Tries to find the corresponding entry for `key`.
/// If not found, returns a null iterator.
///
/// `key_policy` is a possibly heterogenous key policy for comparing `key`'s
/// type to types in the map. `key_policy` may be `&policy->key`.
static inline CWISS_RawIter CWISS_RawTable_find(
    const CWISS_Policy* policy, const CWISS_KeyPolicy* key_policy,
    const CWISS_RawTable* self, const void* key) {
  return CWISS_RawTable_find_hinted(policy, key_policy, self, key,
                                    key_policy->hash(key));
}

/// Erases the element pointed to by the given valid iterator.
/// This function will invalidate the iterator.
static inline void CWISS_RawTable_erase_at(const CWISS_Policy* policy,
                                           CWISS_RawIter it) {
  CWISS_AssertIsFull(it.ctrl_);
  if (policy->slot->del != NULL) {
    policy->slot->del(it.slot_);
  }
  CWISS_RawTable_EraseMetaOnly(policy, it);
}

/// Erases the entry corresponding to `key`, if present. Returns true if
/// deletion occured.
///
/// `key_policy` is a possibly heterogenous key policy for comparing `key`'s
/// type to types in the map. `key_policy` may be `&policy->key`.
static inline bool CWISS_RawTable_erase(const CWISS_Policy* policy,
                                        const CWISS_KeyPolicy* key_policy,
                                        CWISS_RawTable* self, const void* key) {
  CWISS_RawIter it = CWISS_RawTable_find(policy, key_policy, self, key);
  if (it.slot_ == NULL) return false;
  CWISS_RawTable_erase_at(policy, it);
  return true;
}

/// Triggers a rehash, growing to at least a capacity of `n`.
static inline void CWISS_RawTable_rehash(const CWISS_Policy* policy,
                                         CWISS_RawTable* self, size_t n) {
  if (n == 0 && self->capacity_ == 0) return;
  if (n == 0 && self->size_ == 0) {
    CWISS_RawTable_DestroySlots(policy, self);
    // infoz().RecordStorageChanged(0, 0);
    // infoz().RecordClearedReservation();
    return;
  }

  // bitor is a faster way of doing `max` here. We will round up to the next
  // power-of-2-minus-1, so bitor is good enough.
  size_t m = CWISS_NormalizeCapacity(
      n | CWISS_GrowthToLowerboundCapacity(self->size_));
  // n == 0 unconditionally rehashes as per the standard.
  if (n == 0 || m > self->capacity_) {
    CWISS_RawTable_Resize(policy, self, m);

    // This is after resize, to ensure that we have completed the allocation
    // and have potentially sampled the hashtable.
    // infoz().RecordReservation(n);
  }
}

/// Returns whether `key` is contained in this table.
///
/// `key_policy` is a possibly heterogenous key policy for comparing `key`'s
/// type to types in the map. `key_policy` may be `&policy->key`.
static inline bool CWISS_RawTable_contains(const CWISS_Policy* policy,
                                           const CWISS_KeyPolicy* key_policy,
                                           const CWISS_RawTable* self,
                                           const void* key) {
  return CWISS_RawTable_find(policy, key_policy, self, key).slot_ != NULL;
}

CWISS_END_EXTERN
CWISS_END

#endif  // CWISSTABLE_INTERNAL_RAW_TABLE_H_