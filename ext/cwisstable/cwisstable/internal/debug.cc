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

// This library provides APIs to debug the probing behavior of hash tables.
//
// In general, the probing behavior is a black box for users and only the
// side effects can be measured in the form of performance differences.
// These APIs give a glimpse on the actual behavior of the probing algorithms in
// these hashtables given a specified hash function and a set of elements.
//
// The probe count distribution can be used to assess the quality of the hash
// function for that particular hash table. Note that a hash function that
// performs well in one hash table implementation does not necessarily performs
// well in a different one.
//
// This library supports std::unordered_{set,map}, dense_hash_{set,map} and
// absl::{flat,node,string}_hash_{set,map}.

#include "cwisstable/internal/debug.h"

#include <algorithm>

namespace cwisstable::internal {
size_t GetHashtableDebugNumProbes(const CWISS_Policy* policy,
                                  const CWISS_RawTable* set, const void* key) {
  size_t num_probes = 0;
  size_t hash = policy->key->hash(key);
  auto seq = CWISS_ProbeSeq_Start(set->ctrl_, hash, set->capacity_);
  while (true) {
    auto g = CWISS_Group_new(set->ctrl_ + seq.offset_);
    auto match = CWISS_Group_Match(&g, CWISS_H2(hash));
    uint32_t i;
    while (CWISS_BitMask_next(&match, &i)) {
      size_t idx = CWISS_ProbeSeq_offset(&seq, i);
      char* slot = set->slots_ + idx * policy->slot->size;
      if (CWISS_LIKELY(policy->key->eq(slot, key))) return num_probes;

      ++num_probes;
    }
    if (CWISS_LIKELY(CWISS_Group_MatchEmpty(&g).mask)) return num_probes;
    CWISS_ProbeSeq_next(&seq);
    ++num_probes;
  }
}

size_t AllocatedByteSize(const CWISS_Policy* policy,
                         const CWISS_RawTable* set) {
  size_t capacity = set->capacity_;
  if (capacity == 0) return 0;
  size_t m = CWISS_AllocSize(capacity, policy->slot->size, policy->slot->align);

  /* TODO(mcyoung): Ask kfm about this.
  size_t per_slot = Traits::space_used(static_cast<const Slot*>(nullptr));
  if (per_slot != ~size_t{}) {
    m += per_slot * c.size();
  } else {
    for (size_t i = 0; i != capacity; ++i) {
      if (container_internal::IsFull(c.ctrl_[i])) {
        m += Traits::space_used(c.slots_ + i);
      }
    }
  }*/
  return m;
}

size_t LowerBoundAllocatedByteSize(const CWISS_Policy* policy, size_t size) {
  size_t capacity = CWISS_GrowthToLowerboundCapacity(size);
  if (capacity == 0) return 0;
  size_t m = CWISS_AllocSize(CWISS_NormalizeCapacity(capacity),
                             policy->slot->size, policy->slot->align);
  /*size_t per_slot = Traits::space_used(static_cast<const Slot*>(nullptr));
  if (per_slot != ~size_t{}) {
    m += per_slot * size;
  }*/
  return m;
}

std::vector<size_t> GetHashtableDebugNumProbesHistogram(
    const CWISS_Policy* policy, const CWISS_RawTable* set) {
  std::vector<size_t> v;
  for (auto it = CWISS_RawTable_citer(policy, set);
       CWISS_RawIter_get(policy, &it); CWISS_RawIter_next(policy, &it)) {
    size_t num_probes =
        GetHashtableDebugNumProbes(policy, set, CWISS_RawIter_get(policy, &it));
    v.resize((std::max)(v.size(), num_probes + 1));
    v[num_probes]++;
  }
  return v;
}

HashtableDebugProbeSummary GetHashtableDebugProbeSummary(
    const CWISS_Policy* policy, const CWISS_RawTable* set) {
  auto probes = GetHashtableDebugNumProbesHistogram(policy, set);
  HashtableDebugProbeSummary summary = {};
  for (size_t i = 0; i < probes.size(); ++i) {
    summary.total_elements += probes[i];
    summary.total_num_probes += probes[i] * i;
  }
  summary.mean = 1.0 * summary.total_num_probes / summary.total_elements;
  return summary;
}
}  // namespace cwisstable::internal
