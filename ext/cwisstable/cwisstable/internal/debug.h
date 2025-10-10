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
///
/// In general, the probing behavior is a black box for users and only the
/// side effects can be measured in the form of performance differences.
/// These APIs give a glimpse on the actual behavior of the probing algorithms
/// in these hashtables given a specified hash function and a set of elements.
///
/// The probe count distribution can be used to assess the quality of the hash
/// function for that particular hash table. Note that a hash function that
/// performs well in one hash table implementation does not necessarily performs
/// well in a different one.

#ifndef CWISSTABLE_INTERNAL_DEBUG_H_
#define CWISSTABLE_INTERNAL_DEBUG_H_

#include <cstddef>
#include <vector>

#include "cwisstable.h"

namespace cwisstable::internal {
/// Returns the number of probes required to lookup `key`.  Returns 0 for a
/// search with no collisions.  Higher values mean more hash collisions
/// occurred; however, the exact meaning of this number varies according to the
/// container type.
size_t GetHashtableDebugNumProbes(const CWISS_Policy* policy,
                                  const CWISS_RawTable* set, const void* key);

/// Returns the number of bytes requested from the allocator by the container
/// and not freed.
size_t AllocatedByteSize(const CWISS_Policy* policy, const CWISS_RawTable* set);

/// Returns a tight lower bound for AllocatedByteSize(c) where `c` is of type
/// `C` and `c.size()` is equal to `num_elements`.
size_t LowerBoundAllocatedByteSize(const CWISS_Policy* policy, size_t size);

/// Gets a histogram of the number of probes for each elements in the container.
/// The sum of all the values in the vector is equal to container.size().
std::vector<size_t> GetHashtableDebugNumProbesHistogram(
    const CWISS_Policy* policy, const CWISS_RawTable* set);

struct HashtableDebugProbeSummary {
  size_t total_elements;
  size_t total_num_probes;
  double mean;
};

/// Gets a summary of the probe count distribution for the elements in the
/// container.
HashtableDebugProbeSummary GetHashtableDebugProbeSummary(
    const CWISS_Policy* policy, const CWISS_RawTable* set);
}  // namespace cwisstable::internal
#endif  // CWISSTABLE_INTERNAL_DEBUG_H_
