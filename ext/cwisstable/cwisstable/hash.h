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

#ifndef CWISSTABLE_HASH_H_
#define CWISSTABLE_HASH_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "cwisstable/internal/absl_hash.h"
#include "cwisstable/internal/base.h"
#include "cwisstable/internal/bits.h"

/// Hash functions.
///
/// This file provides some hash functions to use with cwisstable types.
///
/// Every hash function defines four symbols:
///   - `CWISS_<Hash>_State`, the state of the hash function.
///   - `CWISS_<Hash>_kInit`, the initial value of the hash state.
///   - `void CWISS_<Hash>_Write(State*, const void*, size_t)`, write some more
///     data into the hash state.
///   - `size_t CWISS_<Hash>_Finish(State)`, digest the state into a final hash
///     value.
///
/// Currently available are two hashes: `FxHash`, which is small and fast, and
/// `AbslHash`, the hash function used by Abseil.
///
/// `AbslHash` is the default hash function.

CWISS_BEGIN
CWISS_BEGIN_EXTERN

typedef size_t CWISS_FxHash_State;
#define CWISS_FxHash_kInit ((CWISS_FxHash_State)0)
static inline void CWISS_FxHash_Write(CWISS_FxHash_State* state,
                                      const void* val, size_t len) {
  const size_t kSeed = (size_t)(UINT64_C(0x517cc1b727220a95));
  const uint32_t kRotate = 5;

  const char* p = (const char*)val;
  CWISS_FxHash_State state_ = *state;
  while (len > 0) {
    size_t word = 0;
    size_t to_read = len >= sizeof(state_) ? sizeof(state_) : len;
    memcpy(&word, p, to_read);

    state_ = CWISS_RotateLeft(state_, kRotate);
    state_ ^= word;
    state_ *= kSeed;

    len -= to_read;
    p += to_read;
  }
  *state = state_;
}
static inline size_t CWISS_FxHash_Finish(CWISS_FxHash_State state) {
  return state;
}

typedef CWISS_AbslHash_State_ CWISS_AbslHash_State;
#define CWISS_AbslHash_kInit CWISS_AbslHash_kInit_
static inline void CWISS_AbslHash_Write(CWISS_AbslHash_State* state,
                                        const void* val, size_t len) {
  const char* val8 = (const char*)val;
  if (CWISS_LIKELY(len < CWISS_AbslHash_kPiecewiseChunkSize)) {
    goto CWISS_AbslHash_Write_small;
  }

  while (len >= CWISS_AbslHash_kPiecewiseChunkSize) {
    CWISS_AbslHash_Mix(
        state, CWISS_AbslHash_Hash64(val8, CWISS_AbslHash_kPiecewiseChunkSize));
    len -= CWISS_AbslHash_kPiecewiseChunkSize;
    val8 += CWISS_AbslHash_kPiecewiseChunkSize;
  }

CWISS_AbslHash_Write_small:;
  uint64_t v;
  if (len > 16) {
    v = CWISS_AbslHash_Hash64(val8, len);
  } else if (len > 8) {
    CWISS_U128 p = CWISS_Load9To16(val8, len);
    CWISS_AbslHash_Mix(state, p.lo);
    v = p.hi;
  } else if (len >= 4) {
    v = CWISS_Load4To8(val8, len);
  } else if (len > 0) {
    v = CWISS_Load1To3(val8, len);
  } else {
    // Empty ranges have no effect.
    return;
  }

  CWISS_AbslHash_Mix(state, v);
}
static inline size_t CWISS_AbslHash_Finish(CWISS_AbslHash_State state) {
  return state;
}

CWISS_END_EXTERN
CWISS_END

#endif  // CWISSTABLE_HASH_H_