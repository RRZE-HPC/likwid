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

#ifndef CWISSTABLE_INTERNAL_ABSL_HASH_H_
#define CWISSTABLE_INTERNAL_ABSL_HASH_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "cwisstable/internal/base.h"
#include "cwisstable/internal/bits.h"

/// Implementation details of AbslHash.

CWISS_BEGIN
CWISS_BEGIN_EXTERN

static inline uint64_t CWISS_AbslHash_LowLevelMix(uint64_t v0, uint64_t v1) {
#ifndef __aarch64__
  // The default bit-mixer uses 64x64->128-bit multiplication.
  CWISS_U128 p = CWISS_Mul128(v0, v1);
  return p.hi ^ p.lo;
#else
  // The default bit-mixer above would perform poorly on some ARM microarchs,
  // where calculating a 128-bit product requires a sequence of two
  // instructions with a high combined latency and poor throughput.
  // Instead, we mix bits using only 64-bit arithmetic, which is faster.
  uint64_t p = v0 ^ CWISS_RotateLeft(v1, 40);
  p *= v1 ^ CWISS_RotateLeft(v0, 39);
  return p ^ (p >> 11);
#endif
}

CWISS_INLINE_NEVER
static uint64_t CWISS_AbslHash_LowLevelHash(const void* data, size_t len,
                                            uint64_t seed,
                                            const uint64_t salt[5]) {
  const char* ptr = (const char*)data;
  uint64_t starting_length = (uint64_t)len;
  uint64_t current_state = seed ^ salt[0];

  if (len > 64) {
    // If we have more than 64 bytes, we're going to handle chunks of 64
    // bytes at a time. We're going to build up two separate hash states
    // which we will then hash together.
    uint64_t duplicated_state = current_state;

    do {
      uint64_t chunk[8];
      memcpy(chunk, ptr, sizeof(chunk));

      uint64_t cs0 = CWISS_AbslHash_LowLevelMix(chunk[0] ^ salt[1],
                                                chunk[1] ^ current_state);
      uint64_t cs1 = CWISS_AbslHash_LowLevelMix(chunk[2] ^ salt[2],
                                                chunk[3] ^ current_state);
      current_state = (cs0 ^ cs1);

      uint64_t ds0 = CWISS_AbslHash_LowLevelMix(chunk[4] ^ salt[3],
                                                chunk[5] ^ duplicated_state);
      uint64_t ds1 = CWISS_AbslHash_LowLevelMix(chunk[6] ^ salt[4],
                                                chunk[7] ^ duplicated_state);
      duplicated_state = (ds0 ^ ds1);

      ptr += 64;
      len -= 64;
    } while (len > 64);

    current_state = current_state ^ duplicated_state;
  }

  // We now have a data `ptr` with at most 64 bytes and the current state
  // of the hashing state machine stored in current_state.
  while (len > 16) {
    uint64_t a = CWISS_Load64(ptr);
    uint64_t b = CWISS_Load64(ptr + 8);

    current_state = CWISS_AbslHash_LowLevelMix(a ^ salt[1], b ^ current_state);

    ptr += 16;
    len -= 16;
  }

  // We now have a data `ptr` with at most 16 bytes.
  uint64_t a = 0;
  uint64_t b = 0;
  if (len > 8) {
    // When we have at least 9 and at most 16 bytes, set A to the first 64
    // bits of the input and B to the last 64 bits of the input. Yes, they will
    // overlap in the middle if we are working with less than the full 16
    // bytes.
    a = CWISS_Load64(ptr);
    b = CWISS_Load64(ptr + len - 8);
  } else if (len > 3) {
    // If we have at least 4 and at most 8 bytes, set A to the first 32
    // bits and B to the last 32 bits.
    a = CWISS_Load32(ptr);
    b = CWISS_Load32(ptr + len - 4);
  } else if (len > 0) {
    // If we have at least 1 and at most 3 bytes, read all of the provided
    // bits into A, with some adjustments.
    a = CWISS_Load1To3(ptr, len);
  }

  uint64_t w = CWISS_AbslHash_LowLevelMix(a ^ salt[1], b ^ current_state);
  uint64_t z = salt[1] ^ starting_length;
  return CWISS_AbslHash_LowLevelMix(w, z);
}

// A non-deterministic seed.
//
// The current purpose of this seed is to generate non-deterministic results
// and prevent having users depend on the particular hash values.
// It is not meant as a security feature right now, but it leaves the door
// open to upgrade it to a true per-process random seed. A true random seed
// costs more and we don't need to pay for that right now.
//
// On platforms with ASLR, we take advantage of it to make a per-process
// random value.
// See https://en.wikipedia.org/wiki/Address_space_layout_randomization
//
// On other platforms this is still going to be non-deterministic but most
// probably per-build and not per-process.
static const void* const CWISS_AbslHash_kSeed = &CWISS_AbslHash_kSeed;

// The salt array used by LowLevelHash. This array is NOT the mechanism used to
// make absl::Hash non-deterministic between program invocations.  See `Seed()`
// for that mechanism.
//
// Any random values are fine. These values are just digits from the decimal
// part of pi.
// https://en.wikipedia.org/wiki/Nothing-up-my-sleeve_number
static const uint64_t CWISS_AbslHash_kHashSalt[5] = {
    0x243F6A8885A308D3, 0x13198A2E03707344, 0xA4093822299F31D0,
    0x082EFA98EC4E6C89, 0x452821E638D01377,
};

#define CWISS_AbslHash_kPiecewiseChunkSize ((size_t)1024)

typedef uint64_t CWISS_AbslHash_State_;
#define CWISS_AbslHash_kInit_ ((CWISS_AbslHash_State_)CWISS_AbslHash_kSeed)

static inline void CWISS_AbslHash_Mix(CWISS_AbslHash_State_* state,
                                      uint64_t v) {
  const uint64_t kMul = sizeof(size_t) == 4 ? 0xcc9e2d51 : 0x9ddfea08eb382d69;
  *state = CWISS_AbslHash_LowLevelMix(*state + v, kMul);
}

CWISS_INLINE_NEVER
static uint64_t CWISS_AbslHash_Hash64(const void* val, size_t len) {
  return CWISS_AbslHash_LowLevelHash(val, len, CWISS_AbslHash_kInit_,
                                     CWISS_AbslHash_kHashSalt);
}

CWISS_END_EXTERN
CWISS_END

#endif  // CWISSTABLE_INTERNAL_ABSL_HASH_H_