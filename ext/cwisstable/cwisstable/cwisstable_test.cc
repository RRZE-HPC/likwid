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

// absl::raw_hash_set's tests ported to run over cwisstable's syntax.
//
// Commented out tests are tests we have yet to port.

#include "cwisstable.h"

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_set>

#include "absl/cleanup/cleanup.h"
#include "cwisstable/internal/debug.h"
#include "cwisstable/internal/test_helpers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace cwisstable {
namespace {

using ::cwisstable::internal::GetHashtableDebugNumProbes;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Ge;
using ::testing::IsEmpty;
using ::testing::Lt;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

TEST(Util, NormalizeCapacity) {
  EXPECT_EQ(1, CWISS_NormalizeCapacity(0));
  EXPECT_EQ(1, CWISS_NormalizeCapacity(1));
  EXPECT_EQ(3, CWISS_NormalizeCapacity(2));
  EXPECT_EQ(3, CWISS_NormalizeCapacity(3));
  EXPECT_EQ(7, CWISS_NormalizeCapacity(4));
  EXPECT_EQ(7, CWISS_NormalizeCapacity(7));
  EXPECT_EQ(15, CWISS_NormalizeCapacity(8));
  EXPECT_EQ(15, CWISS_NormalizeCapacity(15));
  EXPECT_EQ(15 * 2 + 1, CWISS_NormalizeCapacity(15 + 1));
  EXPECT_EQ(15 * 2 + 1, CWISS_NormalizeCapacity(15 + 2));
}

TEST(Util, GrowthAndCapacity) {
  // Verify that GrowthToCapacity gives the minimum capacity that has enough
  // growth.
  for (size_t growth = 0; growth < 10000; ++growth) {
    SCOPED_TRACE(growth);
    size_t capacity =
        CWISS_NormalizeCapacity(CWISS_GrowthToLowerboundCapacity(growth));
    // The capacity is large enough for `growth`.
    EXPECT_THAT(CWISS_CapacityToGrowth(capacity), Ge(growth));
    // For (capacity+1) < kWidth, growth should equal capacity.
    if (capacity + 1 < CWISS_Group_kWidth) {
      EXPECT_THAT(CWISS_CapacityToGrowth(capacity), Eq(capacity));
    } else {
      EXPECT_THAT(CWISS_CapacityToGrowth(capacity), Lt(capacity));
    }
    if (growth != 0 && capacity > 1) {
      // There is no smaller capacity that works.
      EXPECT_THAT(CWISS_CapacityToGrowth(capacity / 2), Lt(growth));
    }
  }

  for (size_t capacity = CWISS_Group_kWidth - 1; capacity < 10000;
       capacity = 2 * capacity + 1) {
    SCOPED_TRACE(capacity);
    size_t growth = CWISS_CapacityToGrowth(capacity);
    EXPECT_THAT(growth, Lt(capacity));
    EXPECT_LE(CWISS_GrowthToLowerboundCapacity(growth), capacity);
    EXPECT_EQ(CWISS_NormalizeCapacity(CWISS_GrowthToLowerboundCapacity(growth)),
              capacity);
  }
}

TEST(Util, probe_seq) {
  CWISS_ProbeSeq seq;
  std::vector<size_t> offsets(8);
  auto gen = [&]() {
    size_t res = CWISS_ProbeSeq_offset(&seq, 0);
    CWISS_ProbeSeq_next(&seq);
    return res;
  };

  CWISS_GCC_PUSH
  CWISS_GCC_ALLOW("-Wunreachable-code")  // Clang seems to whine about this
                                         // specific stanza...
  std::vector<size_t> expected;
  if (CWISS_Group_kWidth == 16) {
    expected = {0, 16, 48, 96, 32, 112, 80, 64};
  } else if (CWISS_Group_kWidth == 8) {
    // Interestingly, OG SwissTable does _not_ test non-SIMD probe sequences.
    expected = {0, 8, 24, 48, 80, 120, 40, 96};
  } else {
    FAIL() << "No test coverage for CWISS_Group_kWidth == "
           << CWISS_Group_kWidth;
  }
  CWISS_GCC_POP

  seq = CWISS_ProbeSeq_new(0, 127);
  std::generate_n(offsets.begin(), 8, gen);
  EXPECT_EQ(offsets, expected);

  seq = CWISS_ProbeSeq_new(128, 127);
  std::generate_n(offsets.begin(), 8, gen);
  EXPECT_EQ(offsets, expected);
}

template <size_t Width, size_t Shift = 0>
CWISS_BitMask MakeMask(uint64_t mask) {
  return {mask, Width, Shift};
}

std::vector<uint32_t> MaskBits(CWISS_BitMask mask) {
  std::vector<uint32_t> v;
  uint32_t x;
  while (CWISS_BitMask_next(&mask, &x)) {
    v.push_back(x);
  }
  return v;
}

TEST(BitMask, Smoke) {
  EXPECT_THAT(MaskBits(MakeMask<8>(0)), ElementsAre());
  EXPECT_THAT(MaskBits(MakeMask<8>(0x1)), ElementsAre(0));
  EXPECT_THAT(MaskBits(MakeMask<8>(0x2)), ElementsAre(1));
  EXPECT_THAT(MaskBits(MakeMask<8>(0x3)), ElementsAre(0, 1));
  EXPECT_THAT(MaskBits(MakeMask<8>(0x4)), ElementsAre(2));
  EXPECT_THAT(MaskBits(MakeMask<8>(0x5)), ElementsAre(0, 2));
  EXPECT_THAT(MaskBits(MakeMask<8>(0x55)), ElementsAre(0, 2, 4, 6));
  EXPECT_THAT(MaskBits(MakeMask<8>(0xAA)), ElementsAre(1, 3, 5, 7));
}

TEST(BitMask, WithShift) {
  // See the non-SSE version of Group for details on what this math is for.
  uint64_t ctrl = 0x1716151413121110;
  uint64_t hash = 0x12;
  constexpr uint64_t msbs = 0x8080808080808080ULL;
  constexpr uint64_t lsbs = 0x0101010101010101ULL;
  auto x = ctrl ^ (lsbs * hash);
  uint64_t mask = (x - lsbs) & ~x & msbs;
  EXPECT_EQ(0x0000000080800000, mask);

  auto b = MakeMask<8, 3>(mask);
  EXPECT_EQ(CWISS_BitMask_LowestBitSet(&b), 2);
}

uint32_t MaskLeading(CWISS_BitMask mask) {
  return CWISS_BitMask_LeadingZeros(&mask);
}
uint32_t MaskTrailing(CWISS_BitMask mask) {
  return CWISS_BitMask_TrailingZeros(&mask);
}

TEST(BitMask, LeadingTrailing) {
  EXPECT_EQ(MaskLeading(MakeMask<16>(0x00001a40)), 3);
  EXPECT_EQ(MaskTrailing(MakeMask<16>(0x00001a40)), 6);

  EXPECT_EQ(MaskLeading(MakeMask<16>(0x00000001)), 15);
  EXPECT_EQ(MaskTrailing(MakeMask<16>(0x00000001)), 0);

  EXPECT_EQ(MaskLeading(MakeMask<16>(0x00008000)), 0);
  EXPECT_EQ(MaskTrailing(MakeMask<16>(0x00008000)), 15);

  EXPECT_EQ(MaskLeading(MakeMask<8, 3>(0x0000008080808000)), 3);
  EXPECT_EQ(MaskTrailing(MakeMask<8, 3>(0x0000008080808000)), 1);

  EXPECT_EQ(MaskLeading(MakeMask<8, 3>(0x0000000000000080)), 7);
  EXPECT_EQ(MaskTrailing(MakeMask<8, 3>(0x0000000000000080)), 0);

  EXPECT_EQ(MaskLeading(MakeMask<8, 3>(0x8000000000000000)), 0);
  EXPECT_EQ(MaskTrailing(MakeMask<8, 3>(0x8000000000000000)), 7);
}

std::vector<uint32_t> GroupMatch(const CWISS_ControlByte* group, CWISS_h2_t h) {
  auto g = CWISS_Group_new(group);
  return MaskBits(CWISS_Group_Match(&g, h));
}

std::vector<uint32_t> GroupMatchEmpty(const CWISS_ControlByte* group) {
  auto g = CWISS_Group_new(group);
  return MaskBits(CWISS_Group_MatchEmpty(&g));
}

std::vector<uint32_t> GroupMatchEmptyOrDeleted(const CWISS_ControlByte* group) {
  auto g = CWISS_Group_new(group);
  return MaskBits(CWISS_Group_MatchEmptyOrDeleted(&g));
}

TEST(Group, EmptyGroup) {
  for (CWISS_h2_t h = 0; h != 128; ++h) {
    EXPECT_THAT(GroupMatch(CWISS_EmptyGroup(), h), IsEmpty());
  }
}

CWISS_ControlByte Control(int i) { return static_cast<CWISS_ControlByte>(i); }

TEST(Group, Match) {
  if (CWISS_Group_kWidth == 16) {
    CWISS_ControlByte group[] = {
        CWISS_kEmpty, Control(1), CWISS_kDeleted,  Control(3),
        CWISS_kEmpty, Control(5), CWISS_kSentinel, Control(7),
        Control(7),   Control(5), Control(3),      Control(1),
        Control(1),   Control(1), Control(1),      Control(1)};
    EXPECT_THAT(GroupMatch(group, 0), ElementsAre());
    EXPECT_THAT(GroupMatch(group, 1), ElementsAre(1, 11, 12, 13, 14, 15));
    EXPECT_THAT(GroupMatch(group, 3), ElementsAre(3, 10));
    EXPECT_THAT(GroupMatch(group, 5), ElementsAre(5, 9));
    EXPECT_THAT(GroupMatch(group, 7), ElementsAre(7, 8));
  } else if (CWISS_Group_kWidth == 8) {
    CWISS_ControlByte group[] = {CWISS_kEmpty,    Control(1), Control(2),
                                 CWISS_kDeleted,  Control(2), Control(1),
                                 CWISS_kSentinel, Control(1)};
    EXPECT_THAT(GroupMatch(group, 0), ElementsAre());
    EXPECT_THAT(GroupMatch(group, 1), ElementsAre(1, 5, 7));
    EXPECT_THAT(GroupMatch(group, 2), ElementsAre(2, 4));
  } else {
    FAIL() << "No test coverage for CWISS_Group_kWidth == "
           << CWISS_Group_kWidth;
  }
}

TEST(Group, MatchEmpty) {
  if (CWISS_Group_kWidth == 16) {
    CWISS_ControlByte group[] = {
        CWISS_kEmpty, Control(1), CWISS_kDeleted,  Control(3),
        CWISS_kEmpty, Control(5), CWISS_kSentinel, Control(7),
        Control(7),   Control(5), Control(3),      Control(1),
        Control(1),   Control(1), Control(1),      Control(1)};
    EXPECT_THAT(GroupMatchEmpty(group), ElementsAre(0, 4));
  } else if (CWISS_Group_kWidth == 8) {
    CWISS_ControlByte group[] = {CWISS_kEmpty,    Control(1), Control(2),
                                 CWISS_kDeleted,  Control(2), Control(1),
                                 CWISS_kSentinel, Control(1)};
    EXPECT_THAT(GroupMatchEmpty(group), ElementsAre(0));
  } else {
    FAIL() << "No test coverage for CWISS_Group_kWidth == "
           << CWISS_Group_kWidth;
  }
}

TEST(Group, MatchEmptyOrDeleted) {
  if (CWISS_Group_kWidth == 16) {
    CWISS_ControlByte group[] = {
        CWISS_kEmpty, Control(1), CWISS_kDeleted,  Control(3),
        CWISS_kEmpty, Control(5), CWISS_kSentinel, Control(7),
        Control(7),   Control(5), Control(3),      Control(1),
        Control(1),   Control(1), Control(1),      Control(1)};
    EXPECT_THAT(GroupMatchEmptyOrDeleted(group), ElementsAre(0, 2, 4));
  } else if (CWISS_Group_kWidth == 8) {
    CWISS_ControlByte group[] = {CWISS_kEmpty,    Control(1), Control(2),
                                 CWISS_kDeleted,  Control(2), Control(1),
                                 CWISS_kSentinel, Control(1)};
    EXPECT_THAT(GroupMatchEmptyOrDeleted(group), ElementsAre(0, 3));
  } else {
    FAIL() << "No test coverage for CWISS_Group_kWidth == "
           << CWISS_Group_kWidth;
  }
}

TEST(Batch, DropDeletes) {
  constexpr size_t kCapacity = 63;
  constexpr size_t kGroupWidth = CWISS_Group_kWidth;

  std::vector<CWISS_ControlByte> ctrl(kCapacity + 1 + kGroupWidth);
  ctrl[kCapacity] = CWISS_kSentinel;

  std::vector<CWISS_ControlByte> pattern = {
      CWISS_kEmpty, Control(2), CWISS_kDeleted, Control(2),
      CWISS_kEmpty, Control(1), CWISS_kDeleted};
  for (size_t i = 0; i != kCapacity; ++i) {
    ctrl[i] = pattern[i % pattern.size()];
    if (i < kGroupWidth - 1) {
      ctrl[i + kCapacity + 1] = pattern[i % pattern.size()];
    }
  }

  CWISS_ConvertDeletedToEmptyAndFullToDeleted(ctrl.data(), kCapacity);
  ASSERT_EQ(ctrl[kCapacity], CWISS_kSentinel);

  for (size_t i = 0; i < kCapacity + kGroupWidth; ++i) {
    CWISS_ControlByte expected = pattern[i % (kCapacity + 1) % pattern.size()];
    if (i == kCapacity) {
      expected = CWISS_kSentinel;
    }
    if (expected == CWISS_kDeleted) {
      expected = CWISS_kEmpty;
    }
    if (CWISS_IsFull(expected)) {
      expected = CWISS_kDeleted;
    }

    EXPECT_EQ(ctrl[i], expected);
  }
}

TEST(Group, CountLeadingEmptyOrDeleted) {
  const std::vector<CWISS_ControlByte> empty_examples = {CWISS_kEmpty,
                                                         CWISS_kDeleted};
  const std::vector<CWISS_ControlByte> full_examples = {
      Control(0), Control(1), Control(2),   Control(3),
      Control(5), Control(9), Control(127), CWISS_kSentinel};

  for (CWISS_ControlByte empty : empty_examples) {
    std::vector<CWISS_ControlByte> e(CWISS_Group_kWidth, empty);
    auto g = CWISS_Group_new(e.data());
    EXPECT_EQ(CWISS_Group_kWidth, CWISS_Group_CountLeadingEmptyOrDeleted(&g));

    for (CWISS_ControlByte full : full_examples) {
      for (size_t i = 0; i != CWISS_Group_kWidth; ++i) {
        std::vector<CWISS_ControlByte> f(CWISS_Group_kWidth, empty);
        f[i] = full;

        auto g = CWISS_Group_new(f.data());
        EXPECT_EQ(i, CWISS_Group_CountLeadingEmptyOrDeleted(&g));
      }

      std::vector<CWISS_ControlByte> f(CWISS_Group_kWidth, empty);
      f[CWISS_Group_kWidth * 2 / 3] = full;
      f[CWISS_Group_kWidth / 2] = full;

      auto g = CWISS_Group_new(f.data());
      EXPECT_EQ(CWISS_Group_kWidth / 2,
                CWISS_Group_CountLeadingEmptyOrDeleted(&g));
    }
  }
}

CWISS_DECLARE_HASHSET_WITH(StringTable, std::string,
                           (FlatPolicy<std::string, HashStdString>()));
CWISS_DECLARE_HASHSET_WITH(IntTable, int64_t, FlatPolicy<int64_t>());

TABLE_HELPERS(StringTable);
TABLE_HELPERS(IntTable);

inline size_t StringTable_View_hash(const std::string_view* self) {
  return HashStdString{}(*self);
}

inline bool StringTable_View_eq(const std::string_view* self,
                                const std::string* that) {
  return *self == *that;
}

CWISS_DECLARE_LOOKUP_NAMED(StringTable, View, std::string_view);

struct BadFastHash {
  size_t operator()(const int&) { return 0; }
};

CWISS_DECLARE_HASHSET_WITH(BadTable, int, (FlatPolicy<int, BadFastHash>()));
TABLE_HELPERS(BadTable);

TEST(Table, Empty) {
  auto t = IntTable_new(0);
  absl::Cleanup c_ = [&] { IntTable_destroy(&t); };

  EXPECT_EQ(IntTable_size(&t), 0);
  EXPECT_TRUE(IntTable_empty(&t));
}

TEST(Table, LookupEmpty) {
  auto t = IntTable_new(0);
  absl::Cleanup c_ = [&] { IntTable_destroy(&t); };

  EXPECT_FALSE(Find(t, 0));
}

TEST(Table, Insert1) {
  auto t = IntTable_new(0);
  absl::Cleanup c_ = [&] { IntTable_destroy(&t); };

  EXPECT_FALSE(Find(t, 0));

  auto [val1, inserted1] = Insert(t, 0);
  EXPECT_TRUE(inserted1);
  EXPECT_EQ(*val1, 0);
  EXPECT_EQ(IntTable_size(&t), 1);

  EXPECT_EQ(*Find(t, 0), 0);
}

TEST(Table, Insert2) {
  auto t = IntTable_new(0);
  absl::Cleanup c_ = [&] { IntTable_destroy(&t); };

  EXPECT_FALSE(Find(t, 0));

  auto [val1, inserted1] = Insert(t, 0);
  EXPECT_TRUE(inserted1);
  EXPECT_EQ(*val1, 0);
  EXPECT_EQ(IntTable_size(&t), 1);

  EXPECT_FALSE(Find(t, 1));

  auto [val2, inserted2] = Insert(t, 1);
  EXPECT_TRUE(inserted2);
  EXPECT_EQ(*val2, 1);
  EXPECT_EQ(IntTable_size(&t), 2);

  EXPECT_EQ(*Find(t, 0), 0);
  EXPECT_EQ(*Find(t, 1), 1);
}

TEST(Table, InsertCollision) {
  auto t = BadTable_new(0);
  absl::Cleanup c_ = [&] { BadTable_destroy(&t); };

  EXPECT_FALSE(Find(t, 0));

  auto [val1, inserted1] = Insert(t, 0);
  EXPECT_TRUE(inserted1);
  EXPECT_EQ(*val1, 0);
  EXPECT_EQ(BadTable_size(&t), 1);

  EXPECT_FALSE(Find(t, 1));

  auto [val2, inserted2] = Insert(t, 1);
  EXPECT_TRUE(inserted2);
  EXPECT_EQ(*val2, 1);
  EXPECT_EQ(BadTable_size(&t), 2);

  EXPECT_EQ(*Find(t, 0), 0);
  EXPECT_EQ(*Find(t, 1), 1);
}

// Test that we do not add existent element in case we need to search through
// many groups with deleted elements
TEST(Table, InsertCollisionAndFindAfterDelete) {
  auto t = BadTable_new(0);  // all elements go to the same group.
  absl::Cleanup c_ = [&] { BadTable_destroy(&t); };

  // Have at least 2 groups with CWISS_Group_kWidth collisions
  // plus some extra collisions in the last group.
  constexpr size_t kNumInserts = CWISS_Group_kWidth * 2 + 5;
  for (size_t i = 0; i < kNumInserts; ++i) {
    auto [val, inserted] = Insert(t, i);
    EXPECT_TRUE(inserted);
    EXPECT_EQ(*val, i);
    EXPECT_EQ(BadTable_size(&t), i + 1);
  }

  // Remove elements one by one and check
  // that we still can find all other elements.
  for (size_t i = 0; i < kNumInserts; ++i) {
    EXPECT_TRUE(Erase(t, i)) << i;

    for (size_t j = i + 1; j < kNumInserts; ++j) {
      EXPECT_EQ(*Find(t, j), j);

      auto [val, inserted] = Insert(t, j);
      EXPECT_FALSE(inserted) << i << " " << j;
      EXPECT_EQ(*val, j);
      EXPECT_EQ(BadTable_size(&t), kNumInserts - i - 1);
    }
  }
  EXPECT_TRUE(BadTable_empty(&t));
}

TEST(Table, InsertWithinCapacity) {
  auto t = IntTable_new(0);
  absl::Cleanup c_ = [&] { IntTable_destroy(&t); };

  IntTable_reserve(&t, 10);
  size_t original_capacity = IntTable_capacity(&t);
  const auto addr = [&](int i) {
    return reinterpret_cast<uintptr_t>(Find(t, i));
  };

  // Inserting an element does not change capacity.
  Insert(t, 0);
  EXPECT_THAT(IntTable_capacity(&t), original_capacity);
  const uintptr_t original_addr_0 = addr(0);

  // Inserting another element does not rehash.
  Insert(t, 0);
  EXPECT_THAT(IntTable_capacity(&t), original_capacity);
  EXPECT_THAT(addr(0), original_addr_0);

  // Inserting lots of duplicate elements does not rehash.
  for (int i = 0; i < 100; ++i) {
    Insert(t, i % 10);
  }
  EXPECT_THAT(IntTable_capacity(&t), original_capacity);
  EXPECT_THAT(addr(0), original_addr_0);
}

TEST(Table, LazyEmplace) {
  auto t = StringTable_new(0);
  absl::Cleanup c_ = [&] { StringTable_destroy(&t); };

  std::string_view sv = "abc";
  auto res = StringTable_deferred_insert_by_View(&t, &sv);
  EXPECT_TRUE(res.inserted);
  new (StringTable_Iter_get(&res.iter)) std::string(sv);

  auto it = StringTable_find_by_View(&t, &sv);
  EXPECT_TRUE(StringTable_Iter_get(&it));
  EXPECT_EQ(*StringTable_Iter_get(&it), sv);

  EXPECT_FALSE(StringTable_deferred_insert_by_View(&t, &sv).inserted);
}

TEST(Table, ContainsEmpty) {
  auto t = IntTable_new(0);
  absl::Cleanup c_ = [&] { IntTable_destroy(&t); };

  int64_t k0 = 0;
  EXPECT_FALSE(IntTable_contains(&t, &k0));
}

TEST(Table, Contains1) {
  auto t = IntTable_new(0);
  absl::Cleanup c_ = [&] { IntTable_destroy(&t); };

  int64_t k0 = 0, k1 = 1;
  EXPECT_THAT(Insert(t, 0), Pair(_, true));
  EXPECT_TRUE(IntTable_contains(&t, &k0));
  EXPECT_FALSE(IntTable_contains(&t, &k1));

  EXPECT_TRUE(Erase(t, 0));
  EXPECT_FALSE(IntTable_contains(&t, &k0));
}

TEST(Table, Contains2) {
  auto t = IntTable_new(0);
  absl::Cleanup c_ = [&] { IntTable_destroy(&t); };

  int64_t k0 = 0, k1 = 1;
  EXPECT_THAT(Insert(t, 0), Pair(_, true));
  EXPECT_TRUE(IntTable_contains(&t, &k0));
  EXPECT_FALSE(IntTable_contains(&t, &k1));

  IntTable_clear(&t);
  EXPECT_FALSE(IntTable_contains(&t, &k0));
}

// Returns the largest m such that a table with m elements has the same number
// of buckets as a table with n elements.
size_t MaxDensitySize(size_t n) {
  auto t = IntTable_new(n);
  absl::Cleanup c_ = [&] { IntTable_destroy(&t); };

  for (size_t i = 0; i != n; ++i) {
    Insert(t, i);
  }
  const size_t c = IntTable_capacity(&t);
  while (c == IntTable_capacity(&t)) {
    Insert(t, n++);
  }
  return IntTable_size(&t) - 1;
}

struct Modulo1000Hash {
  size_t operator()(const int& p) { return p % 1000; }
};

CWISS_DECLARE_HASHSET_WITH(Modulo1000HashTable, int,
                           (FlatPolicy<int, Modulo1000Hash>()));
TABLE_HELPERS(Modulo1000HashTable);

// Test that rehash with no resize happen in case of many deleted slots.
TEST(Table, RehashWithNoResize) {
  auto t = Modulo1000HashTable_new(0);
  absl::Cleanup c_ = [&] { Modulo1000HashTable_destroy(&t); };

  // Adding the same length (and the same hash) strings
  // to have at least kMinFullGroups groups
  // with CWISS_Group_kWidth collisions. Then fill up to MaxDensitySize;
  const size_t kMinFullGroups = 7;
  std::vector<int> keys;
  for (size_t i = 0; i < MaxDensitySize(CWISS_Group_kWidth * kMinFullGroups);
       ++i) {
    int k = i * 1000;
    Insert(t, k);
    keys.push_back(k);
  }
  const size_t capacity = Modulo1000HashTable_capacity(&t);

  // Remove elements from all groups except the first and the last one.
  // All elements removed from full groups will be marked as CWISS_kDeleted.
  const size_t erase_begin = CWISS_Group_kWidth / 2;
  const size_t erase_end =
      (Modulo1000HashTable_size(&t) / CWISS_Group_kWidth - 1) *
      CWISS_Group_kWidth;
  for (size_t i = erase_begin; i < erase_end; ++i) {
    EXPECT_TRUE(Erase(t, keys[i])) << i;
  }
  keys.erase(keys.begin() + erase_begin, keys.begin() + erase_end);

  auto last_key = keys.back();
  auto probes = [&] {
    return GetHashtableDebugNumProbes(Modulo1000HashTable_policy(), &t.set_,
                                      &last_key);
  };
  size_t last_key_num_probes = probes();

  // Make sure that we have to make a lot of probes for last key.
  ASSERT_GE(last_key_num_probes, kMinFullGroups);

  int x = 1;
  // Insert and erase one element, before in-place rehash happens.
  while (last_key_num_probes == probes()) {
    Insert(t, x);
    ASSERT_EQ(Modulo1000HashTable_capacity(&t), capacity);
    // All elements should be there.
    ASSERT_TRUE(Find(t, x)) << x;
    for (const auto& k : keys) {
      ASSERT_TRUE(Find(t, k)) << k;
    }
    Erase(t, x++);
  }
}

TEST(Table, InsertEraseStressTest) {
  auto t = IntTable_new(0);
  absl::Cleanup c_ = [&] { IntTable_destroy(&t); };

  const size_t kMinElementCount = 250;
  std::deque<int> keys;
  size_t i = 0;
  for (; i < MaxDensitySize(kMinElementCount); ++i) {
    Insert(t, i);
    keys.push_back(i);
  }
  const size_t kNumIterations = 1000000;
  for (; i < kNumIterations; ++i) {
    ASSERT_TRUE(Erase(t, keys.front()));
    keys.pop_front();
    Insert(t, i);
    keys.push_back(i);
  }
}

TEST(Table, LargeTable) {
  // Abseil uses a max of 100'000, but this seems very slow with -O1.
  auto t = IntTable_new(0);
  absl::Cleanup c_ = [&] { IntTable_destroy(&t); };

  for (int i = 0; i != 10000; ++i) {
    Insert(t, i << 20);
  }
  for (int i = 0; i != 10000; ++i) {
    auto it = Find(t, i << 20);
    ASSERT_TRUE(it);
    ASSERT_EQ(*it, i << 20);
  }
}

// Timeout if copy is quadratic as it was in Rust.
TEST(Table, EnsureNonQuadraticAsInRust) {
  static const size_t kLargeSize = 1 << 15;

  auto t = IntTable_new(0);
  absl::Cleanup c_ = [&] { IntTable_destroy(&t); };

  for (size_t i = 0; i != kLargeSize; ++i) {
    Insert(t, i);
  }

  // If this is quadratic, the test will timeout.
  auto t2 = IntTable_new(0);
  absl::Cleanup c2_ = [&] { IntTable_destroy(&t2); };

  for (auto it = IntTable_citer(&t); IntTable_CIter_get(&it);
       IntTable_CIter_next(&it)) {
    IntTable_insert(&t2, IntTable_CIter_get(&it));
  }
}

TEST(Table, ClearBug) {
  auto t = IntTable_new(0);
  absl::Cleanup c_ = [&] { IntTable_destroy(&t); };

  constexpr size_t capacity = CWISS_Group_kWidth - 1;
  constexpr size_t max_size = capacity / 2 + 1;

  for (size_t i = 0; i < max_size; ++i) {
    Insert(t, i);
  }
  ASSERT_EQ(IntTable_capacity(&t), capacity);

  intptr_t original = reinterpret_cast<intptr_t>(Find(t, 2));

  IntTable_clear(&t);
  ASSERT_EQ(IntTable_capacity(&t), capacity);

  for (size_t i = 0; i < max_size; ++i) {
    Insert(t, i);
  }
  ASSERT_EQ(IntTable_capacity(&t), capacity);

  intptr_t second = reinterpret_cast<intptr_t>(Find(t, 2));
  // We are checking that original and second are close enough to each other
  // that they are probably still in the same group.  This is not strictly
  // guaranteed.
  EXPECT_LT(std::abs(original - second), capacity * sizeof(int64_t));
}

TEST(Table, Erase) {
  auto t = IntTable_new(0);
  absl::Cleanup c_ = [&] { IntTable_destroy(&t); };

  EXPECT_FALSE(Find(t, 0));

  auto [val, inserted] = Insert(t, 0);
  EXPECT_TRUE(inserted);
  EXPECT_EQ(*val, 0);
  EXPECT_EQ(IntTable_size(&t), 1);

  Erase(t, 0);
  EXPECT_FALSE(Find(t, 0));
}

TEST(Table, EraseMaintainsValidIterator) {
  auto t = IntTable_new(0);
  absl::Cleanup c_ = [&] { IntTable_destroy(&t); };

  const int kNumElements = 100;
  for (int i = 0; i < kNumElements; i++) {
    EXPECT_THAT(Insert(t, i), Pair(_, true));
  }
  EXPECT_EQ(IntTable_size(&t), kNumElements);

  int num_erase_calls = 0;
  auto it = IntTable_iter(&t);
  while (IntTable_Iter_get(&it)) {
    auto prev = it;
    IntTable_Iter_next(&it);
    IntTable_erase_at(prev);
    ++num_erase_calls;
  }

  EXPECT_TRUE(IntTable_empty(&t));
  EXPECT_EQ(num_erase_calls, kNumElements);
}

TEST(Table, EraseCollision) {
  auto t = BadTable_new(0);
  absl::Cleanup c_ = [&] { BadTable_destroy(&t); };

  // 1 2 3
  Insert(t, 1);
  Insert(t, 2);
  Insert(t, 3);
  EXPECT_EQ(*Find(t, 1), 1);
  EXPECT_EQ(*Find(t, 2), 2);
  EXPECT_EQ(*Find(t, 3), 3);

  // 1 DELETED 3
  Erase(t, 2);
  EXPECT_EQ(*Find(t, 1), 1);
  EXPECT_FALSE(Find(t, 2));
  EXPECT_EQ(*Find(t, 3), 3);
  EXPECT_EQ(BadTable_size(&t), 2);

  // DELETED DELETED 3
  Erase(t, 1);
  EXPECT_FALSE(Find(t, 1));
  EXPECT_FALSE(Find(t, 2));
  EXPECT_EQ(*Find(t, 3), 3);
  EXPECT_EQ(BadTable_size(&t), 1);

  // DELETED DELETED DELETED
  Erase(t, 3);
  EXPECT_FALSE(Find(t, 1));
  EXPECT_FALSE(Find(t, 2));
  EXPECT_FALSE(Find(t, 3));
  EXPECT_EQ(BadTable_size(&t), 0);
}

TEST(Table, EraseInsertProbing) {
  auto t = BadTable_new(100);
  absl::Cleanup c_ = [&] { BadTable_destroy(&t); };

  // 1 2 3 4
  Insert(t, 1);
  Insert(t, 2);
  Insert(t, 3);
  Insert(t, 4);

  // 1 DELETED 3 DELETED
  Erase(t, 2);
  Erase(t, 4);

  // 1 10 3 11 12
  Insert(t, 10);
  Insert(t, 11);
  Insert(t, 12);

  EXPECT_EQ(BadTable_size(&t), 5);

  EXPECT_THAT(Collect(t), UnorderedElementsAre(1, 10, 3, 11, 12));
}

TEST(Table, Clear) {
  auto t = IntTable_new(0);
  absl::Cleanup c_ = [&] { IntTable_destroy(&t); };

  EXPECT_FALSE(Find(t, 0));

  IntTable_clear(&t);
  EXPECT_FALSE(Find(t, 0));

  EXPECT_THAT(Insert(t, 0), Pair(_, true));
  EXPECT_EQ(IntTable_size(&t), 1);

  IntTable_clear(&t);
  EXPECT_EQ(IntTable_size(&t), 0);
  EXPECT_FALSE(Find(t, 0));
}

TEST(Table, Rehash) {
  auto t = IntTable_new(0);
  absl::Cleanup c_ = [&] { IntTable_destroy(&t); };

  EXPECT_FALSE(Find(t, 0));

  Insert(t, 0);
  Insert(t, 1);
  EXPECT_EQ(IntTable_size(&t), 2);

  IntTable_rehash(&t, 128);

  EXPECT_EQ(IntTable_size(&t), 2);
  EXPECT_THAT(*Find(t, 0), 0);
  EXPECT_THAT(*Find(t, 1), 1);
}

TEST(Table, RehashDoesNotRehashWhenNotNecessary) {
  auto t = IntTable_new(0);
  absl::Cleanup c_ = [&] { IntTable_destroy(&t); };

  Insert(t, 0);
  Insert(t, 1);
  auto* p = Find(t, 0);
  IntTable_rehash(&t, 1);
  EXPECT_EQ(p, Find(t, 0));
}

TEST(Table, RehashZeroDoesNotAllocateOnEmptyTable) {
  auto t = IntTable_new(0);
  absl::Cleanup c_ = [&] { IntTable_destroy(&t); };

  IntTable_rehash(&t, 0);
  EXPECT_EQ(IntTable_capacity(&t), 0);
}

TEST(Table, RehashZeroDeallocatesEmptyTable) {
  auto t = IntTable_new(0);
  absl::Cleanup c_ = [&] { IntTable_destroy(&t); };

  Insert(t, 0);
  IntTable_clear(&t);
  EXPECT_NE(IntTable_capacity(&t), 0);
  IntTable_rehash(&t, 0);
  EXPECT_EQ(IntTable_capacity(&t), 0);
}

TEST(Table, RehashZeroForcesRehash) {
  auto t = IntTable_new(0);
  absl::Cleanup c_ = [&] { IntTable_destroy(&t); };

  Insert(t, 0);
  Insert(t, 1);
  auto* p = Find(t, 0);
  IntTable_rehash(&t, 1);
  EXPECT_EQ(p, Find(t, 0));
}

TEST(Table, CopyConstruct) {
  auto t = IntTable_new(0);
  absl::Cleanup c_ = [&] { IntTable_destroy(&t); };

  Insert(t, 0);
  EXPECT_EQ(IntTable_size(&t), 1);

  auto u = IntTable_dup(&t);
  absl::Cleanup c2_ = [&] { IntTable_destroy(&u); };

  EXPECT_EQ(IntTable_size(&u), 1);
  EXPECT_EQ(*Find(u, 0), 0);
}

// TEST(Table, Equality) {
//   StringTable t;
//   std::vector<std::pair<std::string, std::string>> v = {{"a", "b"},
//                                                         {"aa", "bb"}};
//   t.insert(std::begin(v), std::end(v));
//   StringTable u = t;
//   EXPECT_EQ(u, t);
// }

// TEST(Table, Equality2) {
//   StringTable t;
//   std::vector<std::pair<std::string, std::string>> v1 = {{"a", "b"},
//                                                          {"aa", "bb"}};
//   t.insert(std::begin(v1), std::end(v1));
//   StringTable u;
//   std::vector<std::pair<std::string, std::string>> v2 = {{"a", "a"},
//                                                          {"aa", "aa"}};
//   u.insert(std::begin(v2), std::end(v2));
//   EXPECT_NE(u, t);
// }

// TEST(Table, Equality3) {
//   StringTable t;
//   std::vector<std::pair<std::string, std::string>> v1 = {{"b", "b"},
//                                                          {"bb", "bb"}};
//   t.insert(std::begin(v1), std::end(v1));
//   StringTable u;
//   std::vector<std::pair<std::string, std::string>> v2 = {{"a", "a"},
//                                                          {"aa", "aa"}};
//   u.insert(std::begin(v2), std::end(v2));
//   EXPECT_NE(u, t);
// }

TEST(Table, NumDeletedRegression) {
  auto t = IntTable_new(0);
  absl::Cleanup c_ = [&] { IntTable_destroy(&t); };

  Insert(t, 0);
  Erase(t, 0);
  // construct over a deleted slot.
  Insert(t, 0);
  IntTable_clear(&t);
}

TEST(Table, FindFullDeletedRegression) {
  auto t = IntTable_new(0);
  absl::Cleanup c_ = [&] { IntTable_destroy(&t); };

  for (int i = 0; i < 1000; ++i) {
    Insert(t, i);
    Erase(t, i);
  }
  EXPECT_EQ(IntTable_size(&t), 0);
}

TEST(Table, ReplacingDeletedSlotDoesNotRehash) {
  size_t n = MaxDensitySize(1);
  auto t = IntTable_new(n);
  absl::Cleanup c_ = [&] { IntTable_destroy(&t); };

  const size_t c = IntTable_capacity(&t);
  for (size_t i = 0; i != n; ++i) {
    Insert(t, i);
  }
  EXPECT_EQ(IntTable_capacity(&t), c) << "rehashing threshold = " << n;
  Erase(t, 0);
  Insert(t, 0);
  EXPECT_EQ(IntTable_capacity(&t), c) << "rehashing threshold = " << n;
}

// Abseil comment: TODO(alkis): Expand iterator tests.

TEST(Iterator, Iterates) {
  auto t = IntTable_new(0);
  absl::Cleanup c_ = [&] { IntTable_destroy(&t); };

  for (size_t i = 3; i != 6; ++i) {
    EXPECT_THAT(Insert(t, i), Pair(_, true));
  }
  EXPECT_THAT(Collect(t), UnorderedElementsAre(3, 4, 5));
}

// TEST(Table, Merge) {
//   StringTable t1, t2;
//   t1.emplace("0", "-0");
//   t1.emplace("1", "-1");
//   t2.emplace("0", "~0");
//   t2.emplace("2", "~2");

//   EXPECT_THAT(t1, UnorderedElementsAre(Pair("0", "-0"), Pair("1", "-1")));
//   EXPECT_THAT(t2, UnorderedElementsAre(Pair("0", "~0"), Pair("2", "~2")));

//   t1.merge(t2);
//   EXPECT_THAT(t1, UnorderedElementsAre(Pair("0", "-0"), Pair("1", "-1"),
//                                        Pair("2", "~2")));
//   EXPECT_THAT(t2, UnorderedElementsAre(Pair("0", "~0")));
// }

IntTable MakeSimpleTable(size_t size) {
  auto t = IntTable_new(0);
  for (size_t i = 0; i < size; ++i) {
    Insert(t, i);
  }
  return t;
}

// These IterationOrderChanges tests depend on non-deterministic behavior.
// We are injecting non-determinism from the pointer of the table, but do so in
// a way that only the page matters. We have to retry enough times to make sure
// we are touching different memory pages to cause the ordering to change.
// We also need to keep the old tables around to avoid getting the same memory
// blocks over and over.
TEST(Table, IterationOrderChangesByInstance) {
  for (size_t size : {2, 6, 12, 20}) {
    auto reference_table = MakeSimpleTable(size);
    auto reference = Collect(reference_table);

    std::vector<IntTable> tables;
    absl::Cleanup c_ = [&] {
      IntTable_destroy(&reference_table);
      for (auto& t : tables) {
        IntTable_destroy(&t);
      }
    };

    bool found_difference = false;
    for (int i = 0; !found_difference && i < 5000; ++i) {
      tables.push_back(MakeSimpleTable(size));
      found_difference = Collect(tables.back()) != reference;
    }
    if (!found_difference) {
      FAIL()
          << "Iteration order remained the same across many attempts with size "
          << size;
    }
  }
}

TEST(Table, IterationOrderChangesOnRehash) {
  std::vector<IntTable> garbage;
  absl::Cleanup c_ = [&] {
    for (auto& t : garbage) {
      IntTable_destroy(&t);
    }
  };

  for (int i = 0; i < 5000; ++i) {
    // Insert now to ensure that the destructor runs on test exit.
    garbage.push_back(MakeSimpleTable(20));
    auto reference = Collect(garbage.back());

    // Force rehash to the same size.
    IntTable_rehash(&garbage.back(), 0);
    auto trial = Collect(garbage.back());

    if (trial != reference) {
      // We are done.
      return;
    }
  }
  FAIL() << "Iteration order remained the same across many attempts.";
}

// Verify that pointers are invalidated as soon as a second element is inserted.
// This prevents dependency on pointer stability on small tables.
TEST(Table, UnstablePointers) {
  auto t = IntTable_new(0);
  absl::Cleanup c_ = [&] { IntTable_destroy(&t); };

  const auto addr = [&](int i) {
    return reinterpret_cast<uintptr_t>(Find(t, i));
  };

  Insert(t, 0);
  const uintptr_t old_ptr = addr(0);

  // This causes a rehash.
  Insert(t, 1);

  EXPECT_NE(old_ptr, addr(0));
}

// Confirm that we assert if we try to erase() end().
TEST(TableDeathTest, EraseOfEndAsserts) {
  // Use an assert with side-effects to figure out if they are actually enabled.
  bool assert_enabled = false;
  assert([&]() {
    assert_enabled = true;
    return true;
  }());
  if (!assert_enabled) return;

  auto t = IntTable_new(0);
  absl::Cleanup c_ = [&] { IntTable_destroy(&t); };

  auto it = IntTable_iter(&t);

  // Extra simple "regexp" as regexp support is highly varied across platforms.
  constexpr char kDeathMsg[] = "Invalid operation on iterator";
  EXPECT_DEATH_IF_SUPPORTED(IntTable_erase_at(it), kDeathMsg);
}

// #if defined(ABSL_INTERNAL_HASHTABLEZ_SAMPLE)
// TEST(RawHashSamplerTest, Sample) {
//   // Enable the feature even if the prod default is off.
//   SetHashtablezEnabled(true);
//   SetHashtablezSampleParameter(100);

//   auto& sampler = GlobalHashtablezSampler();
//   size_t start_size = 0;
//   std::unordered_set<const HashtablezInfo*> preexisting_info;
//   start_size += sampler.Iterate([&](const HashtablezInfo& info) {
//     preexisting_info.insert(&info);
//     ++start_size;
//   });

//   std::vector<IntTable> tables;
//   for (int i = 0; i < 1000000; ++i) {
//     tables.emplace_back();

//     const bool do_reserve = (i % 10 > 5);
//     const bool do_rehash = !do_reserve && (i % 10 > 0);

//     if (do_reserve) {
//       // Don't reserve on all tables.
//       tables.back().reserve(10 * (i % 10));
//     }

//     tables.back().insert(1);
//     tables.back().insert(i % 5);

//     if (do_rehash) {
//       // Rehash some other tables.
//       tables.back().rehash(10 * (i % 10));
//     }
//   }
//   size_t end_size = 0;
//   std::unordered_map<size_t, int> observed_checksums;
//   std::unordered_map<ssize_t, int> reservations;
//   end_size += sampler.Iterate([&](const HashtablezInfo& info) {
//     if (preexisting_info.count(&info) == 0) {
//       observed_checksums[info.hashes_bitwise_xor.load(
//           std::memory_order_relaxed)]++;
//       reservations[info.max_reserve.load(std::memory_order_relaxed)]++;
//     }
//     EXPECT_EQ(info.inline_element_size, sizeof(int64_t));
//     ++end_size;
//   });

//   EXPECT_NEAR((end_size - start_size) / static_cast<double>(tables.size()),
//               0.01, 0.005);
//   EXPECT_EQ(observed_checksums.size(), 5);
//   for (const auto& [_, count] : observed_checksums) {
//     EXPECT_NEAR((100 * count) / static_cast<double>(tables.size()), 0.2,
//     0.05);
//   }

//   EXPECT_EQ(reservations.size(), 10);
//   for (const auto& [reservation, count] : reservations) {
//     EXPECT_GE(reservation, 0);
//     EXPECT_LT(reservation, 100);

//     EXPECT_NEAR((100 * count) / static_cast<double>(tables.size()), 0.1,
//     0.05)
//         << reservation;
//   }
// }
// #endif  // ABSL_INTERNAL_HASHTABLEZ_SAMPLE

#if CWISS_HAVE_ASAN
TEST(Sanitizer, PoisoningUnused) {
  auto t = IntTable_new(0);
  absl::Cleanup c_ = [&] { IntTable_destroy(&t); };
  IntTable_reserve(&t, 5);
  // Insert something to force an allocation.
  auto [v, ignored] = Insert(t, 0);

  // Make sure there is something to test.
  auto cap = IntTable_capacity(&t);
  ASSERT_GT(cap, 1);

  // Reach into the set and grab the slots.
  int64_t* slots = reinterpret_cast<int64_t*>(t.set_.slots_);
  for (size_t i = 0; i < cap; ++i) {
    int64_t* slot = slots + i;
    EXPECT_EQ(__asan_address_is_poisoned(slots + i), slot != v) << i;
  }
}

TEST(Sanitizer, PoisoningOnErase) {
  auto t = IntTable_new(0);
  absl::Cleanup c_ = [&] { IntTable_destroy(&t); };
  auto [v, ignored] = Insert(t, 0);

  EXPECT_FALSE(__asan_address_is_poisoned(v));
  Erase(t, 0);
  EXPECT_TRUE(__asan_address_is_poisoned(v));
}
#endif  // CWISS_HAVE_ASAN

CWISS_DECLARE_HASHSET_WITH(Uint8Table, uint8_t, FlatPolicy<uint8_t>());
TABLE_HELPERS(Uint8Table);

TEST(Table, AlignOne) {
  // We previously had a bug in which we were copying a control byte over the
  // first slot when alignof(value_type) is 1. We test repeated
  // insertions/erases and verify that the behavior is correct.
  auto t = Uint8Table_new(0);
  absl::Cleanup c_ = [&] { Uint8Table_destroy(&t); };

  std::unordered_set<uint8_t> verifier;  // NOLINT

  // Do repeated insertions/erases from the table.
  for (int64_t i = 0; i < 100000; ++i) {
    SCOPED_TRACE(i);
    const uint8_t u = (i * -i) & 0xFF;

    auto it = Find(t, u);
    auto verifier_it = verifier.find(u);
    if (it == NULL) {
      ASSERT_EQ(verifier_it, verifier.end());
      Insert(t, u);
      verifier.insert(u);
    } else {
      ASSERT_NE(verifier_it, verifier.end());
      Erase(t, u);
      verifier.erase(verifier_it);
    }
  }

  EXPECT_EQ(Uint8Table_size(&t), verifier.size());
  for (uint8_t u : Collect(t)) {
    EXPECT_EQ(verifier.count(u), 1);
  }
}
}  // namespace
}  // namespace cwisstable
