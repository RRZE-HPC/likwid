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

#ifndef CWISSTABLE_TEST_HELPERS_H_
#define CWISSTABLE_TEST_HELPERS_H_

// Helpers for interacting with C SwissTables from C++.

#include <cstdint>

#include "cwisstable.h"

namespace cwisstable {
template <typename T>
struct DefaultHash {
  size_t operator()(const T& val) {
    CWISS_AbslHash_State state = CWISS_AbslHash_kInit;
    CWISS_AbslHash_Write(&state, &val, sizeof(T));
    return CWISS_AbslHash_Finish(state);
  }
};

struct HashStdString {
  template <typename S>
  size_t operator()(const S& s) {
    CWISS_AbslHash_State state = CWISS_AbslHash_kInit;
    size_t size = s.size();
    CWISS_AbslHash_Write(&state, &size, sizeof(size_t));
    CWISS_AbslHash_Write(&state, s.data(), s.size());
    return CWISS_AbslHash_Finish(state);
  }
};

template <typename T>
struct DefaultEq {
  bool operator()(const T& a, const T& b) { return a == b; }
};

template <typename T, typename Hash, typename Eq>
struct FlatPolicyWrapper {
  CWISS_GCC_PUSH
  CWISS_GCC_ALLOW("-Waddress")
  // clang-format off
  CWISS_DECLARE_FLAT_SET_POLICY(kPolicy, T,
    (modifiers, static constexpr),
    (obj_copy, [](void* dst, const void* src) {
      new (dst) T(*static_cast<const T*>(src));
    }),
    (obj_dtor, [](void* val) {
      static_cast<T*>(val)->~T();
    }),
    (key_hash, [](const void* val) {
      return Hash{}(*static_cast<const T*>(val));
    }),
    (key_eq, [](const void* a, const void* b) {
      return Eq{}(*static_cast<const T*>(a), *static_cast<const T*>(b));
    }),
    (slot_transfer, [](void* dst, void* src) {
      T* old = static_cast<T*>(src);
      new (dst) T(std::move(*old));
      old->~T();
    }));
  // clang-format on
  CWISS_GCC_POP
};

template <typename T, typename Hash = DefaultHash<T>,
          typename Eq = DefaultEq<T>>
constexpr const CWISS_Policy& FlatPolicy() {
  return FlatPolicyWrapper<T, Hash, Eq>::kPolicy;
}

template <typename K, typename V, typename Hash, typename Eq>
struct FlatMapPolicyWrapper {
  CWISS_GCC_PUSH
  CWISS_GCC_ALLOW("-Waddress")
  // clang-format off
  CWISS_DECLARE_FLAT_MAP_POLICY(kPolicy, K, V,
    (modifiers, static constexpr),
    (obj_copy, [](void* dst, const void* src) {
      new (dst) kPolicy_Entry(*static_cast<const kPolicy_Entry*>(src));
    }),
    (obj_dtor, [](void* val) {
      static_cast<kPolicy_Entry*>(val)->~kPolicy_Entry();
    }),
    (key_hash, [](const void* val) {
      return Hash{}(static_cast<const kPolicy_Entry*>(val)->k);
    }),
    (key_eq, [](const void* a, const void* b) {
      return Eq{}(static_cast<const kPolicy_Entry*>(a)->k,
                  static_cast<const kPolicy_Entry*>(b)->k);
    }),
    (slot_transfer, [](void* dst, void* src) {
      kPolicy_Entry* old = static_cast<kPolicy_Entry*>(src);
      new (dst) kPolicy_Entry(std::move(*old));
      old->~kPolicy_Entry();
    }));
  // clang-format on
  CWISS_GCC_POP
};

template <typename K, typename V, typename Hash = DefaultHash<K>,
          typename Eq = DefaultEq<K>>
constexpr const CWISS_Policy& FlatMapPolicy() {
  return FlatMapPolicyWrapper<K, V, Hash, Eq>::kPolicy;
}

// Helpers for doing some operations on tables with minimal pain.
//
// This macro expands to functions that will form an overload set with other
// table types.
#define TABLE_HELPERS(HashSet_)                                                \
  CWISS_INLINE_ALWAYS                                                          \
  inline HashSet_##_Entry* Find(HashSet_& set, const HashSet_##_Key& needle) { \
    auto it = HashSet_##_find(&set, &needle);                                  \
    return HashSet_##_Iter_get(&it);                                           \
  }                                                                            \
  CWISS_INLINE_ALWAYS                                                          \
  inline std::pair<HashSet_##_Entry*, bool> Insert(                            \
      HashSet_& set, const HashSet_##_Entry& value) {                          \
    auto it = HashSet_##_insert(&set, &value);                                 \
    return {HashSet_##_Iter_get(&it.iter), it.inserted};                       \
  }                                                                            \
  CWISS_INLINE_ALWAYS                                                          \
  inline std::pair<HashSet_##_Entry*, bool> LazyInsert(                        \
      HashSet_& set, const HashSet_##_Key& value) {                            \
    auto it = HashSet_##_deferred_insert(&set, &value);                        \
    return {HashSet_##_Iter_get(&it.iter), it.inserted};                       \
  }                                                                            \
  CWISS_INLINE_ALWAYS                                                          \
  inline std::pair<HashSet_##_Entry*, bool> MoveInsert(                        \
      HashSet_& set, HashSet_##_Entry&& value) {                               \
    auto [ptr, inserted] = LazyInsert(set, value);                             \
    if (inserted) {                                                            \
      new (ptr) HashSet_##_Entry(value);                                       \
    }                                                                          \
    return {ptr, inserted};                                                    \
  }                                                                            \
  CWISS_INLINE_ALWAYS                                                          \
  inline bool Erase(HashSet_& set, const HashSet_##_Key& needle) {             \
    return HashSet_##_erase(&set, &needle);                                    \
  }                                                                            \
  CWISS_INLINE_ALWAYS                                                          \
  inline std::vector<HashSet_##_Entry> Collect(const HashSet_& set) {          \
    std::vector<HashSet_##_Entry> items;                                       \
    items.reserve(HashSet_##_size(&set));                                      \
    for (auto it = HashSet_##_citer(&set); HashSet_##_CIter_get(&it);          \
         HashSet_##_CIter_next(&it)) {                                         \
      items.push_back(*HashSet_##_CIter_get(&it));                             \
    }                                                                          \
    return items;                                                              \
  }
}  // namespace cwisstable

#endif  // CWISSTABLE_TEST_HELPERS_H_
