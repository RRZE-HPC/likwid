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

#ifndef CWISSTABLE_DECLARE_H_
#define CWISSTABLE_DECLARE_H_

#include <stdbool.h>
#include <stddef.h>

#include "cwisstable/internal/base.h"
#include "cwisstable/internal/raw_table.h"
#include "cwisstable/policy.h"

/// SwissTable code generation macros.
///
/// This file is the entry-point for users of `cwisstable`. It exports six
/// macros for generating different kinds of tables. Four correspond to Abseil's
/// four SwissTable containers:
///
/// - `CWISS_DECLARE_FLAT_HASHSET(Set, Type)`
/// - `CWISS_DECLARE_FLAT_HASHMAP(Map, Key, Value)`
/// - `CWISS_DECLARE_NODE_HASHSET(Set, Type)`
/// - `CWISS_DECLARE_NODE_HASHMAP(Map, Key, Value)`
///
/// These expand to a type (with the same name as the first argument) and and
/// a collection of strongly-typed functions associated to it (the generated
/// API is described below). These macros use the default policy (see policy.h)
/// for each of the four containers; custom policies may be used instead via
/// the following macros:
///
/// - `CWISS_DECLARE_HASHSET_WITH(Set, Type, kPolicy)`
/// - `CWISS_DECLARE_HASHMAP_WITH(Map, Key, Value, kPolicy)`
///
/// `kPolicy` must be a constant global variable referring to an appropriate
/// property for the element types of the container.
///
/// The generated API is safe: the functions are well-typed and automatically
/// pass the correct policy pointer. Because the pointer is a constant
/// expression, it promotes devirtualization when inlining.
///
/// # Generated API
///
/// See `set_api.h` and `map_api.h` for detailed listings of what the generated
/// APIs look like.

CWISS_BEGIN
CWISS_BEGIN_EXTERN

/// Generates a new hash set type with inline storage and the default
/// plain-old-data policies.
///
/// See header documentation for examples of generated API.
#define CWISS_DECLARE_FLAT_HASHSET(HashSet_, Type_)                 \
  CWISS_DECLARE_FLAT_SET_POLICY(HashSet_##_kPolicy, Type_, (_, _)); \
  CWISS_DECLARE_HASHSET_WITH(HashSet_, Type_, HashSet_##_kPolicy)

/// Generates a new hash set type with outline storage and the default
/// plain-old-data policies.
///
/// See header documentation for examples of generated API.
#define CWISS_DECLARE_NODE_HASHSET(HashSet_, Type_)                 \
  CWISS_DECLARE_NODE_SET_POLICY(HashSet_##_kPolicy, Type_, (_, _)); \
  CWISS_DECLARE_HASHSET_WITH(HashSet_, Type_, HashSet_##_kPolicy)

/// Generates a new hash map type with inline storage and the default
/// plain-old-data policies.
///
/// See header documentation for examples of generated API.
#define CWISS_DECLARE_FLAT_HASHMAP(HashMap_, K_, V_)                 \
  CWISS_DECLARE_FLAT_MAP_POLICY(HashMap_##_kPolicy, K_, V_, (_, _)); \
  CWISS_DECLARE_HASHMAP_WITH(HashMap_, K_, V_, HashMap_##_kPolicy)

/// Generates a new hash map type with outline storage and the default
/// plain-old-data policies.
///
/// See header documentation for examples of generated API.
#define CWISS_DECLARE_NODE_HASHMAP(HashMap_, K_, V_)                 \
  CWISS_DECLARE_NODE_MAP_POLICY(HashMap_##_kPolicy, K_, V_, (_, _)); \
  CWISS_DECLARE_HASHMAP_WITH(HashMap_, K_, V_, HashMap_##_kPolicy)

/// Generates a new hash set type using the given policy.
///
/// See header documentation for examples of generated API.
#define CWISS_DECLARE_HASHSET_WITH(HashSet_, Type_, kPolicy_) \
  typedef Type_ HashSet_##_Entry;                             \
  typedef Type_ HashSet_##_Key;                               \
  CWISS_DECLARE_COMMON_(HashSet_, HashSet_##_Entry, HashSet_##_Key, kPolicy_)

/// Generates a new hash map type using the given policy.
///
/// See header documentation for examples of generated API.
#define CWISS_DECLARE_HASHMAP_WITH(HashMap_, K_, V_, kPolicy_) \
  typedef struct {                                             \
    K_ key;                                                    \
    V_ val;                                                    \
  } HashMap_##_Entry;                                          \
  typedef K_ HashMap_##_Key;                                   \
  CWISS_DECLARE_COMMON_(HashMap_, HashMap_##_Entry, HashMap_##_Key, kPolicy_)

/// Declares a heterogenous lookup for an existing SwissTable type.
///
/// This macro will expect to find the following functions:
///   - size_t <Table>_<Key>_hash(const Key*);
///   - bool <Table>_<Key>_eq(const Key*, const <Table>_Key*);
///
/// These functions will be used to build the heterogenous key policy.
#define CWISS_DECLARE_LOOKUP(HashSet_, Key_) \
  CWISS_DECLARE_LOOKUP_NAMED(HashSet_, Key_, Key_)

/// Declares a heterogenous lookup for an existing SwissTable type.
///
/// This is like `CWISS_DECLARE_LOOKUP`, but allows customizing the name used
/// in the `_by_` prefix on the names, as well as the names of the extension
/// point functions.
#define CWISS_DECLARE_LOOKUP_NAMED(HashSet_, LookupName_, Key_)                \
  CWISS_BEGIN                                                                  \
  static inline size_t HashSet_##_##LookupName_##_SyntheticHash(               \
      const void* val) {                                                       \
    return HashSet_##_##LookupName_##_hash((const Key_*)val);                  \
  }                                                                            \
  static inline bool HashSet_##_##LookupName_##_SyntheticEq(const void* a,     \
                                                            const void* b) {   \
    return HashSet_##_##LookupName_##_eq((const Key_*)a,                       \
                                         (const HashSet_##_Entry*)b);          \
  }                                                                            \
  static const CWISS_KeyPolicy HashSet_##_##LookupName_##_kPolicy = {          \
      HashSet_##_##LookupName_##_SyntheticHash,                                \
      HashSet_##_##LookupName_##_SyntheticEq,                                  \
  };                                                                           \
                                                                               \
  static inline const CWISS_KeyPolicy* HashSet_##_##LookupName_##_policy(      \
      void) {                                                                  \
    return &HashSet_##_##LookupName_##_kPolicy;                                \
  }                                                                            \
                                                                               \
  static inline HashSet_##_Insert HashSet_##_deferred_insert_by_##LookupName_( \
      HashSet_* self, const Key_* key) {                                       \
    CWISS_Insert ret = CWISS_RawTable_deferred_insert(                         \
        HashSet_##_policy(), &HashSet_##_##LookupName_##_kPolicy, &self->set_, \
        key);                                                                  \
    return (HashSet_##_Insert){{ret.iter}, ret.inserted};                      \
  }                                                                            \
  static inline HashSet_##_CIter HashSet_##_cfind_hinted_by_##LookupName_(     \
      const HashSet_* self, const Key_* key, size_t hash) {                    \
    return (HashSet_##_CIter){CWISS_RawTable_find_hinted(                      \
        HashSet_##_policy(), &HashSet_##_##LookupName_##_kPolicy, &self->set_, \
        key, hash)};                                                           \
  }                                                                            \
  static inline HashSet_##_Iter HashSet_##_find_hinted_by_##LookupName_(       \
      HashSet_* self, const Key_* key, size_t hash) {                          \
    return (HashSet_##_Iter){CWISS_RawTable_find_hinted(                       \
        HashSet_##_policy(), &HashSet_##_##LookupName_##_kPolicy, &self->set_, \
        key, hash)};                                                           \
  }                                                                            \
                                                                               \
  static inline HashSet_##_CIter HashSet_##_cfind_by_##LookupName_(            \
      const HashSet_* self, const Key_* key) {                                 \
    return (HashSet_##_CIter){CWISS_RawTable_find(                             \
        HashSet_##_policy(), &HashSet_##_##LookupName_##_kPolicy, &self->set_, \
        key)};                                                                 \
  }                                                                            \
  static inline HashSet_##_Iter HashSet_##_find_by_##LookupName_(              \
      HashSet_* self, const Key_* key) {                                       \
    return (HashSet_##_Iter){CWISS_RawTable_find(                              \
        HashSet_##_policy(), &HashSet_##_##LookupName_##_kPolicy, &self->set_, \
        key)};                                                                 \
  }                                                                            \
                                                                               \
  static inline bool HashSet_##_contains_by_##LookupName_(                     \
      const HashSet_* self, const Key_* key) {                                 \
    return CWISS_RawTable_contains(HashSet_##_policy(),                        \
                                   &HashSet_##_##LookupName_##_kPolicy,        \
                                   &self->set_, key);                          \
  }                                                                            \
                                                                               \
  static inline bool HashSet_##_erase_by_##LookupName_(HashSet_* self,         \
                                                       const Key_* key) {      \
    return CWISS_RawTable_erase(HashSet_##_policy(),                           \
                                &HashSet_##_##LookupName_##_kPolicy,           \
                                &self->set_, key);                             \
  }                                                                            \
                                                                               \
  CWISS_END                                                                    \
  /* Force a semicolon. */                                                     \
  struct HashSet_##_##LookupName_##_NeedsTrailingSemicolon_ {                  \
    int x;                                                                     \
  }

// ---- PUBLIC API ENDS HERE! ----

#define CWISS_DECLARE_COMMON_(HashSet_, Type_, Key_, kPolicy_)                 \
  CWISS_BEGIN                                                                  \
  static inline const CWISS_Policy* HashSet_##_policy(void) {                  \
    return &kPolicy_;                                                          \
  }                                                                            \
                                                                               \
  typedef struct {                                                             \
    CWISS_RawTable set_;                                                       \
  } HashSet_;                                                                  \
  static inline void HashSet_##_dump(const HashSet_* self) {                   \
    CWISS_RawTable_dump(&kPolicy_, &self->set_);                               \
  }                                                                            \
                                                                               \
  static inline HashSet_ HashSet_##_new(size_t bucket_count) {                 \
    return (HashSet_){CWISS_RawTable_new(&kPolicy_, bucket_count)};            \
  }                                                                            \
  static inline HashSet_ HashSet_##_dup(const HashSet_* that) {                \
    return (HashSet_){CWISS_RawTable_dup(&kPolicy_, &that->set_)};             \
  }                                                                            \
  static inline void HashSet_##_destroy(HashSet_* self) {                      \
    CWISS_RawTable_destroy(&kPolicy_, &self->set_);                            \
  }                                                                            \
                                                                               \
  typedef struct {                                                             \
    CWISS_RawIter it_;                                                         \
  } HashSet_##_Iter;                                                           \
  static inline HashSet_##_Iter HashSet_##_iter(HashSet_* self) {              \
    return (HashSet_##_Iter){CWISS_RawTable_iter(&kPolicy_, &self->set_)};     \
  }                                                                            \
  static inline Type_* HashSet_##_Iter_get(const HashSet_##_Iter* it) {        \
    return (Type_*)CWISS_RawIter_get(&kPolicy_, &it->it_);                     \
  }                                                                            \
  static inline Type_* HashSet_##_Iter_next(HashSet_##_Iter* it) {             \
    return (Type_*)CWISS_RawIter_next(&kPolicy_, &it->it_);                    \
  }                                                                            \
                                                                               \
  typedef struct {                                                             \
    CWISS_RawIter it_;                                                         \
  } HashSet_##_CIter;                                                          \
  static inline HashSet_##_CIter HashSet_##_citer(const HashSet_* self) {      \
    return (HashSet_##_CIter){CWISS_RawTable_citer(&kPolicy_, &self->set_)};   \
  }                                                                            \
  static inline const Type_* HashSet_##_CIter_get(                             \
      const HashSet_##_CIter* it) {                                            \
    return (const Type_*)CWISS_RawIter_get(&kPolicy_, &it->it_);               \
  }                                                                            \
  static inline const Type_* HashSet_##_CIter_next(HashSet_##_CIter* it) {     \
    return (const Type_*)CWISS_RawIter_next(&kPolicy_, &it->it_);              \
  }                                                                            \
  static inline HashSet_##_CIter HashSet_##_Iter_const(HashSet_##_Iter it) {   \
    return (HashSet_##_CIter){it.it_};                                         \
  }                                                                            \
                                                                               \
  static inline void HashSet_##_reserve(HashSet_* self, size_t n) {            \
    CWISS_RawTable_reserve(&kPolicy_, &self->set_, n);                         \
  }                                                                            \
  static inline void HashSet_##_rehash(HashSet_* self, size_t n) {             \
    CWISS_RawTable_rehash(&kPolicy_, &self->set_, n);                          \
  }                                                                            \
                                                                               \
  static inline bool HashSet_##_empty(const HashSet_* self) {                  \
    return CWISS_RawTable_empty(&kPolicy_, &self->set_);                       \
  }                                                                            \
  static inline size_t HashSet_##_size(const HashSet_* self) {                 \
    return CWISS_RawTable_size(&kPolicy_, &self->set_);                        \
  }                                                                            \
  static inline size_t HashSet_##_capacity(const HashSet_* self) {             \
    return CWISS_RawTable_capacity(&kPolicy_, &self->set_);                    \
  }                                                                            \
                                                                               \
  static inline void HashSet_##_clear(HashSet_* self) {                        \
    return CWISS_RawTable_clear(&kPolicy_, &self->set_);                       \
  }                                                                            \
                                                                               \
  typedef struct {                                                             \
    HashSet_##_Iter iter;                                                      \
    bool inserted;                                                             \
  } HashSet_##_Insert;                                                         \
  static inline HashSet_##_Insert HashSet_##_deferred_insert(                  \
      HashSet_* self, const Key_* key) {                                       \
    CWISS_Insert ret = CWISS_RawTable_deferred_insert(&kPolicy_, kPolicy_.key, \
                                                      &self->set_, key);       \
    return (HashSet_##_Insert){{ret.iter}, ret.inserted};                      \
  }                                                                            \
  static inline HashSet_##_Insert HashSet_##_insert(HashSet_* self,            \
                                                    const Type_* val) {        \
    CWISS_Insert ret = CWISS_RawTable_insert(&kPolicy_, &self->set_, val);     \
    return (HashSet_##_Insert){{ret.iter}, ret.inserted};                      \
  }                                                                            \
                                                                               \
  static inline HashSet_##_CIter HashSet_##_cfind_hinted(                      \
      const HashSet_* self, const Key_* key, size_t hash) {                    \
    return (HashSet_##_CIter){CWISS_RawTable_find_hinted(                      \
        &kPolicy_, kPolicy_.key, &self->set_, key, hash)};                     \
  }                                                                            \
  static inline HashSet_##_Iter HashSet_##_find_hinted(                        \
      HashSet_* self, const Key_* key, size_t hash) {                          \
    return (HashSet_##_Iter){CWISS_RawTable_find_hinted(                       \
        &kPolicy_, kPolicy_.key, &self->set_, key, hash)};                     \
  }                                                                            \
  static inline HashSet_##_CIter HashSet_##_cfind(const HashSet_* self,        \
                                                  const Key_* key) {           \
    return (HashSet_##_CIter){                                                 \
        CWISS_RawTable_find(&kPolicy_, kPolicy_.key, &self->set_, key)};       \
  }                                                                            \
  static inline HashSet_##_Iter HashSet_##_find(HashSet_* self,                \
                                                const Key_* key) {             \
    return (HashSet_##_Iter){                                                  \
        CWISS_RawTable_find(&kPolicy_, kPolicy_.key, &self->set_, key)};       \
  }                                                                            \
                                                                               \
  static inline bool HashSet_##_contains(const HashSet_* self,                 \
                                         const Key_* key) {                    \
    return CWISS_RawTable_contains(&kPolicy_, kPolicy_.key, &self->set_, key); \
  }                                                                            \
                                                                               \
  static inline void HashSet_##_erase_at(HashSet_##_Iter it) {                 \
    CWISS_RawTable_erase_at(&kPolicy_, it.it_);                                \
  }                                                                            \
  static inline bool HashSet_##_erase(HashSet_* self, const Key_* key) {       \
    return CWISS_RawTable_erase(&kPolicy_, kPolicy_.key, &self->set_, key);    \
  }                                                                            \
                                                                               \
  CWISS_END                                                                    \
  /* Force a semicolon. */ struct HashSet_##_NeedsTrailingSemicolon_ { int x; }

CWISS_END_EXTERN
CWISS_END

#endif  // CWISSTABLE_DECLARE_H_