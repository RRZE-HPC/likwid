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

#ifndef CWISSTABLE_MAP_API_H_
#define CWISSTABLE_MAP_API_H_

#include <stdbool.h>
#include <stddef.h>

#include "cwisstable/declare.h"
#include "cwisstable/policy.h"

/// Example API expansion of declare.h map macros.
///
/// Should be kept in sync with declare.h; unfortunately we don't have an easy
/// way to test this just yet.

// CWISS_DECLARE_FLAT_HASHMAP(MyMap, K, V) expands to:

/// Returns the policy used with this map type.
static inline const CWISS_Policy* MyMap_policy(void);

/// The generated type.
typedef struct {
  /* ... */
} MyMap;

/// A key-value pair in the map.
typedef struct {
  K key;
  V val;
} MyMap_Entry;

/// Constructs a new map with the given initial capacity.
static inline MyMap MyMap_new(size_t capacity);

/// Creates a deep copy of this map.
static inline MyMap MyMap_dup(const MyMap* self);

/// Destroys this map.
static inline void MyMap_destroy(const MyMap* self);

/// Dumps the internal contents of the table to stderr; intended only for
/// debugging.
///
/// The output of this function is not stable.
static inline void MyMap_dump(const MyMap* self);

/// Ensures that there is at least `n` spare capacity, potentially resizing
/// if necessary.
static inline void MyMap_reserve(MyMap* self, size_t n);

/// Resizes the table to have at least `n` buckets of capacity.
static inline void MyMap_rehash(MyMap* self, size_t n);

/// Returns whether the map is empty.
static inline size_t MyMap_empty(const MyMap* self);

/// Returns the number of elements stored in the table.
static inline size_t MyMap_size(const MyMap* self);

/// Returns the number of buckets in the table.
///
/// Note that this is *different* from the amount of elements that must be
/// in the table before a resize is triggered.
static inline size_t MyMap_capacity(const MyMap* self);

/// Erases every element in the map.
static inline void MyMap_clear(MyMap* self);

/// A non-mutating iterator into a `MyMap`.
typedef struct {
  /* ... */
} MyMap_CIter;

/// Creates a new non-mutating iterator fro this table.
static inline MyMap_CIter MyMap_citer(const MyMap* self);

/// Returns a pointer to the element this iterator is at; returns `NULL` if
/// this iterator has reached the end of the table.
static inline const MyMap_Entry* MyMap_CIter_get(const MyMap_CIter* it);

/// Advances this iterator, returning a pointer to the element the iterator
/// winds up pointing to (see `MyMap_CIter_get()`).
///
/// The iterator must not point to the end of the table.
static inline const MyMap_Entry* MyMap_CIter_next(const MyMap_CIter* it);

/// A mutating iterator into a `MyMap`.
typedef struct {
  /* ... */
} MyMap_Iter;

/// Creates a new mutating iterator fro this table.
static inline MyMap_Iter MyMap_iter(const MyMap* self);

/// Returns a pointer to the element this iterator is at; returns `NULL` if
/// this iterator has reached the end of the table.
static inline MyMap_Entry* MyMap_Iter_get(const MyMap_Iter* it);

/// Advances this iterator, returning a pointer to the element the iterator
/// winds up pointing to (see `MyMap_Iter_get()`).
///
/// The iterator must not point to the end of the table.
static inline MyMap_Entry* MyMap_Iter_next(const MyMap_Iter* it);

/// Checks if this map contains the given element.
///
/// In general, if you plan to use the element and not just check for it,
/// prefer `MyMap_find()` and friends.
static inline bool MyMap_contains(const MyMap* self, const K* key);

/// Searches the table for `key`, non-mutating iterator version.
///
/// If found, returns an iterator at the found element; otherwise, returns
/// an iterator that's already at the end: `get()` will return `NULL`.
static inline MyMap_CIter MyMap_cfind(const MyMap* self, const K* key);

/// Searches the table for `key`, mutating iterator version.
///
/// If found, returns an iterator at the found element; otherwise, returns
/// an iterator that's already at the end: `get()` will return `NULL`.
///
/// This function does not trigger rehashes.
static inline MyMap_Iter MyMap_find(MyMap* self, const K* key);

/// Like `MyMap_cfind`, but takes a pre-computed hash.
///
/// The hash must be correct for `key`.
static inline MyMap_CIter MyMap_cfind_hinted(const MyMap* self, const K* key,
                                             size_t hash);

/// Like `MyMap_find`, but takes a pre-computed hash.
///
/// The hash must be correct for `key`.
///
/// This function does not trigger rehashes.
static inline MyMap_Iter MyMap_find_hinted(MyMap* self, const K* key,
                                           size_t hash);

/// The return type of `MyMap_insert()`.
typedef struct {
  MyMap_Iter iter;
  bool inserted;
} MyMap_Insert;

/// Inserts `val` into the map if it isn't already present, initializing it by
/// copy.
///
/// Returns an iterator pointing to the element in the map and whether it was
/// just inserted or was already present.
static inline MyMap_Insert MyMap_insert(MyMap* self, const MyMap_Entry* val);

/// "Inserts" `val` into the table if it isn't already present.
///
/// This function does not perform insertion; it behaves exactly like
/// `MyMap_insert()` up until it would copy-initialize the new
/// element, instead returning a valid iterator pointing to uninitialized data.
///
/// This allows, for example, lazily constructing the parts of the element that
/// do not figure into the hash or equality. The initialized element must have
/// the same hash value and must compare equal to the value used for the initial
/// lookup; UB may otherwise result.
///
/// If this function returns `true` in `inserted`, the caller has *no choice*
/// but to insert, i.e., they may not change their minds at that point.
static inline MyMap_Insert MyMap_deferred_insert(MyMap* self, const K* key);

/// Looks up `key` and erases it from the map.
///
/// Returns `true` if erasure happened.
static inline bool MyMap_erase(MyMap* self, const K* key);

/// Erases (and destroys) the element pointed to by `it`.
///
/// Although the iterator doesn't point to anything now, this function does
/// not trigger rehashes and the erased iterator can still be safely
/// advanced (although not dereferenced until advanced).
static inline void MyMap_erase_at(MyMap_Iter it);

// CWISS_DECLARE_LOOKUP(MyMap, View) expands to:

/// Returns the policy used with this lookup extension.
static inline const CWISS_KeyPolicy* MyMap_View_policy(void);

/// Checks if this map contains the given element.
///
/// In general, if you plan to use the element and not just check for it,
/// prefer `MyMap_find()` and friends.
static inline bool MyMap_contains_by_View(const MyMap* self, const View* key);

/// Searches the table for `key`, non-mutating iterator version.
///
/// If found, returns an iterator at the found element; otherwise, returns
/// an iterator that's already at the end: `get()` will return `NULL`.
static inline MyMap_CIter MyMap_cfind_by_View(const MyMap* self,
                                              const View* key);

/// Searches the table for `key`, mutating iterator version.
///
/// If found, returns an iterator at the found element; otherwise, returns
/// an iterator that's already at the end: `get()` will return `NULL`.
///
/// This function does not trigger rehashes.
static inline MyMap_Iter MyMap_find_by_View(MyMap* self, const View* key);

/// Like `MyMap_cfind`, but takes a pre-computed hash.
///
/// The hash must be correct for `key`.
static inline MyMap_CIter MyMap_cfind_hinted_by_View(const MyMap* self,
                                                     const View* key,
                                                     size_t hash);

/// Like `MyMap_find`, but takes a pre-computed hash.
///
/// The hash must be correct for `key`.
///
/// This function does not trigger rehashes.
static inline MyMap_Iter MyMap_find_hinted_by_View(MyMap* self, const View* key,
                                                   size_t hash);

/// "Inserts" `key` into the table if it isn't already present.
///
/// This function does not perform insertion; it behaves exactly like
/// `MyMap_insert()` up until it would copy-initialize the new
/// element, instead returning a valid iterator pointing to uninitialized data.
///
/// This allows, for example, lazily constructing the parts of the element that
/// do not figure into the hash or equality. The initialized element must have
/// the same hash value and must compare equal to the value used for the initial
/// lookup; UB may otherwise result.
///
/// If this function returns `true` in `inserted`, the caller has *no choice*
/// but to insert, i.e., they may not change their minds at that point.
static inline MyMap_Insert MyMap_deferred_insert_by_View(MySet* self,
                                                         const View* key);

/// Looks up `key` and erases it from the map.
///
/// Returns `true` if erasure happened.
static inline bool MyMap_erase_by_View(MyMap* self, const View* key);

#error "This file is for demonstration purposes only."

#endif  // CWISSTABLE_MAP_API_H_