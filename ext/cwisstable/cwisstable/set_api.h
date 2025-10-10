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

#ifndef CWISSTABLE_SET_API_H_
#define CWISSTABLE_SET_API_H_

#include <stdbool.h>
#include <stddef.h>

#include "cwisstable/declare.h"
#include "cwisstable/policy.h"

/// Example API expansion of declare.h set macros.
///
/// Should be kept in sync with declare.h; unfortunately we don't have an easy
/// way to test this just yet.

// CWISS_DECLARE_FLAT_HASHSET(MySet, T) expands to:

/// Returns the policy used with this set type.
static inline const CWISS_Policy* MySet_policy();

/// The generated type.
typedef struct {
  /* ... */
} MySet;

/// Constructs a new set with the given initial capacity.
static inline MySet MySet_new(size_t capacity);

/// Creates a deep copy of this set.
static inline MySet MySet_dup(const MySet* self);

/// Destroys this set.
static inline void MySet_destroy(const MySet* self);

/// Dumps the internal contents of the table to stderr; intended only for
/// debugging.
///
/// The output of this function is not stable.
static inline void MySet_dump(const MySet* self);

/// Ensures that there is at least `n` spare capacity, potentially resizing
/// if necessary.
static inline void MySet_reserve(MySet* self, size_t n);

/// Resizes the table to have at least `n` buckets of capacity.
static inline void MySet_rehash(MySet* self, size_t n);

/// Returns whether the set is empty.
static inline size_t MySet_empty(const MySet* self);

/// Returns the number of elements stored in the table.
static inline size_t MySet_size(const MySet* self);

/// Returns the number of buckets in the table.
///
/// Note that this is *different* from the amount of elements that must be
/// in the table before a resize is triggered.
static inline size_t MySet_capacity(const MySet* self);

/// Erases every element in the set.
static inline void MySet_clear(MySet* self);

/// A non-mutating iterator into a `MySet`.
typedef struct {
  /* ... */
} MySet_CIter;

/// Creates a new non-mutating iterator fro this table.
static inline MySet_CIter MySet_citer(const MySet* self);

/// Returns a pointer to the element this iterator is at; returns `NULL` if
/// this iterator has reached the end of the table.
static inline const T* MySet_CIter_get(const MySet_CIter* it);

/// Advances this iterator, returning a pointer to the element the iterator
/// winds up pointing to (see `MySet_CIter_get()`).
///
/// The iterator must not point to the end of the table.
static inline const T* MySet_CIter_next(const MySet_CIter* it);

/// A mutating iterator into a `MySet`.
typedef struct {
  /* ... */
} MySet_Iter;

/// Creates a new mutating iterator fro this table.
static inline MySet_Iter MySet_iter(const MySet* self);

/// Returns a pointer to the element this iterator is at; returns `NULL` if
/// this iterator has reached the end of the table.
static inline T* MySet_Iter_get(const MySet_Iter* it);

/// Advances this iterator, returning a pointer to the element the iterator
/// winds up pointing to (see `MySet_Iter_get()`).
///
/// The iterator must not point to the end of the table.
static inline T* MySet_Iter_next(const MySet_Iter* it);

/// Checks if this set contains the given element.
///
/// In general, if you plan to use the element and not just check for it,
/// prefer `MySet_find()` and friends.
static inline bool MySet_contains(const MySet* self, const T* key);

/// Searches the table for `key`, non-mutating iterator version.
///
/// If found, returns an iterator at the found element; otherwise, returns
/// an iterator that's already at the end: `get()` will return `NULL`.
static inline MySet_CIter MySet_cfind(const MySet* self, const T* key);

/// Searches the table for `key`, mutating iterator version.
///
/// If found, returns an iterator at the found element; otherwise, returns
/// an iterator that's already at the end: `get()` will return `NULL`.
///
/// This function does not trigger rehashes.
static inline MySet_Iter MySet_find(MySet* self, const T* key);

/// Like `MySet_cfind`, but takes a pre-computed hash.
///
/// The hash must be correct for `key`.
static inline MySet_CIter MySet_cfind_hinted(const MySet* self, const T* key,
                                             size_t hash);

/// Like `MySet_find`, but takes a pre-computed hash.
///
/// The hash must be correct for `key`.
///
/// This function does not trigger rehashes.
static inline MySet_Iter MySet_find_hinted(MySet* self, const T* key,
                                           size_t hash);

/// The return type of `MySet_insert()`.
typedef struct {
  MySet_Iter iter;
  bool inserted;
} MySet_Insert;

/// Inserts `val` into the map if it isn't already present, initializing it by
/// copy.
///
/// Returns an iterator pointing to the element in the map and whether it was
/// just inserted or was already present.
static inline MySet_Insert MySet_insert(MySet* self, const T* val);

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
static inline MyMap_Insert MyMap_deferred_insert(MySet* self, const T* key);

/// Looks up `key` and erases it from the set.
///
/// Returns `true` if erasure happened.
static inline bool MySet_erase(MySet* self, const T* key);

/// Erases (and destroys) the element pointed to by `it`.
///
/// Although the iterator doesn't point to anything now, this function does
/// not trigger rehashes and the erased iterator can still be safely
/// advanced (although not dereferenced until advanced).
static inline void MySet_erase_at(MySet_Iter it);

// CWISS_DECLARE_LOOKUP(MySet, View) expands to:

/// Returns the policy used with this lookup extension.
static inline const CWISS_KeyPolicy* MySet_View_policy(void);

/// Checks if this set contains the given element.
///
/// In general, if you plan to use the element and not just check for it,
/// prefer `MySet_find()` and friends.
static inline bool MySet_contains_by_View(const MySet* self, const View* key);

/// Searches the table for `key`, non-mutating iterator version.
///
/// If found, returns an iterator at the found element; otherwise, returns
/// an iterator that's already at the end: `get()` will return `NULL`.
static inline MySet_CIter MySet_cfind_by_View(const MySet* self,
                                              const View* key);

/// Searches the table for `key`, mutating iterator version.
///
/// If found, returns an iterator at the found element; otherwise, returns
/// an iterator that's already at the end: `get()` will return `NULL`.
///
/// This function does not trigger rehashes.
static inline MySet_Iter MySet_find_by_View(MySet* self, const View* key);

/// Like `MySet_cfind`, but takes a pre-computed hash.
///
/// The hash must be correct for `key`.
static inline MySet_CIter MySet_cfind_hinted_by_View(const MySet* self,
                                                     const View* key,
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

/// Like `MySet_find`, but takes a pre-computed hash.
///
/// The hash must be correct for `key`.
///
/// This function does not trigger rehashes.
static inline MySet_Iter MySet_find_hinted_by_View(MySet* self, const View* key,
                                                   size_t hash);

/// "Inserts" `key` into the set if it isn't already present.
///
/// This function does not perform insertion; it behaves exactly like
/// `MySet_insert()` up until it would copy-initialize the new
/// element, instead returning a valid iterator pointing to uninitialized data.
///
/// This allows, for example, lazily constructing the parts of the element that
/// do not figure into the hash or equality. The initialized element must have
/// the same hash value and must compare equal to the value used for the initial
/// lookup; UB may otherwise result.
///
/// If this function returns `true` in `inserted`, the caller has *no choice*
/// but to insert, i.e., they may not change their minds at that point.
static inline MySet_Insert MySet_deferred_insert_by_View(MySet* self,
                                                         const View* key);

/// Looks up `key` and erases it from the map.
///
/// Returns `true` if erasure happened.
static inline bool MySet_erase_by_View(MySet* self, const View* key);

#error "This file is for demonstration purposes only."

#endif  // CWISSTABLE_SET_API_H_