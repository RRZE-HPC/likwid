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

#ifndef CWISSTABLE_POLICY_H_
#define CWISSTABLE_POLICY_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "cwisstable/hash.h"
#include "cwisstable/internal/base.h"
#include "cwisstable/internal/extract.h"

/// Hash table policies.
///
/// Table policies are `cwisstable`'s generic code mechanism. All code in
/// `cwisstable`'s internals is completely agnostic to:
/// - The layout of the elements.
/// - The storage strategy for the elements (inline, indirect in the heap).
/// - Hashing, comparison, and allocation.
///
/// This information is provided to `cwisstable`'s internals by way of a
/// *policy*: a vtable describing how to move elements around, hash them,
/// compare them, allocate storage for them, and so on and on. This design is
/// inspired by Abseil's equivalent, which is a template parameter used for
/// sharing code between all the SwissTable-backed containers.
///
/// Unlike Abseil, policies are part of `cwisstable`'s public interface. Due to
/// C's lack of any mechanism for detecting the gross properties of types,
/// types with unwritten invariants, such as C strings (NUL-terminated byte
/// arrays), users must be able to carefully describe to `cwisstable` how to
/// correctly do things to their type. DESIGN.md goes into detailed rationale
/// for this polymorphism strategy.
///
/// # Defining a Policy
///
/// Policies are defined as read-only globals and passed around by pointer to
/// different `cwisstable` functions; macros are provided for doing this, since
/// most of these functions will not vary significantly from one type to
/// another. There are four of them:
///
/// - `CWISS_DECLARE_FLAT_SET_POLICY(kPolicy, Type, ...)`
/// - `CWISS_DECLARE_FLAT_MAP_POLICY(kPolicy, Key, Value, ...)`
/// - `CWISS_DECLARE_NODE_SET_POLICY(kPolicy, Type, ...)`
/// - `CWISS_DECLARE_NODE_MAP_POLICY(kPolicy, Key, Value, ...)`
///
/// These correspond to the four SwissTable types in Abseil: two map types and
/// two set types; "flat" means that elements are stored inline in the backing
/// array, whereas "node" means that the element is stored in its own heap
/// allocation, making it stable across rehashings (which SwissTable does more
/// or less whenever it feels like it).
///
/// Each macro expands to a read-only global variable definition (with the name
/// `kPolicy`, i.e, the first variable) dedicated for the specified type(s).
/// The arguments that follow are overrides for the default values of each field
/// in the policy; all but the size and alignment fields of `CWISS_ObjectPolicy`
/// may be overridden. To override the field `kPolicy.foo.bar`, pass
/// `(foo_bar, value)` to the macro. If multiple such pairs are passed in, the
/// first one found wins. `examples/stringmap.c` provides an example of how to
/// use this functionality.
///
/// For "common" uses, where the key and value are plain-old-data, `declare.h`
/// has dedicated macros, and fussing with policies directly is unnecessary.

CWISS_BEGIN
CWISS_BEGIN_EXTERN

/// A policy describing the basic laying properties of a type.
///
/// This type describes how to move values of a particular type around.
typedef struct {
  /// The layout of the stored object.
  size_t size, align;

  /// Performs a deep copy of `src` onto a fresh location `dst`.
  void (*copy)(void* dst, const void* src);

  /// Destroys an object.
  ///
  /// This member may, as an optimization, be null. This will cause it to
  /// behave as a no-op, and may be more efficient than making this an empty
  /// function.
  void (*dtor)(void* val);
} CWISS_ObjectPolicy;

/// A policy describing the hashing properties of a type.
///
/// This type describes the necessary information for putting a value into a
/// hash table.
///
/// A *heterogenous* key policy is one whose equality function expects different
/// argument types, which can be used for so-called heterogenous lookup: finding
/// an element of a table by comparing it to a somewhat different type. If the
/// table element is, for example, a `std::string`[1]-like type, it could still
/// be found via a non-owning version like a `std::string_view`[2]. This is
/// important for making efficient use of a SwissTable.
///
/// [1]: For non C++ programmers: a growable string type implemented as a
///      `struct { char* ptr; size_t size, capacity; }`.
/// [2]: Similarly, a `std::string_view` is a pointer-length pair to a string
///      *somewhere*; unlike a C-style string, it might be a substring of a
///      larger allocation elsewhere.
typedef struct {
  /// Computes the hash of a value.
  ///
  /// This function must be such that if two elements compare equal, they must
  /// have the same hash (but not vice-versa).
  ///
  /// If this policy is heterogenous, this function must be defined so that
  /// given the original key policy of the table's element type, if
  /// `hetero->eq(a, b)` holds, then `hetero->hash(a) == original->hash(b)`.
  /// In other words, the obvious condition for a hash table to work correctly
  /// with this policy.
  size_t (*hash)(const void* val);

  /// Compares two values for equality.
  ///
  /// This function is actually not symmetric: the first argument will always be
  /// the value being searched for, and the second will be a pointer to the
  /// candidate entry. In particular, this means they can be different types:
  /// in C++ parlance, `needle` could be a `std::string_view`, while `candidate`
  /// could be a `std::string`.
  bool (*eq)(const void* needle, const void* candidate);
} CWISS_KeyPolicy;

/// A policy for allocation.
///
/// This type provides access to a custom allocator.
typedef struct {
  /// Allocates memory.
  ///
  /// This function must never fail and never return null, unlike `malloc`. This
  /// function does not need to tolerate zero sized allocations.
  void* (*alloc)(size_t size, size_t align);

  /// Deallocates memory allocated by `alloc`.
  ///
  /// This function is passed the same size/alignment as was passed to `alloc`,
  /// allowing for sized-delete optimizations.
  void (*free)(void* array, size_t size, size_t align);
} CWISS_AllocPolicy;

/// A policy for allocating space for slots.
///
/// This allows us to distinguish between inline storage (more cache-friendly)
/// and outline (pointer-stable).
typedef struct {
  /// The layout of a slot value.
  ///
  /// Usually, this will be the same as for the object type, *or* the layout
  /// of a pointer (for outline storage).
  size_t size, align;

  /// Initializes a new slot at the given location.
  ///
  /// This function does not initialize the value *in* the slot; it simply sets
  /// up the slot so that a value can be `memcpy`'d or otherwise emplaced into
  /// the slot.
  void (*init)(void* slot);

  /// Destroys a slot, including the destruction of the value it contains.
  ///
  /// This function may, as an optimization, be null. This will cause it to
  /// behave as a no-op.
  void (*del)(void* slot);

  /// Transfers a slot.
  ///
  /// `dst` must be uninitialized; `src` must be initialized. After this
  /// function, their roles will be switched: `dst` will be initialized and
  /// contain the value from `src`; `src` will be initialized.
  ///
  /// This function need not actually copy the underlying value.
  void (*transfer)(void* dst, void* src);

  /// Extracts a pointer to the value inside the a slot.
  ///
  /// This function does not need to tolerate nulls.
  void* (*get)(void* slot);
} CWISS_SlotPolicy;

/// A hash table policy.
///
/// See the header documentation for more information.
typedef struct {
  const CWISS_ObjectPolicy* obj;
  const CWISS_KeyPolicy* key;
  const CWISS_AllocPolicy* alloc;
  const CWISS_SlotPolicy* slot;
} CWISS_Policy;

/// Declares a hash set policy with inline storage for the given type.
///
/// See the header documentation for more information.
#define CWISS_DECLARE_FLAT_SET_POLICY(kPolicy_, Type_, ...) \
  CWISS_DECLARE_POLICY_(kPolicy_, Type_, Type_, __VA_ARGS__)

/// Declares a hash map policy with inline storage for the given key and value
/// types.
///
/// See the header documentation for more information.
#define CWISS_DECLARE_FLAT_MAP_POLICY(kPolicy_, K_, V_, ...) \
  typedef struct {                                           \
    K_ k;                                                    \
    V_ v;                                                    \
  } kPolicy_##_Entry;                                        \
  CWISS_DECLARE_POLICY_(kPolicy_, kPolicy_##_Entry, K_, __VA_ARGS__)

/// Declares a hash set policy with pointer-stable storage for the given type.
///
/// See the header documentation for more information.
#define CWISS_DECLARE_NODE_SET_POLICY(kPolicy_, Type_, ...)          \
  CWISS_DECLARE_NODE_FUNCTIONS_(kPolicy_, Type_, Type_, __VA_ARGS__) \
  CWISS_DECLARE_POLICY_(kPolicy_, Type_, Type_, __VA_ARGS__,         \
                        CWISS_NODE_OVERRIDES_(kPolicy_))

/// Declares a hash map policy with pointer-stable storage for the given key and
/// value types.
///
/// See the header documentation for more information.
#define CWISS_DECLARE_NODE_MAP_POLICY(kPolicy_, K_, V_, ...)                 \
  typedef struct {                                                           \
    K_ k;                                                                    \
    V_ v;                                                                    \
  } kPolicy_##_Entry;                                                        \
  CWISS_DECLARE_NODE_FUNCTIONS_(kPolicy_, kPolicy_##_Entry, K_, __VA_ARGS__) \
  CWISS_DECLARE_POLICY_(kPolicy_, kPolicy_##_Entry, K_, __VA_ARGS__,         \
                        CWISS_NODE_OVERRIDES_(kPolicy_))

// ---- PUBLIC API ENDS HERE! ----

#define CWISS_DECLARE_POLICY_(kPolicy_, Type_, Key_, ...)                \
  CWISS_BEGIN                                                            \
  CWISS_EXTRACT_RAW(modifiers, static, __VA_ARGS__)                      \
  inline void kPolicy_##_DefaultCopy(void* dst, const void* src) {       \
    memcpy(dst, src, sizeof(Type_));                                     \
  }                                                                      \
  CWISS_EXTRACT_RAW(modifiers, static, __VA_ARGS__)                      \
  inline size_t kPolicy_##_DefaultHash(const void* val) {                \
    CWISS_AbslHash_State state = CWISS_AbslHash_kInit;                   \
    CWISS_AbslHash_Write(&state, val, sizeof(Key_));                     \
    return CWISS_AbslHash_Finish(state);                                 \
  }                                                                      \
  CWISS_EXTRACT_RAW(modifiers, static, __VA_ARGS__)                      \
  inline bool kPolicy_##_DefaultEq(const void* a, const void* b) {       \
    return memcmp(a, b, sizeof(Key_)) == 0;                              \
  }                                                                      \
  CWISS_EXTRACT_RAW(modifiers, static, __VA_ARGS__)                      \
  inline void kPolicy_##_DefaultSlotInit(void* slot) {}                  \
  CWISS_EXTRACT_RAW(modifiers, static, __VA_ARGS__)                      \
  inline void kPolicy_##_DefaultSlotTransfer(void* dst, void* src) {     \
    memcpy(dst, src, sizeof(Type_));                                     \
  }                                                                      \
  CWISS_EXTRACT_RAW(modifiers, static, __VA_ARGS__)                      \
  inline void* kPolicy_##_DefaultSlotGet(void* slot) { return slot; }    \
  CWISS_EXTRACT_RAW(modifiers, static, __VA_ARGS__)                      \
  inline void kPolicy_##_DefaultSlotDtor(void* slot) {                   \
    if (CWISS_EXTRACT(obj_dtor, NULL, __VA_ARGS__) != NULL) {            \
      CWISS_EXTRACT(obj_dtor, (void (*)(void*))NULL, __VA_ARGS__)(slot); \
    }                                                                    \
  }                                                                      \
                                                                         \
  CWISS_EXTRACT_RAW(modifiers, static, __VA_ARGS__)                      \
  const CWISS_ObjectPolicy kPolicy_##_ObjectPolicy = {                   \
      sizeof(Type_),                                                     \
      alignof(Type_),                                                    \
      CWISS_EXTRACT(obj_copy, kPolicy_##_DefaultCopy, __VA_ARGS__),      \
      CWISS_EXTRACT(obj_dtor, NULL, __VA_ARGS__),                        \
  };                                                                     \
  CWISS_EXTRACT_RAW(modifiers, static, __VA_ARGS__)                      \
  const CWISS_KeyPolicy kPolicy_##_KeyPolicy = {                         \
      CWISS_EXTRACT(key_hash, kPolicy_##_DefaultHash, __VA_ARGS__),      \
      CWISS_EXTRACT(key_eq, kPolicy_##_DefaultEq, __VA_ARGS__),          \
  };                                                                     \
  CWISS_EXTRACT_RAW(modifiers, static, __VA_ARGS__)                      \
  const CWISS_AllocPolicy kPolicy_##_AllocPolicy = {                     \
      CWISS_EXTRACT(alloc_alloc, CWISS_DefaultMalloc, __VA_ARGS__),      \
      CWISS_EXTRACT(alloc_free, CWISS_DefaultFree, __VA_ARGS__),         \
  };                                                                     \
  CWISS_EXTRACT_RAW(modifiers, static, __VA_ARGS__)                      \
  const CWISS_SlotPolicy kPolicy_##_SlotPolicy = {                       \
      CWISS_EXTRACT(slot_size, sizeof(Type_), __VA_ARGS__),              \
      CWISS_EXTRACT(slot_align, sizeof(Type_), __VA_ARGS__),             \
      CWISS_EXTRACT(slot_init, kPolicy_##_DefaultSlotInit, __VA_ARGS__), \
      CWISS_EXTRACT(slot_dtor, kPolicy_##_DefaultSlotDtor, __VA_ARGS__), \
      CWISS_EXTRACT(slot_transfer, kPolicy_##_DefaultSlotTransfer,       \
                    __VA_ARGS__),                                        \
      CWISS_EXTRACT(slot_get, kPolicy_##_DefaultSlotGet, __VA_ARGS__),   \
  };                                                                     \
  CWISS_END                                                              \
  CWISS_EXTRACT_RAW(modifiers, static, __VA_ARGS__)                      \
  const CWISS_Policy kPolicy_ = {                                        \
      &kPolicy_##_ObjectPolicy,                                          \
      &kPolicy_##_KeyPolicy,                                             \
      &kPolicy_##_AllocPolicy,                                           \
      &kPolicy_##_SlotPolicy,                                            \
  }

#define CWISS_DECLARE_NODE_FUNCTIONS_(kPolicy_, Type_, ...)                    \
  CWISS_BEGIN                                                                  \
  static inline void kPolicy_##_NodeSlotInit(void* slot) {                     \
    void* node = CWISS_EXTRACT(alloc_alloc, CWISS_DefaultMalloc, __VA_ARGS__)( \
        sizeof(Type_), alignof(Type_));                                        \
    memcpy(slot, &node, sizeof(node));                                         \
  }                                                                            \
  static inline void kPolicy_##_NodeSlotDtor(void* slot) {                     \
    if (CWISS_EXTRACT(obj_dtor, NULL, __VA_ARGS__) != NULL) {                  \
      CWISS_EXTRACT(obj_dtor, (void (*)(void*))NULL, __VA_ARGS__)              \
      (*(void**)slot);                                                         \
    }                                                                          \
    CWISS_EXTRACT(alloc_free, CWISS_DefaultFree, __VA_ARGS__)                  \
    (*(void**)slot, sizeof(Type_), alignof(Type_));                            \
  }                                                                            \
  static inline void kPolicy_##_NodeSlotTransfer(void* dst, void* src) {       \
    memcpy(dst, src, sizeof(void*));                                           \
  }                                                                            \
  static inline void* kPolicy_##_NodeSlotGet(void* slot) {                     \
    return *((void**)slot);                                                    \
  }                                                                            \
  CWISS_END

#define CWISS_NODE_OVERRIDES_(kPolicy_)                     \
  (slot_size, sizeof(void*)), (slot_align, alignof(void*)), \
      (slot_init, kPolicy_##_NodeSlotInit),                 \
      (slot_dtor, kPolicy_##_NodeSlotDtor),                 \
      (slot_transfer, kPolicy_##_NodeSlotTransfer),         \
      (slot_get, kPolicy_##_NodeSlotGet)

static inline void* CWISS_DefaultMalloc(size_t size, size_t align) {
  void* p = malloc(size);  // TODO: Check alignment.
  CWISS_CHECK(p != NULL, "malloc() returned null");
  return p;
}
static inline void CWISS_DefaultFree(void* array, size_t size, size_t align) {
  free(array);
}

CWISS_END_EXTERN
CWISS_END

#endif  // CWISSTABLE_POLICY_H_