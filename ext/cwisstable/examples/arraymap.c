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

// This file demonstrates the heterogenous lookup API.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(CWISS_EXAMPLE_UNIFIED)
  #include "cwisstable.h"
#elif defined(CWISS_EXAMPLE_SPLIT)
  #include "cwisstable/declare.h"
  #include "cwisstable/policy.h"
#else
  #error "must set CWISS_EXAMPLE_UNIFIED or CWISS_EXAMPLE_SPLIT"
#endif

typedef struct {
  int* ptr;
  size_t len, cap;
} IntArray;

typedef struct {
  int* ptr;
  size_t len;
} IntView;

static IntArray IntArray_new(void) { return (IntArray){0}; }

static IntArray IntArray_from_view(IntView view) {
  // This is still private API even if it's used in examples. :)
  uint32_t width = CWISS_BitWidth(view.len);
  uint32_t cap = (((size_t)1) << width) - 1;

  IntArray array = {
      malloc(cap * sizeof(int)),
      view.len,
      cap,
  };
  memcpy(array.ptr, view.ptr, view.len * sizeof(int));
  return array;
}

static IntArray IntArray_dup(const IntArray* self) {
  IntArray array = {
      malloc(self->cap * sizeof(int)),
      self->len,
      self->cap,
  };
  memcpy(array.ptr, self->ptr, self->len * sizeof(int));
  return array;
}

static void IntArray_destroy(IntArray* self) {
  free(self->ptr);
  self->ptr = (int*)0x5a5a5a5a00;
}

static void IntArray_push(IntArray* self, int val) {
  if (self->len == self->cap) {
    if (self->cap == 0) {
      self->cap = 8;
      self->ptr = malloc(self->cap * sizeof(int));
    } else {
      self->cap *= 2;
      self->ptr = realloc(self->ptr, self->cap * sizeof(int));
    }
  }
  self->ptr[self->len] = val;
  ++self->len;
}

static void IntArray_dump(const IntArray* self) {
  printf("%p[%zu:%zu] {", self->ptr, self->len, self->cap);
  for (size_t i = 0; i < self->len; ++i) {
    if (i != 0) {
      printf(", %d", self->ptr[i]);
    } else {
      printf("%d", self->ptr[i]);
    }
  }
  printf("}");
}

static inline void kIntArrayPolicy_copy(void* dst, const void* src) {
  typedef struct {
    IntArray k;
    float v;
  } entry;
  const entry* e = (const entry*)src;
  entry* d = (entry*)dst;

  d->k = IntArray_dup(&e->k);
  d->v = e->v;
}
static inline void kIntArrayPolicy_dtor(void* val) {
  IntArray_destroy((IntArray*)val);
}

static inline size_t kIntArrayPolicy_hash(const void* val) {
  const IntArray* arr = (const IntArray*)val;
  CWISS_FxHash_State state = 0;
  CWISS_FxHash_Write(&state, &arr->len, sizeof(arr->len));
  CWISS_FxHash_Write(&state, arr->ptr, arr->len * sizeof(int));
  return state;
}
static inline bool kIntArrayPolicy_eq(const void* a, const void* b) {
  const IntArray* ap = (const IntArray*)a;
  const IntArray* bp = (const IntArray*)b;
  return ap->len == bp->len &&
         memcmp(ap->ptr, bp->ptr, ap->len * sizeof(int)) == 0;
}

CWISS_DECLARE_NODE_MAP_POLICY(kIntArrayPolicy, IntArray, float,
                              (obj_copy, kIntArrayPolicy_copy),
                              (obj_dtor, kIntArrayPolicy_dtor),
                              (key_hash, kIntArrayPolicy_hash),
                              (key_eq, kIntArrayPolicy_eq));

CWISS_DECLARE_HASHMAP_WITH(MyArrayMap, IntArray, float, kIntArrayPolicy);

static inline size_t MyArrayMap_IntView_hash(const IntView* self) {
  CWISS_FxHash_State state = 0;
  CWISS_FxHash_Write(&state, &self->len, sizeof(self->len));
  CWISS_FxHash_Write(&state, self->ptr, self->len * sizeof(int));
  return state;
}

static inline bool MyArrayMap_IntView_eq(const IntView* self,
                                         const MyArrayMap_Entry* that) {
  IntArray_dump(&that->key);
  return self->len == that->key.len &&
         memcmp(self->ptr, that->key.ptr, self->len * sizeof(int)) == 0;
}

CWISS_DECLARE_LOOKUP(MyArrayMap, IntView);

int main(void) {
  MyArrayMap map = MyArrayMap_new(8);
  IntArray arr = IntArray_new();

  for (int i = 0; i < 8; ++i) {
    IntArray_push(&arr, i);
    IntArray_dump(&arr);
    printf("\n");

    int val = i * i + 1;
    MyArrayMap_Entry e = {arr, sin(val)};
    // MyArrayMap_dump(&map);
    MyArrayMap_insert(&map, &e);
  }
  MyArrayMap_dump(&map);
  printf("\n");

  int buf[5] = {2, 3, 4};
  IntView k = {buf, 3};
  assert(!MyArrayMap_contains_by_IntView(&map, &k));
  k.ptr = arr.ptr;
  MyArrayMap_Iter it = MyArrayMap_find_by_IntView(&map, &k);
  MyArrayMap_Entry* v = MyArrayMap_Iter_get(&it);
  assert(v);
  printf("5: %p: ", v);
  IntArray_dump(&v->key);
  printf("->%f\n", v->val);

  MyArrayMap_rehash(&map, 16);

  it = MyArrayMap_find_by_IntView(&map, &k);
  printf("5: %p: ", v);
  IntArray_dump(&v->key);
  printf("->%f\n", v->val);

  printf("entries:\n");
  it = MyArrayMap_iter(&map);
  for (MyArrayMap_Entry* p = MyArrayMap_Iter_get(&it); p != NULL;
       p = MyArrayMap_Iter_next(&it)) {
    IntArray_dump(&p->key);
    printf("->%f\n", p->val);
  }
  printf("\n");

  MyArrayMap_erase_by_IntView(&map, &k);
  assert(!MyArrayMap_contains_by_IntView(&map, &k));

  MyArrayMap_Insert in = MyArrayMap_deferred_insert_by_IntView(&map, &k);
  assert(in.inserted);
  v = MyArrayMap_Iter_get(&in.iter);
  v->key = IntArray_from_view(k);
  v->val = 42;

  printf("entries:\n");
  it = MyArrayMap_iter(&map);
  for (MyArrayMap_Entry* p = MyArrayMap_Iter_get(&it); p != NULL;
       p = MyArrayMap_Iter_next(&it)) {
    IntArray_dump(&p->key);
    printf("->%f\n", p->val);
  }
  printf("\n");

  MyArrayMap_dump(&map);
  MyArrayMap_destroy(&map);

  return 0;
}