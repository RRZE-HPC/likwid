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

// This file demonstrates the API of a cwisstable map using C-style strings as
// keys.

#include <math.h>
#include <stdio.h>
#include <string.h>

#if defined(CWISS_EXAMPLE_UNIFIED)
  #include "cwisstable.h"
#elif defined(CWISS_EXAMPLE_SPLIT)
  #include "cwisstable/declare.h"
  #include "cwisstable/policy.h"
#else
  #error "must set CWISS_EXAMPLE_UNIFIED or CWISS_EXAMPLE_SPLIT"
#endif

static inline void kCStrPolicy_copy(void* dst, const void* src) {
  typedef struct {
    char* k;
    float v;
  } entry;
  const entry* e = (const entry*)src;
  entry* d = (entry*)dst;

  size_t len = strlen(e->k);
  d->k = malloc(len + 1);
  d->v = e->v;
  memcpy(d->k, e->k, len + 1);
}
static inline void kCStrPolicy_dtor(void* val) {
  char* str = *(char**)val;
  free(str);
}

static inline size_t kCStrPolicy_hash(const void* val) {
  const char* str = *(const char* const*)val;
  size_t len = strlen(str);
  CWISS_FxHash_State state = 0;
  CWISS_FxHash_Write(&state, str, len);
  return state;
}
static inline bool kCStrPolicy_eq(const void* a, const void* b) {
  const char* ap = *(const char* const*)a;
  const char* bp = *(const char* const*)b;
  return strcmp(ap, bp) == 0;
}

CWISS_DECLARE_NODE_MAP_POLICY(kCStrPolicy, const char*, float,
                              (obj_copy, kCStrPolicy_copy),
                              (obj_dtor, kCStrPolicy_dtor),
                              (key_hash, kCStrPolicy_hash),
                              (key_eq, kCStrPolicy_eq));

CWISS_DECLARE_HASHMAP_WITH(MyCStrMap, const char*, float, kCStrPolicy);

static const char* kStrings[] = {
    "abcd", "efgh", "ijkh", "lmno", "pqrs", "tuvw", "xyza", "bcde",
};

int main(void) {
  MyCStrMap map = MyCStrMap_new(8);

  for (int i = 0; i < 8; ++i) {
    int val = i * i + 1;
    MyCStrMap_Entry e = {kStrings[i], sin(val)};
    MyCStrMap_dump(&map);
    MyCStrMap_insert(&map, &e);
  }
  MyCStrMap_dump(&map);
  printf("\n");

  const char* k = "missing";
  assert(!MyCStrMap_contains(&map, &k));
  k = "lmno";
  MyCStrMap_Iter it = MyCStrMap_find(&map, &k);
  MyCStrMap_Entry* v = MyCStrMap_Iter_get(&it);
  assert(v);
  printf("5: %p: \"%s\"->%f\n", v, v->key, v->val);

  MyCStrMap_rehash(&map, 16);

  it = MyCStrMap_find(&map, &k);
  v = MyCStrMap_Iter_get(&it);
  assert(v);
  printf("5: %p: \"%s\"->%f\n", v, v->key, v->val);

  printf("entries:\n");
  it = MyCStrMap_iter(&map);
  for (MyCStrMap_Entry* p = MyCStrMap_Iter_get(&it); p != NULL;
       p = MyCStrMap_Iter_next(&it)) {
    printf("\"%s\"->%f\n", p->key, p->val);
  }
  printf("\n");

  MyCStrMap_erase(&map, &k);
  assert(!MyCStrMap_contains(&map, &k));

  printf("entries:\n");
  it = MyCStrMap_iter(&map);
  for (MyCStrMap_Entry* p = MyCStrMap_Iter_get(&it); p != NULL;
       p = MyCStrMap_Iter_next(&it)) {
    printf("\"%s\"->%f\n", p->key, p->val);
  }
  printf("\n");

  MyCStrMap_dump(&map);
  MyCStrMap_destroy(&map);

  return 0;
}