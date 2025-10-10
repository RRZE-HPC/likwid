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

// This file demonstrates the API of a basic cwisstable map.

#include <math.h>
#include <stdio.h>

#if defined(CWISS_EXAMPLE_UNIFIED)
  #include "cwisstable.h"
#elif defined(CWISS_EXAMPLE_SPLIT)
  #include "cwisstable/declare.h"
  #include "cwisstable/policy.h"
#else
  #error "must set CWISS_EXAMPLE_UNIFIED or CWISS_EXAMPLE_SPLIT"
#endif

CWISS_DECLARE_NODE_HASHMAP(MyIntMap, int, float);

int main(void) {
  MyIntMap map = MyIntMap_new(8);

  for (int i = 0; i < 8; ++i) {
    int val = i * i + 1;
    MyIntMap_Entry e = {val, sin(val)};
    MyIntMap_dump(&map);
    MyIntMap_insert(&map, &e);
  }
  MyIntMap_dump(&map);
  printf("\n");

  int k = 4;
  assert(!MyIntMap_contains(&map, &k));
  k = 5;
  MyIntMap_Iter it = MyIntMap_find(&map, &k);
  MyIntMap_Entry* v = MyIntMap_Iter_get(&it);
  assert(v);
  printf("5: %p: %d->%f\n", v, v->key, v->val);

  MyIntMap_rehash(&map, 16);

  it = MyIntMap_find(&map, &k);
  v = MyIntMap_Iter_get(&it);
  assert(v);
  printf("5: %p: %d->%f\n", v, v->key, v->val);

  printf("entries:\n");
  it = MyIntMap_iter(&map);
  for (MyIntMap_Entry* p = MyIntMap_Iter_get(&it); p != NULL;
       p = MyIntMap_Iter_next(&it)) {
    printf("%d->%f\n", p->key, p->val);
  }
  printf("\n");

  MyIntMap_erase(&map, &k);
  assert(!MyIntMap_contains(&map, &k));

  printf("entries:\n");
  it = MyIntMap_iter(&map);
  for (MyIntMap_Entry* p = MyIntMap_Iter_get(&it); p != NULL;
       p = MyIntMap_Iter_next(&it)) {
    printf("%d->%f\n", p->key, p->val);
  }
  printf("\n");

  MyIntMap_dump(&map);
  MyIntMap_destroy(&map);

  return 0;
}