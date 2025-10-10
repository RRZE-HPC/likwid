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

// This file demonstrates the API of a basic cwisstable set.

#include <stdio.h>

#if defined(CWISS_EXAMPLE_UNIFIED)
  #include "cwisstable.h"
#elif defined(CWISS_EXAMPLE_SPLIT)
  #include "cwisstable/declare.h"
  #include "cwisstable/policy.h"
#else
  #error "must set CWISS_EXAMPLE_UNIFIED or CWISS_EXAMPLE_SPLIT"
#endif

CWISS_DECLARE_FLAT_HASHSET(MyIntSet, int);

int main(void) {
  MyIntSet set = MyIntSet_new(8);

  for (int i = 0; i < 8; ++i) {
    int val = i * i + 1;
    MyIntSet_dump(&set);
    MyIntSet_insert(&set, &val);
  }
  MyIntSet_dump(&set);
  printf("\n");

  int k = 4;
  assert(!MyIntSet_contains(&set, &k));
  k = 5;
  MyIntSet_Iter it = MyIntSet_find(&set, &k);
  int* v = MyIntSet_Iter_get(&it);
  assert(v);
  printf("5: %p: %d\n", v, *v);

  MyIntSet_rehash(&set, 16);

  it = MyIntSet_find(&set, &k);
  v = MyIntSet_Iter_get(&it);
  assert(v);
  printf("5: %p: %d\n", v, *v);

  printf("entries:\n");
  it = MyIntSet_iter(&set);
  for (int* p = MyIntSet_Iter_get(&it); p != NULL;
       p = MyIntSet_Iter_next(&it)) {
    printf("%d\n", *p);
  }
  printf("\n");

  MyIntSet_erase(&set, &k);
  assert(!MyIntSet_contains(&set, &k));

  printf("entries:\n");
  it = MyIntSet_iter(&set);
  for (int* p = MyIntSet_Iter_get(&it); p != NULL;
       p = MyIntSet_Iter_next(&it)) {
    printf("%d\n", *p);
  }
  printf("\n");

  MyIntSet_dump(&set);
  MyIntSet_destroy(&set);

  return 0;
}
