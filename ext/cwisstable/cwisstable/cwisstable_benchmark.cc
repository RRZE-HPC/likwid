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

// absl::raw_hash_set's benchmarks modified to run over cwisstable.

#include <deque>
#include <numeric>
#include <random>
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "absl/strings/str_format.h"
#include "benchmark/benchmark.h"
#include "cwisstable.h"
#include "cwisstable/internal/test_helpers.h"

namespace cwisstable {
namespace {

using benchmark::DoNotOptimize;

CWISS_DECLARE_HASHMAP_WITH(
    StringTable, std::string, std::string,
    (FlatMapPolicy<std::string, std::string, HashStdString>()));
CWISS_DECLARE_HASHSET_WITH(IntTable, int64_t, FlatPolicy<int64_t>());

TABLE_HELPERS(IntTable);

struct StringGen {
  template <class Gen>
  std::string operator()(Gen& rng) const {
    std::string res;
    res.resize(12);
    std::uniform_int_distribution<uint32_t> printable_ascii(0x20, 0x7E);
    std::generate(res.begin(), res.end(), [&] { return printable_ascii(rng); });
    return res;
  }

  size_t size;
};

// Model a cache in steady state.
//
// On a table of size N, keep deleting the LRU entry and add a random one.
void BM_CacheInSteadyState(benchmark::State& state) {
  std::random_device rd;
  std::mt19937 rng(rd());
  StringGen gen{12};
  auto t = StringTable_new(0);
  absl::Cleanup c_ = [&] { StringTable_destroy(&t); };

  std::deque<std::string> keys;
  while (StringTable_size(&t) < state.range(0)) {
    std::string k = gen(rng);
    std::string v = gen(rng);
    auto [it, inserted] = StringTable_deferred_insert(&t, &k);
    if (inserted) {
      auto* ptr = StringTable_Iter_get(&it);
      new (&ptr->key) std::string(std::move(k));
      new (&ptr->val) std::string(std::move(v));
      keys.push_back(ptr->key);
    }
  }

  CWISS_CHECK(state.range(0) >= 10, "n/a");
  while (state.KeepRunning()) {
    // Some cache hits.
    std::deque<std::string>::const_iterator it;
    for (int i = 0; i != 90; ++i) {
      if (i % 10 == 0) {
        it = keys.end();
      }

      DoNotOptimize(StringTable_find(&t, &*--it));
    }

    // Some cache misses.
    for (int i = 0; i != 10; ++i) {
      std::string s = gen(rng);
      DoNotOptimize(StringTable_find(&t, &s));
    }

    CWISS_CHECK(StringTable_erase(&t, &keys.front()), "missing? %s",
                keys.front().c_str());

    keys.pop_front();
    while (true) {
      std::string k = gen(rng);
      std::string v = gen(rng);
      auto [it, inserted] = StringTable_deferred_insert(&t, &k);
      if (inserted) {
        auto* ptr = StringTable_Iter_get(&it);
        new (&ptr->key) std::string(std::move(k));
        new (&ptr->val) std::string(std::move(v));
        keys.push_back(ptr->key);
        break;
      }
    }
  }
  state.SetItemsProcessed(state.iterations());
  double load_factor =
      static_cast<double>(StringTable_size(&t)) / StringTable_capacity(&t);
  state.SetLabel(absl::StrFormat("load_factor=%.2f", load_factor));
}

template <typename Benchmark>
void CacheInSteadyStateArgs(Benchmark* bm) {
  // The default.
  const float max_load_factor = 0.875;
  // When the cache is at the steady state, the probe sequence will equal
  // capacity if there is no reclamation of deleted slots. Pick a number large
  // enough to make the benchmark slow for that case.
  const size_t capacity = 1 << 10;

  // Check N data points to cover load factors in [0.4, 0.8).
  const size_t kNumPoints = 10;
  for (size_t i = 0; i != kNumPoints; ++i)
    bm->Arg(std::ceil(
        capacity * (max_load_factor + i * max_load_factor / kNumPoints) / 2));
}
BENCHMARK(BM_CacheInSteadyState)->Apply(CacheInSteadyStateArgs);

// This is miscompiled by GCC, so the benchmark will be messed up on that
// compiler.
#if !CWISS_IS_GCC
void BM_EndComparison(benchmark::State& state) {
  std::random_device rd;
  std::mt19937 rng(rd());
  StringGen gen{12};
  auto t = StringTable_new(0);
  absl::Cleanup c_ = [&] { StringTable_destroy(&t); };

  while (StringTable_size(&t) < state.range(0)) {
    std::string k = gen(rng);
    std::string v = gen(rng);
    auto [it, inserted] = StringTable_deferred_insert(&t, &k);
    if (inserted) {
      auto* ptr = StringTable_Iter_get(&it);
      new (&ptr->key) std::string(std::move(k));
      new (&ptr->val) std::string(std::move(v));
    }
  }

  for (auto ignored : state) {
    for (auto it = StringTable_iter(&t); StringTable_Iter_get(&it);
         StringTable_Iter_next(&it)) {
      DoNotOptimize(it);
      DoNotOptimize(t);
      DoNotOptimize(StringTable_Iter_get(&it) != NULL);
    }
  }
}
BENCHMARK(BM_EndComparison)->Arg(400);
#endif

void BM_CopyCtor(benchmark::State& state) {
  std::random_device rd;
  std::mt19937 rng(rd());
  auto t = IntTable_new(0);
  absl::Cleanup c_ = [&] { IntTable_destroy(&t); };
  std::uniform_int_distribution<uint64_t> dist(0, ~uint64_t{});
  while (IntTable_size(&t) < state.range(0)) {
    Insert(t, dist(rng));
  }

  for (auto ignored : state) {
    IntTable t2 = IntTable_dup(&t);
    DoNotOptimize(t2);
    IntTable_destroy(&t2);
  }
}
BENCHMARK(BM_CopyCtor)->Range(128, 4096);

void BM_RangeCtor(benchmark::State& state) {
  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_int_distribution<uint64_t> dist(0, ~uint64_t{});
  std::vector<int> values;
  const size_t desired_size = state.range(0);
  while (values.size() < desired_size) {
    values.emplace_back(dist(rng));
  }

  for (auto unused : state) {
    IntTable t = IntTable_new(values.size());
    for (auto v : values) {
      Insert(t, v);
    }
    DoNotOptimize(t);
    IntTable_destroy(&t);
  }
}
BENCHMARK(BM_RangeCtor)->Range(128, 65536);

void BM_NoOpReserveIntTable(benchmark::State& state) {
  auto t = IntTable_new(0);
  absl::Cleanup c_ = [&] { IntTable_destroy(&t); };
  IntTable_reserve(&t, 100'000);

  for (auto ignored : state) {
    DoNotOptimize(t);
    IntTable_reserve(&t, 100'000);
  }
}
BENCHMARK(BM_NoOpReserveIntTable);

void BM_NoOpReserveStringTable(benchmark::State& state) {
  auto t = StringTable_new(0);
  absl::Cleanup c_ = [&] { StringTable_destroy(&t); };
  StringTable_reserve(&t, 100'000);

  for (auto ignored : state) {
    DoNotOptimize(t);
    StringTable_reserve(&t, 100'000);
  }
}
BENCHMARK(BM_NoOpReserveStringTable);

void BM_ReserveIntTable(benchmark::State& state) {
  int reserve_size = state.range(0);
  for (auto ignore : state) {
    state.PauseTiming();
    auto t = IntTable_new(0);
    state.ResumeTiming();
    DoNotOptimize(t);
    IntTable_reserve(&t, reserve_size);
    IntTable_destroy(&t);
  }
}
BENCHMARK(BM_ReserveIntTable)->Range(128, 4096);

void BM_ReserveStringTable(benchmark::State& state) {
  int reserve_size = state.range(0);
  for (auto _ : state) {
    state.PauseTiming();
    auto t = StringTable_new(0);
    state.ResumeTiming();
    DoNotOptimize(t);
    StringTable_reserve(&t, reserve_size);
    StringTable_destroy(&t);
  }
}
BENCHMARK(BM_ReserveStringTable)->Range(128, 4096);

// Like std::iota, except that ctrl_t doesn't support operator++.
template <typename CtrlIter>
void Iota(CtrlIter begin, CtrlIter end, int value) {
  for (; begin != end; ++begin, ++value) {
    *begin = static_cast<CWISS_ControlByte>(value);
  }
}

void BM_Group_Match(benchmark::State& state) {
  std::array<CWISS_ControlByte, CWISS_Group_kWidth> group;
  Iota(group.begin(), group.end(), -4);
  auto g = CWISS_Group_new(group.data());
  CWISS_h2_t h = 1;
  for (auto _ : state) {
    DoNotOptimize(h);
    DoNotOptimize(g);
    DoNotOptimize(CWISS_Group_Match(&g, h));
  }
}
BENCHMARK(BM_Group_Match);

void BM_Group_MatchEmpty(benchmark::State& state) {
  std::array<CWISS_ControlByte, CWISS_Group_kWidth> group;
  Iota(group.begin(), group.end(), -4);
  auto g = CWISS_Group_new(group.data());
  for (auto _ : state) {
    DoNotOptimize(g);
    DoNotOptimize(CWISS_Group_MatchEmpty(&g));
  }
}
BENCHMARK(BM_Group_MatchEmpty);

void BM_Group_MatchEmptyOrDeleted(benchmark::State& state) {
  std::array<CWISS_ControlByte, CWISS_Group_kWidth> group;
  Iota(group.begin(), group.end(), -4);
  auto g = CWISS_Group_new(group.data());
  for (auto _ : state) {
    DoNotOptimize(g);
    DoNotOptimize(CWISS_Group_MatchEmptyOrDeleted(&g));
  }
}
BENCHMARK(BM_Group_MatchEmptyOrDeleted);

void BM_Group_CountLeadingEmptyOrDeleted(benchmark::State& state) {
  std::array<CWISS_ControlByte, CWISS_Group_kWidth> group;
  Iota(group.begin(), group.end(), -2);
  auto g = CWISS_Group_new(group.data());
  for (auto _ : state) {
    DoNotOptimize(g);
    DoNotOptimize(CWISS_Group_CountLeadingEmptyOrDeleted(&g));
  }
}
BENCHMARK(BM_Group_CountLeadingEmptyOrDeleted);

void BM_Group_MatchFirstEmptyOrDeleted(benchmark::State& state) {
  std::array<CWISS_ControlByte, CWISS_Group_kWidth> group;
  Iota(group.begin(), group.end(), -2);
  auto g = CWISS_Group_new(group.data());
  for (auto _ : state) {
    DoNotOptimize(g);
    auto m = CWISS_Group_MatchEmptyOrDeleted(&g);
    DoNotOptimize(CWISS_BitMask_LowestBitSet(&m));
  }
}
BENCHMARK(BM_Group_MatchFirstEmptyOrDeleted);

void BM_DropDeletes(benchmark::State& state) {
  constexpr size_t capacity = (1 << 20) - 1;
  std::vector<CWISS_ControlByte> ctrl(capacity + 1 + CWISS_Group_kWidth);
  ctrl[capacity] = CWISS_kSentinel;
  std::vector<CWISS_ControlByte> pattern = {CWISS_kEmpty, 2, CWISS_kDeleted, 2,
                                            CWISS_kEmpty, 1, CWISS_kDeleted};
  for (size_t i = 0; i != capacity; ++i) {
    ctrl[i] = pattern[i % pattern.size()];
  }
  while (state.KeepRunning()) {
    state.PauseTiming();
    std::vector<CWISS_ControlByte> ctrl_copy = ctrl;
    state.ResumeTiming();
    CWISS_ConvertDeletedToEmptyAndFullToDeleted(ctrl_copy.data(), capacity);
    DoNotOptimize(ctrl_copy[capacity]);
  }
}
BENCHMARK(BM_DropDeletes);

}  // namespace
}  // namespace cwisstable

// These methods are here to make it easy to examine the assembly for targeted
// parts of the API.
auto CWISS_CodegenInt64Find(cwisstable::IntTable* table, int64_t key) {
  return IntTable_find(table, &key);
}

bool CWISS_CodegenInt64FindNeEnd(cwisstable::IntTable* table, int64_t key) {
  auto it = IntTable_find(table, &key);
  return IntTable_Iter_get(&it) != NULL;
}

auto CWISS_CodegenInt64Insert(cwisstable::IntTable* table, int64_t key) {
  return IntTable_insert(table, &key);
}

bool CWISS_CodegenInt64Contains(cwisstable::IntTable* table, int64_t key) {
  return IntTable_contains(table, &key);
}

void CWISS_CodegenInt64Iterate(cwisstable::IntTable* table) {
  for (auto it = IntTable_citer(table); IntTable_CIter_get(&it);
       IntTable_CIter_next(&it)) {
    DoNotOptimize(it);
  }
}

int odr = (benchmark::DoNotOptimize(std::make_tuple(
               &CWISS_CodegenInt64Find, &CWISS_CodegenInt64FindNeEnd,
               &CWISS_CodegenInt64Insert, &CWISS_CodegenInt64Contains,
               &CWISS_CodegenInt64Iterate)),
           1);