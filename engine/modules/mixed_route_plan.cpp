// Fixed GLM TP4 routing plan. No Python, GPU, shared state or heap allocation.
#include <algorithm>
#include <cstdint>
#include <utility>

extern "C" int st_mixed_plan(
    const int32_t* decode, int d, const int32_t* prefill, int p, int quota,
    int32_t* sizes, int32_t* sources, int32_t* experts, int32_t* dc,
    int32_t* hot, int32_t* cc, int32_t* bases, int32_t* tasks,
    int32_t* valid, int32_t* cold_routes, int32_t* cold_sources) {
  if (d < 1 || d > 32 || p < 1 || p > 32768 || quota < 0 || quota > 128)
    return 1;
  int pc[288] = {}, local[288], hot_base[288] = {}, cold_base[288] = {};
  int cursor[288] = {}, seen[288] = {};
  std::fill_n(dc, 288, 0);
  std::fill_n(hot, 288, 0);
  std::fill_n(local, 288, -1);
  int expert_count = 0;
  for (int kind = 0; kind < 2; ++kind) {
    const int32_t* ids = kind ? prefill : decode;
    int rows = kind ? p : d;
    for (int row = 0; row < rows; ++row) {
      for (int slot = 0; slot < 8; ++slot) {
        int e = ids[row * 8 + slot];
        if (e < 0 || e >= 288) return 2;
        for (int prior = 0; prior < slot; ++prior)
          if (ids[row * 8 + prior] == e) return 2;
        if (kind) ++pc[e];
        else {
          if (local[e] < 0) {
            local[e] = expert_count;
            experts[expert_count++] = e;
          }
          ++dc[e];
        }
      }
    }
  }
  const int tile = d <= 8 ? 16 : 32;
  std::pair<int, int> candidates[288];
  int candidate_count = 0;
  for (int i = 0; i < expert_count; ++i) {
    int e = experts[i], spare = ((dc[e] + tile - 1) / tile) * tile - dc[e];
    int tail = pc[e] ? (pc[e] - 1) % 128 + 1 : 0;
    if (tail && tail <= spare) candidates[candidate_count++] = {tail, e};
  }
  std::sort(candidates, candidates + candidate_count);
  for (int i = 0; i < candidate_count; ++i) {
    auto [tail, e] = candidates[i];
    if (tail <= quota) { hot[e] = tail; quota -= tail; }
  }
  int source_count = d * 8;
  for (int i = 0; i < expert_count; ++i) {
    int e = experts[i];
    hot_base[e] = source_count;
    source_count += hot[e];
  }
  for (int i = 0; i < d * 8; ++i) {
    int e = decode[i];
    int32_t* s = sources + i * 5;
    s[0] = local[e]; s[1] = cursor[e]++; s[2] = 0; s[3] = i / 8; s[4] = i % 8;
  }
  int cold_count = 0, task_count = 0;
  bases[0] = 0;
  for (int e = 0; e < 288; ++e) {
    cc[e] = pc[e] - hot[e];
    cold_base[e] = cold_count;
    cold_count += cc[e];
    int tiles = (cc[e] + 127) / 128;
    bases[e + 1] = bases[e] + tiles;
    for (int t = 0; t < tiles; ++t) {
      tasks[task_count] = e | ((bases[e] + t) << 16);
      valid[task_count++] = std::min(128, cc[e] - t * 128) | (4 << 20);
    }
  }
  int original_cold = 0;
  for (int i = 0; i < p * 8; ++i) {
    int e = prefill[i], offset = seen[e]++;
    if (offset < hot[e]) {
      int32_t* s = sources + (hot_base[e] + offset) * 5;
      s[0] = local[e]; s[1] = dc[e] + offset; s[2] = 1; s[3] = i / 8; s[4] = i % 8;
    } else {
      offset -= hot[e];
      int32_t* s = cold_sources + (cold_base[e] + offset) * 4;
      s[0] = e; s[1] = bases[e] * 128 + offset; s[2] = i / 8; s[3] = i % 8;
      cold_routes[original_cold * 2] = i / 8;
      cold_routes[original_cold++ * 2 + 1] = i % 8;
    }
  }
  sizes[0] = expert_count; sizes[1] = source_count;
  sizes[2] = cold_count; sizes[3] = task_count;
  return 0;
}
