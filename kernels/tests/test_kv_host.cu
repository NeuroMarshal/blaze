// blz_kv_host_gather / blz_kv_host_fill: rows of a host pool gathered through the row cache are the pool's rows, over
// rounds of selections with repeats, line collisions and invalidated slots; every cached line holds its tag's row.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "blaze/kv_host.h"

namespace {

void check(cudaError_t e, const char* w) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
    std::exit(1);
  }
}

}  // namespace

int main() {
  const int heads = 2, dim = 256, slots = 5000, lines = 1024, rows = 6, stride = 300;
  const size_t data_row = heads * dim / 2, scale_row = heads * dim / 16;
  std::mt19937 rng(5);
  std::uniform_int_distribution<int> byte(0, 255);
  // host pool (mapped)
  uint8_t *hk, *hv, *hks, *hvs;
  for (uint8_t** p : {&hk, &hv}) check(cudaHostAlloc(p, slots * data_row, cudaHostAllocMapped), "host alloc");
  for (uint8_t** p : {&hks, &hvs}) check(cudaHostAlloc(p, slots * scale_row, cudaHostAllocMapped), "host alloc");
  for (size_t i = 0; i < slots * data_row; ++i) hk[i] = uint8_t(byte(rng)), hv[i] = uint8_t(byte(rng));
  for (size_t i = 0; i < slots * scale_row; ++i) hks[i] = uint8_t(byte(rng)), hvs[i] = uint8_t(byte(rng));
  auto dev = [](void* h) {
    void* d;
    check(cudaHostGetDevicePointer(&d, h, 0), "device pointer");
    return static_cast<uint8_t*>(d);
  };
  blz_nvfp4_pool pool{dev(hk), dev(hv), dev(hks), dev(hvs), nullptr, nullptr, heads, dim, 0};
  // cache and compact pool
  blz_nvfp4_row_cache cache{};
  cache.lines = lines;
  check(cudaMalloc(&cache.tag, lines * 4), "malloc");
  check(cudaMemset(cache.tag, 0xff, lines * 4), "memset");
  check(cudaMalloc(&cache.k_data, lines * data_row), "malloc");
  check(cudaMalloc(&cache.v_data, lines * data_row), "malloc");
  check(cudaMalloc(&cache.k_scale, lines * scale_row), "malloc");
  check(cudaMalloc(&cache.v_scale, lines * scale_row), "malloc");
  uint8_t *ok, *ov, *oks, *ovs;
  const size_t out_rows = size_t(rows) * stride;
  check(cudaMalloc(&ok, out_rows * data_row), "malloc");
  check(cudaMalloc(&ov, out_rows * data_row), "malloc");
  check(cudaMalloc(&oks, out_rows * scale_row), "malloc");
  check(cudaMalloc(&ovs, out_rows * scale_row), "malloc");
  blz_nvfp4_pool out{ok, ov, oks, ovs, nullptr, nullptr, heads, dim, 0};
  int32_t *table, *counts, *fills, *fill_n;
  check(cudaMalloc(&table, rows * stride * 4), "malloc");
  check(cudaMalloc(&counts, rows * 4), "malloc");
  check(cudaMalloc(&fills, out_rows * 12), "malloc");
  check(cudaMalloc(&fill_n, 4), "malloc");
  int bad = 0, filled_total = 0;
  std::vector<int> hot(400);
  for (auto& s : hot) s = std::uniform_int_distribution<int>(0, slots - 1)(rng);
  for (int round = 0; round < 12; ++round) {
    std::vector<int32_t> t(rows * stride, 0), c(rows);
    for (int r = 0; r < rows; ++r) {
      c[r] = std::uniform_int_distribution<int>(1, stride)(rng);
      for (int j = 0; j < c[r]; ++j)  // mostly a recurring set (cache hits), some fresh slots, colliding lines
        t[r * stride + j] = rng() % 4 ? hot[rng() % hot.size()] : int32_t(rng() % slots);
    }
    // some slots are rewritten (new tokens): host rows change, their lines are dropped first
    if (round % 3 == 2) {
      std::vector<int64_t> loc;
      for (int i = 0; i < 20; ++i) {
        const int s = hot[rng() % hot.size()];
        loc.push_back(s);
        for (size_t b = 0; b < data_row; ++b) hk[s * data_row + b] = uint8_t(byte(rng));
      }
      int64_t* dloc;
      check(cudaMalloc(&dloc, loc.size() * 8), "malloc");
      check(cudaMemcpy(dloc, loc.data(), loc.size() * 8, cudaMemcpyHostToDevice), "copy");
      check(blz_nvfp4_row_cache_invalidate(cache, blz_index_ptr{dloc, BLZ_I64}, int64_t(loc.size()), 0), "invalidate");
      check(cudaDeviceSynchronize(), "sync");
      cudaFree(dloc);
    }
    check(cudaMemcpy(table, t.data(), t.size() * 4, cudaMemcpyHostToDevice), "copy");
    check(cudaMemcpy(counts, c.data(), c.size() * 4, cudaMemcpyHostToDevice), "copy");
    check(cudaMemset(fill_n, 0, 4), "memset");
    check(blz_kv_host_gather(pool, cache, table, stride, counts, rows, stride, out, stride, fills, fill_n, int(out_rows), 0),
          "gather");
    check(blz_kv_host_fill(cache, out, fills, fill_n, int(out_rows), 0), "fill");
    check(cudaDeviceSynchronize(), "sync");
    int nf = 0;
    check(cudaMemcpy(&nf, fill_n, 4, cudaMemcpyDeviceToHost), "copy");
    filled_total += nf;
    std::vector<uint8_t> gk(out_rows * data_row), gks(out_rows * scale_row);
    check(cudaMemcpy(gk.data(), ok, gk.size(), cudaMemcpyDeviceToHost), "copy");
    check(cudaMemcpy(gks.data(), oks, gks.size(), cudaMemcpyDeviceToHost), "copy");
    for (int r = 0; r < rows; ++r)
      for (int j = 0; j < c[r]; ++j) {
        const int s = t[r * stride + j];
        const size_t o = size_t(r) * stride + j;
        if (std::memcmp(&gk[o * data_row], &hk[s * data_row], data_row) || std::memcmp(&gks[o * scale_row], &hks[s * scale_row], scale_row))
          ++bad;
      }
    // every line with a tag holds that slot's row
    std::vector<int32_t> tag(lines);
    std::vector<uint8_t> ck(lines * data_row);
    check(cudaMemcpy(tag.data(), cache.tag, lines * 4, cudaMemcpyDeviceToHost), "copy");
    check(cudaMemcpy(ck.data(), cache.k_data, ck.size(), cudaMemcpyDeviceToHost), "copy");
    for (int l = 0; l < lines; ++l)
      if (tag[l] >= 0 && (tag[l] % lines != l || std::memcmp(&ck[l * data_row], &hk[tag[l] * data_row], data_row))) ++bad;
  }
  std::printf("12 rounds: %d rows filled from host memory, %d mismatches %s\n", filled_total, bad, bad ? "FAIL" : "ok");
  return bad ? 1 : 0;
}
