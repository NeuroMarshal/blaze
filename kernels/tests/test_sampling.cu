// blz_topk / blz_apply_penalties against a CPU sort, with ties and a vocabulary shard offset; and the time per row.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

#include "blaze/sampling.h"

namespace {

void check(cudaError_t e, const char* w) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
    std::exit(1);
  }
}

uint16_t bf(float v) { return __bfloat16_as_ushort(__float2bfloat16_rn(v)); }
float fb(uint16_t v) { return __bfloat162float(__ushort_as_bfloat16(v)); }

}  // namespace

int main() {
  std::mt19937 rng(3);
  const int rows = 3, vocab = 248320, k = 64, col0 = 1000;
  std::normal_distribution<float> d(0.0f, 3.0f);
  std::vector<uint16_t> logits(size_t(rows) * vocab);
  for (auto& v : logits) v = bf(d(rng));  // bf16 values: many exact ties
  // penalties: row 1 penalizes its 5 best columns
  std::vector<int32_t> ids, counts, off = {0};
  for (int r = 0; r < rows; ++r) {
    if (r == 1) {
      std::vector<int> idx(vocab);
      std::iota(idx.begin(), idx.end(), 0);
      std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                        [&](int a, int b) { return fb(logits[size_t(r) * vocab + a]) > fb(logits[size_t(r) * vocab + b]); });
      for (int j = 0; j < 5; ++j) ids.push_back(idx[j] + col0), counts.push_back(j + 1);
    }
    off.push_back(int32_t(ids.size()));
  }
  const float presence = 0.5f, frequency = 0.25f;
  std::vector<uint16_t> ref = logits;
  for (int r = 0; r < rows; ++r)
    for (int i = off[r]; i < off[r + 1]; ++i) {
      uint16_t& p = ref[size_t(r) * vocab + ids[i] - col0];
      p = bf(fb(p) - presence - frequency * counts[i]);
    }

  uint16_t* dl;
  int32_t *di, *dc, *doff, *oid;
  float* ov;
  void* ws;
  check(cudaMalloc(&dl, logits.size() * 2), "malloc");
  check(cudaMalloc(&di, ids.size() * 4 + 4), "malloc");
  check(cudaMalloc(&dc, counts.size() * 4 + 4), "malloc");
  check(cudaMalloc(&doff, off.size() * 4), "malloc");
  check(cudaMalloc(&oid, rows * k * 4), "malloc");
  check(cudaMalloc(&ov, rows * k * 4), "malloc");
  check(cudaMalloc(&ws, blz_topk_workspace_bytes(rows, vocab)), "malloc");
  check(cudaMemcpy(dl, logits.data(), logits.size() * 2, cudaMemcpyHostToDevice), "copy");
  check(cudaMemcpy(di, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "copy");
  check(cudaMemcpy(dc, counts.data(), counts.size() * 4, cudaMemcpyHostToDevice), "copy");
  check(cudaMemcpy(doff, off.data(), off.size() * 4, cudaMemcpyHostToDevice), "copy");
  std::vector<float> pres(rows, presence), freq(rows, frequency);
  float *dp, *df;
  check(cudaMalloc(&dp, rows * 4), "malloc");
  check(cudaMalloc(&df, rows * 4), "malloc");
  check(cudaMemcpy(dp, pres.data(), rows * 4, cudaMemcpyHostToDevice), "copy");
  check(cudaMemcpy(df, freq.data(), rows * 4, cudaMemcpyHostToDevice), "copy");
  check(blz_apply_penalties(dl, vocab, rows, col0, vocab, di, dc, doff, dp, df, 0), "penalties");
  check(blz_topk(dl, vocab, rows, vocab, col0, k, ov, oid, ws, 0), "topk");
  std::vector<float> gv(rows * k);
  std::vector<int32_t> gi(rows * k);
  check(cudaMemcpy(gv.data(), ov, gv.size() * 4, cudaMemcpyDeviceToHost), "copy back");
  check(cudaMemcpy(gi.data(), oid, gi.size() * 4, cudaMemcpyDeviceToHost), "copy back");
  int bad = 0;
  for (int r = 0; r < rows; ++r) {
    std::vector<int> idx(vocab);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) {
      const float va = fb(ref[size_t(r) * vocab + a]), vb = fb(ref[size_t(r) * vocab + b]);
      return va > vb || (va == vb && a < b);
    });
    for (int j = 0; j < k; ++j)
      if (gi[r * k + j] != idx[j] + col0 || gv[r * k + j] != fb(ref[size_t(r) * vocab + idx[j]])) ++bad;
  }
  std::printf("top-%d of %d rows x %d: %d mismatches\n", k, rows, vocab, bad);

  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  cudaEventRecord(a);
  for (int i = 0; i < 100; ++i) blz_topk(dl, vocab, 1, vocab, 0, k, ov, oid, ws, 0);
  cudaEventRecord(b);
  check(cudaEventSynchronize(b), "time");
  float ms = 0;
  cudaEventElapsedTime(&ms, a, b);
  std::printf("one row: %.1f us\n", ms * 10.0f);
  std::printf("%s\n", bad == 0 ? "PASS" : "FAIL");
  return bad == 0 ? 0 : 1;
}
