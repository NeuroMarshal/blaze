// EAGLE-4 draft kernels (blaze/eagle.h) against CPU references: strided RMSNorm, neox RoPE, candidate logits from a
// mapped host table, the greedy pick.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "blaze/eagle.h"

namespace {

void check(cudaError_t e, const char* w) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
    std::exit(1);
  }
}
uint16_t bf(float v) { return __bfloat16_as_ushort(__float2bfloat16_rn(v)); }
float fb(uint16_t v) { return __bfloat162float(__ushort_as_bfloat16(v)); }

template <typename T>
T* up(const std::vector<T>& h) {
  T* d;
  check(cudaMalloc(&d, h.size() * sizeof(T)), "malloc");
  check(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "copy");
  return d;
}
template <typename T>
std::vector<T> down(const T* d, size_t n) {
  std::vector<T> h(n);
  check(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "copy back");
  return h;
}

}  // namespace

int main() {
  std::mt19937 rng(7);
  std::normal_distribution<float> n01;
  int fails = 0;
  // strided RMSNorm: 5 rows of 5120 out of a [rows, 3 * 5120] buffer into column block 1 of another
  {
    const int rows = 5, dim = 5120, ldx = 3 * dim, ldo = 3 * dim;
    std::vector<uint16_t> x(size_t(rows) * ldx), w(dim), out(size_t(rows) * ldo, 0);
    for (auto& v : x) v = bf(n01(rng) * 3.0f);
    for (auto& v : w) v = bf(1.0f + 0.1f * n01(rng));
    uint16_t *dx = up(x), *dw = up(w), *dout = up(out);
    check(blz_rmsnorm_ld(dx + dim, ldx, dw, 1e-6f, rows, dim, dout + dim, ldo, 0), "rmsnorm_ld");
    const auto got = down(dout, out.size());
    double worst = 0;
    for (int r = 0; r < rows; ++r) {
      double ss = 0;
      for (int i = 0; i < dim; ++i) ss += double(fb(x[size_t(r) * ldx + dim + i])) * fb(x[size_t(r) * ldx + dim + i]);
      const double inv = 1.0 / std::sqrt(ss / dim + 1e-6);
      for (int i = 0; i < dim; ++i) {
        const double ref = fb(x[size_t(r) * ldx + dim + i]) * inv * fb(w[i]);
        worst = std::max(worst, std::fabs(fb(got[size_t(r) * ldo + dim + i]) - ref) / (std::fabs(ref) + 1e-2));
      }
      for (int i = 0; i < dim; ++i) fails += got[size_t(r) * ldo + i] != 0;  // block 0 untouched
    }
    std::printf("rmsnorm_ld: worst rel %.2e %s\n", worst, worst < 1e-2 ? "ok" : "FAIL");
    fails += worst >= 1e-2;
  }
  // neox RoPE on q-like rows: 3 tokens, 40 heads of 128, row stride 15360
  {
    const int tokens = 3, heads = 40, hd = 128, ld = 15360;
    std::vector<uint16_t> x(size_t(tokens) * ld);
    for (auto& v : x) v = bf(n01(rng));
    std::vector<int64_t> pos = {0, 77, 300000};
    std::vector<float> inv(hd / 2);
    for (int i = 0; i < hd / 2; ++i) inv[i] = float(std::pow(1e7, -2.0 * i / hd));
    uint16_t* dx = up(x);
    int64_t* dp = up(pos);
    float* di = up(inv);
    check(blz_rope_neox(dx, ld, tokens, heads, hd, dp, di, 0), "rope");
    const auto got = down(dx, x.size());
    double worst = 0;
    for (int t = 0; t < tokens; ++t)
      for (int h = 0; h < heads; ++h)
        for (int j = 0; j < hd / 2; ++j) {
          const size_t o = size_t(t) * ld + size_t(h) * hd;
          const double ang = double(float(pos[t]) * inv[j]);
          const double a = fb(x[o + j]), b = fb(x[o + j + hd / 2]);
          const double ra = a * std::cos(ang) - b * std::sin(ang), rb = b * std::cos(ang) + a * std::sin(ang);
          worst = std::max({worst, std::fabs(fb(got[o + j]) - ra), std::fabs(fb(got[o + j + hd / 2]) - rb)});
        }
    // the untouched tail of each row (past 40 * 128 = 5120)
    for (int t = 0; t < tokens; ++t)
      for (int i = heads * hd; i < ld; ++i) fails += got[size_t(t) * ld + i] != x[size_t(t) * ld + i];
    std::printf("rope_neox: worst abs %.2e %s\n", worst, worst < 3e-2 ? "ok" : "FAIL");
    fails += worst >= 3e-2;
  }
  // candidate logits from a mapped host table + pick
  {
    const int vocab = 50000, dim = 5120, rows = 3, c = 128;
    uint16_t* table;
    check(cudaHostAlloc(reinterpret_cast<void**>(&table), size_t(vocab) * dim * 2, cudaHostAllocMapped), "host alloc");
    for (size_t i = 0; i < size_t(vocab) * dim; ++i) table[i] = bf(n01(rng) * 0.02f);
    uint16_t* dtable;
    check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&dtable), table, 0), "mapped");
    std::vector<uint16_t> h(size_t(rows) * dim);
    for (auto& v : h) v = bf(n01(rng));
    std::vector<int32_t> ids(size_t(rows) * c);
    for (auto& v : ids) v = int32_t(rng() % vocab);
    ids[5] = -1;
    uint16_t* dh = up(h);
    int32_t* dids = up(ids);
    float* dl;
    int64_t *dtok, *drec;
    check(cudaMalloc(&dl, ids.size() * 4), "malloc");
    check(cudaMalloc(&dtok, rows * 8), "malloc");
    check(cudaMalloc(&drec, rows * 8), "malloc");
    check(blz_candidate_logits(dh, dim, rows, dim, dtable, dids, c, dl, 0), "cand logits");
    check(blz_candidate_pick(dl, dids, rows, c, dtok, drec, 0), "pick");
    const auto lg = down(dl, ids.size());
    const auto tok = down(dtok, rows), rec = down(drec, rows);
    double worst = 0;
    for (int r = 0; r < rows; ++r) {
      double best = -1e300;
      int bid = -1;
      for (int j = 0; j < c; ++j) {
        const int id = ids[size_t(r) * c + j];
        if (id < 0) continue;
        double ref = 0;
        for (int i = 0; i < dim; ++i) ref += double(fb(table[size_t(id) * dim + i])) * fb(h[size_t(r) * dim + i]);
        worst = std::max(worst, std::fabs(lg[size_t(r) * c + j] - ref));
        if (lg[size_t(r) * c + j] > best || (lg[size_t(r) * c + j] == best && id < bid)) best = lg[size_t(r) * c + j], bid = id;
      }
      fails += tok[r] != bid || rec[r] != bid;
    }
    std::printf("candidate logits: worst abs %.2e, picks %s\n", worst, fails ? "FAIL" : "ok");
    fails += worst >= 1e-3;
    cudaEvent_t a, b;
    cudaEventCreate(&a), cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < 20; ++i) blz_candidate_logits(dh, dim, 1, dim, dtable, dids, c, dl, 0);
    cudaEventRecord(b);
    check(cudaEventSynchronize(b), "time");
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    std::printf("one row x %d candidates from host memory: %.1f us\n", c, ms * 1000.0f / 20);
  }
  std::printf("%s\n", fails ? "FAILED" : "PASS");
  return fails ? 1 : 0;
}
