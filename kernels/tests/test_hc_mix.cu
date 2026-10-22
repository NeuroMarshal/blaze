// blz_hc_mix: agreement with an FP64 reference (same BF16 rounding of t and out), bitwise reproducibility, timing.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "blaze/canon.h"
#include "blaze/hc_mix.h"

namespace {

void check(cudaError_t e, const char* w) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
    std::exit(1);
  }
}

float bf(uint16_t b) {
  uint32_t u = uint32_t(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// Reference with the kernel's rounding points: t rounded to BF16, out rounded to BF16, sums in FP64.
void reference(const std::vector<uint16_t>& x, const std::vector<uint16_t>& wd, const std::vector<uint16_t>& wu,
               int rows, int hc, int hs, int R, std::vector<float>& out) {
  const int K = hc * hs;
  out.assign(size_t(rows) * hs, 0.0f);
  std::vector<float> t(size_t(rows) * R);
  for (int m = 0; m < rows; ++m)
    for (int n = 0; n < R; ++n) {
      double s = 0;
      for (int k = 0; k < K; ++k) s += double(bf(x[size_t(m) * K + k])) * bf(wd[size_t(n) * K + k]);
      const double a = s / hc;
      t[size_t(m) * R + n] = bf(blaze::canon::bf16_rn(float(a / (1.0 + std::exp(-a)))));
    }
  for (int m = 0; m < rows; ++m)
    for (int j = 0; j < hs; ++j) {
      double o = 0;
      for (int g = 0; g < hc; ++g) {
        double u = 0;
        for (int r = 0; r < R; ++r) u += double(t[size_t(m) * R + r]) * bf(wu[(size_t(g) * hs + j) * R + r]);
        o += 1.0 / (1.0 + std::exp(-u)) * bf(x[size_t(m) * K + size_t(g) * hs + j]);
      }
      out[size_t(m) * hs + j] = bf(blaze::canon::bf16_rn(float(o / hc)));
    }
}

}  // namespace

int main() {
  const int hc = 4, hs = 2560, R = 320, K = hc * hs, kMaxRows = 300;
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0, 1);
  std::vector<uint16_t> wd(size_t(R) * K), wu(size_t(K) * R);
  for (auto& v : wd) v = blaze::canon::bf16_rn(nd(rng) * 0.02f);
  for (auto& v : wu) v = blaze::canon::bf16_rn(nd(rng) * 0.05f);
  uint16_t *dwd, *dwu, *dx, *dt, *dout;
  check(cudaMalloc(&dwd, wd.size() * 2), "wd");
  check(cudaMalloc(&dwu, wu.size() * 2), "wu");
  check(cudaMalloc(&dx, size_t(kMaxRows) * K * 2), "x");
  check(cudaMalloc(&dt, size_t(kMaxRows) * R * 2), "t");
  check(cudaMalloc(&dout, size_t(kMaxRows) * hs * 2), "out");
  void* dws;
  check(cudaMalloc(&dws, std::max<size_t>(4, blz_hc_mix_workspace_bytes(kMaxRows, hc, hs, R))), "workspace");
  check(cudaMemcpy(dwd, wd.data(), wd.size() * 2, cudaMemcpyHostToDevice), "wd");
  check(cudaMemcpy(dwu, wu.data(), wu.size() * 2, cudaMemcpyHostToDevice), "wu");
  bool ok = true;
  for (int rows : {1, 2, 3, 4, 7, 8, 16, 37, 300}) {
    std::vector<uint16_t> x(size_t(rows) * K);
    for (auto& v : x) v = blaze::canon::bf16_rn(nd(rng));
    check(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice), "x");
    std::vector<uint16_t> a(size_t(rows) * hs), b(a.size());
    check(blz_hc_mix(dx, dwd, dwu, dt, dout, rows, hc, hs, R, dws, nullptr), "hc_mix");
    check(cudaMemcpy(a.data(), dout, a.size() * 2, cudaMemcpyDeviceToHost), "out");
    check(blz_hc_mix(dx, dwd, dwu, dt, dout, rows, hc, hs, R, dws, nullptr), "hc_mix");
    check(cudaMemcpy(b.data(), dout, b.size() * 2, cudaMemcpyDeviceToHost), "out");
    const bool same = std::memcmp(a.data(), b.data(), a.size() * 2) == 0;
    std::vector<float> ref;
    reference(x, wd, wu, rows, hc, hs, R, ref);
    // Error normalized by the output RMS (relative error blows up on near-zero outputs). Tensor cores (>= 3 rows)
    // accumulate FP32 sequentially, so a t value can round to the other BF16 neighbour more often than FFMA trees.
    double max_err = 0, rms = 0;
    int ulp1 = 0;
    for (size_t i = 0; i < a.size(); ++i) rms += double(ref[i]) * ref[i];
    rms = std::sqrt(rms / a.size());
    for (size_t i = 0; i < a.size(); ++i) {
      max_err = std::max(max_err, std::fabs(double(bf(a[i])) - ref[i]) / rms);
      ulp1 += bf(a[i]) != ref[i];
    }
    const bool good = same && max_err < 2e-2 && ulp1 < int(a.size() / 50 + 4);
    std::printf("rows=%3d: reproducible=%d max err %.2e of rms, %d/%zu differ from the FP64 reference\n", rows, same,
                max_err, ulp1, a.size());
    ok &= good;
  }
  // Timing at decode size, rotating over weight copies that do not fit in L2 (as the model's layers do).
  const int copies = 12;
  std::vector<uint16_t*> cwd(copies), cwu(copies);
  for (int c = 0; c < copies; ++c) {
    check(cudaMalloc(&cwd[c], wd.size() * 2), "wd copy");
    check(cudaMalloc(&cwu[c], wu.size() * 2), "wu copy");
    check(cudaMemcpy(cwd[c], dwd, wd.size() * 2, cudaMemcpyDeviceToDevice), "wd copy");
    check(cudaMemcpy(cwu[c], dwu, wu.size() * 2, cudaMemcpyDeviceToDevice), "wu copy");
  }
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0), cudaEventCreate(&e1);
  for (int rows : {1, 2, 4, 8, 16}) {
    for (int i = 0; i < 2 * copies; ++i) blz_hc_mix(dx, cwd[i % copies], cwu[i % copies], dt, dout, rows, hc, hs, R, dws, nullptr);
    cudaEventRecord(e0);
    const int iters = 50 * copies;
    for (int i = 0; i < iters; ++i)
      blz_hc_mix(dx, cwd[i % copies], cwu[i % copies], dt, dout, rows, hc, hs, R, dws, nullptr);
    cudaEventRecord(e1);
    check(cudaEventSynchronize(e1), "timing");
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    const double us = 1000.0 * ms / iters;
    std::printf("bench rows=%d: %.1f us per mix from DRAM (%.0f GB/s of weights)\n", rows, us,
                (wd.size() + wu.size()) * 2 / us / 1e3);
  }
  for (int c = 0; c < copies; ++c) cudaFree(cwd[c]), cudaFree(cwu[c]);
  std::printf(ok ? "PASSED\n" : "FAILED\n");
  return ok ? 0 : 1;
}
