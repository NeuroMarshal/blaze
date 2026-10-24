// blz_skinny_gemm: agreement with an FP64 reference (one rounding of the FP32 sum to 16 bits), and timing of the
// shapes it serves (Qwen3.8-27B GDN in_proj_ba per TP rank: 48 x 5120).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "blaze/skinny_gemm.h"

namespace {

void check(cudaError_t e, const char* w) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
    std::exit(1);
  }
}

template <typename T>
float to_f(T v);
template <>
float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
template <>
float to_f(__half v) { return __half2float(v); }
template <typename T>
T from_f(float v);
template <>
__nv_bfloat16 from_f(float v) { return __float2bfloat16_rn(v); }
template <>
__half from_f(float v) { return __float2half_rn(v); }

// Worst |y - ref| relative to the largest |ref| of the case; y_stride > 0: blz_skinny_gemm_ex into a wider output
// (the columns outside [0, n) must stay untouched).
template <typename T>
double run(int rows, int n, int k, int64_t x_stride, blz_half_t dt, std::mt19937& rng, int64_t y_stride = 0) {
  std::normal_distribution<float> d(0.0f, 1.0f);
  const int64_t ys = y_stride > 0 ? y_stride : n;
  std::vector<T> x(size_t(rows) * x_stride), w(size_t(n) * k), y(size_t(rows) * ys, from_f<T>(7.0f));
  for (auto& v : x) v = from_f<T>(d(rng));
  for (auto& v : w) v = from_f<T>(d(rng) * 0.05f);
  T *dx, *dw, *dy;
  check(cudaMalloc(&dx, x.size() * sizeof(T)), "malloc");
  check(cudaMalloc(&dw, w.size() * sizeof(T)), "malloc");
  check(cudaMalloc(&dy, y.size() * sizeof(T)), "malloc");
  check(cudaMemcpy(dx, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
  check(cudaMemcpy(dw, w.data(), w.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
  check(cudaMemcpy(dy, y.data(), y.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
  if (y_stride > 0)
    check(blz_skinny_gemm_ex(dx, x_stride, dw, dy, ys, rows, n, k, dt, nullptr), "skinny_ex");
  else
    check(blz_skinny_gemm(dx, x_stride, dw, dy, rows, n, k, dt, nullptr), "skinny");
  check(cudaMemcpy(y.data(), dy, y.size() * sizeof(T), cudaMemcpyDeviceToHost), "download");
  double worst = 0, peak = 0;
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < n; ++c) {
      double ref = 0;
      for (int i = 0; i < k; ++i) ref += double(to_f(x[size_t(r) * x_stride + i])) * to_f(w[size_t(c) * k + i]);
      worst = std::max(worst, std::fabs(double(to_f(y[size_t(r) * ys + c])) - ref));
      peak = std::max(peak, std::fabs(ref));
    }
    for (int64_t c = n; c < ys; ++c)
      if (to_f(y[size_t(r) * ys + c]) != 7.0f) worst = 1e30;  // a write outside the part
  }
  cudaFree(dx), cudaFree(dw), cudaFree(dy);
  return worst / std::max(peak, 1e-30);
}

}  // namespace

int main() {
  std::mt19937 rng(7);
  int fails = 0;
  struct Case { int rows, n, k; int64_t stride; };
  for (const Case c : {Case{1, 48, 5120, 5120}, Case{4, 48, 5120, 5120}, Case{16, 48, 5120, 5248},
                       Case{8, 96, 2560, 2560}, Case{3, 1000, 1024, 1024}, Case{1, 7, 64, 64}}) {
    const double eb = run<__nv_bfloat16>(c.rows, c.n, c.k, c.stride, BLZ_BF16, rng);
    const double eh = run<__half>(c.rows, c.n, c.k, c.stride, BLZ_FP16, rng);
    const bool ok = eb < 8e-3 && eh < 2e-3;  // one 16-bit rounding of the result
    std::printf("rows %2d n %4d k %5d: max error / max |y|: bf16 %.2e, fp16 %.2e %s\n", c.rows, c.n, c.k, eb, eh,
                ok ? "ok" : "FAIL");
    fails += !ok;
  }
  // blz_skinny_gemm_ex: parts of wider outputs, row passes beyond 16, 2 and 4 columns per block
  struct CaseEx { int rows, n, k; int64_t stride, y_stride; };
  for (const CaseEx c : {CaseEx{16, 48, 5120, 5120, 96}, CaseEx{24, 48, 5120, 5248, 50}, CaseEx{64, 48, 5120, 5120, 48},
                         CaseEx{33, 128, 5120, 5120, 300}, CaseEx{64, 600, 1024, 1024, 601}, CaseEx{7, 1000, 2048, 2048, 1003},
                         CaseEx{2, 3, 64, 64, 9}}) {
    const double eb = run<__nv_bfloat16>(c.rows, c.n, c.k, c.stride, BLZ_BF16, rng, c.y_stride);
    const double eh = run<__half>(c.rows, c.n, c.k, c.stride, BLZ_FP16, rng, c.y_stride);
    const bool ok = eb < 8e-3 && eh < 2e-3;
    std::printf("ex rows %2d n %4d k %5d y_stride %4lld: bf16 %.2e, fp16 %.2e %s\n", c.rows, c.n, c.k,
                (long long)c.y_stride, eb, eh, ok ? "ok" : "FAIL");
    fails += !ok;
  }
  // Timing through _ex: the GDN in_proj_ba of a 27B rank (48 x 5120) and the EAGLE-4 low-rank down (128 x 5120)
  for (int n : {48, 128})
    for (int rows : {2, 8, 16, 32, 64}) {
      const int k = 5120, iters = 2000;
      __nv_bfloat16 *dx, *dw, *dy;
      check(cudaMalloc(&dx, size_t(rows) * k * 2), "malloc");
      check(cudaMalloc(&dw, size_t(n) * k * 2), "malloc");
      check(cudaMalloc(&dy, size_t(rows) * n * 2), "malloc");
      cudaMemset(dx, 0, size_t(rows) * k * 2), cudaMemset(dw, 0, size_t(n) * k * 2);
      cudaEvent_t a, b;
      cudaEventCreate(&a), cudaEventCreate(&b);
      for (int i = 0; i < 50; ++i) blz_skinny_gemm_ex(dx, k, dw, dy, n, rows, n, k, BLZ_BF16, nullptr);
      cudaEventRecord(a);
      for (int i = 0; i < iters; ++i) blz_skinny_gemm_ex(dx, k, dw, dy, n, rows, n, k, BLZ_BF16, nullptr);
      cudaEventRecord(b);
      check(cudaEventSynchronize(b), "time");
      float ms = 0;
      cudaEventElapsedTime(&ms, a, b);
      std::printf("ex %3d x 5120 bf16, %2d rows: %.2f us per call\n", n, rows, 1000 * ms / iters);
      cudaFree(dx), cudaFree(dw), cudaFree(dy);
    }
  // Timing: 48 x 5120 bf16, 1 and 4 rows, back to back.
  for (int rows : {1, 4, 32 / 2}) {
    const int n = 48, k = 5120, iters = 2000;
    __nv_bfloat16 *dx, *dw, *dy;
    check(cudaMalloc(&dx, size_t(rows) * k * 2), "malloc");
    check(cudaMalloc(&dw, size_t(n) * k * 2), "malloc");
    check(cudaMalloc(&dy, size_t(rows) * n * 2), "malloc");
    cudaMemset(dx, 0, size_t(rows) * k * 2), cudaMemset(dw, 0, size_t(n) * k * 2);
    cudaEvent_t a, b;
    cudaEventCreate(&a), cudaEventCreate(&b);
    for (int i = 0; i < 50; ++i) blz_skinny_gemm(dx, k, dw, dy, rows, n, k, BLZ_BF16, nullptr);
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) blz_skinny_gemm(dx, k, dw, dy, rows, n, k, BLZ_BF16, nullptr);
    cudaEventRecord(b);
    check(cudaEventSynchronize(b), "time");
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    std::printf("48 x 5120 bf16, %2d rows: %.2f us per call\n", rows, 1000 * ms / iters);
    cudaFree(dx), cudaFree(dw), cudaFree(dy);
  }
  // gemv_wide: LM-head shapes against a double reference on a sample of columns, and the bandwidth
  for (int rows : {1, 2, 4}) {
    const int n = 248320, k = 2048;
    std::normal_distribution<float> d(0.0f, 1.0f);
    std::vector<__nv_bfloat16> x(size_t(rows) * k), w(size_t(n) * k), y(size_t(rows) * n);
    for (auto& v : x) v = from_f<__nv_bfloat16>(d(rng));
    for (auto& v : w) v = from_f<__nv_bfloat16>(d(rng) * 0.05f);
    __nv_bfloat16 *dx, *dw, *dy;
    check(cudaMalloc(&dx, x.size() * 2), "malloc");
    check(cudaMalloc(&dw, w.size() * 2), "malloc");
    check(cudaMalloc(&dy, y.size() * 2), "malloc");
    check(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice), "upload");
    check(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice), "upload");
    check(blz_gemv_wide(dx, k, dw, dy, n, rows, n, k, nullptr), "gemv_wide");
    check(cudaMemcpy(y.data(), dy, y.size() * 2, cudaMemcpyDeviceToHost), "download");
    double worst = 0, peak = 0;
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < n; c += 97) {
        double ref = 0;
        for (int i = 0; i < k; ++i) ref += double(to_f(x[size_t(r) * k + i])) * to_f(w[size_t(c) * k + i]);
        worst = std::max(worst, std::fabs(double(to_f(y[size_t(r) * n + c])) - ref));
        peak = std::max(peak, std::fabs(ref));
      }
    const double e = worst / std::max(peak, 1e-30);
    cudaEvent_t a, b;
    cudaEventCreate(&a), cudaEventCreate(&b);
    for (int i = 0; i < 3; ++i) blz_gemv_wide(dx, k, dw, dy, n, rows, n, k, nullptr);
    cudaEventRecord(a);
    for (int i = 0; i < 20; ++i) blz_gemv_wide(dx, k, dw, dy, n, rows, n, k, nullptr);
    cudaEventRecord(b);
    check(cudaEventSynchronize(b), "time");
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    const double us = 1000.0 * ms / 20;
    std::printf("gemv_wide %d rows, %d x %d: error %.2e %s, %.0f us (%.0f GB/s)\n", rows, n, k, e, e < 8e-3 ? "ok" : "FAIL",
                us, double(n) * k * 2 / us / 1e3);
    fails += e >= 8e-3;
    cudaFree(dx), cudaFree(dw), cudaFree(dy);
  }
  std::printf("%s\n", fails ? "FAILED" : "all passed");
  return fails ? 1 : 0;
}
