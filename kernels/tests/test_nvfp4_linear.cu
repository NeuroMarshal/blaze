// blz_nvfp4_gemv / blz_nvfp4_weight_dequant: agreement with an FP64 reference of the checkpoint formula, and the effective
// weight bandwidth of the GEMV on the Qwen3.8-27B shapes per TP rank (gate|up 17408 x 5120, down 5120 x 8704).
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <map>
#include <string>
#include <vector>

#include "blaze/nvfp4.h"
#include "blaze/nvfp4_linear.h"
#include "blaze/ops.h"

namespace {

void check(cudaError_t e, const char* w) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
    std::exit(1);
  }
}

struct Weights {
  int n, k;
  std::vector<uint8_t> codes, scales;
  std::vector<float> alpha;
  uint8_t *d_codes = nullptr, *d_scales = nullptr;
  float* d_alpha = nullptr;

  Weights(int n_, int k_, std::mt19937& rng) : n(n_), k(k_), codes(size_t(n_) * k_ / 2), scales(size_t(n_) * k_ / 16), alpha(n_) {
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& c : codes) c = uint8_t(byte(rng));
    // E4M3 scales in a checkpoint-like range (about 2^-9 .. 2^-3), never NaN.
    std::uniform_real_distribution<float> s(0.002f, 0.12f);
    for (auto& v : scales) v = blaze::float_to_e4m3_host(s(rng));
    std::uniform_real_distribution<float> a(0.5f, 2.0f);
    for (auto& v : alpha) v = a(rng);
    check(cudaMalloc(&d_codes, codes.size()), "malloc");
    check(cudaMalloc(&d_scales, scales.size()), "malloc");
    check(cudaMalloc(&d_alpha, alpha.size() * 4), "malloc");
    check(cudaMemcpy(d_codes, codes.data(), codes.size(), cudaMemcpyHostToDevice), "copy");
    check(cudaMemcpy(d_scales, scales.data(), scales.size(), cudaMemcpyHostToDevice), "copy");
    check(cudaMemcpy(d_alpha, alpha.data(), alpha.size() * 4, cudaMemcpyHostToDevice), "copy");
  }
  ~Weights() {
    cudaFree(d_codes);
    cudaFree(d_scales);
    cudaFree(d_alpha);
  }
  double w(int row, int col) const {
    const uint8_t b = codes[size_t(row) * k / 2 + col / 2];
    const int code = (col & 1) ? b >> 4 : b & 15;
    return double(blaze::e2m1_value(code)) * blaze::e4m3_to_float(scales[size_t(row) * k / 16 + col / 16]) * alpha[row];
  }
};

uint16_t bf(float v) { return __bfloat16_as_ushort(__float2bfloat16_rn(v)); }
float fb(uint16_t v) { return __bfloat162float(__ushort_as_bfloat16(v)); }

// Worst |y - ref| / max |ref| of `rows` rows through the GEMV (rows <= 8) or dequant + cuBLAS.
double run(const Weights& W, int rows, int64_t ldx, std::mt19937& rng, blz_gemm* gemm, bool mma = false) {
  std::normal_distribution<float> d(0.0f, 1.0f);
  std::vector<uint16_t> x(size_t(rows) * ldx);
  for (auto& v : x) v = bf(d(rng));
  uint16_t *dx, *dy, *dw = nullptr;
  check(cudaMalloc(&dx, x.size() * 2), "malloc");
  check(cudaMalloc(&dy, size_t(rows) * W.n * 2), "malloc");
  check(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice), "copy");
  if (mma) {
    check(blz_nvfp4_gemv_mma(dx, ldx, W.d_codes, W.d_scales, W.d_alpha, dy, W.n, rows, W.n, W.k, 0), "gemv_mma");
  } else if (rows <= BLZ_NVFP4_GEMV_MAX_ROWS) {
    check(blz_nvfp4_gemv(dx, ldx, W.d_codes, W.d_scales, W.d_alpha, dy, W.n, rows, W.n, W.k, 0), "gemv");
  } else {
    check(cudaMalloc(&dw, size_t(W.n) * W.k * 2), "malloc");
    check(blz_nvfp4_weight_dequant(W.d_codes, W.d_scales, W.d_alpha, W.n, W.k, dw, 0), "dequant");
    check(blz_gemm_bf16(gemm, dx, ldx, dw, dy, W.n, rows, W.n, W.k, 0), "gemm");
  }
  std::vector<uint16_t> y(size_t(rows) * W.n);
  check(cudaMemcpy(y.data(), dy, y.size() * 2, cudaMemcpyDeviceToHost), "copy back");
  double worst = 0.0, top = 0.0;
  for (int m = 0; m < rows; ++m)
    for (int j = 0; j < W.n; ++j) {
      double ref = 0.0;
      for (int i = 0; i < W.k; ++i) ref += double(fb(x[size_t(m) * ldx + i])) * W.w(j, i);
      top = std::fmax(top, std::fabs(ref));
      worst = std::fmax(worst, std::fabs(double(fb(y[size_t(m) * W.n + j])) - ref));
    }
  cudaFree(dx);
  cudaFree(dy);
  cudaFree(dw);
  return worst / top;
}

// Cold weights, as in a forward pass: enough copies of the matrix to exceed the L2 cache, used in turn.
double bench(int rows, int n, int k, std::mt19937& rng, bool mma = false) {
  const double bytes = double(n) * k / 2 + double(n) * k / 16;
  const int copies = std::max(2, int(400e6 / bytes));
  // the copies of a shape are made once (random bytes on the host are slow) and kept for later calls
  static std::map<std::pair<int, int>, std::vector<Weights*>> cache;
  auto& ws = cache[{n, k}];
  while (int(ws.size()) < copies) ws.push_back(new Weights(n, k, rng));
  uint16_t *dx, *dy;
  check(cudaMalloc(&dx, size_t(rows) * k * 2), "malloc");
  check(cudaMalloc(&dy, size_t(rows) * n * 2), "malloc");
  check(cudaMemset(dx, 0, size_t(rows) * k * 2), "memset");
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  auto f = mma ? blz_nvfp4_gemv_mma : blz_nvfp4_gemv;
  for (int i = 0; i < copies; ++i) f(dx, k, ws[i]->d_codes, ws[i]->d_scales, ws[i]->d_alpha, dy, n, rows, n, k, 0);
  const int iters = 20 * copies;
  cudaEventRecord(a);
  for (int i = 0; i < iters; ++i) {
    const Weights* W = ws[i % copies];
    f(dx, k, W->d_codes, W->d_scales, W->d_alpha, dy, n, rows, n, k, 0);
  }
  cudaEventRecord(b);
  check(cudaEventSynchronize(b), "bench");
  float ms = 0;
  cudaEventElapsedTime(&ms, a, b);
  const double us = 1000.0 * ms / iters;
  std::printf("  %s %2d x %5d x %5d: %7.1f us, %6.0f GB/s of weights\n", mma ? "mma " : "gemv", rows, n, k, us,
              bytes / us / 1000.0);
  cudaFree(dx);
  cudaFree(dy);
  return us;
}

}  // namespace

int main(int argc, char** argv) {
  std::mt19937 rng(7);
  if (argc > 1 && std::string(argv[1]) == "bench") {
    // quick: the tensor-core route over 8..64 rows on the 27B shapes
    for (int rows : {8, 10, 12, 16, 24, 32, 48, 64}) {
      bench(rows, 17408, 5120, rng, true);
      bench(rows, 5120, 8704, rng, true);
    }
    return 0;
  }
  blz_gemm* gemm = nullptr;
  if (blz_gemm_create(&gemm) != 0) return 1;
  bool ok = true;
  Weights small(200, 1056, rng);  // n not a multiple of the block's rows, k / 32 not a multiple of 32
  for (int rows : {1, 2, 3, 5, 8, 33}) {
    const double e = run(small, rows, 1056 + (rows == 3 ? 32 : 0), rng, gemm);
    const bool pass = e < 6e-3;
    ok &= pass;
    std::printf("rows %2d (%s): max rel err %.2e %s\n", rows, rows <= BLZ_NVFP4_GEMV_MAX_ROWS ? "gemv" : "dequant+gemm",
                e, pass ? "ok" : "FAIL");
  }
  Weights mid(200, 1152, rng);  // k a multiple of 128
  for (int rows : {1, 2, 7, 8, 9, 16, 17, 24, 32, 40, 63, 64}) {
    const double e = run(mid, rows, 1152 + (rows == 9 || rows == 40 ? 128 : 0), rng, gemm, true);
    const bool pass = e < 6e-3;
    ok &= pass;
    std::printf("rows %2d (mma): max rel err %.2e %s\n", rows, e, pass ? "ok" : "FAIL");
  }
  Weights big(64, 8704, rng);
  const double e = run(big, 1, 8704, rng, gemm);
  ok &= e < 6e-3;
  std::printf("rows  1, k 8704: max rel err %.2e %s\n", e, e < 6e-3 ? "ok" : "FAIL");

  std::printf("bandwidth:\n");
  for (int rows : {1, 2, 4, 8, 16, 24, 32, 48, 64}) {
    for (bool mma : {false, true}) {
      if (!mma && rows > BLZ_NVFP4_GEMV_MAX_ROWS) continue;
      bench(rows, 17408, 5120, rng, mma);
      bench(rows, 5120, 8704, rng, mma);
      bench(rows, 5120, 3072, rng, mma);
    }
  }
  bench(1, 7168, 5120, rng);
  blz_gemm_destroy(gemm);
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
