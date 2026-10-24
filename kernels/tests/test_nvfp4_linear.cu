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
// f16: the FP16-accumulate route (blz_nvfp4_f16_rows + blz_nvfp4_gemv_mma_f16_cfg); outliers: a few input channels
// a thousand times larger (Qwen's massive activations reach the down projections).
double run(const Weights& W, int rows, int64_t ldx, std::mt19937& rng, blz_gemm* gemm, bool mma = false, int split = 0,
           int tiles = 0, bool f16 = false, bool outliers = false) {
  std::normal_distribution<float> d(0.0f, 1.0f);
  std::vector<uint16_t> x(size_t(rows) * ldx);
  for (auto& v : x) v = bf(d(rng));
  if (outliers)
    for (int m = 0; m < rows; ++m)
      for (int c : {3, 77, W.k / 2, W.k - 5}) x[size_t(m) * ldx + c] = bf(d(rng) * 1500.0f);
  uint16_t *dx, *dy, *dw = nullptr;
  check(cudaMalloc(&dx, x.size() * 2), "malloc");
  check(cudaMalloc(&dy, size_t(rows) * W.n * 2), "malloc");
  check(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice), "copy");
  if (f16) {
    uint16_t* x16;
    int32_t* sh;
    check(cudaMalloc(&x16, size_t(rows) * W.k * 2), "malloc");
    check(cudaMalloc(&sh, size_t(rows) * 4), "malloc");
    check(blz_nvfp4_f16_rows(dx, ldx, rows, W.k, x16, sh, 0), "f16 rows");
    check(blz_nvfp4_gemv_mma_f16_cfg(x16, sh, W.d_codes, W.d_scales, W.d_alpha, dy, W.n, rows, W.n, W.k, split, tiles, 0),
          "gemv_mma_f16");
    check(cudaDeviceSynchronize(), "f16 sync");
    cudaFree(x16);
    cudaFree(sh);
  } else if (mma) {
    check(blz_nvfp4_gemv_mma_cfg(dx, ldx, W.d_codes, W.d_scales, W.d_alpha, dy, W.n, rows, W.n, W.k, split, tiles, 0),
          "gemv_mma");
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
double bench(int rows, int n, int k, std::mt19937& rng, bool mma = false, bool f16 = false) {
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
  uint16_t* x16;
  int32_t* sh;
  check(cudaMalloc(&x16, size_t(rows) * k * 2), "malloc");
  check(cudaMalloc(&sh, size_t(rows) * 4), "malloc");
  auto f = [&](const Weights* W) {
    if (f16) {
      blz_nvfp4_f16_rows(dx, k, rows, k, x16, sh, 0);
      blz_nvfp4_gemv_mma_f16_cfg(x16, sh, W->d_codes, W->d_scales, W->d_alpha, dy, n, rows, n, k, 0, 0, 0);
    } else if (mma) {
      blz_nvfp4_gemv_mma(dx, k, W->d_codes, W->d_scales, W->d_alpha, dy, n, rows, n, k, 0);
    } else {
      blz_nvfp4_gemv(dx, k, W->d_codes, W->d_scales, W->d_alpha, dy, n, rows, n, k, 0);
    }
  };
  for (int i = 0; i < copies; ++i) f(ws[i]);
  const int iters = 20 * copies;
  cudaEventRecord(a);
  for (int i = 0; i < iters; ++i) f(ws[i % copies]);
  cudaEventRecord(b);
  check(cudaEventSynchronize(b), "bench");
  float ms = 0;
  cudaEventElapsedTime(&ms, a, b);
  const double us = 1000.0 * ms / iters;
  std::printf("  %s %2d x %5d x %5d: %7.1f us, %6.0f GB/s of weights\n", f16 ? "f16 " : mma ? "mma " : "gemv", rows, n,
              k, us, bytes / us / 1000.0);
  cudaFree(x16);
  cudaFree(sh);
  cudaFree(dx);
  cudaFree(dy);
  return us;
}

}  // namespace

int main(int argc, char** argv) {
  std::mt19937 rng(7);
  if (argc > 1 && std::string(argv[1]) == "bench") {
    // quick: the tensor-core route over 8..64 rows on the 27B shapes
    for (int rows : {8, 10, 12, 16, 24, 32, 48, 64})
      for (bool f16 : {false, true}) {
        bench(rows, 17408, 5120, rng, true, f16);
        bench(rows, 5120, 8704, rng, true, f16);
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
  // every launch the tuner may pick (tune.rs): all valid (warps along K, weight tiles per block)
  {
    int checked = 0, bad = 0;
    for (int rows : {1, 7, 8, 9, 16, 17, 32, 33, 48, 64})
      for (int split : {2, 4, 8})
        for (int tiles : {1, 2, 4}) {
          if (!blz_nvfp4_mma_cfg_valid(rows, split, tiles)) continue;
          const double e = run(mid, rows, 1152, rng, gemm, true, split, tiles);
          ++checked;
          if (!(e < 6e-3)) {
            ++bad;
            std::printf("rows %2d split %d tiles %d: max rel err %.2e FAIL\n", rows, split, tiles, e);
          }
        }
    ok &= bad == 0;
    std::printf("mma launches: %d checked, %d failed\n", checked, bad);
  }
  // FP16 accumulation: every launch, beside the BF16 kernel on the same draws (normal inputs and outliers)
  {
    int checked = 0, bad = 0;
    double worst16 = 0, worst_bf = 0;
    for (bool outl : {false, true})
      for (int rows : {1, 8, 9, 16, 17, 32, 33, 48, 64})
        for (int split : {2, 4, 8})
          for (int tiles : {1, 2, 4}) {
            if (!blz_nvfp4_mma_cfg_valid(rows, split, tiles)) continue;
            std::mt19937 r1(rows * 131 + split * 7 + tiles), r2 = r1;
            const double e16 = run(mid, rows, 1152, r1, gemm, true, split, tiles, true, outl);
            const double ebf = run(mid, rows, 1152, r2, gemm, true, split, tiles, false, outl);
            worst16 = std::fmax(worst16, e16), worst_bf = std::fmax(worst_bf, ebf);
            ++checked;
            if (!(e16 < 6e-3)) {
              ++bad;
              std::printf("f16 rows %2d split %d tiles %d outliers %d: max rel err %.2e (bf16 %.2e) FAIL\n", rows, split,
                          tiles, int(outl), e16, ebf);
            }
          }
    ok &= bad == 0;
    std::printf("f16 mma launches: %d checked, %d failed; worst rel err f16 %.2e, bf16 %.2e\n", checked, bad, worst16,
                worst_bf);
  }
  Weights big(64, 8704, rng);
  const double e = run(big, 1, 8704, rng, gemm);
  ok &= e < 6e-3;
  std::printf("rows  1, k 8704: max rel err %.2e %s\n", e, e < 6e-3 ? "ok" : "FAIL");

  if (argc > 1 && std::string(argv[1]) == "check") {  // correctness only
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
  }
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
