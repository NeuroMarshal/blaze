// blz_fp8_gemv_mma / blz_fp8_weight_dequant: agreement with an FP64 reference (per-channel and 128-block scales), and
// the cold-weight bandwidth of the GEMV on Qwen3.8-Flash-Next shapes (GDN in_proj 16384 x 2560, out_proj 2560 x 6144).
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "blaze/fp8_linear.h"
#include "blaze/ops.h"

namespace {

void check(cudaError_t e, const char* w) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
    std::exit(1);
  }
}

float e4m3f(uint8_t b) {
  __nv_fp8_e4m3 f;
  f.__x = b;
  return float(f);
}

struct Weights {
  int n, k, block_n;
  std::vector<uint8_t> codes;
  std::vector<float> alpha, block;
  uint8_t* d_codes = nullptr;
  float *d_alpha = nullptr, *d_block = nullptr;

  Weights(int n_, int k_, bool blocks, std::mt19937& rng) : n(n_), k(k_), block_n(blocks ? 128 : 0), codes(size_t(n_) * k_), alpha(n_) {
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& c : codes) {
      do c = uint8_t(byte(rng));
      while ((c & 0x7f) == 0x7f);  // no NaN
    }
    std::uniform_real_distribution<float> a(0.001f, 0.01f);
    for (auto& v : alpha) v = blocks ? 1.0f : a(rng);
    check(cudaMalloc(&d_codes, codes.size()), "malloc");
    check(cudaMalloc(&d_alpha, alpha.size() * 4), "malloc");
    check(cudaMemcpy(d_codes, codes.data(), codes.size(), cudaMemcpyHostToDevice), "copy");
    check(cudaMemcpy(d_alpha, alpha.data(), alpha.size() * 4, cudaMemcpyHostToDevice), "copy");
    if (blocks) {
      block.resize(size_t((n + 127) / 128) * (k / 128));
      for (auto& v : block) v = a(rng);
      check(cudaMalloc(&d_block, block.size() * 4), "malloc");
      check(cudaMemcpy(d_block, block.data(), block.size() * 4, cudaMemcpyHostToDevice), "copy");
    }
  }
  ~Weights() {
    cudaFree(d_codes);
    cudaFree(d_alpha);
    cudaFree(d_block);
  }
  double w(int row, int col) const {
    double v = e4m3f(codes[size_t(row) * k + col]) * double(alpha[row]);
    if (block_n) v *= block[size_t(row / block_n) * (k / 128) + col / 128];
    return v;
  }
};

uint16_t bf(float v) { return __bfloat16_as_ushort(__float2bfloat16_rn(v)); }
float fb(uint16_t v) { return __bfloat162float(__ushort_as_bfloat16(v)); }

double run(const Weights& W, int rows, int64_t ldx, std::mt19937& rng, blz_gemm* gemm) {
  std::normal_distribution<float> d(0.0f, 1.0f);
  std::vector<uint16_t> x(size_t(rows) * ldx);
  for (auto& v : x) v = bf(d(rng));
  uint16_t *dx, *dy, *dw = nullptr;
  check(cudaMalloc(&dx, x.size() * 2), "malloc");
  check(cudaMalloc(&dy, size_t(rows) * W.n * 2), "malloc");
  check(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice), "copy");
  if (rows <= BLZ_FP8_MMA_MAX_ROWS) {
    check(blz_fp8_gemv_mma(dx, ldx, W.d_codes, W.d_block, W.block_n, W.d_alpha, dy, W.n, rows, W.n, W.k, 0), "gemv");
  } else {
    check(cudaMalloc(&dw, size_t(W.n) * W.k * 2), "malloc");
    check(blz_fp8_weight_dequant(W.d_codes, W.d_block, W.block_n, W.d_alpha, W.n, W.k, dw, 0), "dequant");
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

void bench(int rows, int n, int k, std::mt19937& rng) {
  const double bytes = double(n) * k;
  const int copies = std::max(2, int(400e6 / bytes));
  std::vector<Weights*> ws;
  for (int i = 0; i < copies; ++i) ws.push_back(new Weights(n, k, false, rng));
  uint16_t *dx, *dy;
  check(cudaMalloc(&dx, size_t(rows) * k * 2), "malloc");
  check(cudaMalloc(&dy, size_t(rows) * n * 2), "malloc");
  check(cudaMemset(dx, 0, size_t(rows) * k * 2), "memset");
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  for (int i = 0; i < copies; ++i)
    blz_fp8_gemv_mma(dx, k, ws[i]->d_codes, nullptr, 0, ws[i]->d_alpha, dy, n, rows, n, k, 0);
  const int iters = 20 * copies;
  cudaEventRecord(a);
  for (int i = 0; i < iters; ++i) {
    const Weights* W = ws[i % copies];
    blz_fp8_gemv_mma(dx, k, W->d_codes, nullptr, 0, W->d_alpha, dy, n, rows, n, k, 0);
  }
  cudaEventRecord(b);
  check(cudaEventSynchronize(b), "bench");
  float ms = 0;
  cudaEventElapsedTime(&ms, a, b);
  const double us = 1000.0 * ms / iters;
  std::printf("  fp8 mma %2d x %5d x %5d: %7.1f us, %6.0f GB/s of weights\n", rows, n, k, us, bytes / us / 1000.0);
  for (auto* w : ws) delete w;
  cudaFree(dx);
  cudaFree(dy);
}

}  // namespace

int main() {
  std::mt19937 rng(11);
  blz_gemm* gemm = nullptr;
  if (blz_gemm_create(&gemm) != 0) return 1;
  bool ok = true;
  for (bool blocks : {false, true}) {
    Weights W(264, 1152, blocks, rng);
    for (int rows : {1, 2, 7, 8, 9, 16, 33}) {
      const double e = run(W, rows, 1152 + (rows == 9 ? 128 : 0), rng, gemm);
      const bool pass = e < 6e-3;
      ok &= pass;
      std::printf("%s rows %2d (%s): max rel err %.2e %s\n", blocks ? "block-128" : "per-channel", rows,
                  rows <= BLZ_FP8_MMA_MAX_ROWS ? "mma" : "dequant+gemm", e, pass ? "ok" : "FAIL");
    }
  }
  std::printf("bandwidth (cold):\n");
  for (int rows : {1, 8, 16}) {
    bench(rows, 16384, 2560, rng);
    bench(rows, 2560, 6144, rng);
    bench(rows, 13312, 2560, rng);
  }
  blz_gemm_destroy(gemm);
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
