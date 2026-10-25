// blz_nvfp4_quant_rows + blz_nvfp4_gemm_w4a4: the GEMM against an FP64 product of the same quantized operands
// (exactness of the kernel), the error of W4A4 against W4A16 (the activation quantization), odd m / n edges, and the
// time against the BF16 route (weights widened + cuBLAS) at prefill shapes of the 27B.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <tuple>
#include <random>
#include <vector>

#include "blaze/nvfp4.h"
#include "blaze/nvfp4_gemm.h"
#include "blaze/nvfp4_linear.h"
#include "blaze/ops.h"

namespace {

void check(cudaError_t e, const char* w) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
    std::exit(1);
  }
}

uint16_t bf(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
float fb(uint16_t b) {
  uint32_t u = uint32_t(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

template <typename T>
T* dev(const std::vector<T>& v) {
  T* d;
  check(cudaMalloc(&d, v.size() * sizeof(T) + 16), "malloc");
  check(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "copy");
  return d;
}

struct Weight {
  int n, k;
  std::vector<uint8_t> codes, scales;
  std::vector<float> alpha;
};

Weight random_weight(int n, int k, std::mt19937& rng) {
  Weight w{n, k, std::vector<uint8_t>(size_t(n) * k / 2), std::vector<uint8_t>(size_t(n) * k / 16), std::vector<float>(n)};
  for (auto& c : w.codes) c = uint8_t(rng());
  std::uniform_int_distribution<int> sc(0x20, 0x7e);  // positive E4M3 scales up to 448 (ModelOpt range)
  for (auto& s : w.scales) s = uint8_t(sc(rng));
  for (auto& a : w.alpha) a = std::uniform_real_distribution<float>(1e-4f, 4e-4f)(rng);
  return w;
}

float wval(const Weight& w, int j, int i) {
  const uint8_t b = w.codes[size_t(j) * (w.k / 2) + i / 2];
  const int code = i % 2 ? b >> 4 : b & 15;
  return blaze::e2m1_value(code) * blaze::e4m3_to_float(w.scales[size_t(j) * (w.k / 16) + i / 16]) * w.alpha[j];
}

}  // namespace

int main() {
  if (!blz_nvfp4_gemm_w4a4_supported()) {
    std::printf("W4A4 GEMM: not available on this GPU / build (needs SM120a): skipped\nPASSED\n");
    return 0;
  }
  std::mt19937 rng(11);
  int bad = 0;
  for (auto [m, n, k] : {std::tuple{300, 200, 1024}, std::tuple{1, 72, 256}, std::tuple{129, 131, 512}}) {
    Weight W = random_weight(n, k, rng);
    std::vector<uint16_t> x(size_t(m) * k);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int r = 0; r < m; ++r) {
      const float mag = std::exp(std::uniform_real_distribution<float>(-3.0f, 3.0f)(rng));  // rows of different size
      for (int i = 0; i < k; ++i) x[size_t(r) * k + i] = bf(nd(rng) * mag);
    }
    uint16_t* dx = dev(x);
    uint8_t *xc, *xs;
    float* xr;
    check(cudaMalloc(&xc, size_t(m) * k / 2), "malloc");
    check(cudaMalloc(&xs, size_t(m) * k / 16), "malloc");
    check(cudaMalloc(&xr, m * 4), "malloc");
    uint8_t *wc = dev(W.codes), *ws = dev(W.scales);
    float* wa = dev(W.alpha);
    uint16_t* dy;
    check(cudaMalloc(&dy, size_t(m) * n * 2), "malloc");
    check(blz_nvfp4_quant_rows(dx, k, m, k, xc, xs, xr, 0), "quant");
    check(blz_nvfp4_gemm_w4a4(xc, xs, xr, wc, ws, wa, dy, n, m, n, k, 0), "gemm");
    check(cudaDeviceSynchronize(), "sync");
    std::vector<uint8_t> hc(size_t(m) * k / 2), hs(size_t(m) * k / 16);
    std::vector<float> hr(m);
    std::vector<uint16_t> y(size_t(m) * n);
    check(cudaMemcpy(hc.data(), xc, hc.size(), cudaMemcpyDeviceToHost), "copy");
    check(cudaMemcpy(hs.data(), xs, hs.size(), cudaMemcpyDeviceToHost), "copy");
    check(cudaMemcpy(hr.data(), xr, m * 4, cudaMemcpyDeviceToHost), "copy");
    check(cudaMemcpy(y.data(), dy, y.size() * 2, cudaMemcpyDeviceToHost), "copy");
    double err_kernel = 0, ref_norm = 0, err_quant = 0, exact_norm = 0;
    std::vector<double> wrow(k);
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < k; ++i) wrow[i] = wval(W, j, i);
      for (int r = 0; r < m; ++r) {
        double q = 0, e = 0;
        for (int i = 0; i < k; ++i) {
          const uint8_t b = hc[size_t(r) * (k / 2) + i / 2];
          const double xq = blaze::e2m1_value(i % 2 ? b >> 4 : b & 15) * blaze::e4m3_to_float(hs[size_t(r) * (k / 16) + i / 16]) * hr[r];
          q += xq * wrow[i];
          e += double(fb(x[size_t(r) * k + i])) * wrow[i];
        }
        const double got = fb(y[size_t(r) * n + j]);
        err_kernel += (got - q) * (got - q);
        ref_norm += q * q;
        err_quant += (q - e) * (q - e);
        exact_norm += e * e;
      }
    }
    const double rel = std::sqrt(err_kernel / ref_norm), relq = std::sqrt(err_quant / exact_norm);
    // the kernel against the same quantized operands: BF16 output rounding (2^-9) and FP32 accumulation
    const bool ok = rel < 4e-3;
    bad += !ok;
    std::printf("m %d n %d k %d: kernel vs FP64 of its operands rel L2 %.2e %s | W4A4 vs W4A16 rel L2 %.3f\n", m, n, k, rel,
                ok ? "ok" : "FAIL", relq);
    cudaFree(dx), cudaFree(xc), cudaFree(xs), cudaFree(xr), cudaFree(wc), cudaFree(ws), cudaFree(wa), cudaFree(dy);
  }
  // time at prefill shapes (27B, TP 2: per-rank shards), m = 2048
  blz_gemm* gemm = nullptr;
  if (blz_gemm_create(&gemm) != 0) return 1;
  const int m = 2048;
  for (auto [n, k] : {std::pair{5120, 5120}, std::pair{8704, 5120}, std::pair{5120, 8704}, std::pair{6144, 5120}}) {
    Weight W = random_weight(n, k, rng);
    std::vector<uint16_t> x(size_t(m) * k);
    for (auto& v : x) v = bf(std::normal_distribution<float>(0.0f, 1.0f)(rng));
    uint16_t* dx = dev(x);
    uint8_t *xc, *xs, *wc = dev(W.codes), *ws = dev(W.scales);
    float *xr, *wa = dev(W.alpha);
    uint16_t *dy, *wide;
    check(cudaMalloc(&xc, size_t(m) * k / 2), "malloc");
    check(cudaMalloc(&xs, size_t(m) * k / 16), "malloc");
    check(cudaMalloc(&xr, m * 4), "malloc");
    check(cudaMalloc(&dy, size_t(m) * n * 2), "malloc");
    check(cudaMalloc(&wide, size_t(n) * k * 2), "malloc");
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0), cudaEventCreate(&e1);
    auto time = [&](auto f) {
      f();
      cudaEventRecord(e0);
      for (int i = 0; i < 10; ++i) f();
      cudaEventRecord(e1);
      check(cudaEventSynchronize(e1), "time");
      float ms = 0;
      cudaEventElapsedTime(&ms, e0, e1);
      return ms / 10;
    };
    const float t4 = time([&] {
      blz_nvfp4_quant_rows(dx, k, m, k, xc, xs, xr, 0);
      blz_nvfp4_gemm_w4a4(xc, xs, xr, wc, ws, wa, dy, n, m, n, k, 0);
    });
    const float tq = time([&] { blz_nvfp4_quant_rows(dx, k, m, k, xc, xs, xr, 0); });
    const float tb = time([&] {
      blz_nvfp4_weight_dequant(wc, ws, wa, n, k, wide, 0);
      blz_gemm_bf16(gemm, dx, k, wide, dy, n, m, n, k, 0);
    });
    const double flop = 2.0 * m * n * k;
    std::printf("m %d n %d k %d: W4A4 %.3f ms (%.0f TFLOPS, quant %.3f ms) | widen + cuBLAS BF16 %.3f ms (%.0f TFLOPS): %.2fx\n", m, n,
                k, t4, flop / t4 / 1e9, tq, tb, flop / tb / 1e9, tb / t4);
    cudaFree(dx), cudaFree(xc), cudaFree(xs), cudaFree(xr), cudaFree(wc), cudaFree(ws), cudaFree(wa), cudaFree(dy), cudaFree(wide);
  }
  blz_gemm_destroy(gemm);
  std::printf(bad ? "FAILED\n" : "PASSED\n");
  return bad ? 1 : 0;
}
