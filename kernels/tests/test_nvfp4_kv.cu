// Tests for the NVFP4 KV kernels against the CPU reference in blaze/nvfp4.h.
//   1. host E4M3 rounding == device cvt.rn.satfinite for every bf16 magnitude and random floats
//   2. kv_store (plain and scale search) == reference, bit for bit outside last-ulp error ties
//   3. negative slots are skipped, other pool bytes untouched
//   4. dequant == reference; four over six is never worse than plain per block
//   5. timing of a decode-sized and a prefill-sized store
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "blaze/kv.h"
#include "blaze/nvfp4.h"

#define CHECK_CUDA(x)                                                                    \
  do {                                                                                   \
    cudaError_t err__ = (x);                                                             \
    if (err__ != cudaSuccess) {                                                          \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(err__)); \
      std::exit(2);                                                                      \
    }                                                                                    \
  } while (0)

static int g_failures = 0;
#define EXPECT(cond, ...)                         \
  do {                                            \
    if (!(cond)) {                                \
      ++g_failures;                               \
      std::fprintf(stderr, "FAIL %s:%d ", __FILE__, __LINE__); \
      std::fprintf(stderr, __VA_ARGS__);          \
      std::fprintf(stderr, "\n");                 \
    }                                             \
  } while (0)

template <typename T>
struct DeviceBuffer {
  T* ptr = nullptr;
  size_t n = 0;
  explicit DeviceBuffer(size_t count) : n(count) { CHECK_CUDA(cudaMalloc(&ptr, std::max<size_t>(1, n) * sizeof(T))); }
  ~DeviceBuffer() { cudaFree(ptr); }
  void upload(const std::vector<T>& h) { CHECK_CUDA(cudaMemcpy(ptr, h.data(), n * sizeof(T), cudaMemcpyHostToDevice)); }
  std::vector<T> download() const {
    std::vector<T> h(n);
    CHECK_CUDA(cudaMemcpy(h.data(), ptr, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
  }
};

__global__ void e4m3_probe(const float* in, uint8_t* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const __nv_fp8_e4m3 q(fminf(in[i], 448.0f));
    out[i] = reinterpret_cast<const uint8_t&>(q);
  }
}

static void test_e4m3_rounding() {
  std::vector<float> xs;
  for (uint32_t b = 0; b < 0x7F80; ++b) {  // every finite non-negative bf16 value
    uint32_t bits = b << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    xs.push_back(f);
  }
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> exp_dist(-14.0f, 10.0f);
  for (int i = 0; i < 200000; ++i) xs.push_back(std::exp2(exp_dist(rng)));
  DeviceBuffer<float> din(xs.size());
  DeviceBuffer<uint8_t> dout(xs.size());
  din.upload(xs);
  e4m3_probe<<<(xs.size() + 255) / 256, 256>>>(din.ptr, dout.ptr, int(xs.size()));
  CHECK_CUDA(cudaGetLastError());
  const auto got = dout.download();
  int bad = 0;
  for (size_t i = 0; i < xs.size(); ++i) {
    const uint8_t ref = blaze::float_to_e4m3_host(std::fmin(xs[i], 448.0f));
    if (ref != got[i] && bad++ < 5) std::fprintf(stderr, "  e4m3(%g): host %02x device %02x\n", xs[i], ref, got[i]);
    if (got[i] != 0x7F && std::fabs(blaze::e4m3_to_float(got[i]) - blaze::e4m3_to_float(ref)) > 0) {}
  }
  EXPECT(bad == 0, "E4M3 rounding: %d of %zu values differ", bad, xs.size());
  std::printf("e4m3 rounding: %zu values, %d mismatches\n", xs.size(), bad);
}

static float bf16_round(float x) { return __bfloat162float(__float2bfloat16(x)); }

// Squared error of a stored block against its input (double accumulation, for tie checks).
static double stored_err(const float* xb, const uint8_t* packed, uint8_t scale, float g) {
  float deq[16];
  blaze::nvfp4_dequant_row_ref(packed, &scale, 16, g, deq);
  double e = 0;
  for (int i = 0; i < 16; ++i) e += double(deq[i] - xb[i]) * double(deq[i] - xb[i]);
  return e;
}

struct Case {
  const char* name;
  int tokens, heads, dim;
  float g;  // 0: calibrated from the data as amax / (6 * 256), the four-over-six global scale recipe
  int dist;  // 0 gaussian with per-row scales, 1 outliers, 2 tiny, 3 mixed with zero rows
};

static std::vector<float> make_rows(const Case& c, std::mt19937& rng) {
  std::vector<float> x(size_t(c.tokens) * c.heads * c.dim);
  std::normal_distribution<float> n01(0.0f, 1.0f);
  std::uniform_real_distribution<float> scale_exp(-3.0f, 2.0f);
  std::uniform_int_distribution<int> pick(0, 99);
  for (size_t r = 0; r < x.size() / c.dim; ++r) {
    float s = std::pow(10.0f, scale_exp(rng));
    if (c.dist == 2) s *= 1e-3f;
    const bool zero_row = c.dist == 3 && pick(rng) < 10;
    for (int i = 0; i < c.dim; ++i) {
      float v = zero_row ? 0.0f : n01(rng) * s;
      if (c.dist == 1 && pick(rng) == 0) v *= 30.0f;
      x[r * c.dim + i] = bf16_round(v);
    }
  }
  return x;
}

static void test_store_case(Case c, bool four_over_six, std::mt19937& rng) {
  const int T = c.tokens, H = c.heads, D = c.dim;
  const int slots = T + 7;
  auto kx = make_rows(c, rng), vx = make_rows(c, rng);
  if (c.g == 0.0f) {
    float amax = 0;
    for (float x : kx) amax = std::fmax(amax, std::fabs(x));
    for (float x : vx) amax = std::fmax(amax, std::fabs(x));
    c.g = amax / (6.0f * 256.0f);
  }
  std::vector<__nv_bfloat16> kh(kx.size()), vh(vx.size());
  for (size_t i = 0; i < kx.size(); ++i) kh[i] = __float2bfloat16(kx[i]), vh[i] = __float2bfloat16(vx[i]);
  std::vector<int64_t> loc(T);
  std::vector<int64_t> perm(slots);
  for (int i = 0; i < slots; ++i) perm[i] = i;
  std::shuffle(perm.begin(), perm.end(), rng);
  for (int t = 0; t < T; ++t) loc[t] = (t % 17 == 5) ? -1 : perm[t];

  DeviceBuffer<__nv_bfloat16> dk(kh.size()), dv(vh.size());
  DeviceBuffer<int64_t> dloc(T);
  DeviceBuffer<uint8_t> kd(size_t(slots) * H * D / 2), vd(kd.n), ks(size_t(slots) * H * D / 16), vs(ks.n);
  DeviceBuffer<float> dg(1);
  dk.upload(kh), dv.upload(vh), dloc.upload(loc), dg.upload({c.g});
  for (auto* b : {&kd, &vd, &ks, &vs}) CHECK_CUDA(cudaMemset(b->ptr, 0xAB, b->n));
  CHECK_CUDA(blz_nvfp4_kv_store(dk.ptr, dv.ptr, int64_t(H) * D, int64_t(H) * D, BLZ_BF16, dloc.ptr, BLZ_I64, T, H, D,
                                kd.ptr, vd.ptr, ks.ptr, vs.ptr, dg.ptr, dg.ptr, four_over_six ? 1 : 0, nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());
  const auto gkd = kd.download(), gvd = vd.download(), gks = ks.download(), gvs = vs.download();

  int mismatched_blocks = 0, tie_blocks = 0, untouched_bad = 0;
  double mse = 0, energy = 0;
  std::vector<bool> written(slots, false);
  for (int t = 0; t < T; ++t)
    if (loc[t] >= 0) written[loc[t]] = true;
  std::vector<uint8_t> rp(D / 2), rs(D / 16);
  std::vector<float> deq(D);
  for (int side = 0; side < 2; ++side) {
    const auto& x = side ? vx : kx;
    const auto& gd = side ? gvd : gkd;
    const auto& gs = side ? gvs : gks;
    for (int t = 0; t < T; ++t) {
      if (loc[t] < 0) continue;
      for (int h = 0; h < H; ++h) {
        const float* row = &x[(size_t(t) * H + h) * D];
        blaze::nvfp4_quantize_row_ref(row, D, c.g, four_over_six, rp.data(), rs.data());
        const size_t base = (size_t(loc[t]) * H + h);
        for (int b = 0; b < D / 16; ++b) {
          const bool same = gs[base * (D / 16) + b] == rs[b] &&
                            std::equal(rp.begin() + b * 8, rp.begin() + b * 8 + 8, gd.begin() + base * (D / 2) + b * 8);
          if (same) continue;
          const double er = stored_err(row + b * 16, &rp[b * 8], rs[b], c.g);
          const double eg = stored_err(row + b * 16, &gd[base * (D / 2) + b * 8], gs[base * (D / 16) + b], c.g);
          if (four_over_six && std::fabs(eg - er) <= 1e-6 * std::max(er, 1e-30)) {
            ++tie_blocks;
          } else if (mismatched_blocks++ < 5) {
            std::fprintf(stderr, "  %s t=%d h=%d block=%d: scale ref %02x got %02x\n", c.name, t, h, b, rs[b],
                         gs[base * (D / 16) + b]);
          }
        }
        blaze::nvfp4_dequant_row_ref(&gd[base * (D / 2)], &gs[base * (D / 16)], D, c.g, deq.data());
        for (int i = 0; i < D; ++i) {
          mse += double(deq[i] - row[i]) * (deq[i] - row[i]);
          energy += double(row[i]) * row[i];
        }
      }
    }
    for (int s = 0; s < slots; ++s) {
      if (written[s]) continue;
      for (int i = 0; i < H * D / 2; ++i) untouched_bad += gd[size_t(s) * H * D / 2 + i] != 0xAB;
    }
  }
  EXPECT(mismatched_blocks == 0, "%s 4o6=%d: %d blocks differ from the reference", c.name, four_over_six,
         mismatched_blocks);
  EXPECT(untouched_bad == 0, "%s: %d bytes of skipped slots were written", c.name, untouched_bad);
  std::printf("store %-10s 4o6=%d T=%d H=%d D=%d: mismatches %d, ties %d, rel. MSE %.3e\n", c.name, four_over_six, T,
              H, D, mismatched_blocks, tie_blocks, energy > 0 ? mse / energy : 0.0);
}

static void test_dequant_and_gain(std::mt19937& rng) {
  const int rows = 4096, D = 256;
  Case c{"gain", rows, 1, D, 1.0f, 0};
  auto x = make_rows(c, rng);
  std::vector<uint8_t> p6(size_t(rows) * D / 2), s6(size_t(rows) * D / 16), p4(p6.size()), s4(s6.size());
  double err6 = 0, err4 = 0;
  int worse = 0;
  std::vector<float> d6(D), d4(D);
  for (int r = 0; r < rows; ++r) {
    const float* row = &x[size_t(r) * D];
    blaze::nvfp4_quantize_row_ref(row, D, 1.0f, false, &p6[size_t(r) * D / 2], &s6[size_t(r) * D / 16]);
    blaze::nvfp4_quantize_row_ref(row, D, 1.0f, true, &p4[size_t(r) * D / 2], &s4[size_t(r) * D / 16]);
    blaze::nvfp4_dequant_row_ref(&p6[size_t(r) * D / 2], &s6[size_t(r) * D / 16], D, 1.0f, d6.data());
    blaze::nvfp4_dequant_row_ref(&p4[size_t(r) * D / 2], &s4[size_t(r) * D / 16], D, 1.0f, d4.data());
    for (int b = 0; b < D / 16; ++b) {
      double e6 = 0, e4 = 0;
      for (int i = b * 16; i < b * 16 + 16; ++i) {
        e6 += double(d6[i] - row[i]) * (d6[i] - row[i]);
        e4 += double(d4[i] - row[i]) * (d4[i] - row[i]);
      }
      worse += e4 > e6 * (1 + 1e-6) + 1e-30;
      err6 += e6, err4 += e4;
    }
  }
  EXPECT(worse == 0, "four over six worse than plain in %d blocks", worse);
  std::printf("four over six vs plain (gaussian rows): MSE ratio %.4f (%.1f%% lower)\n", err4 / err6,
              100.0 * (1 - err4 / err6));

  // GPU dequant == CPU dequant (after rounding both to bf16).
  DeviceBuffer<uint8_t> dp(p4.size()), ds(s4.size());
  DeviceBuffer<float> dg(1);
  DeviceBuffer<__nv_bfloat16> dout(size_t(rows) * D);
  dp.upload(p4), ds.upload(s4), dg.upload({1.0f});
  CHECK_CUDA(blz_nvfp4_dequant(dp.ptr, ds.ptr, dg.ptr, rows, D, BLZ_BF16, dout.ptr, nullptr));
  const auto got = dout.download();
  int bad = 0;
  std::vector<float> ref(D);
  for (int r = 0; r < rows; ++r) {
    blaze::nvfp4_dequant_row_ref(&p4[size_t(r) * D / 2], &s4[size_t(r) * D / 16], D, 1.0f, ref.data());
    for (int i = 0; i < D; ++i) bad += __bfloat162float(got[size_t(r) * D + i]) != bf16_round(ref[i]);
  }
  EXPECT(bad == 0, "dequant: %d values differ", bad);
}

static void bench_store(int tokens, int heads, int dim, int layers) {
  DeviceBuffer<__nv_bfloat16> k(size_t(tokens) * heads * dim), v(k.n);
  DeviceBuffer<int32_t> loc(tokens);
  std::vector<int32_t> hl(tokens);
  for (int i = 0; i < tokens; ++i) hl[i] = i;
  loc.upload(hl);
  CHECK_CUDA(cudaMemset(k.ptr, 0x3c, k.n * 2));
  CHECK_CUDA(cudaMemset(v.ptr, 0x3c, v.n * 2));
  DeviceBuffer<uint8_t> kd(size_t(tokens) * heads * dim / 2), vd(kd.n), ks(size_t(tokens) * heads * dim / 16),
      vs(ks.n);
  DeviceBuffer<float> g(1);
  g.upload({1.0f});
  cudaEvent_t a, b;
  cudaEventCreate(&a), cudaEventCreate(&b);
  for (int fos = 0; fos < 2; ++fos) {
    for (int w = 0; w < 3; ++w)
      blz_nvfp4_kv_store(k.ptr, v.ptr, heads * dim, heads * dim, BLZ_BF16, loc.ptr, BLZ_I32, tokens, heads, dim,
                         kd.ptr, vd.ptr, ks.ptr, vs.ptr, g.ptr, g.ptr, fos, nullptr);
    const int iters = 200;
    cudaEventRecord(a);
    for (int i = 0; i < iters * layers; ++i)
      blz_nvfp4_kv_store(k.ptr, v.ptr, heads * dim, heads * dim, BLZ_BF16, loc.ptr, BLZ_I32, tokens, heads, dim,
                         kd.ptr, vd.ptr, ks.ptr, vs.ptr, g.ptr, g.ptr, fos, nullptr);
    cudaEventRecord(b);
    CHECK_CUDA(cudaEventSynchronize(b));
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    const double us = 1000.0 * ms / (iters * layers);
    const double gbs = double(tokens) * heads * dim * 2 * 2 * 1.5625 / 2 / (us * 1e3);
    std::printf("bench store tokens=%d heads=%d dim=%d 4o6=%d: %.2f us per layer call (%.0f GB/s moved)\n", tokens,
                heads, dim, fos, us, gbs);
  }
  cudaEventDestroy(a), cudaEventDestroy(b);
}

int main() {
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  std::printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  test_e4m3_rounding();
  std::mt19937 rng(1234);
  const Case cases[] = {
      {"gaussian", 257, 2, 256, 1.0f, 0},  {"outliers", 129, 2, 256, 1.0f, 1}, {"tiny", 65, 2, 256, 1.0f, 2},
      {"zero-rows", 97, 8, 128, 1.0f, 3},  {"calib-g", 64, 2, 256, 0.0f, 1},
      {"dim64", 33, 4, 64, 1.0f, 0},
  };
  for (const auto& c : cases) {
    test_store_case(c, false, rng);
    test_store_case(c, true, rng);
  }
  test_dequant_and_gain(rng);
  bench_store(1, 2, 256, 12);
  bench_store(8192, 2, 256, 1);
  if (g_failures) {
    std::printf("FAILED: %d check(s)\n", g_failures);
    return 1;
  }
  std::printf("PASSED\n");
  return 0;
}
