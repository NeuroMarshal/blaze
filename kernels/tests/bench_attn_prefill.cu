// Prefill attention on synthetic NVFP4 KV, the 27B's shape per GPU at TP=2 (12 query heads, 2 KV heads, head dim 256):
// a chunk of Q rows at the end of a context of N positions (causal), the time per layer and the TFLOPS of the causal
// work (4 q_len (N - q_len / 2) D heads). Kernels: decode (blz_nvfp4_mha_decode, what prefill ran on) and prefill
// (blz_nvfp4_mha_prefill), optionally checked against each other.
//   bench_attn_prefill [q_len (2048)] [contexts in k (8,32,64,128)] [check (1)]
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <charconv>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "blaze/mha.h"
// Optional hand-built ablations; ordinary CMake builds benchmark the production library itself.
#ifdef BLAZE_PREFILL_ABLATE
#define blz_nvfp4_mha_prefill blz_bench_nvfp4_mha_prefill
#define blz_nvfp4_mha_prefill_supported blz_bench_nvfp4_mha_prefill_supported
#define blz_nvfp4_mha_prefill_group_ok blz_bench_nvfp4_mha_prefill_group_ok
#include "../src/cuda/mha_prefill.cu"
#undef blz_nvfp4_mha_prefill
#undef blz_nvfp4_mha_prefill_supported
#undef blz_nvfp4_mha_prefill_group_ok
#endif

#define CHECK(x)                                                                         \
  do {                                                                                   \
    cudaError_t e_ = (x);                                                                \
    if (e_ != cudaSuccess) {                                                             \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      std::exit(1);                                                                      \
    }                                                                                    \
  } while (0)

namespace {

constexpr int kDim = 256, kKvHeads = 2, kGroup = 6, kQHeads = kKvHeads * kGroup;

template <typename T>
T* dev(size_t n) {
  T* p = nullptr;
  CHECK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
  return p;
}

template <typename T>
T* upload(const std::vector<T>& v) {
  T* p = dev<T>(v.size());
  CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return p;
}

struct Case {
  blz_nvfp4_pool pool{};
  int32_t* r2t = nullptr;
  int64_t* req = nullptr;
  int32_t* lens = nullptr;
  __nv_bfloat16* q = nullptr;
  int ctx = 0, q_len = 0;
};

// Random NVFP4 codes with block scales of about 1 / 4 (values to ~1.5), K and V; Q of normal-ish values.
Case make_case(int ctx, int q_len, std::mt19937& rng) {
  Case c;
  c.ctx = ctx, c.q_len = q_len;
  const size_t slots = size_t(ctx) + 64, rows = slots * kKvHeads;
  std::vector<uint8_t> kd(rows * kDim / 2), vd(rows * kDim / 2), ks(rows * kDim / 16), vs(rows * kDim / 16);
  for (auto& b : kd) b = uint8_t(rng());
  for (auto& b : vd) b = uint8_t(rng());
  for (auto& b : ks) b = uint8_t(0x28 + rng() % 16);  // e4m3 ~ 0.125 .. 0.4
  for (auto& b : vs) b = uint8_t(0x28 + rng() % 16);
  std::vector<float> g{1.0f};
  c.pool = blz_nvfp4_pool{upload(kd), upload(vd), upload(ks), upload(vs), upload(g), upload(g), kKvHeads, kDim, BLZ_KV_NVFP4};
  // the slots of the context in a shuffled order (pages of 64), as a long-running pool has them
  std::vector<int32_t> r2t(size_t(ctx) + 64);
  std::vector<int32_t> pages((ctx + 63) / 64);
  for (size_t i = 0; i < pages.size(); ++i) pages[i] = int32_t(i);
  std::shuffle(pages.begin(), pages.end(), rng);
  for (int t = 0; t < ctx; ++t) r2t[t] = pages[t / 64] * 64 + t % 64;
  c.r2t = upload(r2t);
  c.req = upload(std::vector<int64_t>{0});
  c.lens = upload(std::vector<int32_t>{ctx});
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<__nv_bfloat16> q(size_t(q_len) * kQHeads * kDim);
  for (auto& x : q) x = __float2bfloat16(nd(rng));
  c.q = upload(q);
  return c;
}

double causal_flops(int ctx, int q_len) { return 4.0 * q_len * (double(ctx) - q_len / 2.0) * kDim * kQHeads; }

int positive_int(std::string_view s, int maximum) {
  int value = 0;
  const auto parsed = std::from_chars(s.data(), s.data() + s.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != s.data() + s.size() || value < 1 || value > maximum) {
    std::fprintf(stderr, "invalid positive integer: %.*s (max %d)\n", int(s.size()), s.data(), maximum);
    std::exit(2);
  }
  return value;
}

}  // namespace

int main(int argc, char** argv) {
  const int q_len = argc > 1 ? positive_int(argv[1], INT_MAX - 64) : 2048;
  // Disable shared verify dispatch so the baseline remains the old register kernel at every query length.
  blz_mha_set_verify(0);
  blz_mha_set_f16_accumulate(0);
  std::vector<int> ctxs;
  {
    std::stringstream ss(argc > 2 ? argv[2] : "8,32,64,128");
    for (std::string t; std::getline(ss, t, ',');) ctxs.push_back(positive_int(t, (INT_MAX - 64) / 1000) * 1000);
  }
  if (ctxs.empty() || std::none_of(ctxs.begin(), ctxs.end(), [&](int n) { return n >= q_len; })) {
    std::fprintf(stderr, "at least one context must fit q_len\n");
    return 2;
  }
  const bool check = argc > 3 ? std::atoi(argv[3]) != 0 : true;
  cudaDeviceProp prop{};
  CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("%s, %d SMs; q_len %d, %d q heads / %d KV heads, head dim %d, NVFP4 KV\n", prop.name,
              prop.multiProcessorCount, q_len, kQHeads, kKvHeads, kDim);
  std::mt19937 rng(7);
  for (int ctx : ctxs) {
    if (ctx < q_len) continue;
    Case c = make_case(ctx, q_len, rng);
    const size_t out_n = size_t(q_len) * kQHeads * kDim;
    __nv_bfloat16* out_dec = dev<__nv_bfloat16>(out_n);
    __nv_bfloat16* out_pre = dev<__nv_bfloat16>(out_n);
    const int splits = 1;
    const int parts = blz_nvfp4_mha_parts(kQHeads, kKvHeads, q_len, splits);
    float* po = dev<float>(size_t(q_len) * kQHeads * parts * kDim);
    float* pml = dev<float>(size_t(q_len) * kQHeads * parts * 2);
    const float scale = 1.0f / std::sqrt(float(kDim));
    auto decode = [&] {
      CHECK(blz_nvfp4_mha_decode(c.q, int64_t(kQHeads) * kDim, kDim, kQHeads, c.pool, c.r2t, ctx + 64, {c.req, BLZ_I64},
                                 {c.lens, BLZ_I32}, 1, q_len, splits, scale, po, pml, BLZ_BF16, out_dec,
                                 int64_t(kQHeads) * kDim, kDim, nullptr));
    };
    __nv_bfloat16* out_h = dev<__nv_bfloat16>(out_n);
    auto prefill = [&] {
      CHECK(blz_nvfp4_mha_prefill(c.q, int64_t(kQHeads) * kDim, kDim, kQHeads, c.pool, c.r2t, ctx + 64, 0, ctx, q_len,
                                  scale, 0, BLZ_BF16, out_pre, int64_t(kQHeads) * kDim, kDim, nullptr));
    };
    auto prefill_h = [&] {
      CHECK(blz_nvfp4_mha_prefill(c.q, int64_t(kQHeads) * kDim, kDim, kQHeads, c.pool, c.r2t, ctx + 64, 0, ctx, q_len,
                                  scale, 1, BLZ_BF16, out_h, int64_t(kQHeads) * kDim, kDim, nullptr));
    };
#ifdef BLAZE_PREFILL_ABLATE
    auto cut = [&](auto k, auto f16) {
      return [&, k, f16] {
        CHECK((blz_prefill_ablate<decltype(k)::value, decltype(f16)::value>(c.q, int64_t(kQHeads) * kDim, kDim, c.pool, c.r2t,
                                                                            ctx, q_len, scale, out_h, int64_t(kQHeads) * kDim,
                                                                            kDim)));
      };
    };
#endif
    auto time = [&](auto&& f) {
      f();
      CHECK(cudaDeviceSynchronize());
      cudaEvent_t a, b;
      CHECK(cudaEventCreate(&a));
      CHECK(cudaEventCreate(&b));
      const int iters = ctx >= 64000 ? 3 : 10;
      CHECK(cudaEventRecord(a));
      for (int i = 0; i < iters; ++i) f();
      CHECK(cudaEventRecord(b));
      CHECK(cudaEventSynchronize(b));
      float ms = 0;
      CHECK(cudaEventElapsedTime(&ms, a, b));
      CHECK(cudaEventDestroy(a));
      CHECK(cudaEventDestroy(b));
      return ms / iters;
    };
    const double f = causal_flops(ctx, q_len);
    const float td = time(decode);
    const bool has_prefill = blz_nvfp4_mha_prefill_supported() != 0;
    const float tp = has_prefill ? time(prefill) : 0.0f;
    const float th = has_prefill ? time(prefill_h) : 0.0f;
    auto tf = [&](float ms) { return f / (ms * 1e-3) / 1e12; };
    std::printf("context %4dk: decode kernel %8.2f ms (%5.1f TFLOPS) | prefill FP32 acc %8.2f ms (%5.1f) x%.2f | FP16 acc %8.2f ms (%5.1f) x%.2f",
                ctx / 1000, td, tf(td), tp, tf(tp), td / tp, th, tf(th), td / th);
    auto diff = [&](const __nv_bfloat16* da, const __nv_bfloat16* db) {
      std::vector<__nv_bfloat16> a(out_n), b(out_n);
      CHECK(cudaMemcpy(a.data(), da, out_n * 2, cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(b.data(), db, out_n * 2, cudaMemcpyDeviceToHost));
      double num = 0, den = 0, worst = 0;
      for (size_t i = 0; i < out_n; ++i) {
        const double x = __bfloat162float(a[i]), y = __bfloat162float(b[i]);
        num += (x - y) * (x - y), den += x * x, worst = std::max(worst, std::fabs(x - y));
      }
      std::printf(" %.2e (max %.3g)", std::sqrt(num / std::max(den, 1e-30)), worst);
    };
    if (has_prefill && check) {
      std::printf("\n    rel L2 against the decode kernel: FP32 acc");
      diff(out_dec, out_pre);
      std::printf(", FP16 acc");
      diff(out_dec, out_h);
    }
#ifdef BLAZE_PREFILL_ABLATE
    if (std::getenv("ABLATE")) {
      using blaze::kNoDequant, blaze::kNoQk, blaze::kNoPv;
      std::printf("\n    cuts (FP32 / FP16 acc, ms): no dequant %.2f / %.2f, no QK %.2f / %.2f, no PV %.2f / %.2f, softmax+loads only %.2f",
                  time(cut(std::integral_constant<int, kNoDequant>{}, std::false_type{})),
                  time(cut(std::integral_constant<int, kNoDequant>{}, std::true_type{})),
                  time(cut(std::integral_constant<int, kNoQk>{}, std::false_type{})),
                  time(cut(std::integral_constant<int, kNoQk>{}, std::true_type{})),
                  time(cut(std::integral_constant<int, kNoPv>{}, std::false_type{})),
                  time(cut(std::integral_constant<int, kNoPv>{}, std::true_type{})),
                  time(cut(std::integral_constant<int, kNoDequant | kNoQk | kNoPv>{}, std::false_type{})));
    }
#endif
    std::printf("\n");
    cudaFree(out_h);
    cudaFree(out_dec), cudaFree(out_pre), cudaFree(po), cudaFree(pml);
    cudaFree(c.r2t), cudaFree(c.req), cudaFree(c.lens), cudaFree(c.q);
    cudaFree(const_cast<uint8_t*>(c.pool.k_data)), cudaFree(const_cast<uint8_t*>(c.pool.v_data));
    cudaFree(const_cast<uint8_t*>(c.pool.k_scale)), cudaFree(const_cast<uint8_t*>(c.pool.v_scale));
    cudaFree(const_cast<float*>(c.pool.k_g)), cudaFree(const_cast<float*>(c.pool.v_g));
  }
  return 0;
}
