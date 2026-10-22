// Dense attention straight from an NVFP4 pool (blaze/mha.h) against a double-precision reference on the dequantized
// cache (decode and causal verify, GQA 6 and 8, several context splits), the row dequantization, and the decode time
// of a long context.
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "blaze/mha.h"
#include "blaze/nvfp4.h"

#define CHECK_CUDA(x)                                                                              \
  do {                                                                                             \
    cudaError_t err__ = (x);                                                                       \
    if (err__ != cudaSuccess) {                                                                    \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(err__)); \
      std::exit(2);                                                                                \
    }                                                                                              \
  } while (0)

namespace {

int g_failures = 0;
constexpr int kDim = 256;

template <typename T>
struct Dev {
  T* ptr = nullptr;
  size_t n;
  explicit Dev(size_t count) : n(count) { CHECK_CUDA(cudaMalloc(&ptr, std::max<size_t>(1, n) * sizeof(T))); }
  explicit Dev(const std::vector<T>& h) : Dev(h.size()) {
    CHECK_CUDA(cudaMemcpy(ptr, h.data(), n * sizeof(T), cudaMemcpyHostToDevice));
  }
  ~Dev() { cudaFree(ptr); }
  std::vector<T> get() const {
    std::vector<T> h(n);
    CHECK_CUDA(cudaMemcpy(h.data(), ptr, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
  }
};

struct Pool {
  int slots, heads;
  std::vector<uint8_t> kd, vd, ks, vs;
  float kg = 1.0f, vg = 0.5f;
  float value(bool v, int64_t slot, int head, int d) const {  // bf16-rounded, as the kernels store it
    const int64_t row = slot * heads + head;
    const int code = ((v ? vd : kd)[row * (kDim / 2) + d / 2] >> (4 * (d & 1))) & 15;
    const float x = blaze::e2m1_value(code) * blaze::e4m3_to_float((v ? vs : ks)[row * (kDim / 16) + d / 16]) * (v ? vg : kg);
    return __bfloat162float(__float2bfloat16(x));
  }
};

Pool make_pool(int slots, int heads, std::mt19937& rng) {
  Pool p{slots, heads};
  std::normal_distribution<float> n01;
  p.kd.resize(size_t(slots) * heads * kDim / 2), p.vd.resize(p.kd.size());
  p.ks.resize(size_t(slots) * heads * kDim / 16), p.vs.resize(p.ks.size());
  std::vector<float> row(kDim);
  for (int s = 0; s < slots * heads; ++s)
    for (int side = 0; side < 2; ++side) {
      for (auto& x : row) x = n01(rng) * 2.0f;
      blaze::nvfp4_quantize_row_ref(row.data(), kDim, side ? p.vg : p.kg, true,
                                  &(side ? p.vd : p.kd)[size_t(s) * kDim / 2], &(side ? p.vs : p.ks)[size_t(s) * kDim / 16]);
    }
  return p;
}

struct DevPool {
  Dev<uint8_t> kd, vd, ks, vs;
  Dev<float> kg, vg;
  blz_nvfp4_pool pool;
  explicit DevPool(const Pool& p)
      : kd(p.kd), vd(p.vd), ks(p.ks), vs(p.vs), kg(std::vector<float>{p.kg}), vg(std::vector<float>{p.vg}) {
    pool = {kd.ptr, vd.ptr, ks.ptr, vs.ptr, kg.ptr, vg.ptr, p.heads, kDim};
  }
};

void test_case(const char* what, int kv_heads, int group, int q_len, std::vector<int> kv_lens, int splits,
               std::mt19937& rng) {
  const int batch = int(kv_lens.size()), q_heads = kv_heads * group, slots = 8192;
  const int max_len = *std::max_element(kv_lens.begin(), kv_lens.end());
  Pool p = make_pool(slots, kv_heads, rng);
  DevPool dp(p);
  const int reqs = batch + 2, stride = max_len + 8;
  std::vector<int32_t> r2t(size_t(reqs) * stride);
  for (auto& s : r2t) s = int32_t(rng() % slots);
  std::vector<int64_t> req_idx(batch);
  for (int b = 0; b < batch; ++b) req_idx[b] = (b * 3 + 1) % reqs;
  std::vector<int32_t> lens(kv_lens.begin(), kv_lens.end());
  const int tokens = batch * q_len;
  std::normal_distribution<float> n01;
  std::vector<__nv_bfloat16> q(size_t(tokens) * q_heads * kDim);
  for (auto& v : q) v = __float2bfloat16(n01(rng));
  const float scale = 1.0f / std::sqrt(float(kDim));
  Dev<int32_t> d_r2t(r2t), d_lens(lens);
  Dev<int64_t> d_req(req_idx);
  Dev<__nv_bfloat16> d_q(q), d_out(q.size());
  const int parts = blz_nvfp4_mha_parts(q_heads, kv_heads, q_len, splits);
  Dev<float> po(size_t(tokens) * q_heads * parts * kDim), pml(size_t(tokens) * q_heads * parts * 2);
  CHECK_CUDA(blz_nvfp4_mha_decode(d_q.ptr, int64_t(q_heads) * kDim, kDim, q_heads, dp.pool, d_r2t.ptr, stride,
                                  {d_req.ptr, BLZ_I64}, {d_lens.ptr, BLZ_I32}, batch, q_len, splits, scale, po.ptr,
                                  pml.ptr, BLZ_BF16, d_out.ptr, int64_t(q_heads) * kDim, kDim, nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());
  const auto out = d_out.get();
  double worst = 0, peak = 0;
  std::vector<double> sc;
  for (int b = 0; b < batch; ++b)
    for (int i = 0; i < q_len; ++i)
      for (int h = 0; h < q_heads; ++h) {
        const int g = h / group, L = lens[b] - q_len + i + 1;
        const __nv_bfloat16* qr = &q[((size_t(b) * q_len + i) * q_heads + h) * kDim];
        sc.assign(L, 0.0);
        double mx = -1e300;
        for (int t = 0; t < L; ++t) {
          const int64_t slot = r2t[size_t(req_idx[b]) * stride + t];
          double d = 0;
          for (int k = 0; k < kDim; ++k) d += double(__bfloat162float(qr[k])) * p.value(false, slot, g, k);
          sc[t] = d * scale, mx = std::max(mx, sc[t]);
        }
        double sum = 0;
        for (int t = 0; t < L; ++t) sum += (sc[t] = std::exp(sc[t] - mx));
        for (int k = 0; k < kDim; ++k) {
          double ref = 0;
          for (int t = 0; t < L; ++t) ref += sc[t] * p.value(true, r2t[size_t(req_idx[b]) * stride + t], g, k);
          ref /= sum;
          const double got = __bfloat162float(out[((size_t(b) * q_len + i) * q_heads + h) * kDim + k]);
          worst = std::max(worst, std::fabs(got - ref)), peak = std::max(peak, std::fabs(ref));
        }
      }
  const double rel = worst / std::max(peak, 1e-30);
  const bool ok = rel < 1.5e-2;
  std::printf("%-44s splits %2d: max error / max |out| %.2e %s\n", what, splits, rel, ok ? "ok" : "FAIL");
  g_failures += !ok;
}

void test_dequant_rows(std::mt19937& rng) {
  const int heads = 2, slots = 1000, n = 333;
  Pool p = make_pool(slots, heads, rng);
  DevPool dp(p);
  std::vector<int32_t> idx(n);
  for (auto& s : idx) s = int32_t(rng() % slots);
  Dev<int32_t> d_idx(idx);
  Dev<__nv_bfloat16> k(size_t(n) * heads * kDim), v(k.n);
  CHECK_CUDA(blz_nvfp4_dequant_rows(dp.pool, {d_idx.ptr, BLZ_I32}, n, BLZ_BF16, k.ptr, v.ptr, nullptr));
  const auto hk = k.get(), hv = v.get();
  size_t bad = 0;
  for (int r = 0; r < n; ++r)
    for (int h = 0; h < heads; ++h)
      for (int d = 0; d < kDim; ++d) {
        const size_t o = (size_t(r) * heads + h) * kDim + d;
        bad += __bfloat162float(hk[o]) != p.value(false, idx[r], h, d) ||
               __bfloat162float(hv[o]) != p.value(true, idx[r], h, d);
      }
  std::printf("dequant rows: %zu mismatches %s\n", bad, bad ? "FAIL" : "ok");
  g_failures += bad != 0;
}

void bench(int ctx, int batch, int q_len, int splits, std::mt19937& rng) {
  const int kv_heads = 2, group = 6, q_heads = 12, slots = ctx + 64;
  // Random codes are enough for timing.
  std::vector<uint8_t> kd(size_t(slots) * kv_heads * kDim / 2), ks(size_t(slots) * kv_heads * kDim / 16, 0x38);
  for (auto& c : kd) c = uint8_t(rng());
  Dev<uint8_t> dkd(kd), dvd(kd), dks(ks), dvs(ks);
  Dev<float> g(std::vector<float>{1.0f});
  blz_nvfp4_pool pool{dkd.ptr, dvd.ptr, dks.ptr, dvs.ptr, g.ptr, g.ptr, kv_heads, kDim};
  std::vector<int32_t> r2t(size_t(batch) * (ctx + 8));
  for (size_t i = 0; i < r2t.size(); ++i) r2t[i] = int32_t(i % size_t(ctx));
  std::vector<int32_t> lens(batch, ctx);
  std::vector<int64_t> req(batch);
  for (int b = 0; b < batch; ++b) req[b] = b;
  Dev<int32_t> d_r2t(r2t), d_lens(lens);
  Dev<int64_t> d_req(req);
  const int tokens = batch * q_len;
  Dev<__nv_bfloat16> d_q(size_t(tokens) * q_heads * kDim), d_out(d_q.n);
  CHECK_CUDA(cudaMemset(d_q.ptr, 0, d_q.n * 2));
  const int parts = blz_nvfp4_mha_parts(q_heads, kv_heads, q_len, splits);
  Dev<float> po(size_t(tokens) * q_heads * parts * kDim), pml(size_t(tokens) * q_heads * parts * 2);
  auto run = [&] {
    CHECK_CUDA(blz_nvfp4_mha_decode(d_q.ptr, int64_t(q_heads) * kDim, kDim, q_heads, pool, d_r2t.ptr, ctx + 8,
                                    {d_req.ptr, BLZ_I64}, {d_lens.ptr, BLZ_I32}, batch, q_len, splits, 0.0625f,
                                    po.ptr, pml.ptr, BLZ_BF16, d_out.ptr, int64_t(q_heads) * kDim, kDim, nullptr));
  };
  for (int i = 0; i < 3; ++i) run();
  cudaEvent_t a, e;
  cudaEventCreate(&a), cudaEventCreate(&e);
  const int iters = 20;
  cudaEventRecord(a);
  for (int i = 0; i < iters; ++i) run();
  cudaEventRecord(e);
  CHECK_CUDA(cudaEventSynchronize(e));
  float ms = 0;
  cudaEventElapsedTime(&ms, a, e);
  const double us = 1000.0 * ms / iters;
  const double bytes = double(batch) * ctx * kv_heads * (kDim + kDim / 8);  // K and V codes + scales
  std::printf("bench %d x %d tokens over %dk context, %d splits: %.1f us per layer (%.0f GB/s of KV)\n", batch, q_len,
              ctx / 1000, splits, us, bytes / us / 1e3);
}

}  // namespace

int main() {
  std::mt19937 rng(11);
  test_case("decode, GQA 6, lengths 1 / 37 / 3000", 2, 6, 1, {1, 37, 3000}, 1, rng);
  test_case("decode, GQA 6, lengths 1 / 37 / 3000", 2, 6, 1, {1, 37, 3000}, 16, rng);
  test_case("verify 8 tokens, GQA 6, lengths 8 / 2500", 2, 6, 8, {8, 2500}, 8, rng);
  test_case("verify 3 tokens, GQA 8, lengths 40 / 1200", 1, 8, 3, {40, 1200}, 5, rng);
  test_case("decode, GQA 16, length 777", 1, 16, 1, {777}, 3, rng);
  test_dequant_rows(rng);
  bench(122000, 1, 1, 64, rng);
  bench(122000, 4, 1, 32, rng);
  bench(122000, 1, 8, 32, rng);
  bench(122000, 1, 8, 64, rng);
  bench(524000, 1, 1, 64, rng);
  bench(524000, 1, 8, 64, rng);
  std::printf("%s\n", g_failures ? "FAILED" : "all passed");
  return g_failures ? 1 : 0;
}
