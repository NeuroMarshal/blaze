// Dense attention straight from an NVFP4, FP8 or BF16 pool (blaze/mha.h) against a double-precision reference on the
// dequantized cache (decode and causal verify, GQA 6 and 8, several context splits; the prefill kernel in both
// accumulation modes), the FP8 / BF16 store, the row dequantization, and the decode time of a long context per format.
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_fp8.h>

#include "blaze/kv.h"
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

float e4m3_value(uint8_t b) { return blaze::e4m3_to_float(b); }

struct Pool {
  int slots, heads, format = BLZ_KV_NVFP4;
  std::vector<uint8_t> kd, vd, ks, vs;
  float kg = 1.0f, vg = 0.5f;
  float value(bool v, int64_t slot, int head, int d) const {  // bf16-rounded, as the kernels store it
    const int64_t row = slot * heads + head;
    const auto& data = v ? vd : kd;
    const float g = v ? vg : kg;
    float x;
    if (format == BLZ_KV_FP8) {
      x = e4m3_value(data[row * kDim + d]) * g;
    } else if (format == BLZ_KV_BF16) {
      uint16_t b;
      std::memcpy(&b, &data[(row * kDim + d) * 2], 2);
      x = __bfloat162float(__ushort_as_bfloat16(b)) * g;
    } else {
      const int code = (data[row * (kDim / 2) + d / 2] >> (4 * (d & 1))) & 15;
      x = blaze::e2m1_value(code) * blaze::e4m3_to_float((v ? vs : ks)[row * (kDim / 16) + d / 16]) * g;
    }
    return __bfloat162float(__float2bfloat16(x));
  }
};

Pool make_pool(int slots, int heads, std::mt19937& rng, int format = BLZ_KV_NVFP4) {
  Pool p{slots, heads, format};
  std::normal_distribution<float> n01;
  if (format != BLZ_KV_NVFP4) {
    const size_t n = size_t(slots) * heads * kDim;
    p.kd.resize(n * (format == BLZ_KV_BF16 ? 2 : 1)), p.vd.resize(p.kd.size());
    p.ks.resize(1), p.vs.resize(1);
    for (int side = 0; side < 2; ++side)
      for (size_t i = 0; i < n; ++i) {
        const float x = n01(rng) * 2.0f / (side ? p.vg : p.kg);
        auto& data = side ? p.vd : p.kd;
        if (format == BLZ_KV_FP8) {
          const __nv_fp8_e4m3 q(x);
          data[i] = reinterpret_cast<const uint8_t&>(q);
        } else {
          const uint16_t b = __bfloat16_as_ushort(__float2bfloat16(x));
          std::memcpy(&data[2 * i], &b, 2);
        }
      }
    return p;
  }
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
    pool = {kd.ptr, vd.ptr, ks.ptr, vs.ptr, kg.ptr, vg.ptr, p.heads, kDim, p.format};
  }
};

const char* format_name(int f) { return f == BLZ_KV_FP8 ? "fp8" : f == BLZ_KV_BF16 ? "bf16" : "nvfp4"; }

// FP8 / BF16 store: bf16 rows (strided, some slots skipped) against a CPU conversion.
void test_store_dense(int format, std::mt19937& rng) {
  const int heads = 2, tokens = 37, slots = 100, stride = heads * kDim + 64;
  std::normal_distribution<float> n01;
  std::vector<__nv_bfloat16> k(size_t(tokens) * stride), v(k.size());
  for (auto& x : k) x = __float2bfloat16(n01(rng) * 3.0f);
  for (auto& x : v) x = __float2bfloat16(n01(rng) * 900.0f);  // FP8: some values saturate at 448 * g
  std::vector<int32_t> loc(tokens);
  for (int t = 0; t < tokens; ++t) loc[t] = t % 7 == 3 ? -1 : (t * 13) % slots;
  const float g = 2.0f;
  const int bytes = format == BLZ_KV_BF16 ? 2 : 1;
  Dev<__nv_bfloat16> dk(k), dv(v);
  Dev<int32_t> dl(loc);
  Dev<uint8_t> kd(size_t(slots) * heads * kDim * bytes), vd(kd.n);
  CHECK_CUDA(cudaMemset(kd.ptr, 0, kd.n));
  CHECK_CUDA(cudaMemset(vd.ptr, 0, vd.n));
  Dev<float> dg(std::vector<float>{g});
  CHECK_CUDA(blz_kv_store_dense(dk.ptr, dv.ptr, stride, stride, dl.ptr, BLZ_I32, tokens, heads, kDim,
                                blz_kv_format_t(format), kd.ptr, vd.ptr, dg.ptr, dg.ptr, nullptr));
  const auto hk = kd.get(), hv = vd.get();
  size_t bad = 0;
  for (int t = 0; t < tokens; ++t) {
    if (loc[t] < 0) continue;
    for (int side = 0; side < 2; ++side)
      for (int e = 0; e < heads * kDim; ++e) {
        const float x = __bfloat162float((side ? v : k)[size_t(t) * stride + e]);
        const size_t o = size_t(loc[t]) * heads * kDim + e;
        const auto& h = side ? hv : hk;
        if (format == BLZ_KV_FP8) {
          const __nv_fp8_e4m3 q(x / g);
          bad += h[o] != reinterpret_cast<const uint8_t&>(q);
        } else {
          uint16_t b;
          std::memcpy(&b, &h[2 * o], 2);
          bad += b != __bfloat16_as_ushort(__float2bfloat16(x));
        }
      }
  }
  std::printf("store %s: %zu mismatches %s\n", format_name(format), bad, bad ? "FAIL" : "ok");
  g_failures += bad != 0;
}

void test_case(const char* what, int kv_heads, int group, int q_len, std::vector<int> kv_lens, int splits,
               std::mt19937& rng, int format = BLZ_KV_NVFP4, bool padded = false) {
  const int batch = int(kv_lens.size()), q_heads = kv_heads * group, slots = 8192;
  const int max_len = *std::max_element(kv_lens.begin(), kv_lens.end());
  Pool p = make_pool(slots, kv_heads, rng, format);
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
  const int out_head = kDim + int(padded), out_tok = q_heads * out_head + (padded ? 3 : 0);
  Dev<__nv_bfloat16> d_q(q), d_out(size_t(tokens) * out_tok);
  CHECK_CUDA(cudaMemset(d_out.ptr, 0x42, d_out.n * 2));
  const int parts = blz_nvfp4_mha_parts(q_heads, kv_heads, q_len, splits);
  Dev<float> po(size_t(tokens) * q_heads * parts * kDim), pml(size_t(tokens) * q_heads * parts * 2);
  CHECK_CUDA(blz_nvfp4_mha_decode(d_q.ptr, int64_t(q_heads) * kDim, kDim, q_heads, dp.pool, d_r2t.ptr, stride,
                                  {d_req.ptr, BLZ_I64}, {d_lens.ptr, BLZ_I32}, batch, q_len, splits, scale, po.ptr,
                                  pml.ptr, BLZ_BF16, d_out.ptr, out_tok, out_head, nullptr));
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
          const double got = __bfloat162float(out[(size_t(b) * q_len + i) * out_tok + h * out_head + k]);
          worst = std::max(worst, std::fabs(got - ref)), peak = std::max(peak, std::fabs(ref));
        }
      }
  const double rel = worst / std::max(peak, 1e-30);
  bool padding_ok = true;
  if (padded)
    for (int t = 0; t < tokens; ++t) {
      for (int h = 0; h < q_heads; ++h)
        padding_ok &= __bfloat16_as_ushort(out[size_t(t) * out_tok + h * out_head + kDim]) == 0x4242;
      for (int j = q_heads * out_head; j < out_tok; ++j)
        padding_ok &= __bfloat16_as_ushort(out[size_t(t) * out_tok + j]) == 0x4242;
    }
  const bool ok = rel < 1.5e-2 && padding_ok;
  std::printf("%-5s %-44s splits %2d: max error / max |out| %.2e %s\n", format_name(format), what, splits, rel,
              ok ? "ok" : "FAIL");
  g_failures += !ok;
}

// The prefill kernel (blz_nvfp4_mha_prefill, one sequence, causal) against the same double reference: FP32 and FP16
// accumulation, groups the CTAs take whole or in parts, chunks shorter than a CTA's 16 tokens.
void test_prefill(const char* what, int kv_heads, int group, int q_len, int kv_len, bool f16, std::mt19937& rng) {
  const int q_heads = kv_heads * group, slots = 8192;
  Pool p = make_pool(slots, kv_heads, rng, BLZ_KV_NVFP4);
  DevPool dp(p);
  const int reqs = 3, stride = kv_len + 8, req = 1;
  std::vector<int32_t> r2t(size_t(reqs) * stride);
  for (auto& s : r2t) s = int32_t(rng() % slots);
  std::normal_distribution<float> n01;
  std::vector<__nv_bfloat16> q(size_t(q_len) * q_heads * kDim);
  for (auto& v : q) v = __float2bfloat16(n01(rng));
  const float scale = 1.0f / std::sqrt(float(kDim));
  Dev<int32_t> d_r2t(r2t);
  Dev<__nv_bfloat16> d_q(q), d_out(size_t(q_len) * q_heads * kDim);
  CHECK_CUDA(blz_nvfp4_mha_prefill(d_q.ptr, int64_t(q_heads) * kDim, kDim, q_heads, dp.pool, d_r2t.ptr, stride, req, kv_len,
                                   q_len, scale, f16, BLZ_BF16, d_out.ptr, int64_t(q_heads) * kDim, kDim, nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());
  const auto out = d_out.get();
  double worst = 0, peak = 0;
  std::vector<double> sc;
  for (int i = 0; i < q_len; ++i)
    for (int h = 0; h < q_heads; ++h) {
      const int g = h / group, L = kv_len - q_len + i + 1;
      const __nv_bfloat16* qr = &q[(size_t(i) * q_heads + h) * kDim];
      sc.assign(L, 0.0);
      double mx = -1e300;
      for (int t = 0; t < L; ++t) {
        const int64_t slot = r2t[size_t(req) * stride + t];
        double d = 0;
        for (int k = 0; k < kDim; ++k) d += double(__bfloat162float(qr[k])) * p.value(false, slot, g, k);
        sc[t] = d * scale, mx = std::max(mx, sc[t]);
      }
      double sum = 0;
      for (int t = 0; t < L; ++t) sum += (sc[t] = std::exp(sc[t] - mx));
      for (int k = 0; k < kDim; ++k) {
        double ref = 0;
        for (int t = 0; t < L; ++t) ref += sc[t] * p.value(true, r2t[size_t(req) * stride + t], g, k);
        ref /= sum;
        const double got = __bfloat162float(out[(size_t(i) * q_heads + h) * kDim + k]);
        worst = std::max(worst, std::fabs(got - ref)), peak = std::max(peak, std::fabs(ref));
      }
    }
  const double rel = worst / std::max(peak, 1e-30);
  const bool ok = rel < (f16 ? 3e-2 : 1.5e-2);
  std::printf("prefill %-4s %-44s: max error / max |out| %.2e %s\n", f16 ? "f16" : "f32", what, rel, ok ? "ok" : "FAIL");
  g_failures += !ok;
}

// Head dim 128 from a BF16 pool with a sliding window (the EAGLE-4 draft layer: MHA, window 4096) against a double
// reference over positions max(0, p - window + 1) .. p of each query.
void test_window(const char* what, int heads, int q_len, std::vector<int> kv_lens, int window, int splits,
                 std::mt19937& rng, int causal = 1) {
  constexpr int D = 128;
  const int batch = int(kv_lens.size()), slots = 12000;
  const int max_len = *std::max_element(kv_lens.begin(), kv_lens.end());
  std::normal_distribution<float> n01;
  std::vector<__nv_bfloat16> kd(size_t(slots) * heads * D), vd(kd.size());
  for (auto& x : kd) x = __float2bfloat16(n01(rng) * 2.0f);
  for (auto& x : vd) x = __float2bfloat16(n01(rng));
  const int reqs = batch + 1, stride = max_len + 8;
  std::vector<int32_t> r2t(size_t(reqs) * stride);
  for (auto& x : r2t) x = int32_t(rng() % slots);
  std::vector<int32_t> req_idx(batch), lens(kv_lens.begin(), kv_lens.end());
  for (int b = 0; b < batch; ++b) req_idx[b] = (b + 1) % reqs;
  const int tokens = batch * q_len;
  std::vector<__nv_bfloat16> q(size_t(tokens) * heads * D);
  for (auto& x : q) x = __float2bfloat16(n01(rng));
  Dev<__nv_bfloat16> dk(kd), dv(vd), dq(q), dout(q.size());
  Dev<int32_t> d_r2t(r2t), d_req(req_idx), d_lens(lens);
  Dev<float> g(std::vector<float>{1.0f});
  Dev<uint8_t> dummy(16);
  blz_nvfp4_pool pool{reinterpret_cast<uint8_t*>(dk.ptr), reinterpret_cast<uint8_t*>(dv.ptr), dummy.ptr, dummy.ptr,
                      g.ptr, g.ptr, heads, D, BLZ_KV_BF16};
  const int parts = blz_nvfp4_mha_parts(heads, heads, q_len, splits);
  Dev<float> po(size_t(tokens) * heads * parts * D), pml(size_t(tokens) * heads * parts * 2);
  const float scale = 1.0f / std::sqrt(float(D));
  CHECK_CUDA(blz_mha_decode_window(dq.ptr, int64_t(heads) * D, D, heads, pool, d_r2t.ptr, stride, {d_req.ptr, BLZ_I32},
                                   {d_lens.ptr, BLZ_I32}, batch, q_len, splits, scale, window, causal, po.ptr, pml.ptr,
                                   BLZ_BF16, dout.ptr, int64_t(heads) * D, D, nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());
  const auto out = dout.get();
  double worst = 0, peak = 0;
  std::vector<double> sc;
  for (int b = 0; b < batch; ++b)
    for (int i = 0; i < q_len; ++i)
      for (int h = 0; h < heads; ++h) {
        const int p = lens[b] - q_len + i, lo = window > 0 ? std::max(0, p - window + 1) : 0;
        const int hi = causal ? p : lens[b] - 1;  // non-causal: the whole query block
        const __nv_bfloat16* qr = &q[((size_t(b) * q_len + i) * heads + h) * D];
        sc.assign(hi + 1, 0.0);
        double mx = -1e300;
        for (int t = lo; t <= hi; ++t) {
          const size_t row = size_t(r2t[size_t(req_idx[b]) * stride + t]) * heads + h;
          double d = 0;
          for (int k = 0; k < D; ++k) d += double(__bfloat162float(qr[k])) * __bfloat162float(kd[row * D + k]);
          sc[t] = d * scale, mx = std::max(mx, sc[t]);
        }
        double sum = 0;
        for (int t = lo; t <= hi; ++t) sum += (sc[t] = std::exp(sc[t] - mx));
        for (int k = 0; k < D; ++k) {
          double ref = 0;
          for (int t = lo; t <= hi; ++t)
            ref += sc[t] * __bfloat162float(vd[(size_t(r2t[size_t(req_idx[b]) * stride + t]) * heads + h) * D + k]);
          ref /= sum;
          const double got = __bfloat162float(out[((size_t(b) * q_len + i) * heads + h) * D + k]);
          worst = std::max(worst, std::fabs(got - ref)), peak = std::max(peak, std::fabs(ref));
        }
      }
  const double rel = worst / std::max(peak, 1e-30);
  const bool ok = rel < 1.5e-2;
  std::printf("bf16 d128 %-38s window %4d %s splits %2d: max error / max |out| %.2e %s\n", what, window,
              causal ? "causal" : "block ", splits, rel,
              ok ? "ok" : "FAIL");
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

void bench(int ctx, int batch, int q_len, int splits, std::mt19937& rng, int format = BLZ_KV_NVFP4) {
  const int kv_heads = 2, group = 6, q_heads = 12, slots = ctx + 64;
  // Random codes are enough for timing (BF16: small values from random low bytes and a fixed high byte).
  const size_t row_bytes = format == BLZ_KV_NVFP4 ? kDim / 2 : format == BLZ_KV_FP8 ? kDim : 2 * kDim;
  std::vector<uint8_t> kd(size_t(slots) * kv_heads * row_bytes), ks(size_t(slots) * kv_heads * kDim / 16, 0x38);
  for (size_t i = 0; i < kd.size(); ++i) kd[i] = format == BLZ_KV_BF16 && i % 2 ? 0x3f : uint8_t(rng() & 0x77);
  Dev<uint8_t> dkd(kd), dvd(kd), dks(ks), dvs(ks);
  Dev<float> g(std::vector<float>{1.0f});
  blz_nvfp4_pool pool{dkd.ptr, dvd.ptr, dks.ptr, dvs.ptr, g.ptr, g.ptr, kv_heads, kDim, format};
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
  const double bytes = double(batch) * ctx * kv_heads * 2 * (row_bytes + (format == BLZ_KV_NVFP4 ? kDim / 16 : 0));
  std::printf("bench %-5s %d x %d tokens over %dk context, %d splits: %.1f us per layer (%.0f GB/s of KV)\n",
              format_name(format), batch, q_len,
              ctx / 1000, splits, us, bytes / us / 1e3);
}

}  // namespace

int main() {
  std::mt19937 rng(11);
  // Fixed-shape prefill measurements, runnable independently of the numerical/reference cases.
  if (std::getenv("BLAZE_MHA_PREFILL_BENCH")) {
    for (int repeat = 0; repeat < 3; ++repeat) {
      bench(8192, 1, 2048, 1, rng);
      bench(32768, 1, 2048, 1, rng);
    }
    return 0;
  }
  {
    blz_nvfp4_pool pool{};
    pool.heads = 1, pool.dim = kDim, pool.format = BLZ_KV_NVFP4;
    const auto err = blz_nvfp4_mha_decode(nullptr, 0, 0, 0, pool, nullptr, 0, {nullptr, BLZ_I32},
                                         {nullptr, BLZ_I32}, 1, 1, 1, 1.0f, nullptr, nullptr,
                                         BLZ_BF16, nullptr, 0, 0, nullptr);
    const bool ok = err == cudaErrorNotSupported && blz_nvfp4_mha_parts(0, 1, 1, 1) == 0 &&
                    blz_nvfp4_mha_parts(-1, 1, 1, 1) == 0 && blz_nvfp4_mha_parts(12, 2, 1, 0) == 0 &&
                    blz_nvfp4_mha_parts(12, 2, INT_MAX, 1) == 1;
    std::printf("invalid shapes and overflow-safe parts: %s\n", ok ? "ok" : "FAIL");
    g_failures += !ok;
  }
  for (int f : {BLZ_KV_NVFP4, BLZ_KV_FP8, BLZ_KV_BF16})
    test_case("single-part prefill, padded strides", 2, 6, 17, {17, 97}, 1, rng, f, true);
  test_case("decode, GQA 6, lengths 1 / 37 / 3000", 2, 6, 1, {1, 37, 3000}, 1, rng);
  // Flash Next (24 q heads on 2 KV heads) and its sparse prefill rows: one-row sequences of lengths 1 .. 19
  {
    std::vector<int> lens;
    for (int l = 1; l <= 19; ++l) lens.push_back(l);
    for (int sp : {1, 7, 16}) test_case("decode, GQA 12, lengths 1 .. 19", 2, 12, 1, lens, sp, rng);
    test_case("decode, GQA 12, lengths 1 / 37 / 3000", 2, 12, 1, {1, 37, 3000}, 16, rng);
  }
  test_case("decode, GQA 6, lengths 1 / 37 / 3000", 2, 6, 1, {1, 37, 3000}, 16, rng);
  test_case("verify 8 tokens, GQA 6, lengths 8 / 2500", 2, 6, 8, {8, 2500}, 8, rng);
  test_case("verify 3 tokens, GQA 8, lengths 40 / 1200", 1, 8, 3, {40, 1200}, 5, rng);
  test_case("decode, GQA 16, length 777", 1, 16, 1, {777}, 3, rng);
  for (int f : {BLZ_KV_FP8, BLZ_KV_BF16}) {
    test_case("decode, GQA 6, lengths 1 / 37 / 3000", 2, 6, 1, {1, 37, 3000}, 16, rng, f);
    test_case("verify 8 tokens, GQA 6, lengths 8 / 2500", 2, 6, 8, {8, 2500}, 8, rng, f);
    test_case("verify 3 tokens, GQA 8, lengths 40 / 1200", 1, 8, 3, {40, 1200}, 5, rng, f);
    test_case("decode, GQA 16, length 777", 1, 16, 1, {777}, 3, rng, f);
    test_store_dense(f, rng);
  }
  test_window("decode, MHA 8 heads, lengths 1 / 300 / 9000", 8, 1, {1, 300, 9000}, 4096, 16, rng);
  test_window("verify 9 tokens, lengths 9 / 5000", 8, 9, {9, 5000}, 4096, 8, rng);
  test_window("extend 300 tokens, length 5000", 4, 300, {5000}, 4096, 1, rng);
  test_window("verify 5 tokens, short window", 4, 5, {700}, 37, 4, rng);
  test_window("decode, no window", 4, 1, {2000}, 0, 8, rng);
  test_window("DFlash block 8", 2, 8, {8, 3000}, 2048, 4, rng, 0);
  test_window("DFlash block 8, short window", 2, 8, {500}, 37, 2, rng, 0);
  for (bool f16 : {false, true}) {
    test_prefill("GQA 6, 37 tokens, a first chunk", 2, 6, 37, 37, f16, rng);
    test_prefill("GQA 6, 300 tokens after 1700", 2, 6, 300, 2000, f16, rng);
    test_prefill("GQA 8 (two CTAs of 4), 65 tokens after 935", 1, 8, 65, 1000, f16, rng);
    test_prefill("GQA 12 (two CTAs of 6), 33 tokens after 467", 2, 12, 33, 500, f16, rng);
    test_prefill("MHA, 129 tokens after 2000", 3, 1, 129, 2129, f16, rng);
  }
  test_dequant_rows(rng);
  for (int f : {BLZ_KV_FP8, BLZ_KV_BF16}) {
    bench(122000, 1, 1, 64, rng, f);
    bench(122000, 1, 8, 64, rng, f);
  }
  bench(122000, 1, 1, 64, rng);
  bench(122000, 4, 1, 32, rng);
  bench(122000, 1, 8, 32, rng);
  bench(122000, 1, 8, 64, rng);
  bench(524000, 1, 1, 64, rng);
  bench(524000, 1, 8, 64, rng);
  std::printf("%s\n", g_failures ? "FAILED" : "all passed");
  return g_failures ? 1 : 0;
}
