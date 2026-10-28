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
#include <limits>
#include <random>
#include <thread>
#include <vector>

#include <cuda_fp8.h>

#include "blaze/kv.h"
#include "blaze/mha.h"
#include "blaze/nvfp4.h"
#include "blaze/tq3.h"
#include "blaze/ops.h"

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

// std::max(worst, abs(NaN)) keeps worst unchanged and can report a corrupt kernel as passing.
void observe_error(double got, double ref, double& worst, double& peak) {
  if (!std::isfinite(got) || !std::isfinite(ref)) {
    worst = std::numeric_limits<double>::infinity();
    return;
  }
  worst = std::max(worst, std::fabs(got - ref));
  peak = std::max(peak, std::fabs(ref));
}

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
  std::vector<float> k_ref, v_ref;
  float kg = 1.0f, vg = 0.5f;
  float decoded_value(bool v, int64_t slot, int head, int d) const {  // bf16-rounded, as the kernels store it
    const int64_t row = slot * heads + head;
    const auto& data = v ? vd : kd;
    const float g = v ? vg : kg;
    float x;
    if (format == BLZ_KV_TQ3) {
      uint16_t sc;
      std::memcpy(&sc, &(v ? vs : ks)[size_t(row) * 2], 2);
      return blaze::tq3_value_ref(&data[size_t(row) * blaze::kTq3RowBytes], sc, d) * g;  // an FP16 value, as the kernel's
    } else if (format == BLZ_KV_FP8) {
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
  // Decode the independent reference once. Repeating E4M3 conversion inside every query's O(N D) reference loop
  // dominated the test time, while the GPU was idle. No GPU output is used to construct this reference.
  void cache_reference() {
    const size_t n = size_t(slots) * heads * kDim;
    k_ref.resize(n), v_ref.resize(n);
    for (int slot = 0; slot < slots; ++slot)
      for (int head = 0; head < heads; ++head)
        for (int d = 0; d < kDim; ++d) {
          const size_t i = (size_t(slot) * heads + head) * kDim + d;
          k_ref[i] = decoded_value(false, slot, head, d);
          v_ref[i] = decoded_value(true, slot, head, d);
        }
  }
  float value(bool v, int64_t slot, int head, int d) const {
    return (v ? v_ref : k_ref)[(slot * heads + head) * kDim + d];
  }
};

Pool make_pool(int slots, int heads, std::mt19937& rng, int format = BLZ_KV_NVFP4) {
  Pool p{slots, heads, format};
  std::normal_distribution<float> n01;
  if (format == BLZ_KV_TQ3) {
    const size_t rows = size_t(slots) * heads;
    p.kd.resize(rows * blaze::kTq3RowBytes), p.vd.resize(p.kd.size());
    p.ks.resize(rows * 2), p.vs.resize(p.ks.size());
    std::vector<float> row(kDim);
    for (size_t r = 0; r < rows; ++r)
      for (int side = 0; side < 2; ++side) {
        for (auto& x : row) x = n01(rng) * 2.0f / (side ? p.vg : p.kg);
        uint16_t sc;
        blaze::tq3_quantize_row_ref(row.data(), &(side ? p.vd : p.kd)[r * blaze::kTq3RowBytes], &sc);
        std::memcpy(&(side ? p.vs : p.ks)[r * 2], &sc, 2);
      }
    p.cache_reference();
    return p;
  }
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
    p.cache_reference();
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
  p.cache_reference();
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

const char* format_name(int f) {
  return f == BLZ_KV_FP8 ? "fp8" : f == BLZ_KV_BF16 ? "bf16" : f == BLZ_KV_TQ3 ? "tq3" : "nvfp4";
}

// TQ3 store: bf16 rows (strided, some slots skipped) against the CPU reference quantizer; the row scale may differ by
// an FP16 step (the GPU sums in float), a magnitude only where |x| / rms sits at a cut.
void test_store_tq3(std::mt19937& rng) {
  const int heads = 2, tokens = 37, slots = 100, stride = heads * kDim + 64;
  std::normal_distribution<float> n01;
  std::vector<__nv_bfloat16> k(size_t(tokens) * stride), v(k.size());
  for (auto& x : k) x = __float2bfloat16(n01(rng) * 3.0f);
  for (auto& x : v) x = __float2bfloat16(n01(rng) * 0.05f);
  std::vector<int32_t> loc(tokens);
  for (int t = 0; t < tokens; ++t) loc[t] = t % 7 == 3 ? -1 : (t * 13) % slots;
  Dev<__nv_bfloat16> dk(k), dv(v);
  Dev<int32_t> dl(loc);
  Dev<uint8_t> kd(size_t(slots) * heads * blaze::kTq3RowBytes), vd(kd.n), ks(size_t(slots) * heads * 2), vs(ks.n);
  CHECK_CUDA(blz_tq3_kv_store(dk.ptr, dv.ptr, stride, stride, dl.ptr, BLZ_I32, tokens, heads, kDim, kd.ptr, vd.ptr,
                              ks.ptr, vs.ptr, nullptr));
  const auto hk = kd.get(), hv = vd.get(), hks = ks.get(), hvs = vs.get();
  size_t bad_codes = 0, bad_scale = 0, total = 0;
  std::vector<float> row(kDim);
  uint8_t codes[blaze::kTq3RowBytes];
  for (int t = 0; t < tokens; ++t) {
    if (loc[t] < 0) continue;
    for (int side = 0; side < 2; ++side)
      for (int h = 0; h < heads; ++h) {
        for (int d = 0; d < kDim; ++d) row[d] = __bfloat162float((side ? v : k)[size_t(t) * stride + h * kDim + d]);
        uint16_t sc, got;
        blaze::tq3_quantize_row_ref(row.data(), codes, &sc);
        const size_t r = size_t(loc[t]) * heads + h;
        std::memcpy(&got, &(side ? hvs : hks)[r * 2], 2);
        bad_scale += std::abs(int(got) - int(sc)) > 1;
        for (int i = 0; i < blaze::kTq3RowBytes; ++i) bad_codes += codes[i] != (side ? hv : hk)[r * blaze::kTq3RowBytes + i];
        total += blaze::kTq3RowBytes;
      }
  }
  const bool ok = bad_scale == 0 && bad_codes * 1000 <= total;
  std::printf("store tq3: %zu of %zu code bytes differ, %zu scales off by more than an FP16 step %s\n", bad_codes,
              total, bad_scale, ok ? "ok" : "FAIL");
  g_failures += !ok;
}

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
          observe_error(got, ref, worst, peak);
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
        observe_error(got, ref, worst, peak);
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
          observe_error(got, ref, worst, peak);
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

// the splits the runtime picks (model.rs attn_splits) for the bench shape on this GPU
int auto_splits(int batch, int q_len) {
  const int kv_heads = 2, group = 6, q_heads = 12;
  int dev = 0, sms = 0;
  CHECK_CUDA(cudaGetDevice(&dev));
  CHECK_CUDA(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  const int vt = blz_nvfp4_mha_verify_tiles(q_heads, kv_heads, q_len);
  if (vt > 0) return std::clamp(sms / (batch * vt * kv_heads), 1, 64);
  const int mt = std::min(3, (q_len * group + 15) / 16), per = std::max(1, 16 * mt / group);
  return std::clamp(4 * sms / (batch * ((q_len + per - 1) / per) * kv_heads), 1, 64);
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
  CHECK_CUDA(cudaEventDestroy(a));
  CHECK_CUDA(cudaEventDestroy(e));
  const double us = 1000.0 * ms / iters;
  const double bytes = double(batch) * ctx * kv_heads * 2 * (row_bytes + (format == BLZ_KV_NVFP4 ? kDim / 16 : 0));
  std::printf("bench %-5s %d x %d tokens over %dk context, %d splits: %.1f us per layer (%.0f GB/s of KV)\n",
              format_name(format), batch, q_len,
              ctx / 1000, splits, us, bytes / us / 1e3);
}

// The rotated cache (blz_rht_rows): the inverse undoes the forward and dot products stay. Then keys with a few large
// channels nearly constant over positions (as LLM keys have) plus the varying part: the softmax sees only the scores'
// differences between keys (centred per query), and those lose more to NVFP4 unrotated (a block holding a large
// channel scales its 15 small ones away) than rotated. Both errors are printed.
void test_rotation(std::mt19937& rng, int kind) {
  static const char* names[] = {"hadamard", "planar", "iso", "rotor"};
  const int n = 512, nq = 64;
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<__nv_bfloat16> k(size_t(n) * kDim), q(size_t(nq) * kDim);
  for (int r = 0; r < n; ++r)
    for (int d = 0; d < kDim; ++d) k[size_t(r) * kDim + d] = __float2bfloat16(nd(rng) + (d % 64 == 5 ? 40.0f : 0.0f));
  for (auto& x : q) x = __float2bfloat16(nd(rng));
  auto f = [](__nv_bfloat16 x) { return double(__bfloat162float(x)); };
  Dev<__nv_bfloat16> dk(k), dq(q);
  auto rot = [&](Dev<__nv_bfloat16>& x, int rows, int inverse) {
    CHECK_CUDA(blz_rht_rows(reinterpret_cast<uint16_t*>(x.ptr), rows, 1, kDim, kDim, kDim, inverse, kind, nullptr));
  };
  rot(dk, n, 0);
  rot(dq, nq, 0);
  const auto rk = dk.get(), rq = dq.get();
  rot(dk, n, 1);
  const auto back = dk.get();
  double trip = 0, dots = 0;
  for (int r = 0; r < n; ++r) {  // relative L2 of a row: two bf16 roundings, of large values where a block keeps them
    double norm = 0, err = 0;
    for (int d = 0; d < kDim; ++d) {
      const double x = f(k[size_t(r) * kDim + d]), y = f(back[size_t(r) * kDim + d]);
      norm += x * x, err += (y - x) * (y - x);
    }
    trip = std::max(trip, std::sqrt(err / norm));
  }
  // NVFP4 of the plain and the rotated keys, decoded back
  auto quantized = [&](const std::vector<__nv_bfloat16>& rows) {
    Dev<__nv_bfloat16> src(rows), out(rows.size());
    std::vector<int64_t> loc(n);
    for (int i = 0; i < n; ++i) loc[i] = i;
    Dev<int64_t> dloc(loc);
    Dev<uint8_t> kd(size_t(n) * kDim / 2), vd(kd.n), ks(size_t(n) * kDim / 16), vs(ks.n);
    Dev<float> g(std::vector<float>{1.0f});
    CHECK_CUDA(blz_nvfp4_kv_store(src.ptr, src.ptr, kDim, kDim, BLZ_BF16, dloc.ptr, BLZ_I64, n, 1, kDim, kd.ptr, vd.ptr,
                                  ks.ptr, vs.ptr, g.ptr, g.ptr, 1, nullptr));
    CHECK_CUDA(blz_nvfp4_dequant(kd.ptr, ks.ptr, g.ptr, n, kDim, BLZ_BF16, out.ptr, nullptr));
    return out.get();
  };
  const auto qk = quantized(k), qrk = quantized(rk);
  double e_plain = 0, e_rot = 0, ref2 = 0, raw2 = 0;
  std::vector<double> exact(n), plain(n), rotated(n);
  auto centre = [](std::vector<double>& x) {
    double m = 0;
    for (double v : x) m += v;
    m /= double(x.size());
    for (double& v : x) v -= m;
  };
  for (int i = 0; i < nq; ++i) {
    for (int r = 0; r < n; ++r) {
      double e = 0, rdot = 0, p = 0, rr = 0;
      for (int d = 0; d < kDim; ++d) {
        const size_t a = size_t(i) * kDim + d, b = size_t(r) * kDim + d;
        e += f(q[a]) * f(k[b]);
        rdot += f(rq[a]) * f(rk[b]);
        p += f(q[a]) * f(qk[b]);
        rr += f(rq[a]) * f(qrk[b]);
      }
      raw2 += e * e;
      dots = std::max(dots, std::fabs(rdot - e));
      exact[r] = e, plain[r] = p, rotated[r] = rr;
    }
    centre(exact), centre(plain), centre(rotated);
    for (int r = 0; r < n; ++r) {
      ref2 += exact[r] * exact[r];
      e_plain += (plain[r] - exact[r]) * (plain[r] - exact[r]);
      e_rot += (rotated[r] - exact[r]) * (rotated[r] - exact[r]);
    }
  }
  const double rms = std::sqrt(ref2 / (double(nq) * n));
  e_plain = std::sqrt(e_plain / (double(nq) * n)) / rms, e_rot = std::sqrt(e_rot / (double(nq) * n)) / rms;
  dots /= std::sqrt(raw2 / (double(nq) * n));
  // the fused launches: Q, K, V in one (bit for bit the separate calls), the inverse with the output gate (within a
  // bf16 rounding of the inverse followed by blz_mul_sigmoid)
  bool fused = true;
  {
    const int t = 7, qh = 3, kvh = 2, ld = (qh + 2 * kvh) * kDim + 8;
    std::vector<__nv_bfloat16> h(size_t(t) * ld);
    for (auto& x : h) x = __float2bfloat16(nd(rng));
    Dev<__nv_bfloat16> one(h), sep(h);
    auto* p1 = reinterpret_cast<uint16_t*>(one.ptr);
    auto* p2 = reinterpret_cast<uint16_t*>(sep.ptr);
    CHECK_CUDA(blz_rht_qkv(p1, qh, ld, p1 + qh * kDim, p1 + (qh + kvh) * kDim, kvh, ld, t, kDim, kind, nullptr));
    CHECK_CUDA(blz_rht_rows(p2, t, qh, ld, kDim, kDim, 0, kind, nullptr));
    CHECK_CUDA(blz_rht_rows(p2 + qh * kDim, t, kvh, ld, kDim, kDim, 0, kind, nullptr));
    CHECK_CUDA(blz_rht_rows(p2 + (qh + kvh) * kDim, t, kvh, ld, kDim, kDim, 0, kind, nullptr));
    fused = std::memcmp(one.get().data(), sep.get().data(), h.size() * 2) == 0;
    std::vector<__nv_bfloat16> x(size_t(t) * qh * kDim), gt(x.size());
    for (auto& v : x) v = __float2bfloat16(nd(rng));
    for (auto& v : gt) v = __float2bfloat16(nd(rng));
    Dev<__nv_bfloat16> a(x), b(x), dg(gt);
    auto* pa = reinterpret_cast<uint16_t*>(a.ptr);
    auto* pb = reinterpret_cast<uint16_t*>(b.ptr);
    CHECK_CUDA(blz_rht_inverse_mul_sigmoid(pa, reinterpret_cast<uint16_t*>(dg.ptr), t, qh, kDim, kind, nullptr));
    CHECK_CUDA(blz_rht_rows(pb, t, qh, qh * kDim, kDim, kDim, 1, kind, nullptr));
    CHECK_CUDA(blz_mul_sigmoid(pb, reinterpret_cast<uint16_t*>(dg.ptr), int64_t(x.size()), nullptr));
    const auto ra = a.get(), rb = b.get();
    for (size_t i = 0; i < ra.size(); ++i)
      fused = fused && std::fabs(f(ra[i]) - f(rb[i])) <= 1e-2 * std::max(1.0, std::fabs(f(rb[i])));
  }
  const bool ok = trip < 2e-2 && dots < 2e-2 && (kind != 0 || e_rot < e_plain) && std::isfinite(e_rot) && fused;
  g_failures += !ok;
  std::printf("rotated cache (%s): round trip %.2e, dot products %.2e; NVFP4 error of the centred scores plain %.4f, "
              "rotated %.4f; fused launches %s %s\n", names[kind], trip,
              dots, e_plain, e_rot, fused ? "match" : "differ", ok ? "ok" : "FAIL");
}

void check_verify(int mode, std::mt19937& rng) {
  blz_mha_set_verify(mode);
  std::printf("-- verify kernel %d\n", mode);
  if (mode >= 5) {
    const bool ok = blz_nvfp4_mha_verify_tiles(12, 2, 3) == 0 &&
                    blz_nvfp4_mha_verify_tiles(12, 2, 8) == 0 &&
                    blz_nvfp4_mha_verify_tiles(12, 2, 9) == 1;
    g_failures += !ok;
    std::printf("hybrid geometry selection: %s\n", ok ? "ok" : "FAIL");
  }
  test_case("verify 8 tokens, GQA 6, lengths 8 / 2500", 2, 6, 8, {8, 2500}, 8, rng);
  test_case("verify 9 tokens, GQA 6, lengths 9 / 33 / 4100", 2, 6, 9, {9, 33, 4100}, 12, rng);
  test_case("verify 9 tokens, GQA 8, lengths 70 / 3001", 2, 8, 9, {70, 3001}, 7, rng);
  test_case("verify 9, GQA 6, empty splits / tile tails", 1, 6, 9, {9, 33, 79}, 64, rng);
  test_case("verify 16 tokens, GQA 6 (two tiles), 40 / 2222", 2, 6, 16, {40, 2222}, 5, rng);
  test_case("verify 3 tokens, GQA 8, lengths 40 / 1200", 1, 8, 3, {40, 1200}, 5, rng);
  test_case("chunk 37 tokens, GQA 12, lengths 37 / 1500", 2, 12, 37, {37, 1500}, 3, rng);
  test_case("verify 2 tokens, GQA 16, splits past the context", 1, 16, 2, {5, 900}, 64, rng);
}

}  // namespace

int main() {
  std::mt19937 rng(11);
  if (std::getenv("BLAZE_MHA_HYBRID_CHECK")) {
    for (int mode : {5, 6}) check_verify(mode, rng);
    std::printf("%s\n", g_failures ? "FAILED" : "all passed");
    return g_failures ? 1 : 0;
  }
  if (std::getenv("BLAZE_MHA_TQ3_CHECK")) {
    test_case("decode, GQA 6, lengths 1 / 37 / 3000", 2, 6, 1, {1, 37, 3000}, 16, rng, BLZ_KV_TQ3);
    test_case("verify 8 tokens, GQA 6, lengths 8 / 2500", 2, 6, 8, {8, 2500}, 8, rng, BLZ_KV_TQ3);
    test_case("verify 9 tokens, GQA 6, lengths 9 / 33 / 4100", 2, 6, 9, {9, 33, 4100}, 12, rng, BLZ_KV_TQ3);
    test_case("verify 3 tokens, GQA 8, lengths 40 / 1200", 1, 8, 3, {40, 1200}, 5, rng, BLZ_KV_TQ3);
    test_case("decode, GQA 16, length 777", 1, 16, 1, {777}, 3, rng, BLZ_KV_TQ3);
    test_store_tq3(rng);
    blz_mha_set_f16_accumulate(1);
    test_case("verify 8 tokens, GQA 6, FP16 accumulation", 2, 6, 8, {8, 2500}, 8, rng, BLZ_KV_TQ3);
    blz_mha_set_f16_accumulate(0);
    std::printf("%s\n", g_failures ? "FAILED" : "all passed");
    return g_failures ? 1 : 0;
  }
  if (std::getenv("BLAZE_MHA_ROTATION_CHECK")) {
    for (int kind = 0; kind < 4; ++kind) test_rotation(rng, kind);
    std::printf("%s\n", g_failures ? "FAILED" : "all passed");
    return g_failures ? 1 : 0;
  }
  // only the verify cases of the listed kernels (BLAZE_MHA_VERIFY_MODES=3,4): a quick check of one kernel's change
  if (const char* modes = std::getenv("BLAZE_MHA_VERIFY_MODES")) {
    for (const char* m = modes; *m; ++m)
      if (*m >= '0' && *m <= '9') check_verify(*m - '0', rng);
    std::printf("%s\n", g_failures ? "FAILED" : "all passed");
    return g_failures ? 1 : 0;
  }
  // Performance measurements are opt-in: correctness tests do not allocate 524k pools or run timing loops.
  if (std::getenv("BLAZE_MHA_VERIFY_BENCH")) {
    for (int mode : {0, 1, 2, 3, 4, 5, 6}) {
      blz_mha_set_verify(mode);
      std::printf("-- verify kernel %d\n", mode);
      for (int ql : {3, 8, 9}) {
        bench(122000, 1, ql, auto_splits(1, ql), rng);
        bench(524000, 1, ql, auto_splits(1, ql), rng);
      }
      bench(524000, 2, 9, auto_splits(2, 9), rng);
    }
    return 0;
  }
  if (std::getenv("BLAZE_MHA_FORMAT_BENCH")) {
    blz_mha_set_verify(0);
    for (int f : {BLZ_KV_NVFP4, BLZ_KV_FP8, BLZ_KV_BF16}) {
      bench(122000, 1, 1, 64, rng, f);
      bench(122000, 1, 8, 64, rng, f);
    }
    return 0;
  }
  // Fixed-shape prefill measurements, runnable independently of the numerical/reference cases.
  if (std::getenv("BLAZE_MHA_PREFILL_BENCH")) {
    blz_mha_set_verify(0);  // measure the actual register baseline, independent of new verify dispatch
    for (int repeat = 0; repeat < 3; ++repeat) {
      bench(8192, 1, 2048, 1, rng);
      bench(32768, 1, 2048, 1, rng);
    }
    return 0;
  }
  {
    blz_mha_set_verify(0);
    bool child_ok = false;
    std::thread rank([&] {
      blz_mha_set_verify(4);
      child_ok = blz_nvfp4_mha_verify_tiles(12, 2, 3) == 1;
    });
    rank.join();
    const bool ok = child_ok && blz_nvfp4_mha_verify_tiles(12, 2, 3) == 0;
    std::printf("host thread launch-policy isolation: %s\n", ok ? "ok" : "FAIL");
    g_failures += !ok;
    blz_mha_set_verify(1);
  }
  {
    blz_nvfp4_pool pool{};
    pool.heads = 1, pool.dim = kDim, pool.format = BLZ_KV_NVFP4;
    const auto err = blz_nvfp4_mha_decode(nullptr, 0, 0, 0, pool, nullptr, 0, {nullptr, BLZ_I32},
                                         {nullptr, BLZ_I32}, 1, 1, 1, 1.0f, nullptr, nullptr,
                                         BLZ_BF16, nullptr, 0, 0, nullptr);
    const bool ok = err == cudaErrorNotSupported && blz_nvfp4_mha_parts(0, 1, 1, 1) == 0 &&
                    blz_nvfp4_mha_parts(-1, 1, 1, 1) == 0 && blz_nvfp4_mha_parts(12, 2, 1, 0) == 0 &&
                    blz_nvfp4_mha_parts(12, 2, INT_MAX, 1) == 1 &&
                    blz_nvfp4_mha_verify_tiles(0, 2, 9) == 0 &&
                    blz_nvfp4_mha_verify_tiles(-12, 2, 9) == 0 &&
                    blz_nvfp4_mha_verify_tiles(34, 2, 9) == 0 &&
                    blz_nvfp4_mha_verify_tiles(12, 2, INT_MAX) == 1 + (INT_MAX - 1) / 13;
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
  for (int f : {BLZ_KV_FP8, BLZ_KV_BF16, BLZ_KV_TQ3}) {
    test_case("decode, GQA 6, lengths 1 / 37 / 3000", 2, 6, 1, {1, 37, 3000}, 16, rng, f);
    test_case("verify 8 tokens, GQA 6, lengths 8 / 2500", 2, 6, 8, {8, 2500}, 8, rng, f);
    test_case("verify 3 tokens, GQA 8, lengths 40 / 1200", 1, 8, 3, {40, 1200}, 5, rng, f);
    test_case("decode, GQA 16, length 777", 1, 16, 1, {777}, 3, rng, f);
    if (f == BLZ_KV_TQ3) test_store_tq3(rng);
    else test_store_dense(f, rng);
  }
  test_window("decode, MHA 8 heads, lengths 1 / 300 / 9000", 8, 1, {1, 300, 9000}, 4096, 16, rng);
  test_window("verify 9 tokens, lengths 9 / 5000", 8, 9, {9, 5000}, 4096, 8, rng);
  test_window("extend 300 tokens, length 5000", 4, 300, {5000}, 4096, 1, rng);
  test_window("verify 5 tokens, short window", 4, 5, {700}, 37, 4, rng);
  test_window("decode, no window", 4, 1, {2000}, 0, 8, rng);
  test_window("DFlash block 8", 2, 8, {8, 3000}, 2048, 4, rng, 0);
  test_window("DFlash block 8, short window", 2, 8, {500}, 37, 2, rng, 0);
  // the verify shapes on every kernel: the register kernel, the shared-dequantization one in FP32 / FP16 accumulation,
  // its pair kernel in FP32 / FP16
  for (int mode : {0, 1, 2, 3, 4, 5, 6}) check_verify(mode, rng);
  blz_mha_set_verify(1);
  // the decode / verify kernels with FP16 accumulation
  blz_mha_set_f16_accumulate(1);
  std::printf("-- decode / verify with FP16 accumulation\n");
  test_case("decode, GQA 6, lengths 1 / 37 / 3000", 2, 6, 1, {1, 37, 3000}, 16, rng);
  test_case("verify 8 tokens, GQA 6, lengths 8 / 2500", 2, 6, 8, {8, 2500}, 8, rng);
  test_case("verify 3 tokens, GQA 8, lengths 40 / 1200", 1, 8, 3, {40, 1200}, 5, rng);
  test_case("decode, GQA 12, lengths 1 / 37 / 3000", 2, 12, 1, {1, 37, 3000}, 16, rng);
  for (int f : {BLZ_KV_FP8, BLZ_KV_BF16}) test_case("verify 8 tokens, GQA 6, lengths 8 / 2500", 2, 6, 8, {8, 2500}, 8, rng, f);
  test_window("verify 9 tokens, lengths 9 / 5000", 8, 9, {9, 5000}, 4096, 8, rng);
  test_window("DFlash block 8", 2, 8, {8, 3000}, 2048, 4, rng, 0);
  blz_mha_set_f16_accumulate(0);
  for (bool f16 : {false, true}) {
    test_prefill("GQA 6, 37 tokens, a first chunk", 2, 6, 37, 37, f16, rng);
    test_prefill("GQA 6, 300 tokens after 1700", 2, 6, 300, 2000, f16, rng);
    test_prefill("GQA 8 (two CTAs of 4), 65 tokens after 935", 1, 8, 65, 1000, f16, rng);
    test_prefill("GQA 12 (two CTAs of 6), 33 tokens after 467", 2, 12, 33, 500, f16, rng);
    test_prefill("MHA, 129 tokens after 2000", 3, 1, 129, 2129, f16, rng);
  }
  test_dequant_rows(rng);
  for (int kind = 0; kind < 4; ++kind) test_rotation(rng, kind);
  blz_mha_set_verify(1);
  std::printf("%s\n", g_failures ? "FAILED" : "all passed");
  return g_failures ? 1 : 0;
}
