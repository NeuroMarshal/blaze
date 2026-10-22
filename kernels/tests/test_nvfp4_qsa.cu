// Tests for the QSA NVFP4 readers against a CPU model of SGLang's Triton gathers, and for the sparse prefill attention
// straight from the NVFP4 pool against a double-precision reference.
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "blaze/kv.h"
#include "blaze/nvfp4.h"
#include "blaze/qsa.h"

#define CHECK_CUDA(x)                                                                              \
  do {                                                                                             \
    cudaError_t err__ = (x);                                                                       \
    if (err__ != cudaSuccess) {                                                                    \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(err__)); \
      std::exit(2);                                                                                \
    }                                                                                              \
  } while (0)

static int g_failures = 0;

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
  int slots, heads, dim;
  std::vector<uint8_t> kd, vd, ks, vs;
  float kg = 1.0f, vg = 0.5f;
  // dequantized value of (slot, head, d), bf16-rounded like the kernel's store
  float value(bool v, int64_t slot, int head, int d) const {
    const int64_t row = slot * heads + head;
    const auto& data = v ? vd : kd;
    const auto& scale = v ? vs : ks;
    const int code = (data[row * (dim / 2) + d / 2] >> (4 * (d & 1))) & 15;
    const float x = blaze::e2m1_value(code) * blaze::e4m3_to_float(scale[row * (dim / 16) + d / 16]) * (v ? vg : kg);
    return __bfloat162float(__float2bfloat16(x));
  }
};

static Pool make_pool(int slots, int heads, int dim, std::mt19937& rng) {
  Pool p{slots, heads, dim};
  std::normal_distribution<float> n01;
  p.kd.resize(size_t(slots) * heads * dim / 2), p.vd.resize(p.kd.size());
  p.ks.resize(size_t(slots) * heads * dim / 16), p.vs.resize(p.ks.size());
  std::vector<float> row(dim);
  for (int s = 0; s < slots * heads; ++s) {
    for (int side = 0; side < 2; ++side) {
      for (auto& x : row) x = n01(rng) * 3.0f;
      blaze::nvfp4_quantize_row_ref(row.data(), dim, side ? p.vg : p.kg, true,
                                  &(side ? p.vd : p.kd)[size_t(s) * dim / 2], &(side ? p.vs : p.ks)[size_t(s) * dim / 16]);
    }
  }
  return p;
}

template <typename ReqIdx, typename SeqIdx>
static void test_compact(int zero_fill_stride, std::mt19937& rng) {
  const int heads = 2, dim = 256, slots = 4096, reqs = 6, max_len = 600, batch = 4, topk = 37;
  Pool p = make_pool(slots, heads, dim, rng);
  std::vector<int32_t> r2t(size_t(reqs) * max_len);
  for (auto& s : r2t) s = int32_t(rng() % slots);
  std::vector<ReqIdx> req(batch);
  std::vector<SeqIdx> seq(batch);
  std::vector<int32_t> idx(size_t(batch) * topk, -1), cu(batch + 1, 0);
  for (int b = 0; b < batch; ++b) {
    req[b] = ReqIdx(rng() % reqs);
    seq[b] = SeqIdx(5 + rng() % (max_len - 5));
    const int nvalid = int(rng() % (topk + 1));
    for (int c = 0; c < nvalid; ++c) idx[b * topk + c] = int32_t(rng() % (seq[b] + 3));  // some >= seq_len
    cu[b + 1] = cu[b] + (zero_fill_stride ? zero_fill_stride : nvalid);
  }
  const int rows = cu[batch];
  Dev<uint8_t> kd(p.kd), vd(p.vd), ks(p.ks), vs(p.vs);
  Dev<float> kg(std::vector<float>{p.kg}), vg(std::vector<float>{p.vg});
  Dev<int32_t> dr2t(r2t), didx(idx), dcu(cu);
  Dev<ReqIdx> dreq(req);
  Dev<SeqIdx> dseq(seq);
  Dev<__nv_bfloat16> ok(size_t(rows) * heads * dim), ov(ok.n);
  const uint16_t sentinel = 0x7fc1;  // a NaN pattern: any unwritten value that is read shows up
  CHECK_CUDA(cudaMemset(ok.ptr, 0xff, ok.n * 2));
  CHECK_CUDA(cudaMemset(ov.ptr, 0xff, ov.n * 2));
  (void)sentinel;
  const blz_nvfp4_pool pool{kd.ptr, vd.ptr, ks.ptr, vs.ptr, kg.ptr, vg.ptr, heads, dim};
  const blz_index_t rt = sizeof(ReqIdx) == 8 ? BLZ_I64 : BLZ_I32, st = sizeof(SeqIdx) == 8 ? BLZ_I64 : BLZ_I32;
  CHECK_CUDA(blz_nvfp4_qsa_compact(pool, dr2t.ptr, max_len, {dreq.ptr, rt}, didx.ptr, topk, {dseq.ptr, st}, dcu.ptr,
                                   batch, topk, zero_fill_stride, nullptr, BLZ_BF16, ok.ptr, ov.ptr, nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());
  const auto gk = ok.get(), gv = ov.get();
  {
    // Decode reads straight from a pool in mapped host memory (zero-copy): identical rows.
    void* host[4];
    void* dev[4];
    const std::vector<uint8_t>* src[4] = {&p.kd, &p.vd, &p.ks, &p.vs};
    for (int i = 0; i < 4; ++i) {
      CHECK_CUDA(cudaHostAlloc(&host[i], src[i]->size(), cudaHostAllocMapped));
      std::memcpy(host[i], src[i]->data(), src[i]->size());
      CHECK_CUDA(cudaHostGetDevicePointer(&dev[i], host[i], 0));
    }
    const blz_nvfp4_pool hpool{static_cast<uint8_t*>(dev[0]), static_cast<uint8_t*>(dev[1]),
                               static_cast<uint8_t*>(dev[2]), static_cast<uint8_t*>(dev[3]), kg.ptr, vg.ptr, heads, dim};
    Dev<__nv_bfloat16> hk(ok.n), hv(ov.n);
    CHECK_CUDA(cudaMemset(hk.ptr, 0xff, hk.n * 2));
    CHECK_CUDA(cudaMemset(hv.ptr, 0xff, hv.n * 2));
    CHECK_CUDA(blz_nvfp4_qsa_compact(hpool, dr2t.ptr, max_len, {dreq.ptr, rt}, didx.ptr, topk, {dseq.ptr, st}, dcu.ptr,
                                     batch, topk, zero_fill_stride, nullptr, BLZ_BF16, hk.ptr, hv.ptr, nullptr));
    CHECK_CUDA(cudaDeviceSynchronize());
    const auto hgk = hk.get(), hgv = hv.get();
    if (std::memcmp(hgk.data(), gk.data(), gk.size() * 2) || std::memcmp(hgv.data(), gv.data(), gv.size() * 2)) {
      std::printf("qsa compact from a host pool differs FAIL\n");
      ++g_failures;
    }
    // Through a small row cache (many lines shared by several slots): the first pass fills it, the second reads it.
    const int lines = 64;
    std::vector<int32_t> tags(lines, -1);
    Dev<int32_t> ctag(tags);
    Dev<uint8_t> ckd(size_t(lines) * heads * dim / 2), cvd(ckd.n), cks(size_t(lines) * heads * dim / 16), cvs(cks.n);
    const blz_nvfp4_row_cache cache{ctag.ptr, ckd.ptr, cvd.ptr, cks.ptr, cvs.ptr, lines};
    for (int pass = 0; pass < 2; ++pass) {
      CHECK_CUDA(cudaMemset(hk.ptr, 0xff, hk.n * 2));
      CHECK_CUDA(cudaMemset(hv.ptr, 0xff, hv.n * 2));
      CHECK_CUDA(blz_nvfp4_qsa_compact(hpool, dr2t.ptr, max_len, {dreq.ptr, rt}, didx.ptr, topk, {dseq.ptr, st},
                                       dcu.ptr, batch, topk, zero_fill_stride, &cache, BLZ_BF16, hk.ptr, hv.ptr,
                                       nullptr));
      CHECK_CUDA(cudaDeviceSynchronize());
      const auto ck = hk.get(), cv = hv.get();
      if (std::memcmp(ck.data(), gk.data(), gk.size() * 2) || std::memcmp(cv.data(), gv.data(), gv.size() * 2)) {
        std::printf("qsa compact through the row cache (pass %d) differs FAIL\n", pass);
        ++g_failures;
      }
    }
    // Rewritten slots: invalidated lines must not serve the old rows.
    std::memset(host[0], 0x77, p.kd.size());
    std::vector<int64_t> all(slots);
    for (int i = 0; i < slots; ++i) all[i] = i;
    Dev<int64_t> dall(all);
    CHECK_CUDA(blz_nvfp4_row_cache_invalidate(cache, {dall.ptr, BLZ_I64}, slots, nullptr));
    Dev<__nv_bfloat16> rk(ok.n), rv(ov.n), ek(ok.n), ev(ov.n);
    CHECK_CUDA(blz_nvfp4_qsa_compact(hpool, dr2t.ptr, max_len, {dreq.ptr, rt}, didx.ptr, topk, {dseq.ptr, st}, dcu.ptr,
                                     batch, topk, zero_fill_stride, &cache, BLZ_BF16, rk.ptr, rv.ptr, nullptr));
    CHECK_CUDA(blz_nvfp4_qsa_compact(hpool, dr2t.ptr, max_len, {dreq.ptr, rt}, didx.ptr, topk, {dseq.ptr, st}, dcu.ptr,
                                     batch, topk, zero_fill_stride, nullptr, BLZ_BF16, ek.ptr, ev.ptr, nullptr));
    CHECK_CUDA(cudaDeviceSynchronize());
    const auto rkh = rk.get(), ekh = ek.get();
    if (std::memcmp(rkh.data(), ekh.data(), rkh.size() * 2)) {
      std::printf("qsa compact after invalidation serves stale rows FAIL\n");
      ++g_failures;
    }
    for (void* h : host) cudaFreeHost(h);
  }
  int bad = 0, checked = 0;
  for (int b = 0; b < batch; ++b) {
    const int valid_count = cu[b + 1] - cu[b];
    const int ncols = zero_fill_stride ? zero_fill_stride : topk;
    for (int c = 0; c < ncols; ++c) {
      const int64_t pos = c < topk ? idx[b * topk + c] : -1;
      const bool valid = c < valid_count && pos >= 0 && pos < int64_t(seq[b]);
      if (!valid && !zero_fill_stride) continue;
      const int64_t slot = valid ? r2t[size_t(req[b]) * max_len + pos] : 0;
      for (int h = 0; h < heads; ++h) {
        for (int d = 0; d < dim; ++d) {
          const size_t o = (size_t(cu[b] + c) * heads + h) * dim + d;
          const float ek = valid ? p.value(false, slot, h, d) : 0.0f;
          const float ev = valid ? p.value(true, slot, h, d) : 0.0f;
          bad += __bfloat162float(gk[o]) != ek || __bfloat162float(gv[o]) != ev;
          ++checked;
        }
      }
    }
  }
  if (bad) ++g_failures;
  std::printf("qsa compact zero_fill=%d req%zu seq%zu: %d of %d values differ %s\n", zero_fill_stride,
              sizeof(ReqIdx) * 8, sizeof(SeqIdx) * 8, bad, checked, bad ? "FAIL" : "ok");
}

static void test_pack(std::mt19937& rng) {
  const int heads = 2, dim = 256, slots = 3000, reqs = 5, max_len = 900, batch = 3;
  Pool p = make_pool(slots, heads, dim, rng);
  std::vector<int32_t> r2t(size_t(reqs) * max_len);
  for (auto& s : r2t) s = int32_t(rng() % slots);
  std::vector<int64_t> req(batch);
  std::vector<int32_t> cu(batch + 1, 0);
  int64_t max_k = 0;
  for (int b = 0; b < batch; ++b) {
    req[b] = int64_t(rng() % reqs);
    const int len = 1 + int(rng() % max_len);
    cu[b + 1] = cu[b] + len;
    max_k = std::max<int64_t>(max_k, len);
  }
  Dev<uint8_t> kd(p.kd), vd(p.vd), ks(p.ks), vs(p.vs);
  Dev<float> kg(std::vector<float>{p.kg}), vg(std::vector<float>{p.vg});
  Dev<int32_t> dr2t(r2t), dcu(cu);
  Dev<int64_t> dreq(req);
  Dev<__nv_bfloat16> ok(size_t(cu[batch]) * heads * dim), ov(ok.n);
  const blz_nvfp4_pool pool{kd.ptr, vd.ptr, ks.ptr, vs.ptr, kg.ptr, vg.ptr, heads, dim};
  CHECK_CUDA(blz_nvfp4_qsa_pack(pool, dr2t.ptr, max_len, 1, {dreq.ptr, BLZ_I64}, dcu.ptr, batch, max_k, BLZ_BF16,
                                ok.ptr, ov.ptr, nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());
  const auto gk = ok.get(), gv = ov.get();
  int bad = 0;
  for (int b = 0; b < batch; ++b)
    for (int t = 0; t < cu[b + 1] - cu[b]; ++t) {
      const int64_t slot = r2t[size_t(req[b]) * max_len + t];
      for (int h = 0; h < heads; ++h)
        for (int d = 0; d < dim; ++d) {
          const size_t o = (size_t(cu[b] + t) * heads + h) * dim + d;
          bad += __bfloat162float(gk[o]) != p.value(false, slot, h, d) ||
                 __bfloat162float(gv[o]) != p.value(true, slot, h, d);
        }
    }
  if (bad) ++g_failures;
  std::printf("qsa pack: %d rows, %d values differ %s\n", cu[batch], bad, bad ? "FAIL" : "ok");
}

static void bench_compact(int topk, bool host_pool, bool cached = false) {
  // Decode-shaped: one row, ~2051 selected positions of a 262K context, 2 heads x 256; the pool in device memory or
  // in mapped host memory (read over PCIe).
  const int heads = 2, dim = 256, slots = 262144, batch = 1, stride = (topk + 63) / 64 * 64;
  Dev<uint8_t> kd(host_pool ? 0 : size_t(slots) * heads * dim / 2), vd(kd.n),
      ks(host_pool ? 0 : size_t(slots) * heads * dim / 16), vs(ks.n);
  void* host[4] = {nullptr, nullptr, nullptr, nullptr};
  if (host_pool) {
    const size_t bytes[4] = {size_t(slots) * heads * dim / 2, size_t(slots) * heads * dim / 2,
                             size_t(slots) * heads * dim / 16, size_t(slots) * heads * dim / 16};
    uint8_t** dst[4] = {&kd.ptr, &vd.ptr, &ks.ptr, &vs.ptr};
    for (int i = 0; i < 4; ++i) {
      cudaFree(*dst[i]);
      CHECK_CUDA(cudaHostAlloc(&host[i], bytes[i], cudaHostAllocMapped));
      std::memset(host[i], i < 2 ? 0x11 * (i + 1) : 0x30, bytes[i]);
      CHECK_CUDA(cudaHostGetDevicePointer(reinterpret_cast<void**>(dst[i]), host[i], 0));
    }
  } else {
    CHECK_CUDA(cudaMemset(kd.ptr, 0x11, kd.n));
    CHECK_CUDA(cudaMemset(vd.ptr, 0x22, vd.n));
    CHECK_CUDA(cudaMemset(ks.ptr, 0x30, ks.n));
    CHECK_CUDA(cudaMemset(vs.ptr, 0x30, vs.n));
  }
  std::vector<int32_t> r2t(slots), idx(topk), cu{0, stride};
  for (int i = 0; i < slots; ++i) r2t[i] = (i * 7919) % slots;
  for (int i = 0; i < topk; ++i) idx[i] = int32_t((int64_t(i) * 127) % slots);
  Dev<int32_t> dr2t(r2t), didx(idx), dcu(cu), dseq(std::vector<int32_t>{slots});
  Dev<int64_t> dreq(std::vector<int64_t>{0});
  Dev<float> g(std::vector<float>{1.0f});
  Dev<__nv_bfloat16> ok(size_t(stride) * heads * dim), ov(ok.n);
  const blz_nvfp4_pool pool{kd.ptr, vd.ptr, ks.ptr, vs.ptr, g.ptr, g.ptr, heads, dim};
  const int lines = 32768;
  Dev<int32_t> ctag(std::vector<int32_t>(lines, -1));
  Dev<uint8_t> ckd(size_t(lines) * heads * dim / 2), cvd(ckd.n), cks(size_t(lines) * heads * dim / 16), cvs(cks.n);
  const blz_nvfp4_row_cache cache_v{ctag.ptr, ckd.ptr, cvd.ptr, cks.ptr, cvs.ptr, lines};
  const blz_nvfp4_row_cache* cache = cached ? &cache_v : nullptr;
  cudaEvent_t a, b;
  cudaEventCreate(&a), cudaEventCreate(&b);
  for (int w = 0; w < 5; ++w)
    blz_nvfp4_qsa_compact(pool, dr2t.ptr, slots, {dreq.ptr, BLZ_I64}, didx.ptr, topk, {dseq.ptr, BLZ_I32}, dcu.ptr,
                          batch, topk, stride, cache, BLZ_BF16, ok.ptr, ov.ptr, nullptr);
  const int iters = 500;
  cudaEventRecord(a);
  for (int i = 0; i < iters; ++i)
    blz_nvfp4_qsa_compact(pool, dr2t.ptr, slots, {dreq.ptr, BLZ_I64}, didx.ptr, topk, {dseq.ptr, BLZ_I32}, dcu.ptr,
                          batch, topk, stride, cache, BLZ_BF16, ok.ptr, ov.ptr, nullptr);
  cudaEventRecord(b);
  CHECK_CUDA(cudaEventSynchronize(b));
  float ms;
  cudaEventElapsedTime(&ms, a, b);
  std::printf("bench qsa compact topk=%d (262K pool in %s memory%s): %.2f us per layer\n", topk,
              host_pool ? "host" : "device", cached ? ", warm row cache" : "", 1000.0f * ms / iters);
  if (host_pool) {
    kd.ptr = vd.ptr = ks.ptr = vs.ptr = nullptr;  // not cudaMalloc'ed
    for (void* h : host) cudaFreeHost(h);
  }
}


// Sparse prefill attention from the NVFP4 pool vs exact attention (double) over the same dequantized K/V and the same
// bf16-rounded scaled q. Requests with prefixes; indices shared by the KV groups (2D) or per group (3D), with -1 and
// out-of-range entries and fewer valid entries than topk for early rows.
static void test_sparse_prefill(bool per_group, int topk, std::mt19937& rng) {
  const int hkv = 2, group = 12, hq = hkv * group, dim = 256, slots = 6000, reqs = 4, max_len = 2500, batch = 3;
  const float scale = 1.0f / 16.0f;
  Pool p = make_pool(slots, hkv, dim, rng);
  std::vector<int32_t> r2t(size_t(reqs) * max_len);
  for (auto& s : r2t) s = int32_t(rng() % slots);
  std::vector<int64_t> req(batch);
  std::vector<int32_t> cu_q(batch + 1, 0), kv(batch);
  for (int b = 0; b < batch; ++b) {
    req[b] = int64_t(rng() % reqs);
    const int q_len = 1 + int(rng() % 40);
    kv[b] = q_len + int(rng() % (max_len - 50));
    cu_q[b + 1] = cu_q[b] + q_len;
  }
  const int total_q = cu_q[batch];
  const int G = per_group ? hkv : 1;
  std::vector<int32_t> idx(size_t(total_q) * G * topk, -1);
  for (int b = 0; b < batch; ++b)
    for (int i = cu_q[b]; i < cu_q[b + 1]; ++i) {
      const int visible = (i - cu_q[b]) + kv[b] - (cu_q[b + 1] - cu_q[b]) + 1;
      for (int g = 0; g < G; ++g)
        for (int j = 0; j < std::min(topk, visible); ++j) {
          const int r = int(rng() % 20);
          idx[(size_t(i) * G + g) * topk + j] = r == 0 ? -1 : (r == 1 ? visible + int(rng() % 5) : int(rng() % visible));
        }
    }
  std::vector<__nv_bfloat16> q(size_t(total_q) * hq * dim);
  std::normal_distribution<float> n01;
  for (auto& x : q) x = __float2bfloat16(n01(rng) * 2.0f);
  Dev<uint8_t> kd(p.kd), vd(p.vd), ks(p.ks), vs(p.vs);
  Dev<float> kg(std::vector<float>{p.kg}), vg(std::vector<float>{p.vg});
  Dev<int32_t> dr2t(r2t), didx(idx), dcu(cu_q), dkv(kv);
  Dev<int64_t> dreq(req);
  Dev<__nv_bfloat16> dq(q), dout(q.size());
  const blz_nvfp4_pool pool{kd.ptr, vd.ptr, ks.ptr, vs.ptr, kg.ptr, vg.ptr, hkv, dim};
  CHECK_CUDA(blz_nvfp4_qsa_sparse_prefill(dq.ptr, int64_t(hq) * dim, dim, hq, pool, dr2t.ptr, max_len,
                                          {dreq.ptr, BLZ_I64}, didx.ptr, int64_t(G) * topk, per_group ? topk : 0, topk,
                                          dcu.ptr, dkv.ptr, nullptr, batch, total_q, scale, BLZ_BF16, dout.ptr,
                                          int64_t(hq) * dim, dim, nullptr));
  CHECK_CUDA(cudaDeviceSynchronize());
  const auto got = dout.get();

  // The same pool in mapped host memory, staged into a contiguous device pool: identical outputs.
  {
    void* host[4];
    const size_t bytes[4] = {p.kd.size(), p.vd.size(), p.ks.size(), p.vs.size()};
    const std::vector<uint8_t>* src[4] = {&p.kd, &p.vd, &p.ks, &p.vs};
    void* dev[4];
    for (int i = 0; i < 4; ++i) {
      CHECK_CUDA(cudaHostAlloc(&host[i], bytes[i], cudaHostAllocMapped));
      std::memcpy(host[i], src[i]->data(), bytes[i]);
      CHECK_CUDA(cudaHostGetDevicePointer(&dev[i], host[i], 0));
    }
    const blz_nvfp4_pool hpool{static_cast<uint8_t*>(dev[0]), static_cast<uint8_t*>(dev[1]),
                               static_cast<uint8_t*>(dev[2]), static_cast<uint8_t*>(dev[3]), kg.ptr, vg.ptr, hkv, dim};
    std::vector<int32_t> cu_k(batch + 1, 0);
    int max_k = 0;
    for (int b = 0; b < batch; ++b) cu_k[b + 1] = cu_k[b] + kv[b], max_k = std::max(max_k, kv[b]);
    const size_t rows = size_t(cu_k[batch]) * hkv;
    Dev<uint8_t> sk(rows * dim / 2), sv(sk.n), sks(rows * dim / 16), svs(sks.n);
    Dev<int32_t> dcuk(cu_k);
    Dev<__nv_bfloat16> dout2(q.size());
    const blz_nvfp4_pool spool{sk.ptr, sv.ptr, sks.ptr, svs.ptr, kg.ptr, vg.ptr, hkv, dim};
    CHECK_CUDA(blz_nvfp4_qsa_stage(hpool, dr2t.ptr, max_len, {dreq.ptr, BLZ_I64}, dcuk.ptr, batch, max_k, spool,
                                   nullptr));
    CHECK_CUDA(blz_nvfp4_qsa_sparse_prefill(dq.ptr, int64_t(hq) * dim, dim, hq, spool, nullptr, 0, {nullptr, BLZ_I64},
                                            didx.ptr, int64_t(G) * topk, per_group ? topk : 0, topk, dcu.ptr, dkv.ptr,
                                            dcuk.ptr, batch, total_q, scale, BLZ_BF16, dout2.ptr, int64_t(hq) * dim,
                                            dim, nullptr));
    CHECK_CUDA(cudaDeviceSynchronize());
    const auto got2 = dout2.get();
    const bool same = std::memcmp(got.data(), got2.data(), got.size() * sizeof(got[0])) == 0;
    std::printf("qsa sparse prefill from a host pool (staged): %s\n", same ? "identical ok" : "DIFFERENT FAIL");
    if (!same) ++g_failures;
    for (void* h : host) cudaFreeHost(h);
  }
  double worst = 0;
  int bad_rows = 0;
  std::vector<double> sc, acc(dim);
  for (int b = 0; b < batch; ++b)
    for (int i = cu_q[b]; i < cu_q[b + 1]; ++i) {
      const int visible = (i - cu_q[b]) + kv[b] - (cu_q[b + 1] - cu_q[b]) + 1;
      for (int h = 0; h < hq; ++h) {
        const int g = h / group;
        const int32_t* row = &idx[(size_t(i) * G + (per_group ? g : 0)) * topk];
        std::vector<int64_t> slot_of;
        for (int j = 0; j < std::min(topk, visible); ++j)
          if (row[j] >= 0 && row[j] < kv[b]) slot_of.push_back(r2t[size_t(req[b]) * max_len + row[j]]);
        sc.assign(slot_of.size(), 0.0);
        double mx = -INFINITY;
        for (size_t j = 0; j < slot_of.size(); ++j) {
          for (int d = 0; d < dim; ++d) {
            const float qs = __bfloat162float(__float2bfloat16(
                __bfloat162float(q[(size_t(i) * hq + h) * dim + d]) * scale * 1.4426950408f));
            sc[j] += double(qs) * p.value(false, slot_of[j], g, d);
          }
          mx = std::max(mx, sc[j]);
        }
        std::fill(acc.begin(), acc.end(), 0.0);
        double l = 0;
        for (size_t j = 0; j < slot_of.size(); ++j) {
          const double w = std::exp2(sc[j] - mx);
          l += w;
          for (int d = 0; d < dim; ++d) acc[d] += w * p.value(true, slot_of[j], g, d);
        }
        double err = 0, ref_max = 1e-6;
        for (int d = 0; d < dim; ++d) {
          const double ref = l > 0 ? acc[d] / l : 0.0;
          ref_max = std::max(ref_max, std::fabs(ref));
          err = std::max(err, std::fabs(ref - double(__bfloat162float(got[(size_t(i) * hq + h) * dim + d]))));
        }
        worst = std::max(worst, err / ref_max);
        bad_rows += err / ref_max > 2e-2 || !std::isfinite(err);
      }
    }
  if (bad_rows) ++g_failures;
  std::printf("qsa sparse prefill %s topk=%d: %d rows x %d heads, worst error %.2e of the row's max, %d bad %s\n",
              per_group ? "per-group" : "shared", topk, total_q, hq, worst, bad_rows, bad_rows ? "FAIL" : "ok");
}

// Prefill-chunk shape: 4096 rows, 24 q / 2 KV heads, top-2048 of a 500K-token context.
static void bench_sparse_prefill() {
  const int hkv = 2, hq = 24, dim = 256, total_q = 4096, topk = 2048, kv_len = 500000;
  Dev<uint8_t> kd(size_t(kv_len) * hkv * dim / 2), vd(kd.n), ks(size_t(kv_len) * hkv * dim / 16), vs(ks.n);
  CHECK_CUDA(cudaMemset(kd.ptr, 0x35, kd.n));
  CHECK_CUDA(cudaMemset(vd.ptr, 0x53, vd.n));
  CHECK_CUDA(cudaMemset(ks.ptr, 0x30, ks.n));
  CHECK_CUDA(cudaMemset(vs.ptr, 0x30, vs.n));
  std::vector<int32_t> r2t(kv_len), idx(size_t(total_q) * topk), cu{0, total_q}, kvl{kv_len};
  for (int t = 0; t < kv_len; ++t) r2t[t] = int32_t((int64_t(t) * 7919) % kv_len);
  std::mt19937 rng(3);
  for (int i = 0; i < total_q; ++i) {  // a recent window plus scattered older positions, like the indexer's picks
    const int pos = kv_len - total_q + i;
    for (int j = 0; j < topk; ++j) idx[size_t(i) * topk + j] = j < 512 ? pos - j : int32_t(rng() % (pos + 1));
  }
  Dev<int32_t> dr2t(r2t), didx(idx), dcu(cu), dkv(kvl);
  Dev<int64_t> dreq(std::vector<int64_t>{0});
  Dev<float> g(std::vector<float>{1.0f});
  Dev<__nv_bfloat16> dq(size_t(total_q) * hq * dim), dout(dq.n);
  CHECK_CUDA(cudaMemset(dq.ptr, 0, dq.n * 2));
  const blz_nvfp4_pool pool{kd.ptr, vd.ptr, ks.ptr, vs.ptr, g.ptr, g.ptr, hkv, dim};
  auto run = [&] {
    return blz_nvfp4_qsa_sparse_prefill(dq.ptr, int64_t(hq) * dim, dim, hq, pool, dr2t.ptr, kv_len, {dreq.ptr, BLZ_I64},
                                        didx.ptr, topk, 0, topk, dcu.ptr, dkv.ptr, nullptr, 1, total_q, 0.0625f, BLZ_BF16,
                                        dout.ptr, int64_t(hq) * dim, dim, nullptr);
  };
  CHECK_CUDA(run());
  cudaEvent_t a, b;
  cudaEventCreate(&a), cudaEventCreate(&b);
  const int iters = 5;
  cudaEventRecord(a);
  for (int i = 0; i < iters; ++i) CHECK_CUDA(run());
  cudaEventRecord(b);
  CHECK_CUDA(cudaEventSynchronize(b));
  float ms;
  cudaEventElapsedTime(&ms, a, b);
  std::printf("bench qsa sparse prefill: 4096 rows x top-2048, 500K context: %.2f ms per layer (the BF16 prefix copy it "
              "replaces would be %.0f MB per layer)\n", ms / iters, 2.0 * kv_len * hkv * dim * 2 / 1e6);
}

int main() {
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  std::printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  std::mt19937 rng(7);
  test_compact<int64_t, int32_t>(0, rng);
  test_compact<int64_t, int32_t>(64, rng);
  test_compact<int32_t, int64_t>(0, rng);
  test_compact<int32_t, int64_t>(128, rng);
  test_pack(rng);
  test_sparse_prefill(false, 100, rng);
  test_sparse_prefill(true, 77, rng);
  test_sparse_prefill(true, 2048, rng);
  bench_compact(2051, false);
  bench_compact(2051, true);
  bench_compact(2051, true, true);
  bench_sparse_prefill();
  std::printf(g_failures ? "FAILED\n" : "PASSED\n");
  return g_failures ? 1 : 0;
}
