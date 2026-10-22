// blaze/ops.h against fp64/fp32 CPU references written from the Hugging Face Qwen3.5 implementation.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "blaze/ops.h"

#define CHECK_CUDA(x)                                                                    \
  do {                                                                                   \
    cudaError_t e_ = (x);                                                                \
    if (e_ != cudaSuccess) {                                                             \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      std::exit(1);                                                                      \
    }                                                                                    \
  } while (0)

static int g_fail = 0;
#define EXPECT(cond, ...)               \
  do {                                  \
    if (!(cond)) {                      \
      std::printf("FAIL: " __VA_ARGS__); \
      std::printf("\n");                \
      ++g_fail;                         \
    }                                   \
  } while (0)

static float bf(uint16_t b) {
  uint32_t u = uint32_t(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
static uint16_t to_bf(float f) {
  __nv_bfloat16 h = __float2bfloat16_rn(f);
  uint16_t b;
  std::memcpy(&b, &h, 2);
  return b;
}
static float rbf(float f) { return bf(to_bf(f)); }
static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
static float silu(float x) { return x * sigmoid(x); }

template <typename T>
struct Dev {
  T* p = nullptr;
  size_t n = 0;
  explicit Dev(size_t n_) : n(n_) { CHECK_CUDA(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T))); }
  Dev(const std::vector<T>& h) : Dev(h.size()) { up(h); }
  ~Dev() { cudaFree(p); }
  void up(const std::vector<T>& h) { CHECK_CUDA(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); }
  std::vector<T> down() const {
    std::vector<T> h(n);
    CHECK_CUDA(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
  }
};

static std::mt19937 rng(7);
static std::vector<uint16_t> rand_bf(size_t n, float scale) {
  std::normal_distribution<float> d(0.0f, scale);
  std::vector<uint16_t> v(n);
  for (auto& x : v) x = to_bf(d(rng));
  return v;
}

static double max_rel(const std::vector<uint16_t>& got, const std::vector<float>& ref) {
  double mx = 0, ma = 1e-30;
  for (size_t i = 0; i < ref.size(); ++i) mx = std::max(mx, double(std::fabs(bf(got[i]) - ref[i]))), ma = std::max(ma, double(std::fabs(ref[i])));
  return mx / ma;
}

static void test_rmsnorm() {
  const int rows = 5, dim = 2048;
  auto x = rand_bf(rows * dim, 3.0f), res = rand_bf(rows * dim, 3.0f), w = rand_bf(dim, 0.1f);
  Dev<uint16_t> dx(x), dres(res), dw(w), dout(rows * dim);
  CHECK_CUDA(blz_rmsnorm(dx.p, dres.p, dw.p, 1e-6f, rows, dim, 1, dout.p, 0));
  std::vector<float> ref(rows * dim);
  std::vector<float> resref(rows * dim);
  for (int r = 0; r < rows; ++r) {
    double ss = 0;
    std::vector<float> h(dim);
    for (int i = 0; i < dim; ++i) h[i] = rbf(bf(x[r * dim + i]) + bf(res[r * dim + i])), ss += double(h[i]) * h[i], resref[r * dim + i] = h[i];
    const double inv = 1.0 / std::sqrt(ss / dim + 1e-6);
    for (int i = 0; i < dim; ++i) ref[r * dim + i] = float(h[i] * inv * (1.0 + bf(w[i])));
  }
  const double e = max_rel(dout.down(), ref), er = max_rel(dres.down(), resref);
  EXPECT(e < 8e-3 && er == 0, "rmsnorm: rel err %.2e, residual %.2e", e, er);
  std::printf("rmsnorm (residual, zero-centered): rel err %.2e\n", e);
}

static void test_attn_prep() {
  const int T = 3, H = 4, KVH = 2, D = 256, RD = 64;
  auto qg = rand_bf(T * H * 2 * D, 2.0f), k = rand_bf(T * KVH * D, 2.0f), qn = rand_bf(D, 0.1f), kn = rand_bf(D, 0.1f);
  std::vector<int64_t> pos = {0, 7, 123456};
  std::vector<float> inv(RD / 2);
  for (int i = 0; i < RD / 2; ++i) inv[i] = float(1.0 / std::pow(1e7, 2.0 * i / RD));
  Dev<uint16_t> dqg(qg), dk(k), dqn(qn), dkn(kn), dq(T * H * D), dg(T * H * D);
  Dev<int64_t> dpos(pos);
  Dev<float> dinv(inv);
  CHECK_CUDA(blz_attn_prep(dqg.p, dk.p, dpos.p, dqn.p, dkn.p, 1e-6f, 1, T, H, KVH, D, RD, dinv.p, 1.0f, dq.p, dg.p, 0));
  auto ref_row = [&](const uint16_t* src, const std::vector<uint16_t>& nw, int64_t p, std::vector<float>& out) {
    double ss = 0;
    for (int i = 0; i < D; ++i) ss += double(bf(src[i])) * bf(src[i]);
    const double invn = 1.0 / std::sqrt(ss / D + 1e-6);
    std::vector<float> n(D);
    for (int i = 0; i < D; ++i) n[i] = rbf(float(bf(src[i]) * invn * (1.0 + bf(nw[i]))));
    out.assign(n.begin(), n.end());
    for (int i = 0; i < RD / 2; ++i) {
      const float f = float(p) * inv[i];
      const float c = std::cos(f), s = std::sin(f);
      out[i] = n[i] * c - n[i + RD / 2] * s;
      out[i + RD / 2] = n[i + RD / 2] * c + n[i] * s;
    }
  };
  const auto gq = dq.down(), gk = dk.down(), gg = dg.down();
  double eq = 0, ek = 0;
  int gate_bad = 0;
  for (int t = 0; t < T; ++t) {
    for (int h = 0; h < H; ++h) {
      std::vector<float> r;
      ref_row(&qg[(t * H + h) * 2 * D], qn, pos[t], r);
      for (int i = 0; i < D; ++i) eq = std::max(eq, double(std::fabs(bf(gq[(t * H + h) * D + i]) - r[i])));
      for (int i = 0; i < D; ++i) gate_bad += gg[(t * H + h) * D + i] != qg[(t * H + h) * 2 * D + D + i];
    }
    for (int h = 0; h < KVH; ++h) {
      std::vector<float> r;
      ref_row(&k[(t * KVH + h) * D], kn, pos[t], r);
      for (int i = 0; i < D; ++i) ek = std::max(ek, double(std::fabs(bf(gk[(t * KVH + h) * D + i]) - r[i])));
    }
  }
  EXPECT(eq < 0.05 && ek < 0.05 && gate_bad == 0, "attn_prep: q err %.3e, k err %.3e, gate mismatches %d", eq, ek, gate_bad);
  std::printf("attn_prep (qk norm, rope 64/256 incl. position 123456): max abs err q %.2e, k %.2e\n", eq, ek);
}

static void test_router() {
  const int T = 9, E = 256, K = 8;
  auto logits = rand_bf(T * E, 2.0f);
  Dev<uint16_t> dl(logits);
  Dev<int32_t> dids(T * K);
  Dev<float> dw(T * K);
  CHECK_CUDA(blz_router_topk(dl.p, T, E, K, 1, dids.p, dw.p, 0));
  const auto ids = dids.down();
  const auto w = dw.down();
  int bad = 0;
  double werr = 0;
  for (int t = 0; t < T; ++t) {
    std::vector<double> p(E);
    double mx = -1e30, sum = 0;
    for (int e = 0; e < E; ++e) mx = std::max(mx, double(bf(logits[t * E + e])));
    for (int e = 0; e < E; ++e) p[e] = std::exp(bf(logits[t * E + e]) - mx), sum += p[e];
    std::vector<int> order(E);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return p[a] > p[b]; });
    double top = 0;
    for (int r = 0; r < K; ++r) top += p[order[r]];
    for (int r = 0; r < K; ++r) {
      bad += ids[t * K + r] != order[r];
      werr = std::max(werr, std::fabs(w[t * K + r] - p[order[r]] / top));
    }
  }
  EXPECT(bad == 0 && werr < 1e-5, "router: %d wrong ids, weight err %.2e", bad, werr);
  std::printf("router softmax top-8 of 256: %d wrong ids, weight err %.2e\n", bad, werr);
}

// Reference GDN layer core: conv (with state) + gating + recurrent delta rule, per HF formulas.
static void test_gdn() {
  const int KH = 2, VH = 4, DK = 128, DV = 64, KER = 4;
  const int C = 2 * KH * DK + VH * DV;
  const std::vector<int32_t> starts = {0, 5, 6, 13};  // three sequences: 5, 1, 7 tokens
  const std::vector<int32_t> slots = {2, 0, 1};
  const int S = 3, T = starts.back(), NSLOT = 3;
  auto x = rand_bf(T * C, 1.0f), wconv = rand_bf(C * KER, 0.5f);
  auto conv_state = rand_bf(NSLOT * C * (KER - 1), 1.0f);
  auto a = rand_bf(T * VH, 1.0f), b = rand_bf(T * VH, 1.0f), alog = rand_bf(VH, 0.5f), dtb = rand_bf(VH, 0.5f);
  std::normal_distribution<float> nd(0.0f, 0.05f);
  std::vector<float> state(NSLOT * VH * DK * DV);
  for (auto& v : state) v = nd(rng);

  Dev<uint16_t> dx(x), dw(wconv), dcs(conv_state), dconv(T * C), da(a), db(b), dal(alog), ddt(dtb), dout(T * VH * DV);
  Dev<int32_t> dst(starts), dsl(slots);
  Dev<float> dg(T * VH), dbeta(T * VH), dstate(state);
  CHECK_CUDA(blz_gdn_conv(dx.p, C, dst.p, dsl.p, S, C, KER, dw.p, dcs.p, dconv.p, 0));
  CHECK_CUDA(blz_gdn_gate(da.p, VH, db.p, VH, dal.p, ddt.p, T, VH, dg.p, dbeta.p, 0));
  CHECK_CUDA(blz_gdn_recurrent(dconv.p, dg.p, dbeta.p, dst.p, dsl.p, S, T, KH, VH, DK, DV, dstate.p, dout.p, 0));

  // reference
  std::vector<float> conv(T * C), cs(conv_state.size());
  for (size_t i = 0; i < cs.size(); ++i) cs[i] = bf(conv_state[i]);
  for (int s = 0; s < S; ++s) {
    for (int c = 0; c < C; ++c) {
      std::vector<float> hist(cs.begin() + (slots[s] * C + c) * (KER - 1), cs.begin() + (slots[s] * C + c + 1) * (KER - 1));
      for (int t = starts[s]; t < starts[s + 1]; ++t) {
        const float xt = bf(x[t * C + c]);
        float acc = bf(wconv[c * KER + KER - 1]) * xt;
        for (int j = 0; j < KER - 1; ++j) acc += bf(wconv[c * KER + j]) * hist[j];
        conv[t * C + c] = rbf(silu(rbf(acc)));
        hist.erase(hist.begin());
        hist.push_back(xt);
      }
    }
  }
  std::vector<float> ref(T * VH * DV);
  std::vector<double> st(state.begin(), state.end());
  for (int s = 0; s < S; ++s) {
    for (int t = starts[s]; t < starts[s + 1]; ++t) {
      for (int h = 0; h < VH; ++h) {
        const int kh = h / (VH / KH);
        const float* q = &conv[t * C + kh * DK];
        const float* k = &conv[t * C + KH * DK + kh * DK];
        const float* v = &conv[t * C + 2 * KH * DK + h * DV];
        double qq = 0, kk = 0;
        for (int i = 0; i < DK; ++i) qq += double(q[i]) * q[i], kk += double(k[i]) * k[i];
        const double iq = 1.0 / std::sqrt(qq + 1e-6) / std::sqrt(double(DK)), ik = 1.0 / std::sqrt(kk + 1e-6);
        const double xa = bf(a[t * VH + h]) + bf(dtb[h]);
        const double g = -std::exp(double(bf(alog[h]))) * std::log1p(std::exp(xa));
        const double beta = rbf(sigmoid(bf(b[t * VH + h])));
        double* S2 = &st[(size_t(slots[s]) * VH + h) * DK * DV];
        for (int i = 0; i < DK * DV; ++i) S2[i] *= std::exp(g);
        for (int j = 0; j < DV; ++j) {
          double kv = 0;
          for (int i = 0; i < DK; ++i) kv += S2[i * DV + j] * k[i] * ik;
          const double delta = (v[j] - kv) * beta;
          for (int i = 0; i < DK; ++i) S2[i * DV + j] += k[i] * ik * delta;
        }
        for (int j = 0; j < DV; ++j) {
          double o = 0;
          for (int i = 0; i < DK; ++i) o += S2[i * DV + j] * q[i] * iq;
          ref[(t * VH + h) * DV + j] = float(o);
        }
      }
    }
  }
  const double econv = max_rel(dconv.down(), conv), eout = max_rel(dout.down(), ref);
  const auto gstate = dstate.down();
  double es = 0, ms = 1e-30;
  for (size_t i = 0; i < st.size(); ++i) es = std::max(es, std::fabs(gstate[i] - st[i])), ms = std::max(ms, std::fabs(st[i]));
  EXPECT(econv < 1e-2 && eout < 2e-2 && es / ms < 1e-3, "gdn: conv %.2e, out %.2e, state %.2e", econv, eout, es / ms);
  std::printf("gdn conv + gate + recurrent (3 sequences, GQA 2): rel err conv %.2e, out %.2e, state %.2e\n", econv, eout, es / ms);
}

static void test_gemm() {
  const int M = 7, N = 300, K = 2048;
  auto x = rand_bf(M * K, 1.0f), w = rand_bf(N * K, 0.05f);
  Dev<uint16_t> dx(x), dw(w), dout(M * N);
  blz_gemm* g = nullptr;
  EXPECT(blz_gemm_create(&g) == 0, "gemm create");
  CHECK_CUDA(blz_gemm_bf16(g, dx.p, K, dw.p, dout.p, N, M, N, K, 0));
  std::vector<float> ref(M * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double acc = 0;
      for (int k = 0; k < K; ++k) acc += double(bf(x[m * K + k])) * bf(w[n * K + k]);
      ref[m * N + n] = float(acc);
    }
  const double e = max_rel(dout.down(), ref);
  EXPECT(e < 1e-2, "gemm: rel err %.2e", e);
  std::printf("gemm bf16 7x300x2048: rel err %.2e\n", e);
  blz_gemm_destroy(g);
}

static void test_fp8_dequant() {
  const int N = 256, K = 384;
  std::vector<uint8_t> w(N * K);
  for (auto& v : w) v = uint8_t(rng() & 0x7e);  // finite positive and negative codes
  auto scale = rand_bf((N / 128) * (K / 128), 0.01f);
  Dev<uint8_t> dw(w);
  Dev<uint16_t> ds(scale), dout(N * K);
  CHECK_CUDA(blz_fp8_block_dequant(dw.p, ds.p, 0, N, K, 128, 128, dout.p, 0));
  const auto got = dout.down();
  int bad = 0;
  for (int r = 0; r < N; ++r)
    for (int c = 0; c < K; ++c) {
      __nv_fp8_e4m3 q;
      q.__x = w[r * K + c];
      bad += got[r * K + c] != to_bf(float(q) * bf(scale[(r / 128) * (K / 128) + c / 128]));
    }
  EXPECT(bad == 0, "fp8 dequant: %d mismatches", bad);
  std::printf("fp8 block dequant: %d mismatches\n", bad);
}

int main() {
  test_rmsnorm();
  test_attn_prep();
  test_router();
  test_gdn();
  test_gemm();
  test_fp8_dequant();
  std::printf(g_fail ? "FAILED (%d)\n" : "PASSED\n", g_fail);
  return g_fail ? 1 : 0;
}
