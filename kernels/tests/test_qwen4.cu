// Qwen4-Exp kernels (blaze/qwen4.h) against CPU references written from SGLang's qwen4_exp.py / hyperconnection.py.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "blaze/qwen4.h"

namespace {

void check(cudaError_t e, const char* w) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
    std::exit(1);
  }
}

uint16_t bf(float v) { return __bfloat16_as_ushort(__float2bfloat16_rn(v)); }
float fb(uint16_t v) { return __bfloat162float(__ushort_as_bfloat16(v)); }
float rb(float v) { return fb(bf(v)); }

template <typename T>
T* dev(const std::vector<T>& h) {
  T* d;
  check(cudaMalloc(&d, h.size() * sizeof(T) + 16), "malloc");
  check(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "copy");
  return d;
}

template <typename T>
std::vector<T> host(const T* d, size_t n) {
  std::vector<T> h(n);
  check(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "copy back");
  return h;
}

std::vector<uint16_t> randbf(size_t n, std::mt19937& rng, float sd = 1.0f) {
  std::normal_distribution<float> d(0.0f, sd);
  std::vector<uint16_t> v(n);
  for (auto& x : v) x = bf(d(rng));
  return v;
}

double max_err(const std::vector<uint16_t>& a, const std::vector<float>& ref) {
  double e = 0, top = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    e = std::fmax(e, std::fabs(fb(a[i]) - ref[i]));
    top = std::fmax(top, std::fabs(ref[i]));
  }
  return e / std::fmax(top, 1e-30);
}

bool report(const char* what, double e, double tol) {
  std::printf("%-22s max rel err %.2e %s\n", what, e, e <= tol ? "ok" : "FAIL");
  return e <= tol;
}

}  // namespace

int main() {
  std::mt19937 rng(5);
  bool ok = true;
  const int H = 2560, HC = 4, rows = 3;
  const float eps = 1e-6f;

  // grouped norm
  {
    auto x = randbf(size_t(rows) * HC * H, rng), w = randbf(size_t(HC) * H, rng, 0.1f);
    std::vector<float> ref(x.size());
    for (int r = 0; r < rows; ++r)
      for (int g = 0; g < HC; ++g) {
        double ss = 0;
        const size_t b = (size_t(r) * HC + g) * H;
        for (int i = 0; i < H; ++i) ss += double(fb(x[b + i])) * fb(x[b + i]);
        const float inv = 1.0f / std::sqrt(float(ss / H) + eps);
        for (int i = 0; i < H; ++i) ref[b + i] = fb(x[b + i]) * inv * (1.0f + fb(w[size_t(g) * H + i]));
      }
    uint16_t *dx = dev(x), *dw = dev(w), *dout;
    check(cudaMalloc(&dout, x.size() * 2), "malloc");
    check(blz_grouped_rmsnorm(dx, dw, eps, rows, HC, H, dout, 0), "grouped norm");
    ok &= report("grouped rmsnorm", max_err(host(dout, x.size()), ref), 8e-3);
  }

  // hc combine
  {
    auto st = randbf(size_t(rows) * HC * H, rng), nm = randbf(size_t(rows) * HC * H, rng);
    auto inj = randbf(size_t(HC) * HC * H, rng, 0.01f), y = randbf(size_t(rows) * H, rng);
    std::vector<float> ref(st.size());
    for (int r = 0; r < rows; ++r) {
      float a[HC];
      for (int c = 0; c < HC; ++c) {
        double d = 0;
        for (int i = 0; i < HC * H; ++i) d += double(fb(nm[size_t(r) * HC * H + i])) * fb(inj[size_t(c) * HC * H + i]);
        a[c] = 2.0f / (1.0f + std::exp(-float(d) / HC));
      }
      for (int i = 0; i < HC * H; ++i)
        ref[size_t(r) * HC * H + i] = fb(st[size_t(r) * HC * H + i]) + a[i / H] * fb(y[size_t(r) * H + i % H]);
    }
    uint16_t *ds = dev(st), *dn = dev(nm), *di = dev(inj), *dy = dev(y);
    check(blz_hc_combine(ds, dn, di, dy, H, rows, HC, H, 0), "hc combine");
    ok &= report("hc combine", max_err(host(ds, st.size()), ref), 8e-3);
  }

  // n-gram hash: two sequences, EOS inside, history carried over
  {
    const int64_t eos = 248044;
    const std::vector<int64_t> mult = {123456789123LL, 98765432101LL, 55555555555LL};
    std::vector<int64_t> sizes(16), offs(16);
    int64_t total = 0;
    for (int h = 0; h < 16; ++h) sizes[h] = 20000003 + 2 * h, offs[h] = total, total += sizes[h];
    std::vector<int64_t> tokens = {11, 12, eos, 13, 14, 15, 7, eos, eos, 9};
    std::vector<int32_t> seq_start = {0, 6, 10}, slots = {1, 0};
    std::vector<int64_t> hist = {300, 301, eos, eos};  // slot 0: history (300, 301); slot 1: a new sequence
    // reference: window = history ++ tokens
    std::vector<int64_t> ref;
    for (int s = 0; s < 2; ++s) {
      std::vector<int64_t> win = {hist[slots[s] * 2], hist[slots[s] * 2 + 1]};
      for (int j = seq_start[s]; j < seq_start[s + 1]; ++j) win.push_back(tokens[j]);
      for (size_t i = 2; i < win.size(); ++i) {
        const int64_t w0 = win[i - 2], w1 = win[i - 1], w2 = win[i];
        const int reach = w1 == eos ? 0 : (w0 == eos ? 1 : 2);
        const uint64_t t0 = w2, t1 = reach >= 1 ? w1 : eos, t2 = reach >= 2 ? w0 : eos;
        const uint64_t m2 = (t0 * uint64_t(mult[0])) ^ (t1 * uint64_t(mult[1])), m3 = m2 ^ (t2 * uint64_t(mult[2]));
        for (int h = 0; h < 16; ++h) {
          const int64_t mix = int64_t(h < 8 ? m2 : m3);
          int64_t r = mix % sizes[h];
          if (r < 0) r += sizes[h];
          ref.push_back(r + offs[h]);
        }
      }
    }
    int64_t *dt = dev(tokens), *dh = dev(hist), *dm = dev(mult), *dsz = dev(sizes), *dof = dev(offs), *dids;
    int32_t *dss = dev(seq_start), *dsl = dev(slots);
    check(cudaMalloc(&dids, ref.size() * 8), "malloc");
    check(blz_ngram_hash(dt, dss, dsl, 2, 16, dh, 1, dm, dsz, dof, 8, eos, dids, 0), "hash");
    auto got = host(dids, ref.size());
    auto h2 = host(dh, 4);
    int bad = 0;
    for (size_t i = 0; i < ref.size(); ++i) bad += got[i] != ref[i];
    // slot 1 took sequence 0 (ends ..., 14, 15), slot 0 sequence 1 (ends eos, 9)
    const bool hist_ok = h2[2] == 14 && h2[3] == 15 && h2[0] == eos && h2[1] == 9;
    std::printf("%-22s %d of %zu ids differ, history %s\n", "n-gram hash", bad, ref.size(), hist_ok ? "ok" : "FAIL");
    ok &= bad == 0 && hist_ok;
  }

  // PLE gate + conv over two chunks (state carried)
  {
    const int C = HC * H, T = 5;
    auto kn = randbf(size_t(T) * C, rng), qn = randbf(size_t(T) * C, rng), val = randbf(size_t(T) * H, rng);
    std::vector<float> gref(size_t(T) * C);
    for (int t = 0; t < T; ++t)
      for (int c = 0; c < HC; ++c) {
        double d = 0;
        for (int i = 0; i < H; ++i) d += double(fb(kn[(size_t(t) * HC + c) * H + i])) * fb(qn[(size_t(t) * HC + c) * H + i]);
        const float x = float(d) / std::sqrt(float(H));
        const float sq = std::copysign(std::sqrt(std::fmax(std::fabs(x), 1e-6f)), x);
        const float g = 1.0f / (1.0f + std::exp(-sq));
        for (int i = 0; i < H; ++i) gref[(size_t(t) * HC + c) * H + i] = g * fb(val[size_t(t) * H + i]);
      }
    uint16_t *dk = dev(kn), *dq = dev(qn), *dv = dev(val), *dg;
    check(cudaMalloc(&dg, gref.size() * 2), "malloc");
    check(blz_ple_gate(dk, dq, dv, T, HC, H, dg, 0), "gate");
    auto gated = host(dg, gref.size());
    ok &= report("ple gate", max_err(gated, gref), 8e-3);

    auto gn = randbf(size_t(T) * C, rng), w = randbf(size_t(C) * 4, rng, 0.3f), streams = randbf(size_t(T) * C, rng);
    std::vector<uint16_t> state(size_t(2) * C * 9);
    for (auto& s : state) s = bf(0.0f);
    // reference over the whole 5-token sequence in slot 1, starting from zero state
    std::vector<float> sref(streams.size());
    for (int c = 0; c < C; ++c) {
      float hbuf[9] = {0};
      for (int t = 0; t < T; ++t) {
        const size_t idx = size_t(t) * C + c;
        const float x = fb(gn[idx]);
        float acc = fb(w[size_t(c) * 4 + 3]) * x;
        for (int j = 0; j < 3; ++j) acc += fb(w[size_t(c) * 4 + j]) * hbuf[j * 3];
        const float cv = rb(acc), sl = rb(cv / (1.0f + std::exp(-cv))), o = rb(fb(gated[idx]) + sl);
        sref[idx] = fb(streams[idx]) + o;
        for (int j = 0; j < 8; ++j) hbuf[j] = hbuf[j + 1];
        hbuf[8] = x;
      }
    }
    uint16_t *dgn = dev(gn), *dw = dev(w), *dst = dev(state), *dsm = dev(streams);
    // two chunks: tokens 0..2, then 3..4, same sequence (slot 1)
    std::vector<int32_t> s1 = {0, 3}, s2 = {0, 2}, sl = {1};
    int32_t *ds1 = dev(s1), *ds2 = dev(s2), *dsl = dev(sl);
    check(blz_ple_conv(dgn, dg, ds1, dsl, 1, C, 4, 3, dw, dst, 1, dsm, 0), "conv 1");
    check(blz_ple_conv(dgn + size_t(3) * C, dg + size_t(3) * C, ds2, dsl, 1, C, 4, 3, dw, dst, 1, dsm + size_t(3) * C, 0),
          "conv 2");
    ok &= report("ple conv (2 chunks)", max_err(host(dsm, streams.size()), sref), 8e-3);
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
