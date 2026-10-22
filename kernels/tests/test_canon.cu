// Canonical arithmetic: CPU == GPU bit for bit.
//   1. exp_c / silu_c / swiglu_bf16 over every finite BF16 input (and BF16 u values), plus exp_c accuracy
//   2. scale_sum and a16_row_output over random sums and exponents
//   3. A16 encoding exactness within 13 binades
//   4. whole expert: a naive GPU forward (the same canon functions in device code) == forward_ref
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "blaze/canon.h"
#include "blaze/expert.h"

#define CHECK_CUDA(x)                                                                              \
  do {                                                                                             \
    cudaError_t err__ = (x);                                                                       \
    if (err__ != cudaSuccess) {                                                                    \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(err__)); \
      std::exit(2);                                                                                \
    }                                                                                              \
  } while (0)

using namespace blaze;
static int g_fail = 0;

template <typename T>
static T* to_dev(const std::vector<T>& h) {
  T* d;
  CHECK_CUDA(cudaMalloc(&d, std::max<size_t>(1, h.size()) * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

template <typename T>
static std::vector<T> from_dev(const T* d, size_t n) {
  std::vector<T> h(n);
  CHECK_CUDA(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}

__global__ void k_silu(const uint16_t* g, const uint16_t* u, int ng, int nu, uint32_t* exp_out, uint32_t* silu_out,
                       uint16_t* sw_out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ng) return;
  const float gf = canon::bf16_to_f32(g[i]);
  exp_out[i] = canon::f32_bits(canon::exp_c(gf));
  silu_out[i] = canon::f32_bits(canon::silu_c(gf));
  for (int j = 0; j < nu; ++j) sw_out[size_t(i) * nu + j] = canon::swiglu_bf16(g[i], u[j]);
}

static void test_silu() {
  std::vector<uint16_t> g;
  for (uint32_t b = 0; b < 65536; ++b)
    if (!canon::bf16_nonfinite(uint16_t(b))) g.push_back(uint16_t(b));
  std::mt19937 rng(3);
  std::vector<uint16_t> u;
  for (int i = 0; i < 48; ++i) {
    uint16_t b;
    do b = uint16_t(rng()); while (canon::bf16_nonfinite(b));
    u.push_back(b);
  }
  for (uint16_t s : {0x3F80, 0xBF80, 0x0000, 0x8000, 0x7F7F, 0x0001}) u.push_back(s);
  const int ng = int(g.size()), nu = int(u.size());
  uint16_t *dg = to_dev(g), *du = to_dev(u);
  uint32_t *dexp, *dsilu;
  uint16_t* dsw;
  CHECK_CUDA(cudaMalloc(&dexp, ng * 4));
  CHECK_CUDA(cudaMalloc(&dsilu, ng * 4));
  CHECK_CUDA(cudaMalloc(&dsw, size_t(ng) * nu * 2));
  k_silu<<<(ng + 255) / 256, 256>>>(dg, du, ng, nu, dexp, dsilu, dsw);
  CHECK_CUDA(cudaGetLastError());
  const auto gexp = from_dev(dexp, ng), gsilu = from_dev(dsilu, ng);
  const auto gsw = from_dev(dsw, size_t(ng) * nu);
  int bad_exp = 0, bad_silu = 0, bad_sw = 0;
  double max_ulp = 0;
  for (int i = 0; i < ng; ++i) {
    const float gf = canon::bf16_to_f32(g[i]);
    const float e = canon::exp_c(gf);
    bad_exp += canon::f32_bits(e) != gexp[i];
    bad_silu += canon::f32_bits(canon::silu_c(gf)) != gsilu[i];
    for (int j = 0; j < nu; ++j) bad_sw += canon::swiglu_bf16(g[i], u[j]) != gsw[size_t(i) * nu + j];
    const double ref = std::exp(double(gf));
    if (std::isfinite(ref) && ref > 1.2e-38 && ref < 3.4e38) {
      const double ulp = std::ldexp(1.0, std::ilogb(ref) - 23);
      max_ulp = std::max(max_ulp, std::fabs(double(e) - ref) / ulp);
    }
  }
  std::printf("canon silu: %d inputs x %d u: exp mismatches %d, silu %d, swiglu %d; exp_c max error %.3f ulp\n", ng,
              nu, bad_exp, bad_silu, bad_sw, max_ulp);
  if (bad_exp || bad_silu || bad_sw || max_ulp > 2.0) ++g_fail;
  cudaFree(dg), cudaFree(du), cudaFree(dexp), cudaFree(dsilu), cudaFree(dsw);
}

__global__ void k_rowout(const int64_t* s, const int* emax, const float* m, int n, uint16_t* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = canon::a16_row_output(s[i], emax[i], m[i]);
}

static void test_row_output() {
  std::mt19937_64 rng(5);
  const int n = 1 << 20;
  std::vector<int64_t> s(n);
  std::vector<int> e(n);
  std::vector<float> m(n);
  std::uniform_int_distribution<int> bits(0, 55), ed(1, 254);
  std::uniform_real_distribution<float> md(-8.0f, 8.0f);
  for (int i = 0; i < n; ++i) {
    const int b = bits(rng);
    s[i] = int64_t(rng() >> (64 - b)) * ((rng() & 1) ? -1 : 1);
    e[i] = ed(rng);
    m[i] = std::ldexp(md(rng), int(rng() % 40) - 20);
  }
  int64_t* ds = to_dev(s);
  int* de = to_dev(e);
  float* dm = to_dev(m);
  uint16_t* dout;
  CHECK_CUDA(cudaMalloc(&dout, n * 2));
  k_rowout<<<(n + 255) / 256, 256>>>(ds, de, dm, n, dout);
  const auto got = from_dev(dout, n);
  int bad = 0;
  for (int i = 0; i < n; ++i) bad += canon::a16_row_output(s[i], e[i], m[i]) != got[i];
  std::printf("canon row output: %d random sums, %d mismatches\n", n, bad);
  if (bad) ++g_fail;
  cudaFree(ds), cudaFree(de), cudaFree(dm), cudaFree(dout);
}

static void test_a16_exact() {
  std::mt19937 rng(9);
  int bad = 0, total = 0;
  for (int trial = 0; trial < 2000; ++trial) {
    const int top = 1 + int(rng() % 250);
    std::vector<uint16_t> v(64);
    for (auto& b : v) {
      const int e = std::max(1, top - int(rng() % 14));  // within 13 binades of the column's largest
      b = uint16_t(((rng() & 1) << 15) | (uint32_t(std::min(e, 254)) << 7) | (rng() & 0x7F));
    }
    v[0] = uint16_t((uint32_t(top) << 7) | (rng() & 0x7F));
    const int emax = expert::a16_emax(v.data(), 64);
    for (uint16_t b : v) {
      const int32_t x = canon::a16_encode(b, emax);
      const double back = std::ldexp(double(x), emax - 147);
      bad += back != double(canon::bf16_to_f32(b)) || std::abs(x) >= (1 << 21);
      ++total;
    }
  }
  std::printf("canon a16 exactness: %d of %d values not exact\n", bad, total);
  if (bad) ++g_fail;
}

// Naive GPU forward: one thread per output row, same canon functions as forward_ref.
__global__ void k_gu_naive(expert::Geometry geo, const uint8_t* rec, expert::Multipliers m, const int32_t* x, int emax,
                           uint16_t* h) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= geo.inter) return;
  int64_t sg = 0, su = 0;
  for (int b = 0; b < geo.gu_blocks(); ++b) {
    int32_t pg = 0, pu = 0;
    for (int j = 0; j < 16; ++j) {
      const int k = 16 * b + j;
      pg += canon::c2(expert::unit_code(rec, geo.gu_blocks(), 2 * i, k)) * x[k];
      pu += canon::c2(expert::unit_code(rec, geo.gu_blocks(), 2 * i + 1, k)) * x[k];
    }
    sg += int64_t(pg) * canon::s_hat(expert::unit_scale(rec, geo.gu_blocks(), 2 * i, b));
    su += int64_t(pu) * canon::s_hat(expert::unit_scale(rec, geo.gu_blocks(), 2 * i + 1, b));
  }
  h[i] = canon::swiglu_bf16(canon::a16_row_output(sg, emax, m.gate), canon::a16_row_output(su, emax, m.up));
}

__global__ void k_down_naive(expert::Geometry geo, const uint8_t* dn, float m, const int32_t* x, int emax,
                             uint16_t* y) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= geo.hidden) return;
  int64_t s = 0;
  for (int b = 0; b < geo.d_blocks(); ++b) {
    int32_t p = 0;
    for (int j = 0; j < 16; ++j) p += canon::c2(expert::unit_code(dn, geo.d_blocks(), r, 16 * b + j)) * x[16 * b + j];
    s += int64_t(p) * canon::s_hat(expert::unit_scale(dn, geo.d_blocks(), r, b));
  }
  y[r] = canon::a16_row_output(s, emax, m);
}

// Random NVFP4 expert in row-major form, packed into a record. Scales spread over many E4M3 binades.
std::vector<uint8_t> random_record(const expert::Geometry& geo, std::mt19937& rng) {
  auto mat = [&](int rows, int k, std::vector<uint8_t>& c, std::vector<uint8_t>& s) {
    c.resize(size_t(rows) * k / 2), s.resize(size_t(rows) * k / 16);
    for (auto& b : c) b = uint8_t(rng());
    for (auto& b : s) b = uint8_t(0x20 + rng() % 0x40);  // E4M3 2^-4 .. 2^4
  };
  std::vector<uint8_t> gc, gs, uc, us, dc, ds;
  mat(geo.inter, geo.hidden, gc, gs), mat(geo.inter, geo.hidden, uc, us), mat(geo.hidden, geo.inter, dc, ds);
  std::vector<uint8_t> rec(geo.record_bytes(), 0);
  expert::pack_record(geo, gc.data(), gs.data(), uc.data(), us.data(), dc.data(), ds.data(), rec.data());
  // Spot-check the layout against the row-major source.
  for (int t = 0; t < 1000; ++t) {
    const int r = int(rng() % geo.inter), k = int(rng() % geo.hidden);
    const int src = (gc[size_t(r) * geo.hidden / 2 + k / 2] >> (4 * (k & 1))) & 15;
    if (expert::unit_code(rec.data(), geo.gu_blocks(), 2 * r, k) != src) {
      std::printf("layout mismatch at r=%d k=%d\n", r, k);
      ++g_fail;
      break;
    }
  }
  return rec;
}

std::vector<uint16_t> random_column(int n, std::mt19937& rng, float scale) {
  std::normal_distribution<float> nd(0.0f, scale);
  std::vector<uint16_t> v(n);
  for (auto& b : v) b = canon::bf16_rn(nd(rng));
  return v;
}

static void test_expert_naive(const expert::Geometry& geo) {
  std::mt19937 rng(11);
  const auto rec = random_record(geo, rng);
  const expert::Multipliers m{0.0123f, 0.0456f, 0.0078f};
  int bad = 0;
  uint8_t* drec = to_dev(rec);
  for (int trial = 0; trial < 4; ++trial) {
    const auto x = random_column(geo.hidden, rng, trial == 3 ? 1e-3f : 1.0f);
    std::vector<uint16_t> yref(geo.hidden);
    expert::forward_ref(geo, rec.data(), m, x.data(), yref.data());
    // GPU: encode on the host (shared code), compute rows on the device.
    const int emax = expert::a16_emax(x.data(), geo.hidden);
    std::vector<int32_t> xi(geo.hidden);
    for (int j = 0; j < geo.hidden; ++j) xi[j] = canon::a16_encode(x[j], emax);
    int32_t* dx = to_dev(xi);
    uint16_t* dh;
    CHECK_CUDA(cudaMalloc(&dh, geo.inter * 2));
    k_gu_naive<<<(geo.inter + 127) / 128, 128>>>(geo, drec, m, dx, emax, dh);
    const auto h = from_dev(dh, geo.inter);
    const int hemax = expert::a16_emax(h.data(), geo.inter);
    std::vector<int32_t> hi(geo.inter);
    for (int j = 0; j < geo.inter; ++j) hi[j] = canon::a16_encode(h[j], hemax);
    int32_t* dhi = to_dev(hi);
    uint16_t* dy;
    CHECK_CUDA(cudaMalloc(&dy, geo.hidden * 2));
    k_down_naive<<<(geo.hidden + 127) / 128, 128>>>(geo, drec + geo.gu_bytes(), m.down, dhi, hemax, dy);
    const auto y = from_dev(dy, geo.hidden);
    for (int r = 0; r < geo.hidden; ++r) bad += y[r] != yref[r];
    // Sanity against float64 math on the dequantized weights: relative L2 error of the BF16 pipeline.
    if (trial == 0) {
      double num = 0, den = 0;
      std::vector<double> hd(geo.inter);
      for (int i = 0; i < geo.inter; ++i) {
        double g = 0, u = 0;
        for (int k = 0; k < geo.hidden; ++k) {
          const double xv = canon::bf16_to_f32(x[k]);
          auto w = [&](int row) {
            return 0.5 * canon::c2(expert::unit_code(rec.data(), geo.gu_blocks(), row, k)) *
                   e4m3_to_float(expert::unit_scale(rec.data(), geo.gu_blocks(), row, k / 16));
          };
          g += w(2 * i) * xv, u += w(2 * i + 1) * xv;
        }
        g *= m.gate, u *= m.up;
        hd[i] = g / (1 + std::exp(-g)) * u;
      }
      for (int r = 0; r < geo.hidden; ++r) {
        double acc = 0;
        for (int i = 0; i < geo.inter; ++i)
          acc += 0.5 * canon::c2(expert::unit_code(rec.data() + geo.gu_bytes(), geo.d_blocks(), r, i)) *
                 e4m3_to_float(expert::unit_scale(rec.data() + geo.gu_bytes(), geo.d_blocks(), r, i / 16)) * hd[i];
        acc *= m.down;
        num += (acc - canon::bf16_to_f32(yref[r])) * (acc - canon::bf16_to_f32(yref[r]));
        den += acc * acc;
      }
      std::printf("expert H=%d I=%d: relative L2 error vs float64 %.2e\n", geo.hidden, geo.inter, std::sqrt(num / den));
      if (std::sqrt(num / den) > 2e-2) ++g_fail;
    }
    cudaFree(dx), cudaFree(dh), cudaFree(dhi), cudaFree(dy);
  }
  cudaFree(drec);
  std::printf("expert H=%d I=%d: naive GPU vs forward_ref: %d mismatches\n", geo.hidden, geo.inter, bad);
  if (bad) ++g_fail;
}

int main() {
  test_silu();
  test_row_output();
  test_a16_exact();
  test_expert_naive({256, 64});
  test_expert_naive(expert::kFlashNext);
  std::printf(g_fail ? "FAILED\n" : "PASSED\n");
  return g_fail ? 1 : 0;
}
