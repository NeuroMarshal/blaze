// Vision tower on the CPU (blaze/vision_cpu.h). fp32 activations, bf16 weights converted on the fly.
//
// GEMM: C[M, N] = A[M, K] * W[N, K]^T with A rows read directly (six row pointers, broadcast loads) and W packed per
// (n-block, k-block) into fp32 panels [k][16] that stay in L1/L2 while a block of rows streams through: a 6 x 16 FMA
// micro-kernel (12 accumulators). Attention: flash-style over 192-row query tiles; the keys are written once in the
// panel layout of the S = Q K^T micro-kernel and the values in the layout of the P V one, straight from the rotary pass.
#include "blaze/vision_cpu.h"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace {

inline float bf2f(uint16_t v) {
  const uint32_t u = uint32_t(v) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

inline uint16_t f2bf(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return uint16_t((u >> 16) | 0x40);
  u += 0x7fffu + ((u >> 16) & 1u);
  return uint16_t(u >> 16);
}

// threads

// Workers wait between jobs; run() hands out task indices dynamically and the calling thread works too.
class Pool {
 public:
  explicit Pool(int threads) : n_(std::max(threads, 1)) {
    for (int t = 1; t < n_; ++t) workers_.emplace_back([this, t] { loop(t); });
  }
  ~Pool() {
    {
      std::lock_guard<std::mutex> l(m_);
      quit_ = true;
      ++gen_;
    }
    cv_.notify_all();
    for (auto& w : workers_) w.join();
  }
  int size() const { return n_; }

  void run(int64_t tasks, const std::function<void(int64_t, int)>& fn) {
    if (tasks <= 0) return;
    if (n_ == 1 || tasks == 1) {
      for (int64_t i = 0; i < tasks; ++i) fn(i, 0);
      return;
    }
    {
      std::lock_guard<std::mutex> l(m_);
      fn_ = &fn;
      total_ = tasks;
      next_.store(0);
      pending_ = n_ - 1;
      ++gen_;
    }
    cv_.notify_all();
    work(0);
    std::unique_lock<std::mutex> l(m_);
    done_.wait(l, [&] { return pending_ == 0; });
  }

 private:
  void work(int t) {
    for (;;) {
      const int64_t i = next_.fetch_add(1);
      if (i >= total_) break;
      (*fn_)(i, t);
    }
  }
  void loop(int t) {
    int64_t seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> l(m_);
        cv_.wait(l, [&] { return gen_ != seen; });
        seen = gen_;
        if (quit_) return;
      }
      work(t);
      std::lock_guard<std::mutex> l(m_);
      if (--pending_ == 0) done_.notify_one();
    }
  }

  int n_;
  std::vector<std::thread> workers_;
  std::mutex m_;
  std::condition_variable cv_, done_;
  const std::function<void(int64_t, int)>* fn_ = nullptr;
  int64_t total_ = 0;
  std::atomic<int64_t> next_{0};
  int pending_ = 0;
  int64_t gen_ = 0;
  bool quit_ = false;
};

// Rows [0, n) in chunks over the pool: fn(first, last).
void rows(Pool& pool, int64_t n, int64_t chunk, const std::function<void(int64_t, int64_t)>& fn) {
  pool.run((n + chunk - 1) / chunk, [&](int64_t c, int) { fn(c * chunk, std::min(n, (c + 1) * chunk)); });
}

// micro-kernel

constexpr int MR = 6, NR = 16;

// c[6][16] (rows ldc apart) = (acc ? c : 0) + sum_k a_i[k] * b[k][0..16); a_i: six row pointers, b: [kc][16] fp32.
inline void micro6x16(int kc, const float* const a[MR], const float* b, float* c, int64_t ldc, bool acc) {
  __m256 c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
  if (acc) {
    c00 = _mm256_loadu_ps(c + 0 * ldc), c01 = _mm256_loadu_ps(c + 0 * ldc + 8);
    c10 = _mm256_loadu_ps(c + 1 * ldc), c11 = _mm256_loadu_ps(c + 1 * ldc + 8);
    c20 = _mm256_loadu_ps(c + 2 * ldc), c21 = _mm256_loadu_ps(c + 2 * ldc + 8);
    c30 = _mm256_loadu_ps(c + 3 * ldc), c31 = _mm256_loadu_ps(c + 3 * ldc + 8);
    c40 = _mm256_loadu_ps(c + 4 * ldc), c41 = _mm256_loadu_ps(c + 4 * ldc + 8);
    c50 = _mm256_loadu_ps(c + 5 * ldc), c51 = _mm256_loadu_ps(c + 5 * ldc + 8);
  } else {
    c00 = c01 = c10 = c11 = c20 = c21 = c30 = c31 = c40 = c41 = c50 = c51 = _mm256_setzero_ps();
  }
  const float *a0 = a[0], *a1 = a[1], *a2 = a[2], *a3 = a[3], *a4 = a[4], *a5 = a[5];
  for (int k = 0; k < kc; ++k, b += NR) {
    const __m256 b0 = _mm256_loadu_ps(b), b1 = _mm256_loadu_ps(b + 8);
    __m256 x = _mm256_broadcast_ss(a0 + k);
    c00 = _mm256_fmadd_ps(x, b0, c00), c01 = _mm256_fmadd_ps(x, b1, c01);
    x = _mm256_broadcast_ss(a1 + k);
    c10 = _mm256_fmadd_ps(x, b0, c10), c11 = _mm256_fmadd_ps(x, b1, c11);
    x = _mm256_broadcast_ss(a2 + k);
    c20 = _mm256_fmadd_ps(x, b0, c20), c21 = _mm256_fmadd_ps(x, b1, c21);
    x = _mm256_broadcast_ss(a3 + k);
    c30 = _mm256_fmadd_ps(x, b0, c30), c31 = _mm256_fmadd_ps(x, b1, c31);
    x = _mm256_broadcast_ss(a4 + k);
    c40 = _mm256_fmadd_ps(x, b0, c40), c41 = _mm256_fmadd_ps(x, b1, c41);
    x = _mm256_broadcast_ss(a5 + k);
    c50 = _mm256_fmadd_ps(x, b0, c50), c51 = _mm256_fmadd_ps(x, b1, c51);
  }
  _mm256_storeu_ps(c + 0 * ldc, c00), _mm256_storeu_ps(c + 0 * ldc + 8, c01);
  _mm256_storeu_ps(c + 1 * ldc, c10), _mm256_storeu_ps(c + 1 * ldc + 8, c11);
  _mm256_storeu_ps(c + 2 * ldc, c20), _mm256_storeu_ps(c + 2 * ldc + 8, c21);
  _mm256_storeu_ps(c + 3 * ldc, c30), _mm256_storeu_ps(c + 3 * ldc + 8, c31);
  _mm256_storeu_ps(c + 4 * ldc, c40), _mm256_storeu_ps(c + 4 * ldc + 8, c41);
  _mm256_storeu_ps(c + 5 * ldc, c50), _mm256_storeu_ps(c + 5 * ldc + 8, c51);
}

// Natural exponential of 8 floats (Cephes expf: range reduction by ln 2, degree-5 polynomial; relative error ~1e-7).
inline __m256 exp256(__m256 x) {
  x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.3f)), _mm256_set1_ps(88.3f));
  const __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375f), x);
  r = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.12194440e-4f), r);
  __m256 p = _mm256_set1_ps(1.9875691500e-4f);
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507e-3f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073e-3f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894e-2f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459e-1f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201e-1f));
  const __m256 y = _mm256_add_ps(_mm256_fmadd_ps(_mm256_mul_ps(p, r), r, r), _mm256_set1_ps(1.0f));
  const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
  return _mm256_mul_ps(y, _mm256_castsi256_ps(e));
}

inline float hsum(__m256 v) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
  return _mm_cvtss_f32(s);
}

inline float hmax(__m256 v) {
  __m128 s = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_max_ps(s, _mm_movehl_ps(s, s));
  s = _mm_max_ss(s, _mm_shuffle_ps(s, s, 1));
  return _mm_cvtss_f32(s);
}

// GEMM

struct Scratch {
  std::vector<float> b;    // packed weight block
  std::vector<float> s, o;  // attention tiles
  std::vector<float> m, l;
};

constexpr int KC = 256, NC = 64;

// C[M, N] = A[M, K] * W[N, K]^T, A fp32 rows lda apart, W bf16 [N, K], C fp32 rows ldc apart.
void gemm(Pool& pool, std::vector<Scratch>& sc, const float* A, int64_t lda, const uint16_t* W, int M, int N, int K, float* C,
          int64_t ldc) {
  const int nt = (N + NC - 1) / NC;
  // row blocks of 192 rows, 96 when there would be too few tasks for the threads
  int MC = 192;
  if (int64_t((M + MC - 1) / MC) * nt < 2 * pool.size()) MC = 96;
  const int mt = (M + MC - 1) / MC;
  pool.run(int64_t(mt) * nt, [&](int64_t task, int tid) {
    const int m0 = int(task / nt) * MC, n0 = int(task % nt) * NC;
    const int mc = std::min(MC, M - m0), nc = std::min(NC, N - n0);
    float* bp = sc[tid].b.data();
    const int np = (nc + NR - 1) / NR, mp = (mc + MR - 1) / MR;
    for (int k0 = 0; k0 < K; k0 += KC) {
      const int kc = std::min(KC, K - k0);
      for (int q = 0; q < np; ++q) {
        float* dst = bp + int64_t(q) * kc * NR;
        for (int j = 0; j < NR; ++j) {
          const int c = q * NR + j;
          if (c < nc) {
            const uint16_t* src = W + int64_t(n0 + c) * K + k0;
            for (int k = 0; k < kc; ++k) dst[k * NR + j] = bf2f(src[k]);
          } else {
            for (int k = 0; k < kc; ++k) dst[k * NR + j] = 0.0f;
          }
        }
      }
      for (int q = 0; q < np; ++q) {
        const int cols = std::min(NR, nc - q * NR);
        for (int p = 0; p < mp; ++p) {
          const int r0 = m0 + p * MR, rows_left = std::min(MR, M - r0);
          const float* a[MR];
          for (int i = 0; i < MR; ++i) a[i] = A + int64_t(r0 + std::min(i, rows_left - 1)) * lda + k0;
          float* c = C + int64_t(r0) * ldc + n0 + q * NR;
          const float* b = bp + int64_t(q) * kc * NR;
          if (rows_left == MR && cols == NR) {
            micro6x16(kc, a, b, c, ldc, k0 > 0);
          } else {
            float tmp[MR * NR] = {};
            if (k0 > 0)
              for (int i = 0; i < rows_left; ++i)
                for (int j = 0; j < cols; ++j) tmp[i * NR + j] = c[i * ldc + j];
            micro6x16(kc, a, b, tmp, NR, k0 > 0);
            for (int i = 0; i < rows_left; ++i)
              for (int j = 0; j < cols; ++j) c[i * ldc + j] = tmp[i * NR + j];
          }
        }
      }
    }
  });
}

// elementwise

void layernorm(Pool& pool, const float* x, const uint16_t* g, const uint16_t* b, float eps, int n, int dim, float* out) {
  std::vector<float> gf(dim), bf(dim);
  for (int i = 0; i < dim; ++i) gf[i] = bf2f(g[i]), bf[i] = bf2f(b[i]);
  rows(pool, n, 16, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const float* xr = x + r * dim;
      double s = 0;
      for (int i = 0; i < dim; ++i) s += xr[i];
      const float mean = float(s / dim);
      double q = 0;
      for (int i = 0; i < dim; ++i) q += double(xr[i] - mean) * (xr[i] - mean);
      const float inv = 1.0f / std::sqrt(float(q / dim) + eps);
      float* o = out + r * dim;
      for (int i = 0; i < dim; ++i) o[i] = (xr[i] - mean) * inv * gf[i] + bf[i];
    }
  });
}

// x[r][c] += bias[c]; optionally gelu.
void bias_act(Pool& pool, float* x, const uint16_t* bias, int n, int dim, int act /*0 none, 1 tanh, 2 erf*/) {
  std::vector<float> bf(dim);
  for (int i = 0; i < dim; ++i) bf[i] = bf2f(bias[i]);
  rows(pool, n, 16, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      float* xr = x + r * dim;
      for (int i = 0; i < dim; ++i) {
        float v = xr[i] + bf[i];
        if (act == 1) {
          v = 0.5f * v * (1.0f + std::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
        } else if (act == 2) {
          v = 0.5f * v * (1.0f + std::erf(v * 0.7071067811865476f));
        }
        xr[i] = v;
      }
    }
  });
}

// x += g + bias (projection output, bias and residual).
void bias_residual(Pool& pool, float* x, const float* g, const uint16_t* bias, int n, int dim) {
  std::vector<float> bf(dim);
  for (int i = 0; i < dim; ++i) bf[i] = bf2f(bias[i]);
  rows(pool, n, 16, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r)
      for (int i = 0; i < dim; ++i) x[r * dim + i] += g[r * dim + i] + bf[i];
  });
}

// Patch i (merge order) of the grid: its row and column.
inline void patch_rc(int64_t i, int grid_w, int merge, int* row, int* col) {
  const int unit = merge * merge;
  const int64_t b = i / unit;
  const int j = int(i % unit);
  const int bw = grid_w / merge;
  *row = int(b / bw) * merge + j / merge;
  *col = int(b % bw) * merge + j % merge;
}

// x += bias + the learned position table resampled bilinearly (corners aligned) to the grid.
void embed_pos(Pool& pool, float* x, const uint16_t* bias, const uint16_t* table, int side, int gh, int gw, int merge, int dim) {
  rows(pool, int64_t(gh) * gw, 8, [&](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; ++i) {
      int row, col;
      patch_rc(i, gw, merge, &row, &col);
      const float sh = float(row) * float(side - 1) / float(std::max(gh - 1, 1));
      const float sw = float(col) * float(side - 1) / float(std::max(gw - 1, 1));
      const float fh = std::floor(sh), fw = std::floor(sw);
      int th[2], tw[2];
      float wh[2], ww[2];
      for (int o = 0; o < 2; ++o) {
        th[o] = std::min(std::max(int(fh) + o, 0), side - 1);
        tw[o] = std::min(std::max(int(fw) + o, 0), side - 1);
        wh[o] = std::max(1.0f - std::fabs(sh - fh - float(o)), 0.0f);
        ww[o] = std::max(1.0f - std::fabs(sw - fw - float(o)), 0.0f);
      }
      float* xr = x + i * dim;
      for (int c = 0; c < dim; ++c) {
        float pe = 0.0f;
        for (int a = 0; a < 2; ++a)
          for (int b = 0; b < 2; ++b) pe += bf2f(table[int64_t(th[a]) * side * dim + int64_t(tw[b]) * dim + c]) * (wh[a] * ww[b]);
        xr[c] += bf2f(bias[c]) + pe;
      }
    }
  });
}

// attention

constexpr int QB = 192, KB = 64;

struct AttnLayout {
  int n, nk, nkb, heads, hd, npv;
  // keys: [heads][nk / 16][hd][16]; values: [heads][nkb][npv][KB][16]
  int64_t kp(int h, int j, int d) const { return ((int64_t(h) * (nk / 16) + j / 16) * hd + d) * 16 + j % 16; }
  int64_t vp(int h, int j, int d) const { return (((int64_t(h) * nkb + j / KB) * npv + d / 16) * KB + j % KB) * 16 + d % 16; }
};

// qkv [n][3 * hidden] (bias not yet added): q (rotated) [n][hidden], keys (rotated) and values (+ bias) in the panel layouts.
void qkv_rope(Pool& pool, const AttnLayout& L, const float* qkv, const uint16_t* bias, int gw, int merge, float theta, float* q,
              float* kp, float* vp) {
  const int hd = L.hd, hd2 = hd / 2, nf = hd / 4, heads = L.heads;
  const int64_t width = int64_t(heads) * hd;
  std::vector<float> inv(nf), bf(3 * width);
  for (int f = 0; f < nf; ++f) inv[f] = 1.0f / std::pow(theta, float(2 * f) / float(hd2));
  for (int64_t i = 0; i < 3 * width; ++i) bf[i] = bf2f(bias[i]);
  rows(pool, L.n, 4, [&](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; ++i) {
      int row, col;
      patch_rc(i, gw, merge, &row, &col);
      float cs[64], sn[64];  // hd <= 128
      for (int d = 0; d < hd2; ++d) {
        const float ang = float(d < nf ? row : col) * inv[d % nf];
        cs[d] = std::cos(ang), sn[d] = std::sin(ang);
      }
      for (int h = 0; h < heads; ++h) {
        const float* src = qkv + i * 3 * width + int64_t(h) * hd;
        for (int t = 0; t < 2; ++t) {
          const float* s = src + t * width;
          const float* b = bf.data() + t * width + int64_t(h) * hd;
          for (int d = 0; d < hd2; ++d) {
            const float x1 = s[d] + b[d], x2 = s[d + hd2] + b[d + hd2];
            const float r1 = x1 * cs[d] - x2 * sn[d], r2 = x2 * cs[d] + x1 * sn[d];
            if (t == 0) {
              q[i * width + int64_t(h) * hd + d] = r1;
              q[i * width + int64_t(h) * hd + d + hd2] = r2;
            } else {
              kp[L.kp(h, int(i), d)] = r1;
              kp[L.kp(h, int(i), d + hd2)] = r2;
            }
          }
        }
        for (int d = 0; d < hd; ++d) vp[L.vp(h, int(i), d)] = src[2 * width + d] + bf[2 * width + int64_t(h) * hd + d];
      }
    }
  });
}

// o [n][heads * hd] = softmax(q k^T / sqrt(hd)) v per head, over the whole image.
void attention(Pool& pool, std::vector<Scratch>& sc, const AttnLayout& L, const float* q, const float* kp, const float* vp, float* o) {
  const int hd = L.hd, heads = L.heads, n = L.n;
  const int64_t width = int64_t(heads) * hd;
  const int npv = L.npv, ow = npv * 16;
  const int qt = (n + QB - 1) / QB;
  const float scale = 1.0f / std::sqrt(float(hd));
  pool.run(int64_t(heads) * qt, [&](int64_t task, int tid) {
    const int h = int(task / qt), q0 = int(task % qt) * QB;
    float* S = sc[tid].s.data();
    float* O = sc[tid].o.data();
    float* mx = sc[tid].m.data();
    float* ls = sc[tid].l.data();
    for (int r = 0; r < QB; ++r) mx[r] = -1e30f, ls[r] = 0.0f;
    std::memset(O, 0, sizeof(float) * QB * ow);
    const float* qa[QB / MR][MR];
    for (int p = 0; p < QB / MR; ++p)
      for (int i = 0; i < MR; ++i) qa[p][i] = q + int64_t(std::min(q0 + p * MR + i, n - 1)) * width + int64_t(h) * hd;
    for (int kb = 0; kb < L.nkb; ++kb) {
      // S = Q K^T for the 64 keys of the block
      for (int pq = 0; pq < KB / NR; ++pq) {
        const float* b = kp + (int64_t(h) * (L.nk / 16) + kb * (KB / NR) + pq) * hd * 16;
        for (int p = 0; p < QB / MR; ++p) micro6x16(hd, qa[p], b, S + int64_t(p) * MR * KB + pq * NR, KB, false);
      }
      const int valid = std::min(KB, n - kb * KB);
      for (int r = 0; r < QB; ++r) {
        float* s = S + int64_t(r) * KB;
        for (int j = valid; j < KB; ++j) s[j] = -1e30f;
        __m256 m8 = _mm256_set1_ps(-1e30f);
        for (int j = 0; j < KB; j += 8) m8 = _mm256_max_ps(m8, _mm256_mul_ps(_mm256_loadu_ps(s + j), _mm256_set1_ps(scale)));
        const float m_new = std::max(mx[r], hmax(m8));
        const float alpha = std::exp(std::max(mx[r] - m_new, -87.0f));
        const __m256 mn = _mm256_set1_ps(m_new), sc8 = _mm256_set1_ps(scale);
        __m256 sum = _mm256_setzero_ps();
        for (int j = 0; j < KB; j += 8) {
          const __m256 pj = exp256(_mm256_sub_ps(_mm256_mul_ps(_mm256_loadu_ps(s + j), sc8), mn));
          _mm256_storeu_ps(s + j, pj);
          sum = _mm256_add_ps(sum, pj);
        }
        ls[r] = ls[r] * alpha + hsum(sum);
        mx[r] = m_new;
        float* orow = O + int64_t(r) * ow;
        const __m256 al = _mm256_set1_ps(alpha);
        for (int d = 0; d < ow; d += 8) _mm256_storeu_ps(orow + d, _mm256_mul_ps(_mm256_loadu_ps(orow + d), al));
      }
      // O += P V
      for (int pn = 0; pn < npv; ++pn) {
        const float* b = vp + ((int64_t(h) * L.nkb + kb) * npv + pn) * KB * 16;
        for (int p = 0; p < QB / MR; ++p) {
          const float* a[MR];
          for (int i = 0; i < MR; ++i) a[i] = S + int64_t(p * MR + i) * KB;
          micro6x16(KB, a, b, O + int64_t(p) * MR * ow + pn * NR, ow, true);
        }
      }
    }
    for (int r = 0; r < QB && q0 + r < n; ++r) {
      const float inv = 1.0f / ls[r];
      float* dst = o + int64_t(q0 + r) * width + int64_t(h) * hd;
      for (int d = 0; d < hd; ++d) dst[d] = O[int64_t(r) * ow + d] * inv;
    }
  });
}

}  // namespace

extern "C" int blz_vis_cpu_encode(const blz_vis_weights* w, const uint16_t* pixels, int gh, int gw, int threads, uint16_t* out) {
  const blz_vis_cfg& c = w->cfg;
  const int H = c.hidden, heads = c.heads, hd = H / heads, merge = c.merge;
  if (H % heads || hd % 4 || hd > 128 || gh % merge || gw % merge || c.intermediate % 16 || H % 16 || c.out_hidden % 16) return 1;
  const int n = gh * gw;
  const int kpatch = c.channels * c.temporal * c.patch * c.patch;
  const int unit = merge * merge, nm = n / unit, mh = H * unit;
  Pool pool(threads);
  std::vector<Scratch> sc(pool.size());
  AttnLayout L{n, (n + KB - 1) / KB * KB, 0, heads, hd, (hd + 15) / 16};
  L.nkb = L.nk / KB;
  for (auto& s : sc) {
    s.b.resize(size_t(KC) * NC);
    s.s.resize(size_t(QB) * KB);
    s.o.resize(size_t(QB) * L.npv * 16);
    s.m.resize(QB);
    s.l.resize(QB);
  }
  const int64_t nh = int64_t(n) * H;
  std::vector<float> x(nh), xn(nh), q(nh), ob(nh), qkv(int64_t(n) * 3 * H), hbuf(int64_t(n) * c.intermediate);
  std::vector<float> kp(int64_t(heads) * L.nk * hd, 0.0f), vp(int64_t(heads) * L.nk * L.npv * 16, 0.0f);

  // patch embedding: Conv3d with stride = kernel is a GEMM over the flattened patches
  {
    std::vector<float> px(int64_t(n) * kpatch);
    rows(pool, int64_t(n) * kpatch, 1 << 16, [&](int64_t a, int64_t b) {
      for (int64_t i = a; i < b; ++i) px[i] = bf2f(pixels[i]);
    });
    gemm(pool, sc, px.data(), kpatch, w->patch_w, n, H, kpatch, x.data(), H);
    embed_pos(pool, x.data(), w->patch_b, w->pos_table, c.side, gh, gw, merge, H);
  }
  for (int l = 0; l < c.depth; ++l) {
    const blz_vis_layer& y = w->layers[l];
    layernorm(pool, x.data(), y.n1_w, y.n1_b, c.eps, n, H, xn.data());
    gemm(pool, sc, xn.data(), H, y.qkv_w, n, 3 * H, H, qkv.data(), 3 * H);
    qkv_rope(pool, L, qkv.data(), y.qkv_b, gw, merge, c.theta, q.data(), kp.data(), vp.data());
    attention(pool, sc, L, q.data(), kp.data(), vp.data(), ob.data());
    gemm(pool, sc, ob.data(), H, y.proj_w, n, H, H, xn.data(), H);
    bias_residual(pool, x.data(), xn.data(), y.proj_b, n, H);
    layernorm(pool, x.data(), y.n2_w, y.n2_b, c.eps, n, H, xn.data());
    gemm(pool, sc, xn.data(), H, y.fc1_w, n, c.intermediate, H, hbuf.data(), c.intermediate);
    bias_act(pool, hbuf.data(), y.fc1_b, n, c.intermediate, c.act_tanh ? 1 : 2);
    gemm(pool, sc, hbuf.data(), c.intermediate, y.fc2_w, n, H, c.intermediate, xn.data(), H);
    bias_residual(pool, x.data(), xn.data(), y.fc2_b, n, H);
  }
  // patch merger: norm per patch, merge x merge neighbours side by side, MLP with the exact GELU
  layernorm(pool, x.data(), w->mrg_norm_w, w->mrg_norm_b, c.eps, n, H, xn.data());
  gemm(pool, sc, xn.data(), mh, w->mrg_fc1_w, nm, mh, mh, hbuf.data(), mh);
  bias_act(pool, hbuf.data(), w->mrg_fc1_b, nm, mh, 2);
  std::vector<float> res(int64_t(nm) * c.out_hidden);
  gemm(pool, sc, hbuf.data(), mh, w->mrg_fc2_w, nm, c.out_hidden, mh, res.data(), c.out_hidden);
  bias_act(pool, res.data(), w->mrg_fc2_b, nm, c.out_hidden, 0);
  rows(pool, int64_t(nm) * c.out_hidden, 1 << 16, [&](int64_t a, int64_t b) {
    for (int64_t i = a; i < b; ++i) out[i] = f2bf(res[i]);
  });
  return 0;
}
