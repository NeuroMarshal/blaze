// Exact W4A16 expert kernels for x86 AVX2 (Zen 2 has no AVX-512 / VNNI), bit-identical to the GPU route.
//
// Per rg16 unit (16 rows x 16 k) and column: each 32-byte code quad q is one YMM whose dword d holds four k of row d
// (low nibbles) and of row 8 + d (high nibbles). vpshufb maps nibbles to c2 bytes; vpmaddubsw multiplies the
// unsigned activation limbs (lo, mid) by the signed c2 and the biased c2 + 12 (unsigned) by the signed hi limb, so
// every product is a byte product with no saturation (pairs stay below 2^13); the four quads are summed in int16
// (below 2^15), vpmaddwd folds pairs to int32, and P = lo + 256 mid + 65536 (hi - 12 sum(hi)) is exact in int32.
// P times the row's block scale S^ is accumulated in int64 with vpmuldq on even and odd dword lanes.

#include <immintrin.h>

#include <algorithm>

#include <cstdlib>
#include <cstring>
#include <new>

#include "blaze/cpu_expert.h"

namespace blaze::cpu {

void encode_column(const uint16_t* v, int n, ColumnBuffer& buf) {
  buf.emax = expert::a16_emax(v, n);
  for (int b = 0; b < n / 16; ++b) {
    int32_t hs = 0;
    for (int j = 16 * b; j < 16 * b + 16; ++j) {
      const int32_t x = buf.emax < 0 ? 0 : canon::a16_encode(v[j], buf.emax);
      buf.lo[j] = uint8_t(x & 0xFF);
      buf.mid[j] = uint8_t((x >> 8) & 0xFF);
      buf.hi[j] = int8_t(x >> 16);
      hs += buf.hi[j];
    }
    buf.hi_sum[b] = hs;
  }
}

void row_sums_scalar(const uint8_t* matrix, int blocks, int g0, int g1, const Column* cols, int ncols,
                     int64_t* out) {
  for (int g = g0; g < g1; ++g) {
    for (int r = 0; r < 16; ++r) {
      const int row = 16 * g + r;
      for (int c = 0; c < ncols; ++c) {
        int64_t s = 0;
        for (int b = 0; b < blocks; ++b) {
          int32_t p = 0;
          for (int j = 0; j < 16; ++j) {
            const int k = 16 * b + j;
            const int32_t x = int32_t(cols[c].lo[k]) | (int32_t(cols[c].mid[k]) << 8) | (int32_t(cols[c].hi[k]) * 65536);
            p += canon::c2(expert::unit_code(matrix, blocks, row, k)) * x;
          }
          s += int64_t(p) * canon::s_hat(expert::unit_scale(matrix, blocks, row, b));
        }
        out[size_t(row) * ncols + c] = s;
      }
    }
  }
}

namespace {

__attribute__((target("avx2,fma"))) inline __m256i s_hat8(__m128i bytes8) {
  // S^(w) = (e == 0) ? m : (8 + m) << (e - 1), for the low 8 scale bytes.
  const __m256i w = _mm256_cvtepu8_epi32(bytes8);
  const __m256i m = _mm256_and_si256(w, _mm256_set1_epi32(7));
  const __m256i e = _mm256_and_si256(_mm256_srli_epi32(w, 3), _mm256_set1_epi32(15));
  const __m256i normal = _mm256_sllv_epi32(_mm256_add_epi32(m, _mm256_set1_epi32(8)),
                                           _mm256_sub_epi32(e, _mm256_set1_epi32(1)));
  const __m256i is_sub = _mm256_cmpeq_epi32(e, _mm256_setzero_si256());
  return _mm256_blendv_epi8(normal, m, is_sub);
}

// acc += P * S^ in int64: even dword lanes into acc_even (rows 0,2,4,6), odd lanes into acc_odd (rows 1,3,5,7).
__attribute__((target("avx2,fma"))) inline void mac64(__m256i p, __m256i s, __m256i& acc_even, __m256i& acc_odd) {
  acc_even = _mm256_add_epi64(acc_even, _mm256_mul_epi32(p, s));
  acc_odd = _mm256_add_epi64(acc_odd, _mm256_mul_epi32(_mm256_srli_epi64(p, 32), _mm256_srli_epi64(s, 32)));
}

__attribute__((target("avx2,fma"))) inline void store_rows(__m256i even, __m256i odd, int64_t* out, int ncols, int c,
                                                         int row0) {
  alignas(32) int64_t e[4], o[4];
  _mm256_store_si256(reinterpret_cast<__m256i*>(e), even);
  _mm256_store_si256(reinterpret_cast<__m256i*>(o), odd);
  for (int j = 0; j < 4; ++j) {
    out[size_t(row0 + 2 * j) * ncols + c] = e[j];
    out[size_t(row0 + 2 * j + 1) * ncols + c] = o[j];
  }
}

// One row group, all columns in one pass over its units: each unit's codes are decoded once (c2 and the biased
// c2 + 12 for both row halves) and every column's int64 accumulators live in a small L1 array, so a unit is read
// once whatever the column count.
__attribute__((target("avx2,fma"))) void group_sums(const uint8_t* unit0, int blocks, const Column* cols, int ncols,
                                                    int64_t* out, int row0) {
  const __m256i lut = _mm256_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,  //
                                       0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
  const __m256i nib = _mm256_set1_epi8(0x0F);
  const __m256i twelve = _mm256_set1_epi8(12);
  const __m256i ones16 = _mm256_set1_epi16(1);
  alignas(32) __m256i acc[8][4];
  for (int c = 0; c < ncols; ++c)
    for (auto& a : acc[c]) a = _mm256_setzero_si256();

  const uint8_t* unit = unit0;
  for (int b = 0; b < blocks; ++b, unit += expert::kUnitBytes) {
    _mm_prefetch(reinterpret_cast<const char*>(unit + 8 * expert::kUnitBytes), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(unit + 8 * expert::kUnitBytes + 64), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(unit + 8 * expert::kUnitBytes + 128), _MM_HINT_T0);
    __m256i clo[4], chi[4], blo[4], bhi[4];
    for (int q = 0; q < 4; ++q) {
      const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(unit + 32 * q));
      clo[q] = _mm256_shuffle_epi8(lut, _mm256_and_si256(w, nib));
      chi[q] = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(w, 4), nib));
      blo[q] = _mm256_add_epi8(clo[q], twelve);
      bhi[q] = _mm256_add_epi8(chi[q], twelve);
    }
    const __m128i sb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(unit + expert::kUnitCodeBytes));
    const __m256i s_lo = s_hat8(sb), s_hi = s_hat8(_mm_srli_si128(sb, 8));
    const int k0 = 16 * b;
    for (int c = 0; c < ncols; ++c) {
      const Column& col = cols[c];
      __m256i l0 = _mm256_setzero_si256(), m0 = l0, h0 = l0, l1 = l0, m1 = l0, h1 = l0;
      for (int q = 0; q < 4; ++q) {
        int32_t lw, mw, hw;
        std::memcpy(&lw, col.lo + k0 + 4 * q, 4);
        std::memcpy(&mw, col.mid + k0 + 4 * q, 4);
        std::memcpy(&hw, col.hi + k0 + 4 * q, 4);
        const __m256i al = _mm256_set1_epi32(lw), am = _mm256_set1_epi32(mw), ah = _mm256_set1_epi32(hw);
        l0 = _mm256_add_epi16(l0, _mm256_maddubs_epi16(al, clo[q]));
        m0 = _mm256_add_epi16(m0, _mm256_maddubs_epi16(am, clo[q]));
        h0 = _mm256_add_epi16(h0, _mm256_maddubs_epi16(blo[q], ah));
        l1 = _mm256_add_epi16(l1, _mm256_maddubs_epi16(al, chi[q]));
        m1 = _mm256_add_epi16(m1, _mm256_maddubs_epi16(am, chi[q]));
        h1 = _mm256_add_epi16(h1, _mm256_maddubs_epi16(bhi[q], ah));
      }
      const __m256i corr = _mm256_set1_epi32(12 * col.hi_sum[b]);
      auto fold = [&](__m256i l, __m256i m, __m256i h) {
        const __m256i pl = _mm256_madd_epi16(l, ones16), pm = _mm256_madd_epi16(m, ones16);
        const __m256i ph = _mm256_sub_epi32(_mm256_madd_epi16(h, ones16), corr);
        return _mm256_add_epi32(_mm256_add_epi32(pl, _mm256_slli_epi32(pm, 8)), _mm256_slli_epi32(ph, 16));
      };
      mac64(fold(l0, m0, h0), s_lo, acc[c][0], acc[c][1]);
      mac64(fold(l1, m1, h1), s_hi, acc[c][2], acc[c][3]);
    }
  }
  for (int c = 0; c < ncols; ++c) {
    store_rows(acc[c][0], acc[c][1], out, ncols, c, row0);
    store_rows(acc[c][2], acc[c][3], out, ncols, c, row0 + 8);
  }
}

}  // namespace

void row_sums_avx2(const uint8_t* matrix, int blocks, int g0, int g1, const Column* cols, int ncols, int64_t* out) {
  for (int g = g0; g < g1; ++g) {
    group_sums(matrix + expert::unit_offset(g, 0, blocks), blocks, cols, ncols, out, 16 * g);
  }
}

// approximate (Q8)

void quantize_column_q8(const Column& col, int n, int8_t* q, uint8_t* shift) {
  for (int b = 0; b < n / 16; ++b) {
    int32_t x[16], amax = 0;
    for (int j = 0; j < 16; ++j) {
      const int k = 16 * b + j;
      x[j] = int32_t(col.lo[k]) | (int32_t(col.mid[k]) << 8) | (int32_t(col.hi[k]) * 65536);
      amax = std::max(amax, x[j] < 0 ? -x[j] : x[j]);
    }
    int sh = 0;
    while ((amax >> sh) > 127) ++sh;
    shift[b] = uint8_t(sh);
    const int32_t half = sh ? 1 << (sh - 1) : 0;
    for (int j = 0; j < 16; ++j) {
      int32_t v = (x[j] + half) >> sh;  // arithmetic shift: round half up
      q[16 * b + j] = int8_t(std::min(127, std::max(-127, v)));
    }
  }
}

namespace {

__attribute__((target("avx2,fma"))) void group_sums_q8(const uint8_t* unit0, int blocks, const QColumn* cols,
                                                       int ncols, int64_t* out, int row0) {
  const __m256i lut = _mm256_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,  //
                                       0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
  const __m256i nib = _mm256_set1_epi8(0x0F);
  const __m256i ones16 = _mm256_set1_epi16(1);
  alignas(32) __m256i acc[8][4];
  for (int c = 0; c < ncols; ++c)
    for (auto& a : acc[c]) a = _mm256_setzero_si256();
  const uint8_t* unit = unit0;
  for (int b = 0; b < blocks; ++b, unit += expert::kUnitBytes) {
    _mm_prefetch(reinterpret_cast<const char*>(unit + 8 * expert::kUnitBytes), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(unit + 8 * expert::kUnitBytes + 64), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(unit + 8 * expert::kUnitBytes + 128), _MM_HINT_T0);
    __m256i clo[4], chi[4], alo[4], ahi[4];
    for (int q = 0; q < 4; ++q) {
      const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(unit + 32 * q));
      clo[q] = _mm256_shuffle_epi8(lut, _mm256_and_si256(w, nib));
      chi[q] = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(w, 4), nib));
      alo[q] = _mm256_abs_epi8(clo[q]);
      ahi[q] = _mm256_abs_epi8(chi[q]);
    }
    const __m128i sb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(unit + expert::kUnitCodeBytes));
    const __m256i s_lo = s_hat8(sb), s_hi = s_hat8(_mm_srli_si128(sb, 8));
    const int k0 = 16 * b;
    for (int c = 0; c < ncols; ++c) {
      const QColumn& col = cols[c];
      __m256i p0 = _mm256_setzero_si256(), p1 = p0;
      for (int q = 0; q < 4; ++q) {
        int32_t qw;
        std::memcpy(&qw, col.q + k0 + 4 * q, 4);
        const __m256i xq = _mm256_set1_epi32(qw);
        // |c2| (unsigned) x q * sign(c2) (signed): every product is c2 * q.
        p0 = _mm256_add_epi16(p0, _mm256_maddubs_epi16(alo[q], _mm256_sign_epi8(xq, clo[q])));
        p1 = _mm256_add_epi16(p1, _mm256_maddubs_epi16(ahi[q], _mm256_sign_epi8(xq, chi[q])));
      }
      const __m128i sh = _mm_cvtsi32_si128(col.shift[b]);
      mac64(_mm256_sll_epi32(_mm256_madd_epi16(p0, ones16), sh), s_lo, acc[c][0], acc[c][1]);
      mac64(_mm256_sll_epi32(_mm256_madd_epi16(p1, ones16), sh), s_hi, acc[c][2], acc[c][3]);
    }
  }
  for (int c = 0; c < ncols; ++c) {
    store_rows(acc[c][0], acc[c][1], out, ncols, c, row0);
    store_rows(acc[c][2], acc[c][3], out, ncols, c, row0 + 8);
  }
}

}  // namespace

void row_sums_q8_avx2(const uint8_t* matrix, int blocks, int g0, int g1, const QColumn* cols, int ncols, int64_t* out) {
  for (int g = g0; g < g1; ++g) group_sums_q8(matrix + expert::unit_offset(g, 0, blocks), blocks, cols, ncols, out, 16 * g);
}

// expert forward

Scratch::Scratch(const expert::Geometry& g) : geo(g) {
  const int H = g.hidden, I = g.inter;
  const size_t per_x = 3 * size_t(H) + 4 * size_t(H / 16) + 64, per_h = 3 * size_t(I) + 4 * size_t(I / 16) + 64;
  const size_t rows = size_t(std::max(2 * I, H));
  const size_t bytes = 8 * per_x + 8 * per_h + 8 * size_t(I) * 2 + rows * 8 * 8 + 4096;
  mem_ = std::aligned_alloc(64, (bytes + 63) / 64 * 64);
  if (!mem_) throw std::bad_alloc();
  auto* p = static_cast<uint8_t*>(mem_);
  auto take = [&](size_t n) {
    uint8_t* r = p;
    p += (n + 63) / 64 * 64;
    return r;
  };
  sums = reinterpret_cast<int64_t*>(take(rows * 8 * 8));
  h_bits = reinterpret_cast<uint16_t*>(take(8 * size_t(I) * 2));
  for (int c = 0; c < 8; ++c) {
    x[c].lo = take(H), x[c].mid = take(H), x[c].hi = reinterpret_cast<int8_t*>(take(H));
    x[c].hi_sum = reinterpret_cast<int32_t*>(take(4 * size_t(H / 16)));
    h[c].lo = take(I), h[c].mid = take(I), h[c].hi = reinterpret_cast<int8_t*>(take(I));
    h[c].hi_sum = reinterpret_cast<int32_t*>(take(4 * size_t(I / 16)));
  }
}

Scratch::~Scratch() { std::free(mem_); }

void expert_forward(const uint8_t* record, const expert::Multipliers& m, const uint16_t* const* x, int ncols,
                    uint16_t* const* y, Scratch& s, bool use_avx2) {
  const expert::Geometry& geo = s.geo;
  const int H = geo.hidden, I = geo.inter;
  auto sums_fn = use_avx2 ? row_sums_avx2 : row_sums_scalar;
  Column cols[8] = {};
  for (int c = 0; c < ncols; ++c) {
    encode_column(x[c], H, s.x[c]);
    cols[c] = s.x[c].view();
  }
  sums_fn(record, geo.gu_blocks(), 0, geo.gu_groups(), cols, ncols, s.sums);
  for (int c = 0; c < ncols; ++c) {
    uint16_t* h = s.h_bits + size_t(c) * I;
    for (int i = 0; i < I; ++i) {
      if (cols[c].emax < 0) {
        h[i] = canon::kCanonicalNaN;
        continue;
      }
      const uint16_t gv = canon::a16_row_output(s.sums[size_t(2 * i) * ncols + c], cols[c].emax, m.gate);
      const uint16_t uv = canon::a16_row_output(s.sums[size_t(2 * i + 1) * ncols + c], cols[c].emax, m.up);
      h[i] = canon::swiglu_bf16(gv, uv);
    }
    encode_column(h, I, s.h[c]);
    cols[c] = s.h[c].view();
  }
  sums_fn(record + geo.gu_bytes(), geo.d_blocks(), 0, geo.d_groups(), cols, ncols, s.sums);
  for (int c = 0; c < ncols; ++c) {
    for (int r = 0; r < H; ++r) {
      y[c][r] = cols[c].emax < 0 ? canon::kCanonicalNaN
                                 : canon::a16_row_output(s.sums[size_t(r) * ncols + c], cols[c].emax, m.down);
    }
  }
}

}  // namespace blaze::cpu
