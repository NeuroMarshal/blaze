// rg16 packing with SSE (see blaze/repack.h and the record layout in blaze/expert.h).
#include "blaze/repack.h"

#include <immintrin.h>

#include "blaze/expert.h"

namespace blaze::expert {

namespace {

// 8 code bytes (16 codes, even k in the low nibble) -> 16 bytes, code k in byte k.
inline __m128i expand_codes(const uint8_t* p) {
  const __m128i mask = _mm_set1_epi8(0x0F);
  const __m128i x = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
  return _mm_unpacklo_epi8(_mm_and_si128(x, mask), _mm_and_si128(_mm_srli_epi16(x, 4), mask));
}

}  // namespace

void pack_groups(const PackSource& src, int k_dim, int g0, int g1, uint8_t* matrix) {
  const int blocks = k_dim / 16;
  for (int g = g0; g < g1; ++g) {
    const uint8_t* codes[kRowGroup];
    const uint8_t* scales[kRowGroup];
    for (int ri = 0; ri < kRowGroup; ++ri) {
      const int d = kRowGroup * g + ri;
      const int m = src.interleaved ? (d & 1) : 0;
      const size_t r = size_t((src.interleaved ? d >> 1 : d) - src.row0);
      codes[ri] = src.codes[m] + r * src.code_stride;
      scales[ri] = src.scales[m] + r * src.scale_stride;
    }
    uint8_t* u = matrix + unit_offset(g, 0, blocks);
    for (int b = 0; b < blocks; ++b, u += kUnitBytes) {
      // Byte 32q + 4i + j of the unit: row i (low nibble) and row i + 8 (high nibble) at k = 16b + 4q + j, so the
      // 16 bytes (row i | row i+8 << 4) for k = 16b .. 16b + 15 split into four dwords at 32q + 4i.
      for (int ri = 0; ri < 8; ++ri) {
        const __m128i v = _mm_or_si128(expand_codes(codes[ri] + 8 * b),
                                       _mm_slli_epi16(expand_codes(codes[ri + 8] + 8 * b), 4));
        uint8_t* w = u + 4 * ri;
        _mm_storeu_si32(w, v);
        _mm_storeu_si32(w + 32, _mm_srli_si128(v, 4));
        _mm_storeu_si32(w + 64, _mm_srli_si128(v, 8));
        _mm_storeu_si32(w + 96, _mm_srli_si128(v, 12));
      }
      for (int ri = 0; ri < kRowGroup; ++ri) u[kUnitCodeBytes + ri] = scales[ri][b];
    }
  }
}

}  // namespace blaze::expert
