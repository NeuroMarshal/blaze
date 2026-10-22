// Canonical routed-expert arithmetic (W4A16, exact integer sums). Shared verbatim by the GPU kernels and the
// CPU engine so that an expert's BF16 output does not depend on where it was computed.
//
// Algorithm after Infernix's design (docs/maintainer/qwen3_8-flash-next-design.md, sections 16.2 and 16.2.1,
// Apache-2.0), reimplemented here:
//   A16 encoding of a BF16 column v (K values):  M_j = 128|mant (normal) or mant (subnormal), e_j = max(E_j, 1),
//     emax = max_j e_j, X_j = sign_j * rne(M_j * 2^13 / 2^(emax - e_j)), |X_j| < 2^21, v_j = X_j * 2^(emax-147)
//     exactly for elements within 13 binades of the largest.
//   Row product: P_b = sum_j c2(code) * X_j (exact int32), S = sum_b P_b * S^(scale) (exact int64, any order),
//     y = bf16_rn((fl32_rn(S) * 2^(emax - 157)) * m), m = fl32(1 / global weight scale).
//   SwiGLU: h = bf16_rn(silu_c(g) * u) with exp_c below; then h is A16-encoded for the down projection.
//
// Rules: every floating-point step is an explicit IEEE operation (device: __fmaf_rn / __fmul_rn / __fadd_rn /
// __fdiv_rn; host: __builtin_fmaf and plain operators compiled with -ffp-contract=off, no fast math).
#pragma once

#include <cstdint>
#include <cstring>

#include "blaze/nvfp4.h"

#if defined(__CUDA_ARCH__)
#define BLZ_FMA(a, b, c) __fmaf_rn((a), (b), (c))
#define BLZ_MUL(a, b) __fmul_rn((a), (b))
#define BLZ_ADD(a, b) __fadd_rn((a), (b))
#define BLZ_DIV(a, b) __fdiv_rn((a), (b))
#define BLZ_I64_TO_F32(x) __ll2float_rn(x)
#define BLZ_RINT(x) rintf(x)
#else
#define BLZ_FMA(a, b, c) __builtin_fmaf((a), (b), (c))
#define BLZ_MUL(a, b) ((a) * (b))
#define BLZ_ADD(a, b) ((a) + (b))
#define BLZ_DIV(a, b) ((a) / (b))
#define BLZ_I64_TO_F32(x) static_cast<float>(x)
#define BLZ_RINT(x) __builtin_rintf(x)
#endif

namespace blaze::canon {

BLZ_HD uint32_t f32_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}

BLZ_HD float bits_f32(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

/// \brief 2^k for k in [-126, 127] (a normal float, exact).
BLZ_HD float pow2(int k) { return bits_f32(uint32_t(k + 127) << 23); }

/// \brief Round FP32 to BF16 bits, nearest even; any NaN becomes the canonical 0x7FC0.
BLZ_HD uint16_t bf16_rn(float f) {
  const uint32_t u = f32_bits(f);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return 0x7FC0;
  return uint16_t((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

BLZ_HD float bf16_to_f32(uint16_t b) { return bits_f32(uint32_t(b) << 16); }

/// \brief Doubled E2M1 value of a 4-bit code: {0, 1, 2, 3, 4, 6, 8, 12} with the code's sign.
BLZ_HD int c2(int code) {
  const int idx = code & 7;
  const int mag = idx < 5 ? idx : (idx == 5 ? 6 : (idx == 6 ? 8 : 12));
  return (code & 8) ? -mag : mag;
}

/// \brief E4M3FN scale word times 2^9 as an integer: (e == 0) ? m : (8 + m) << (e - 1).
BLZ_HD int32_t s_hat(uint8_t w) {
  const int e = (w >> 3) & 15, m = w & 7;
  return e == 0 ? m : (8 + m) << (e - 1);
}

/// \brief Scale an integer sum to its float value: fl32_rn(S) * 2^k with a single rounding.
/// The first factor keeps the value normal (|fl32(S)| >= 1), so only the second can round (into subnormals).
BLZ_HD float scale_sum(int64_t s, int k) {
  float f = BLZ_I64_TO_F32(s);
  if (k > 127) {
    f = BLZ_MUL(f, pow2(127));
    k -= 127;
  } else if (k < -126) {
    f = BLZ_MUL(f, pow2(-126));
    k += 126;
    if (k < -126) {  // two more halvings at most (k >= -157 - 126 + ... is bounded by the caller's emax range)
      f = BLZ_MUL(f, pow2(-126));
      k += 126;
    }
  }
  return BLZ_MUL(f, pow2(k));
}

/// \brief exp_c: Cody-Waite reduction and a degree-7 Taylor polynomial by explicit FMAs.
/// Defined overflow (+inf above 88.72283) and underflow (+0 below -103.97208).
BLZ_HD float exp_c(float x) {
  if (!(x == x)) return x;
  if (x > 88.72283f) return bits_f32(0x7F800000u);
  if (x < -103.97208f) return 0.0f;
  const float log2e = 1.44269504088896341f;
  const float ln2_hi = 0.693145751953125f;   // 13 bits of mantissa: n * ln2_hi is exact for |n| < 2^11
  const float ln2_lo = 1.42860682030941723e-6f;
  const float n = BLZ_RINT(BLZ_MUL(x, log2e));
  float r = BLZ_FMA(-n, ln2_hi, x);
  r = BLZ_FMA(-n, ln2_lo, r);
  // Horner: 1 + r(1 + r/2(1 + r/3(...))) written with the Taylor coefficients 1/k!.
  float p = 1.98412698e-4f;                  // 1/5040
  p = BLZ_FMA(p, r, 1.38888889e-3f);         // 1/720
  p = BLZ_FMA(p, r, 8.33333333e-3f);         // 1/120
  p = BLZ_FMA(p, r, 4.16666667e-2f);         // 1/24
  p = BLZ_FMA(p, r, 1.66666667e-1f);         // 1/6
  p = BLZ_FMA(p, r, 0.5f);
  p = BLZ_FMA(p, r, 1.0f);
  p = BLZ_FMA(p, r, 1.0f);
  int k = static_cast<int>(n);
  // p in about [0.7, 1.42]: scale by 2^k in two exact steps so a subnormal result is rounded once.
  if (k < -126) {
    p = BLZ_MUL(p, pow2(-126));
    k += 126;
  } else if (k > 127) {
    p = BLZ_MUL(p, pow2(127));
    k -= 127;
  }
  return BLZ_MUL(p, pow2(k));
}

/// \brief SiLU with exp_c; two branches keep the sign and magnitude for large negative g.
BLZ_HD float silu_c(float g) {
  if (g >= 0.0f) return BLZ_DIV(g, BLZ_ADD(1.0f, exp_c(-g)));
  const float e = exp_c(g);
  return BLZ_DIV(BLZ_MUL(g, e), BLZ_ADD(1.0f, e));
}

/// \brief SwiGLU of BF16 gate/up outputs to BF16 h.
BLZ_HD uint16_t swiglu_bf16(uint16_t g, uint16_t u) {
  return bf16_rn(BLZ_MUL(silu_c(bf16_to_f32(g)), bf16_to_f32(u)));
}

/// \brief Exponent field used by the A16 encoding: max(E, 1).
BLZ_HD int a16_exp(uint16_t b) {
  const int e = (b >> 7) & 0xFF;
  return e < 1 ? 1 : e;
}

BLZ_HD bool bf16_nonfinite(uint16_t b) { return ((b >> 7) & 0xFF) == 0xFF; }

/// \brief A16 code of one BF16 element given the column's emax (|X| < 2^21).
BLZ_HD int32_t a16_encode(uint16_t b, int emax) {
  const int e_field = (b >> 7) & 0xFF;
  const int32_t m = (e_field != 0 ? 128 : 0) | (b & 0x7F);
  const int shift = emax - (e_field < 1 ? 1 : e_field);  // >= 0
  int32_t x;
  if (shift <= 13) {
    x = m << (13 - shift);
  } else {
    const int s = shift - 13;
    if (s >= 9) {
      x = 0;  // m < 2^8: below half a unit
    } else {
      const int32_t q = m >> s, rem = m & ((1 << s) - 1), half = 1 << (s - 1);
      x = q + ((rem > half || (rem == half && (q & 1))) ? 1 : 0);
    }
  }
  return (b & 0x8000) ? -x : x;
}

/// \brief Final value of a row: bf16_rn((fl32_rn(S) * 2^(emax - 157)) * m).
BLZ_HD uint16_t a16_row_output(int64_t s, int emax, float m) {
  return bf16_rn(BLZ_MUL(scale_sum(s, emax - 157), m));
}

constexpr uint16_t kCanonicalNaN = 0x7FC0;

}  // namespace blaze::canon
