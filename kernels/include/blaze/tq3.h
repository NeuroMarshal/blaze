// TQ3 KV rows (TurboQuant-style, BLZ_KV_TQ3): 3 bits a value for a head vector of 256 already rotated by the cache's
// randomized Hadamard transform (blz_rht_rows), so its values are close to independent Gaussians of one scale. A value
// is sign * kTq3Level[m] * scale: m a 2-bit magnitude index into the Lloyd-Max levels of N(0, 1) for 3 bits, the scale
// one FP16 per row, the least-squares fit of the row to its levels (TurboQuant's norm correction). 96 code bytes and 2
// scale bytes a row, against NVFP4's 128 + 16.
//
// Code layout (the register attention kernel's: thread c of an MMA fragment reads dims 64c .. 64c + 63 of a key, thread
// g dims 32g .. 32g + 31 of a value): four groups of 64 dims, 24 bytes each; in a group bytes 0..15 hold the
// magnitudes (byte s: dims 4s .. 4s + 3, two bits each from bit 0), bytes 16..23 the signs (dims 4s .. 4s + 3 in
// nibble s: byte 16 + s / 2, bits from 4 (s & 1)).
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#if defined(__CUDACC__)
#include <cuda_fp16.h>
#define BLZ_TQ_HD __host__ __device__ __forceinline__
#else
#define BLZ_TQ_HD inline
#endif

namespace blaze {

constexpr int kTq3Dim = 256;
constexpr int kTq3RowBytes = kTq3Dim * 3 / 8;  // 96
constexpr int kTq3GroupBytes = 24;
// The levels in FP16 (0.2454, 0.757, 1.345, 2.154: the Lloyd-Max quantizer of N(0, 1), MSE 0.0346) and the cuts
// between them (midpoints, of the unrounded levels).
constexpr uint16_t kTq3LevelBits[4] = {0x33da, 0x3a0e, 0x3d61, 0x404f};
constexpr float kTq3Cut0 = 0.50111f, kTq3Cut1 = 1.05058f, kTq3Cut2 = 1.74891f;
// byte_perm tables: the low and the high bytes of the four FP16 levels
constexpr uint32_t kTq3Lo = 0x4f610edau, kTq3Hi = 0x403d3a33u;

BLZ_TQ_HD int tq3_magnitude(float a) { return (a > kTq3Cut0) + (a > kTq3Cut1) + (a > kTq3Cut2); }

/// \brief Code byte offset and bit of the magnitude / sign of dim d in a row.
BLZ_TQ_HD int tq3_mag_byte(int d) { return (d / 64) * kTq3GroupBytes + (d % 64) / 4; }
BLZ_TQ_HD int tq3_mag_shift(int d) { return 2 * (d % 4); }
BLZ_TQ_HD int tq3_sign_byte(int d) { return (d / 64) * kTq3GroupBytes + 16 + (d % 64) / 8; }
BLZ_TQ_HD int tq3_sign_shift(int d) { return d % 8; }

// host references (the device code converts with cuda_fp16)
inline float tq3_f16_bits_to_float(uint16_t h) {
  const int sign = h >> 15, exp = (h >> 10) & 31, man = h & 1023;
  float v = exp == 0 ? std::ldexp(float(man), -24) : std::ldexp(float(1024 + man), exp - 25);
  return sign ? -v : v;
}
inline uint16_t tq3_float_to_f16_bits(float v) {  // round to nearest even, finite non-negative inputs below 65504
  if (!(v > 0.0f)) return 0;
  int e;
  std::frexp(v, &e);
  const int q = e - 11 > -24 ? e - 11 : -24;  // quantum 2^q (subnormals: 2^-24)
  long n = long(std::nearbyint(std::ldexp(double(v), -q)));
  int exp = q + 25;  // biased exponent of n * 2^q as 1.m * 2^(q + 10)
  if (n >= 2048) n >>= 1, ++exp;
  if (n < 1024) return uint16_t(n);  // subnormal
  if (exp > 30) return 0x7bff;
  return uint16_t((exp << 10) | (n - 1024));
}

/// \brief CPU reference: quantize one rotated row of 256 floats to TQ3 (codes: 96 bytes, scale: FP16 bits).
inline void tq3_quantize_row_ref(const float* x, uint8_t* codes, uint16_t* scale) {
  std::memset(codes, 0, kTq3RowBytes);
  double ss = 0;
  for (int d = 0; d < kTq3Dim; ++d) ss += double(x[d]) * x[d];
  const float rms = float(std::sqrt(ss / kTq3Dim));
  if (!(rms > 0.0f)) {
    *scale = 0;
    return;
  }
  double num = 0, den = 0;
  for (int d = 0; d < kTq3Dim; ++d) {
    const int m = tq3_magnitude(std::fabs(x[d]) / rms);
    const float level = tq3_f16_bits_to_float(kTq3LevelBits[m]);
    num += std::fabs(double(x[d])) * level, den += double(level) * level;
    codes[tq3_mag_byte(d)] |= uint8_t(m << tq3_mag_shift(d));
    if (x[d] < 0.0f) codes[tq3_sign_byte(d)] |= uint8_t(1 << tq3_sign_shift(d));
  }
  *scale = tq3_float_to_f16_bits(float(num / den));
}

/// \brief CPU reference: value of dim d of a TQ3 row, rounded to FP16 as the kernels' f16 product.
inline float tq3_value_ref(const uint8_t* codes, uint16_t scale, int d) {
  const int m = (codes[tq3_mag_byte(d)] >> tq3_mag_shift(d)) & 3;
  const float v = tq3_f16_bits_to_float(tq3_float_to_f16_bits(tq3_f16_bits_to_float(kTq3LevelBits[m]) *
                                                                tq3_f16_bits_to_float(scale)));
  return (codes[tq3_sign_byte(d)] >> tq3_sign_shift(d)) & 1 ? -v : v;
}

#if defined(__CUDACC__)
/// \brief Two TQ3 values as f16x2 times the row scale: `mags` the two 2-bit magnitudes (bits 0..3), `signs` their two
/// sign bits (bits 0, 1).
__device__ __forceinline__ __half2 tq3_pair(uint32_t mags, uint32_t signs, __half2 scale) {
  const uint32_t t = (mags & 3u) | ((mags & 0xcu) << 6);  // m0 in nibble 0, m1 in nibble 2
  uint32_t bits = __byte_perm(kTq3Lo, kTq3Hi, t * 0x11u + 0x4040u);
  bits ^= ((signs & 1u) << 15) | ((signs & 2u) << 30);
  return __hmul2(*reinterpret_cast<const __half2*>(&bits), scale);
}
#endif

}  // namespace blaze
