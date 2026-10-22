// NVFP4 helpers shared by host and device code: E2M1 rounding, E4M3 conversion, and the CPU reference
// quantizer used by the tests (and by host-side tools).
//
// NVFP4 layout used everywhere in blaze: per 16 values one E4M3 block scale and 16 E2M1 codes, two codes per
// byte with the even element in the low nibble, plus an FP32 global scale g:
//     x ~= e2m1(code) * e4m3(scale) * g
// Code bits: [2:0] magnitude index into {0, 0.5, 1, 1.5, 2, 3, 4, 6}, bit 3 sign.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#include <cuda_fp8.h>
#define BLZ_HD __host__ __device__ __forceinline__
#else
#define BLZ_HD inline
#endif

namespace blaze {

constexpr int kNvfp4Block = 16;
constexpr float kE4m3Max = 448.0f;
// Block scale search of the four-over-six store: besides the max/6 and max/4 scales, the E4M3 codes these many steps
// from max/6's. On Qwen3.6 K/V the steps -1..5 give 95% of an exhaustive search's gain: tensor error 0.0868 -> 0.0824
// (max/4 alone: 0.0868, max/6 alone: 0.0925); one step below max/6 clips the block maximum yet wins 18% of blocks.
constexpr int kScaleSearchFirst = -1;
constexpr int kScaleSearchLast = 5;

/// \brief Nearest E2M1 magnitude index for a >= 0, ties to the even code (as cvt.rn.satfinite.e2m1x2.f32).
BLZ_HD int e2m1_index(float a) {
  return (a > 0.25f) + (a >= 0.75f) + (a > 1.25f) + (a >= 1.75f) + (a > 2.5f) + (a >= 3.5f) + (a > 5.0f);
}

/// \brief Magnitude of an E2M1 index (0..7).
BLZ_HD float e2m1_level(int idx) { return idx < 4 ? 0.5f * idx : (idx < 7 ? float(idx - 2) : 6.0f); }

/// \brief Value of a 4-bit E2M1 code (sign in bit 3).
BLZ_HD float e2m1_value(int code) {
  const float m = e2m1_level(code & 7);
  return (code & 8) ? -m : m;
}

/// \brief E4M3 (fn: no infinities, max 448) bits to float.
BLZ_HD float e4m3_to_float(uint8_t b) {
#if defined(__CUDA_ARCH__)
  __nv_fp8_e4m3 q;
  q.__x = b;
  return float(q);
#else
  const int sign = b >> 7, exp = (b >> 3) & 15, man = b & 7;
  float v;
  if (exp == 15 && man == 7) {
    v = NAN;
  } else if (exp == 0) {
    v = ldexpf(float(man), -9);  // subnormal: man / 8 * 2^-6
  } else {
    v = ldexpf(float(8 + man), exp - 10);  // (1 + man / 8) * 2^(exp - 7)
  }
  return sign ? -v : v;
#endif
}

/// \brief Round a non-negative float to E4M3 bits, nearest with ties to even, saturating at 448.
/// Host reference for __nv_fp8_e4m3(float) (cvt.rn.satfinite.e4m3x2.f32) on non-negative finite inputs.
inline uint8_t float_to_e4m3_host(float v) {
  if (!(v > 0.0f)) return 0;
  if (v >= kE4m3Max) return 0x7E;
  int e;
  std::frexp(v, &e);  // v = f * 2^e, f in [0.5, 1)
  int exp = e - 1 + 7;  // biased exponent of v as 1.m * 2^(e-1)
  // Quantum: 2^(e-1-3) for normals, 2^-9 for subnormals (exp <= 0).
  const int q_exp = exp >= 1 ? e - 1 - 3 : -9;
  const double scaled = std::ldexp(double(v), -q_exp);  // exact in double
  double r = std::nearbyint(scaled);                     // ties to even (default rounding mode)
  long n = long(r);
  // n counts quanta; rebuild bits, letting a carry move to the next binade.
  if (exp >= 1) {
    if (n >= 16) {  // carried into the next binade
      n >>= 1;
      ++exp;
    }
    if (exp > 15 || (exp == 15 && n - 8 > 6)) return 0x7E;
    return uint8_t((exp << 3) | (n - 8));
  }
  if (n >= 8) return uint8_t((1 << 3) | (n - 8));  // subnormal rounded up to the smallest normal
  return uint8_t(n);
}

/// \brief CPU reference: quantize one row of `dim` floats (dim % 16 == 0) to NVFP4.
/// \param x            input values
/// \param dim          row length
/// \param g            global scale (> 0)
/// \param four_over_six search the block scale (max/6, max/4, then the codes kScaleSearchFirst..kScaleSearchLast
///                     steps from max/6's) for the lowest squared reconstruction error; else max/6
/// \param packed       out: dim/2 bytes
/// \param scales       out: dim/16 E4M3 bytes
inline void nvfp4_quantize_row_ref(const float* x, int dim, float g, bool four_over_six, uint8_t* packed,
                                   uint8_t* scales) {
  for (int b = 0; b < dim / kNvfp4Block; ++b) {
    const float* xb = x + b * kNvfp4Block;
    float amax = 0.0f;
    for (int i = 0; i < kNvfp4Block; ++i) amax = std::fmax(amax, std::fabs(xb[i]));
    const int lo = float_to_e4m3_host(std::fmin(amax / (6.0f * g), kE4m3Max));
    int cand[2 + kScaleSearchLast - kScaleSearchFirst];
    int ncand = 0;
    cand[ncand++] = lo;
    if (four_over_six) {
      cand[ncand++] = float_to_e4m3_host(std::fmin(amax / (4.0f * g), kE4m3Max));
      for (int s = kScaleSearchFirst; s <= kScaleSearchLast; ++s)
        if (s != 0) cand[ncand++] = lo + s;
    }
    uint8_t best_codes[kNvfp4Block];
    uint8_t best_scale = 0;
    float best_err = INFINITY;
    for (int c = 0; c < ncand; ++c) {
      if (cand[c] < 0 || cand[c] > 0x7E) continue;
      const uint8_t sb = uint8_t(cand[c]);
      const float t = e4m3_to_float(sb) * g;
      const float inv = t > 0.0f ? 1.0f / t : 0.0f;
      uint8_t codes[kNvfp4Block];
      float half[2] = {0.0f, 0.0f};  // summed per 8 values and then added, as the CUDA store's lane pair does
      for (int i = 0; i < kNvfp4Block; ++i) {
        const float v = xb[i] * inv;
        const int idx = e2m1_index(std::fabs(v));
        const float d = (v < 0.0f ? -e2m1_level(idx) : e2m1_level(idx)) * t - xb[i];
        half[i / 8] += d * d;
        codes[i] = uint8_t(idx | (v < 0.0f ? 8 : 0));
      }
      const float err = half[0] + half[1];
      if (c == 0 || err < best_err) {  // a later candidate wins only when strictly better
        best_err = err;
        best_scale = sb;
        std::memcpy(best_codes, codes, sizeof(codes));
      }
    }
    scales[b] = best_scale;
    for (int i = 0; i < kNvfp4Block; i += 2) {
      packed[(b * kNvfp4Block + i) / 2] = uint8_t(best_codes[i] | (best_codes[i + 1] << 4));
    }
  }
}

/// \brief CPU reference: dequantize one NVFP4 row.
inline void nvfp4_dequant_row_ref(const uint8_t* packed, const uint8_t* scales, int dim, float g, float* out) {
  for (int i = 0; i < dim; ++i) {
    const int code = (packed[i / 2] >> (4 * (i & 1))) & 15;
    out[i] = e2m1_value(code) * e4m3_to_float(scales[i / kNvfp4Block]) * g;
  }
}

}  // namespace blaze
