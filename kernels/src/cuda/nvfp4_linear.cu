// NVFP4 W4A16 linear layers (blaze/nvfp4_linear.h).
//
// GEMV: one warp per R output rows; lane l reads 16-byte chunks (32 codes = two scale blocks) l, l + 32, ... of each
// row, so a warp load covers 512 contiguous bytes. Codes become floats through a 256-entry table (byte -> two
// values) in shared memory; each block's 16 products are summed in FP32 and multiplied by the block scale once.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>


#include "blaze/nvfp4.h"
#include "blaze/nvfp4_linear.h"

namespace {

constexpr int kWarps = 4;

// the format's decoders (blaze/nvfp4.h)
__device__ __forceinline__ float e2m1(int code) { return blaze::e2m1_value(code); }
__device__ __forceinline__ float e4m3(uint8_t v) { return blaze::e4m3_to_float(v); }

__device__ __forceinline__ uint4 ldg_stream(const void* p) {
  uint4 v;
  asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

__device__ __forceinline__ float2 bf2(uint32_t v) {
  return make_float2(__uint_as_float(v << 16), __uint_as_float(v & 0xffff0000u));
}

// Sum over one 32-bit word of codes (8 values) times 8 inputs (4 words of bf16 pairs).
__device__ __forceinline__ float dot8(uint32_t codes, const uint32_t* xw, const float2* lut) {
  float s = 0.0f;
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    const float2 w = lut[(codes >> (8 * b)) & 0xff];
    const float2 x = bf2(xw[b]);
    s = fmaf(w.x, x.x, s);
    s = fmaf(w.y, x.y, s);
  }
  return s;
}

template <int M, int R>
__global__ void __launch_bounds__(kWarps * 32) nvfp4_gemv_kernel(const uint16_t* __restrict__ x, int64_t ldx,
                                                                 const uint8_t* __restrict__ w,
                                                                 const uint8_t* __restrict__ scale,
                                                                 const float* __restrict__ alpha,
                                                                 uint16_t* __restrict__ y, int64_t ldy, int n, int k) {
  __shared__ float2 lut[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) lut[i] = make_float2(e2m1(i & 15), e2m1(i >> 4));
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row0 = (blockIdx.x * kWarps + warp) * R;
  if (row0 >= n) return;
  const int chunks = k >> 5;
  const int64_t wrow = k >> 1, srow = k >> 4;
  float acc[R][M];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = 0.0f;

  // Register double buffer: the next chunk's codes and scales are in flight while this one is summed.
  uint4 wv[R];
  uint16_t sv[R];
  auto load = [&](int c, uint4* dw, uint16_t* ds) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int row = min(row0 + r, n - 1);
      dw[r] = ldg_stream(w + row * wrow + c * 16);
      ds[r] = __ldg(reinterpret_cast<const uint16_t*>(scale + row * srow + c * 2));
    }
  };
  if (lane < chunks) load(lane, wv, sv);
  for (int c = lane; c < chunks; c += 32) {
    uint4 nw[R];
    uint16_t ns[R];
    if (c + 32 < chunks) load(c + 32, nw, ns);
    float s0[R], s1[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      s0[r] = e4m3(sv[r] & 0xff);
      s1[r] = e4m3(sv[r] >> 8);
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const uint4* xp = reinterpret_cast<const uint4*>(x + m * ldx + c * 32);
      uint32_t xw[16];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint4 v = __ldg(xp + q);
        xw[4 * q] = v.x;
        xw[4 * q + 1] = v.y;
        xw[4 * q + 2] = v.z;
        xw[4 * q + 3] = v.w;
      }
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const float b0 = dot8(wv[r].x, xw, lut) + dot8(wv[r].y, xw + 4, lut);
        const float b1 = dot8(wv[r].z, xw + 8, lut) + dot8(wv[r].w, xw + 12, lut);
        acc[r][m] = fmaf(b0, s0[r], fmaf(b1, s1[r], acc[r][m]));
      }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      wv[r] = nw[r];
      sv[r] = ns[r];
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float v = acc[r][m];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      acc[r][m] = v;
    }
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int row = row0 + r;
      if (row >= n) break;
      const float a = alpha[row];
#pragma unroll
      for (int m = 0; m < M; ++m) y[m * ldy + row] = __bfloat16_as_ushort(__float2bfloat16(acc[r][m] * a));
    }
  }
}

template <int M, int R>
cudaError_t launch_gemv(const uint16_t* x, int64_t ldx, const uint8_t* w, const uint8_t* scale, const float* alpha,
                        uint16_t* y, int64_t ldy, int n, int k, cudaStream_t stream) {
  const int rows_per_block = kWarps * R;
  nvfp4_gemv_kernel<M, R><<<(n + rows_per_block - 1) / rows_per_block, kWarps * 32, 0, stream>>>(x, ldx, w, scale,
                                                                                                 alpha, y, ldy, n, k);
  return cudaGetLastError();
}

// tensor-core route
// For 2..16 rows: mma.m16n8k16 (BF16 in, FP32 accumulate) with A = the input rows (tokens, zero-padded to 16) and
// B = 8 weight rows. The reduction order is free, so the K index is permuted to fit the stored layout: over a 128-wide
// chunk, thread (g, t) of a warp holds the 32 consecutive codes [32t, 32t + 32) of weight row g (one 16-byte load,
// two scale blocks), and mma step s takes its codes 4s .. 4s + 3 as the fragment's k slots (2t, 2t+1, 2t+8, 2t+9);
// the input fragments are read with the same permutation. Codes become BF16 through two byte-permute table lookups
// (the E2M1 magnitudes are exact in BF16), times the block scale (exact). Four warps split K, partial sums meet in
// shared memory; each block covers 8 * kTiles weight rows.


__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Four E2M1 codes (16 bits, first code lowest) -> two BF16 pairs.
__device__ __forceinline__ void e2m1x4_to_bf16(uint32_t q, uint32_t& w0, uint32_t& w1) {
  const uint32_t sel = q & 0x7777u;
  const uint32_t hi = __byte_perm(0x3F3F3F00u, 0x40404040u, sel);  // high bytes of 0, .5, 1, 1.5, 2, 3, 4, 6
  const uint32_t lo = __byte_perm(0xC0800000u, 0xC0804000u, sel);  // low bytes
  w0 = __byte_perm(lo, hi, 0x5140) | ((q & 0x8u) << 12) | ((q & 0x80u) << 24);
  w1 = __byte_perm(lo, hi, 0x7362) | ((q & 0x800u) << 4) | ((q & 0x8000u) << 16);
}

__device__ __forceinline__ uint32_t bf16x2_mul(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ uint32_t bf16x2_splat(float v) {
  const uint32_t h = __bfloat16_as_ushort(__float2bfloat16(v));
  return h | (h << 16);
}

// kMT 16-row tiles of tokens (up to 64): the weight fragments are widened once per chunk and multiply every tile.
template <int kTiles, int kSplit, bool kHi, int kMT>
__global__ void __launch_bounds__(kSplit * 32) nvfp4_mma_kernel(const uint16_t* __restrict__ x, int64_t ldx,
                                                                const uint8_t* __restrict__ w,
                                                                const uint8_t* __restrict__ scale,
                                                                const float* __restrict__ alpha,
                                                                uint16_t* __restrict__ y, int64_t ldy, int m, int n,
                                                                int k) {
  constexpr int kTok = 16 * kMT;
  __shared__ float red[kSplit][kTiles * 8][kTok + 1];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int f0 = blockIdx.x * kTiles * 8;
  const int64_t wrow = k >> 1, srow = k >> 4;
  const uint8_t* wp[kTiles];
  const uint8_t* sp[kTiles];
#pragma unroll
  for (int j = 0; j < kTiles; ++j) {
    const int f = min(f0 + 8 * j + g, n - 1);
    wp[j] = w + f * wrow + 16 * t;
    sp[j] = scale + f * srow + 2 * t;
  }
  float acc[kTiles][kMT][4];
#pragma unroll
  for (int j = 0; j < kTiles; ++j)
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt) acc[j][mt][0] = acc[j][mt][1] = acc[j][mt][2] = acc[j][mt][3] = 0.0f;

  const int chunks = k >> 7;
  // Register double buffer of the weight codes and scales (the input rows come from L1/L2).
  uint4 wq[kTiles];
  uint16_t sq[kTiles];
  auto load = [&](int c, uint4* dw, uint16_t* ds) {
#pragma unroll
    for (int j = 0; j < kTiles; ++j) {
      dw[j] = ldg_stream(wp[j] + c * 64);
      ds[j] = __ldg(reinterpret_cast<const uint16_t*>(sp[j] + c * 8));
    }
  };
  if (warp < chunks) load(warp, wq, sq);
  for (int c = warp; c < chunks; c += kSplit) {
    uint4 nw[kTiles];
    uint16_t ns[kTiles];
    if (c + kSplit < chunks) load(c + kSplit, nw, ns);
    // the chunk's weight fragments of every step, scaled
    uint32_t bq[kTiles][8][2];
#pragma unroll
    for (int j = 0; j < kTiles; ++j) {
      const uint32_t s0 = bf16x2_splat(e4m3(sq[j] & 0xff)), s1 = bf16x2_splat(e4m3(sq[j] >> 8));
      const uint32_t codes[4] = {wq[j].x, wq[j].y, wq[j].z, wq[j].w};
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        uint32_t b0, b1;
        e2m1x4_to_bf16(codes[s >> 1] >> (16 * (s & 1)), b0, b1);
        const uint32_t sc = s < 4 ? s0 : s1;
        bq[j][s][0] = bf16x2_mul(b0, sc), bq[j][s][1] = bf16x2_mul(b1, sc);
      }
      wq[j] = nw[j];
      sq[j] = ns[j];
    }
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt) {
      const int rlo = 16 * mt + g, rhi = rlo + 8;
      const bool lo_ok = rlo < m, hi_ok = kHi && rhi < m;
      const uint16_t* xlo = x + int64_t(lo_ok ? rlo : 0) * ldx + 32 * t;
      const uint16_t* xhi = x + int64_t(hi_ok ? rhi : 0) * ldx + 32 * t;
      uint32_t xa[16], xb[16];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint4 v = lo_ok ? __ldg(reinterpret_cast<const uint4*>(xlo + c * 128) + q) : make_uint4(0, 0, 0, 0);
        xa[4 * q] = v.x, xa[4 * q + 1] = v.y, xa[4 * q + 2] = v.z, xa[4 * q + 3] = v.w;
        if (kHi) {
          const uint4 u = hi_ok ? __ldg(reinterpret_cast<const uint4*>(xhi + c * 128) + q) : make_uint4(0, 0, 0, 0);
          xb[4 * q] = u.x, xb[4 * q + 1] = u.y, xb[4 * q + 2] = u.z, xb[4 * q + 3] = u.w;
        } else {
          xb[4 * q] = xb[4 * q + 1] = xb[4 * q + 2] = xb[4 * q + 3] = 0;
        }
      }
#pragma unroll
      for (int j = 0; j < kTiles; ++j)
#pragma unroll
        for (int s = 0; s < 8; ++s)
          mma_bf16(acc[j][mt], xa[2 * s], xb[2 * s], xa[2 * s + 1], xb[2 * s + 1], bq[j][s][0], bq[j][s][1]);
    }
  }
  // c0, c1: token 16 mt + g, rows 8j + 2t, +1; c2, c3: token 16 mt + g + 8
#pragma unroll
  for (int j = 0; j < kTiles; ++j)
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt) {
      red[warp][8 * j + 2 * t][16 * mt + g] = acc[j][mt][0];
      red[warp][8 * j + 2 * t + 1][16 * mt + g] = acc[j][mt][1];
      red[warp][8 * j + 2 * t][16 * mt + g + 8] = acc[j][mt][2];
      red[warp][8 * j + 2 * t + 1][16 * mt + g + 8] = acc[j][mt][3];
    }
  __syncthreads();
  for (int i = threadIdx.x; i < kTiles * 8 * kTok; i += blockDim.x) {
    const int r = i / kTok, tok = i % kTok, f = f0 + r;
    if (tok >= m || f >= n) continue;
    float v = 0.0f;
#pragma unroll
    for (int q = 0; q < kSplit; ++q) v += red[q][r][tok];
    y[tok * ldy + f] = __bfloat16_as_ushort(__float2bfloat16(v * alpha[f]));
  }
}

// FP16 inputs and FP16 accumulation within each 128-wide chunk of K (8 mma steps), FP32 across chunks: GeForce cards run
// FP16-accumulate mma at twice the FP32-accumulate rate, which the BF16 kernel is bound by from ~16 rows. The codes
// become FP16 exactly (E2M1 magnitudes have zero low bytes in FP16; times an E4M3 block scale: at most 6 significant
// bits); the inputs come from nvfp4_f16_rows_kernel, each row scaled by a power of two to max |x| <= 64, and the block
// scales by 2^-wshift (the weight's largest scale to <= 1; ModelOpt's span the whole E4M3 range), so that a chunk's
// partial sums stay below 128 * 64 * 6 < 65504; both powers of two come back in the epilogue.
__device__ __forceinline__ void mma_f16(uint32_t* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
      : "+r"(c[0]), "+r"(c[1])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Four E2M1 codes (16 bits, first code lowest) -> two FP16 pairs.
__device__ __forceinline__ void e2m1x4_to_f16(uint32_t q, uint32_t& w0, uint32_t& w1) {
  const uint32_t hi = __byte_perm(0x3E3C3800u, 0x46444240u, q & 0x7777u);  // high bytes of 0, .5, 1, 1.5, 2, 3, 4, 6
  w0 = __byte_perm(hi, 0u, 0x1404) | ((q & 0x8u) << 12) | ((q & 0x80u) << 24);
  w1 = __byte_perm(hi, 0u, 0x3424) | ((q & 0x800u) << 4) | ((q & 0x8000u) << 16);
}

__device__ __forceinline__ uint32_t f16x2_mul(uint32_t a, uint32_t b) {
  __half2 r = __hmul2(*reinterpret_cast<__half2*>(&a), *reinterpret_cast<__half2*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ uint32_t f16x2_splat(float v) {
  const uint32_t h = __half_as_ushort(__float2half_rn(v));
  return h | (h << 16);
}

template <int kTiles, int kSplit, bool kHi, int kMT>
__global__ void __launch_bounds__(kSplit * 32) nvfp4_mma_f16_kernel(const uint16_t* __restrict__ x, int64_t ldx,
                                                                    const int32_t* __restrict__ shift,
                                                                    const uint8_t* __restrict__ w,
                                                                    const uint8_t* __restrict__ scale,
                                                                    const float* __restrict__ alpha,
                                                                    uint16_t* __restrict__ y, int64_t ldy, int m, int n,
                                                                    int k, int wshift) {
  constexpr int kTok = 16 * kMT;
  __shared__ float red[kSplit][kTiles * 8][kTok + 1];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int f0 = blockIdx.x * kTiles * 8;
  const int64_t wrow = k >> 1, srow = k >> 4;
  const float wscale = ldexpf(1.0f, -wshift);
  const uint8_t* wp[kTiles];
  const uint8_t* sp[kTiles];
#pragma unroll
  for (int j = 0; j < kTiles; ++j) {
    const int f = min(f0 + 8 * j + g, n - 1);
    wp[j] = w + f * wrow + 16 * t;
    sp[j] = scale + f * srow + 2 * t;
  }
  float acc[kTiles][kMT][4];
#pragma unroll
  for (int j = 0; j < kTiles; ++j)
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt) acc[j][mt][0] = acc[j][mt][1] = acc[j][mt][2] = acc[j][mt][3] = 0.0f;
  const int chunks = k >> 7;
  uint4 wq[kTiles];
  uint16_t sq[kTiles];
  auto load = [&](int c, uint4* dw, uint16_t* ds) {
#pragma unroll
    for (int j = 0; j < kTiles; ++j) {
      dw[j] = ldg_stream(wp[j] + c * 64);
      ds[j] = __ldg(reinterpret_cast<const uint16_t*>(sp[j] + c * 8));
    }
  };
  if (warp < chunks) load(warp, wq, sq);
  for (int c = warp; c < chunks; c += kSplit) {
    uint4 nw[kTiles];
    uint16_t ns[kTiles];
    if (c + kSplit < chunks) load(c + kSplit, nw, ns);
    uint32_t bq[kTiles][8][2];
#pragma unroll
    for (int j = 0; j < kTiles; ++j) {
      const uint32_t s0 = f16x2_splat(e4m3(sq[j] & 0xff) * wscale), s1 = f16x2_splat(e4m3(sq[j] >> 8) * wscale);
      const uint32_t codes[4] = {wq[j].x, wq[j].y, wq[j].z, wq[j].w};
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        uint32_t b0, b1;
        e2m1x4_to_f16(codes[s >> 1] >> (16 * (s & 1)), b0, b1);
        const uint32_t sc = s < 4 ? s0 : s1;
        bq[j][s][0] = f16x2_mul(b0, sc), bq[j][s][1] = f16x2_mul(b1, sc);
      }
      wq[j] = nw[j];
      sq[j] = ns[j];
    }
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt) {
      const int rlo = 16 * mt + g, rhi = rlo + 8;
      const bool lo_ok = rlo < m, hi_ok = kHi && rhi < m;
      const uint16_t* xlo = x + int64_t(lo_ok ? rlo : 0) * ldx + 32 * t;
      const uint16_t* xhi = x + int64_t(hi_ok ? rhi : 0) * ldx + 32 * t;
      uint32_t xa[16], xb[16];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint4 v = lo_ok ? __ldg(reinterpret_cast<const uint4*>(xlo + c * 128) + q) : make_uint4(0, 0, 0, 0);
        xa[4 * q] = v.x, xa[4 * q + 1] = v.y, xa[4 * q + 2] = v.z, xa[4 * q + 3] = v.w;
        if (kHi) {
          const uint4 u = hi_ok ? __ldg(reinterpret_cast<const uint4*>(xhi + c * 128) + q) : make_uint4(0, 0, 0, 0);
          xb[4 * q] = u.x, xb[4 * q + 1] = u.y, xb[4 * q + 2] = u.z, xb[4 * q + 3] = u.w;
        } else {
          xb[4 * q] = xb[4 * q + 1] = xb[4 * q + 2] = xb[4 * q + 3] = 0;
        }
      }
#pragma unroll
      for (int j = 0; j < kTiles; ++j) {
        uint32_t h[2] = {0u, 0u};
#pragma unroll
        for (int s = 0; s < 8; ++s) mma_f16(h, xa[2 * s], xb[2 * s], xa[2 * s + 1], xb[2 * s + 1], bq[j][s][0], bq[j][s][1]);
        const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&h[0]));
        const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&h[1]));
        acc[j][mt][0] += lo.x, acc[j][mt][1] += lo.y, acc[j][mt][2] += hi.x, acc[j][mt][3] += hi.y;
      }
    }
  }
#pragma unroll
  for (int j = 0; j < kTiles; ++j)
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt) {
      red[warp][8 * j + 2 * t][16 * mt + g] = acc[j][mt][0];
      red[warp][8 * j + 2 * t + 1][16 * mt + g] = acc[j][mt][1];
      red[warp][8 * j + 2 * t][16 * mt + g + 8] = acc[j][mt][2];
      red[warp][8 * j + 2 * t + 1][16 * mt + g + 8] = acc[j][mt][3];
    }
  __syncthreads();
  for (int i = threadIdx.x; i < kTiles * 8 * kTok; i += blockDim.x) {
    const int r = i / kTok, tok = i % kTok, f = f0 + r;
    if (tok >= m || f >= n) continue;
    float v = 0.0f;
#pragma unroll
    for (int q = 0; q < kSplit; ++q) v += red[q][r][tok];
    y[tok * ldy + f] = __bfloat16_as_ushort(__float2bfloat16(ldexpf(v * alpha[f], shift[tok] + wshift)));
  }
}

// grid: rows, block: 256. Row r of x (BF16) as FP16 times 2^-shift[r], max |x| in (32, 64] (0: shift 0).
__global__ void nvfp4_f16_rows_kernel(const uint16_t* __restrict__ x, int64_t ldx, int k, uint16_t* __restrict__ out,
                                      int32_t* __restrict__ shift) {
  const int r = blockIdx.x;
  const uint16_t* xr = x + int64_t(r) * ldx;
  float mx = 0.0f;
  for (int i = threadIdx.x; i < k; i += blockDim.x) mx = fmaxf(mx, fabsf(__bfloat162float(__ushort_as_bfloat16(xr[i]))));
  __shared__ float red[8];
#pragma unroll
  for (int o = 16; o > 0; o /= 2) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
  __syncthreads();
  mx = red[0];
#pragma unroll
  for (int i = 1; i < 8; ++i) mx = fmaxf(mx, red[i]);
  int e = 0;
  if (mx > 0.0f && isfinite(mx)) {
    frexpf(mx / 64.0f, &e);  // mx / 64 = f * 2^e, f in [0.5, 1): mx * 2^-e in [32, 64)
    e = max(-60, min(60, e));
  }
  if (threadIdx.x == 0) shift[r] = e;
  uint16_t* o = out + int64_t(r) * k;
  for (int i = threadIdx.x; i < k; i += blockDim.x)
    o[i] = __half_as_ushort(__float2half_rn(ldexpf(__bfloat162float(__ushort_as_bfloat16(xr[i])), -e)));
}

__global__ void nvfp4_dequant_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ scale,
                                     const float* __restrict__ alpha, int n, int k, uint16_t* __restrict__ out) {
  const int64_t chunks = int64_t(n) * (k >> 5);
  for (int64_t c = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; c < chunks; c += int64_t(gridDim.x) * blockDim.x) {
    const int64_t row = c / (k >> 5), col = c % (k >> 5);
    const uint4 v = *reinterpret_cast<const uint4*>(w + row * (k >> 1) + col * 16);
    const uint32_t words[4] = {v.x, v.y, v.z, v.w};
    const float a = alpha[row];
    const float s[2] = {e4m3(scale[row * (k >> 4) + col * 2]) * a, e4m3(scale[row * (k >> 4) + col * 2 + 1]) * a};
    uint32_t o[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const uint32_t byte = (words[i >> 2] >> (8 * (i & 3))) & 0xff;
      const float sc = s[i >> 3];
      const uint16_t lo = __bfloat16_as_ushort(__float2bfloat16(e2m1(byte & 15) * sc));
      const uint16_t hi = __bfloat16_as_ushort(__float2bfloat16(e2m1(byte >> 4) * sc));
      o[i] = uint32_t(lo) | (uint32_t(hi) << 16);
    }
    uint4* dst = reinterpret_cast<uint4*>(out + row * k + col * 32);
#pragma unroll
    for (int q = 0; q < 4; ++q) dst[q] = make_uint4(o[4 * q], o[4 * q + 1], o[4 * q + 2], o[4 * q + 3]);
  }
}

}  // namespace

extern "C" cudaError_t blz_nvfp4_gemv(const uint16_t* x, int64_t ldx, const uint8_t* w, const uint8_t* scale,
                                      const float* alpha, uint16_t* y, int64_t ldy, int rows, int n, int k,
                                      cudaStream_t stream) {
  if (rows <= 0 || n <= 0) return cudaSuccess;
  if (k % 32 != 0 || rows > BLZ_NVFP4_GEMV_MAX_ROWS) return cudaErrorInvalidValue;
  switch (rows) {
    case 1: return launch_gemv<1, 2>(x, ldx, w, scale, alpha, y, ldy, n, k, stream);
    case 2: return launch_gemv<2, 2>(x, ldx, w, scale, alpha, y, ldy, n, k, stream);
    case 3: return launch_gemv<3, 2>(x, ldx, w, scale, alpha, y, ldy, n, k, stream);
    case 4: return launch_gemv<4, 2>(x, ldx, w, scale, alpha, y, ldy, n, k, stream);
    case 5: return launch_gemv<5, 2>(x, ldx, w, scale, alpha, y, ldy, n, k, stream);
    case 6: return launch_gemv<6, 2>(x, ldx, w, scale, alpha, y, ldy, n, k, stream);
    case 7: return launch_gemv<7, 2>(x, ldx, w, scale, alpha, y, ldy, n, k, stream);
    default: return launch_gemv<8, 2>(x, ldx, w, scale, alpha, y, ldy, n, k, stream);
  }
}

namespace {

// Partial sums of a block: kSplit x 8 kTiles x (16 kMT + 1) floats, within the 48 KiB of static shared memory.
constexpr bool mma_cfg_fits(int tiles, int split, int mt) { return split * tiles * 8 * (16 * mt + 1) * 4 <= 48 * 1024; }

template <int T, int S>
cudaError_t launch_mma(const uint16_t* x, int64_t ldx, const uint8_t* w, const uint8_t* scale, const float* alpha,
                       uint16_t* y, int64_t ldy, int rows, int n, int k, cudaStream_t stream) {
  const int blocks = (n + 8 * T - 1) / (8 * T), mt = (rows + 15) / 16;
  if (mt == 1 && rows <= 8)
    nvfp4_mma_kernel<T, S, false, 1><<<blocks, S * 32, 0, stream>>>(x, ldx, w, scale, alpha, y, ldy, rows, n, k);
  else if (mt == 1)
    nvfp4_mma_kernel<T, S, true, 1><<<blocks, S * 32, 0, stream>>>(x, ldx, w, scale, alpha, y, ldy, rows, n, k);
  else if (mt == 2)
    nvfp4_mma_kernel<T, S, true, 2><<<blocks, S * 32, 0, stream>>>(x, ldx, w, scale, alpha, y, ldy, rows, n, k);
  else if constexpr (mma_cfg_fits(T, S, 4)) {
    if (mt == 3)
      nvfp4_mma_kernel<T, S, true, 3><<<blocks, S * 32, 0, stream>>>(x, ldx, w, scale, alpha, y, ldy, rows, n, k);
    else
      nvfp4_mma_kernel<T, S, true, 4><<<blocks, S * 32, 0, stream>>>(x, ldx, w, scale, alpha, y, ldy, rows, n, k);
  } else {
    return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

}  // namespace

extern "C" int blz_nvfp4_mma_default_cfg(int rows, int k, int* split, int* tiles) {
  // Measured on the RTX 5070 Ti with cold weights: 8 warps along K for a few rows of long inputs (17408 x 5120: 879
  // GB/s), 2 for short inputs and 9..16 rows; above 16 rows several weight tiles per block, so that fewer blocks re-read
  // the inputs from L2 (32 rows: 110 us with 2 tiles, 401 with 1; 64 rows: 202 us with 4, 835 with 1). The tuner
  // (crates/blaze/src/tune.rs) measures the real shapes instead.
  const int mt = (rows + 15) / 16;
  *split = mt == 1 && rows <= 8 && k >= 5120 ? 8 : 2;
  *tiles = mt == 1 ? 1 : (mt == 2 ? 2 : 4);
  return 0;
}

extern "C" int blz_nvfp4_mma_cfg_valid(int rows, int split, int tiles) {
  const int mt = (rows + 15) / 16;
  return (split == 2 || split == 4 || split == 8) && (tiles == 1 || tiles == 2 || tiles == 4) && mt >= 1 && mt <= 4 &&
         mma_cfg_fits(tiles, split, mt) && !(tiles == 4 && split == 8);
}

extern "C" cudaError_t blz_nvfp4_gemv_mma_cfg(const uint16_t* x, int64_t ldx, const uint8_t* w, const uint8_t* scale,
                                              const float* alpha, uint16_t* y, int64_t ldy, int rows, int n, int k,
                                              int split, int tiles, cudaStream_t stream) {
  if (rows <= 0 || n <= 0) return cudaSuccess;
  if (k % 128 != 0 || rows > BLZ_NVFP4_MMA_MAX_ROWS) return cudaErrorInvalidValue;
  if (split == 0 || tiles == 0) blz_nvfp4_mma_default_cfg(rows, k, &split, &tiles);
  if (!blz_nvfp4_mma_cfg_valid(rows, split, tiles)) return cudaErrorInvalidValue;
#define BLZ_MMA_S(T)                                                                                    \
  switch (split) {                                                                                      \
    case 2: return launch_mma<T, 2>(x, ldx, w, scale, alpha, y, ldy, rows, n, k, stream);               \
    case 4: return launch_mma<T, 4>(x, ldx, w, scale, alpha, y, ldy, rows, n, k, stream);               \
    default: return launch_mma<T, 8>(x, ldx, w, scale, alpha, y, ldy, rows, n, k, stream);              \
  }
  switch (tiles) {
    case 1: BLZ_MMA_S(1)
    case 2: BLZ_MMA_S(2)
    default:
      if (split == 2) return launch_mma<4, 2>(x, ldx, w, scale, alpha, y, ldy, rows, n, k, stream);
      return launch_mma<4, 4>(x, ldx, w, scale, alpha, y, ldy, rows, n, k, stream);
  }
#undef BLZ_MMA_S
}

extern "C" cudaError_t blz_nvfp4_gemv_mma(const uint16_t* x, int64_t ldx, const uint8_t* w, const uint8_t* scale,
                                          const float* alpha, uint16_t* y, int64_t ldy, int rows, int n, int k,
                                          cudaStream_t stream) {
  return blz_nvfp4_gemv_mma_cfg(x, ldx, w, scale, alpha, y, ldy, rows, n, k, 0, 0, stream);
}

namespace {

template <int T, int S>
cudaError_t launch_mma_f16(const uint16_t* x, const int32_t* shift, const uint8_t* w, const uint8_t* scale,
                           const float* alpha, uint16_t* y, int64_t ldy, int rows, int n, int k, int ws, cudaStream_t stream) {
  const int blocks = (n + 8 * T - 1) / (8 * T), mt = (rows + 15) / 16;
  if (mt == 1 && rows <= 8)
    nvfp4_mma_f16_kernel<T, S, false, 1><<<blocks, S * 32, 0, stream>>>(x, k, shift, w, scale, alpha, y, ldy, rows, n, k, ws);
  else if (mt == 1)
    nvfp4_mma_f16_kernel<T, S, true, 1><<<blocks, S * 32, 0, stream>>>(x, k, shift, w, scale, alpha, y, ldy, rows, n, k, ws);
  else if (mt == 2)
    nvfp4_mma_f16_kernel<T, S, true, 2><<<blocks, S * 32, 0, stream>>>(x, k, shift, w, scale, alpha, y, ldy, rows, n, k, ws);
  else if constexpr (mma_cfg_fits(T, S, 4)) {
    if (mt == 3)
      nvfp4_mma_f16_kernel<T, S, true, 3><<<blocks, S * 32, 0, stream>>>(x, k, shift, w, scale, alpha, y, ldy, rows, n, k, ws);
    else
      nvfp4_mma_f16_kernel<T, S, true, 4><<<blocks, S * 32, 0, stream>>>(x, k, shift, w, scale, alpha, y, ldy, rows, n, k, ws);
  } else {
    return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

}  // namespace

extern "C" cudaError_t blz_nvfp4_f16_rows(const uint16_t* x, int64_t ldx, int rows, int k, uint16_t* x16, int32_t* shift,
                                          cudaStream_t stream) {
  if (rows <= 0) return cudaSuccess;
  nvfp4_f16_rows_kernel<<<rows, 256, 0, stream>>>(x, ldx, k, x16, shift);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_nvfp4_gemv_mma_f16_cfg(const uint16_t* x16, const int32_t* shift, const uint8_t* w,
                                                  const uint8_t* scale, const float* alpha, uint16_t* y, int64_t ldy,
                                                  int rows, int n, int k, int wshift, int split, int tiles,
                                                  cudaStream_t stream) {
  const int ws = wshift;
  if (rows <= 0 || n <= 0) return cudaSuccess;
  if (k % 128 != 0 || rows > BLZ_NVFP4_MMA_MAX_ROWS) return cudaErrorInvalidValue;
  if (split == 0 || tiles == 0) blz_nvfp4_mma_default_cfg(rows, k, &split, &tiles);
  if (!blz_nvfp4_mma_cfg_valid(rows, split, tiles)) return cudaErrorInvalidValue;
#define BLZ_MMA16_S(T)                                                                                  \
  switch (split) {                                                                                      \
    case 2: return launch_mma_f16<T, 2>(x16, shift, w, scale, alpha, y, ldy, rows, n, k, ws, stream);       \
    case 4: return launch_mma_f16<T, 4>(x16, shift, w, scale, alpha, y, ldy, rows, n, k, ws, stream);       \
    default: return launch_mma_f16<T, 8>(x16, shift, w, scale, alpha, y, ldy, rows, n, k, ws, stream);      \
  }
  switch (tiles) {
    case 1: BLZ_MMA16_S(1)
    case 2: BLZ_MMA16_S(2)
    default:
      if (split == 2) return launch_mma_f16<4, 2>(x16, shift, w, scale, alpha, y, ldy, rows, n, k, ws, stream);
      return launch_mma_f16<4, 4>(x16, shift, w, scale, alpha, y, ldy, rows, n, k, ws, stream);
  }
#undef BLZ_MMA16_S
}

extern "C" cudaError_t blz_nvfp4_weight_dequant(const uint8_t* w, const uint8_t* scale, const float* alpha, int n, int k,
                                         uint16_t* out, cudaStream_t stream) {
  if (n <= 0) return cudaSuccess;
  if (k % 32 != 0) return cudaErrorInvalidValue;
  const int64_t chunks = int64_t(n) * (k >> 5);
  const int blocks = int((chunks + 255) / 256 < 65535 * 8 ? (chunks + 255) / 256 : 65535 * 8);
  nvfp4_dequant_kernel<<<blocks, 256, 0, stream>>>(w, scale, alpha, n, k, out);
  return cudaGetLastError();
}
