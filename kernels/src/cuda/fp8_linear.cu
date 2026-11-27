// FP8 W8A16 linear layers (blaze/fp8_linear.h), the tensor-core route of nvfp4_linear.cu for E4M3 codes.
//
// mma.m16n8k16 with A = the input rows (zero-padded to 16) and B = 8 weight rows; over a 128-wide chunk thread (g, t)
// holds the 32 codes [32t, 32t + 32) of weight row g (two 16-byte loads) and mma step s takes codes 4s .. 4s + 3 as its
// k slots (2t, 2t+1, 2t+8, 2t+9); the input fragments follow the same permutation. Codes become BF16 through a
// 256-entry table (every E4M3 value is exact in BF16). Warps split K; partial sums meet in shared memory.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdlib>

#include "blaze/fp8_linear.h"

namespace {

__device__ __forceinline__ uint4 ldg_stream(const void* p) {
  uint4 v;
  asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
      : "l"(p));
  return v;
}

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint16_t e4m3_bf16(int code) {
  __nv_fp8_e4m3 f;
  f.__x = uint8_t(code);
  return __bfloat16_as_ushort(__float2bfloat16(float(f)));
}

template <int kSplit, bool kHi, bool kBlock>
__global__ void __launch_bounds__(kSplit * 32) fp8_mma_kernel(const uint16_t* __restrict__ x, int64_t ldx,
                                                             const uint8_t* __restrict__ w,
                                                             const float* __restrict__ block_scale, int block_n,
                                                             const float* __restrict__ alpha,
                                                             uint16_t* __restrict__ y, int64_t ldy, int m, int n,
                                                             int k) {
  __shared__ uint16_t lut[256];
  __shared__ float red[kSplit][8][17];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) lut[i] = e4m3_bf16(i);
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int f0 = blockIdx.x * 8;
  const int f = min(f0 + g, n - 1);
  const uint8_t* wp = w + int64_t(f) * k + 32 * t;
  const int chunks = k >> 7;
  const float* bs = kBlock ? block_scale + int64_t(f / block_n) * chunks : nullptr;
  const bool lo_ok = g < m, hi_ok = kHi && g + 8 < m;
  const uint16_t* xlo = x + int64_t(lo_ok ? g : 0) * ldx + 32 * t;
  const uint16_t* xhi = x + int64_t(hi_ok ? g + 8 : 0) * ldx + 32 * t;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  uint4 wq[2];
  if (warp < chunks) {
    wq[0] = ldg_stream(wp + warp * 128);
    wq[1] = ldg_stream(wp + warp * 128 + 16);
  }
  for (int c = warp; c < chunks; c += kSplit) {
    uint4 nw[2];
    if (c + kSplit < chunks) {
      nw[0] = ldg_stream(wp + (c + kSplit) * 128);
      nw[1] = ldg_stream(wp + (c + kSplit) * 128 + 16);
    }
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
    const uint32_t codes[8] = {wq[0].x, wq[0].y, wq[0].z, wq[0].w, wq[1].x, wq[1].y, wq[1].z, wq[1].w};
    float part[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float* tgt = kBlock ? part : acc;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const uint32_t q = codes[s];
      const uint32_t b0 = uint32_t(lut[q & 0xff]) | (uint32_t(lut[(q >> 8) & 0xff]) << 16);
      const uint32_t b1 = uint32_t(lut[(q >> 16) & 0xff]) | (uint32_t(lut[q >> 24]) << 16);
      mma_bf16(tgt, xa[2 * s], xb[2 * s], xa[2 * s + 1], xb[2 * s + 1], b0, b1);
    }
    if (kBlock) {
      // c0, c1 belong to rows 2t, 2t + 1 of the tile; their block scales (same block for the whole 8-row tile)
      const float sc = bs[c];
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[i] = fmaf(part[i], sc, acc[i]);
    }
    wq[0] = nw[0];
    wq[1] = nw[1];
  }
  red[warp][2 * t][g] = acc[0];
  red[warp][2 * t + 1][g] = acc[1];
  red[warp][2 * t][g + 8] = acc[2];
  red[warp][2 * t + 1][g + 8] = acc[3];
  __syncthreads();
  for (int i = threadIdx.x; i < 8 * 16; i += blockDim.x) {
    const int r = i >> 4, tok = i & 15, row = f0 + r;
    if (tok >= m || row >= n) continue;
    float v = 0.0f;
#pragma unroll
    for (int q = 0; q < kSplit; ++q) v += red[q][r][tok];
    y[tok * ldy + row] = __bfloat16_as_ushort(__float2bfloat16(v * alpha[row]));
  }
}

__global__ void fp8_dequant_kernel(const uint8_t* __restrict__ w, const float* __restrict__ block_scale, int block_n,
                                   const float* __restrict__ alpha, int n, int k, uint16_t* __restrict__ out) {
  const int64_t vecs = int64_t(n) * (k >> 4);
  const int chunks = k >> 7;
  for (int64_t v = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; v < vecs; v += int64_t(gridDim.x) * blockDim.x) {
    const int64_t row = v / (k >> 4), col = (v % (k >> 4)) * 16;
    const uint4 q = *reinterpret_cast<const uint4*>(w + row * k + col);
    float s = alpha[row];
    if (block_scale) s *= block_scale[(row / block_n) * chunks + col / 128];
    const uint32_t words[4] = {q.x, q.y, q.z, q.w};
    uint32_t o[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t wd = words[i >> 1] >> (16 * (i & 1));
      __nv_fp8_e4m3 a, b;
      a.__x = uint8_t(wd & 0xff);
      b.__x = uint8_t((wd >> 8) & 0xff);
      o[i] = uint32_t(__bfloat16_as_ushort(__float2bfloat16(float(a) * s))) |
             (uint32_t(__bfloat16_as_ushort(__float2bfloat16(float(b) * s))) << 16);
    }
    uint4* dst = reinterpret_cast<uint4*>(out + row * k + col);
    dst[0] = make_uint4(o[0], o[1], o[2], o[3]);
    dst[1] = make_uint4(o[4], o[5], o[6], o[7]);
  }
}

// MXFP8 (OCP microscaling): E4M3 codes with an E8M0 scale per 32 values of a row. Over a 128-wide chunk thread (g, t)
// holds exactly one scale block of row g, so its widened BF16 codes take the block's power of two (exact); kMT 16-row
// tiles of tokens share the widened fragments, as in nvfp4_linear.cu.
__device__ __forceinline__ uint32_t e8m0_bf16x2(uint32_t e) {
  const uint32_t h = e == 0 ? 0x0040u : (e << 7);  // 2^(e - 127); e = 0: the BF16 subnormal 2^-127
  return h | (h << 16);
}

__device__ __forceinline__ uint32_t bf16x2_mul(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}

template <int kTiles, int kSplit, bool kHi, int kMT>
__global__ void __launch_bounds__(kSplit * 32) mxfp8_mma_kernel(const uint16_t* __restrict__ x, int64_t ldx,
                                                               const uint8_t* __restrict__ w,
                                                               const uint8_t* __restrict__ scale,
                                                               uint16_t* __restrict__ y, int64_t ldy, int m, int n,
                                                               int k) {
  constexpr int kTok = 16 * kMT;
  __shared__ uint16_t lut[256];
  __shared__ float red[kSplit][kTiles * 8][kTok + 1];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) lut[i] = e4m3_bf16(i);
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int f0 = blockIdx.x * kTiles * 8;
  const uint8_t* wp[kTiles];
  const uint8_t* sp[kTiles];
#pragma unroll
  for (int j = 0; j < kTiles; ++j) {
    const int f = min(f0 + 8 * j + g, n - 1);
    wp[j] = w + int64_t(f) * k + 32 * t;
    sp[j] = scale + int64_t(f) * (k >> 5) + t;
  }
  const int chunks = k >> 7;
  float acc[kTiles][kMT][4];
#pragma unroll
  for (int j = 0; j < kTiles; ++j)
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt) acc[j][mt][0] = acc[j][mt][1] = acc[j][mt][2] = acc[j][mt][3] = 0.0f;

  uint4 wq[kTiles][2];
  uint32_t sq[kTiles];
  auto load = [&](int c, uint4 (*dw)[2], uint32_t* ds) {
#pragma unroll
    for (int j = 0; j < kTiles; ++j) {
      dw[j][0] = ldg_stream(wp[j] + c * 128);
      dw[j][1] = ldg_stream(wp[j] + c * 128 + 16);
      ds[j] = __ldg(sp[j] + c * 4);
    }
  };
  if (warp < chunks) load(warp, wq, sq);
  for (int c = warp; c < chunks; c += kSplit) {
    uint4 nw[kTiles][2];
    uint32_t ns[kTiles];
    if (c + kSplit < chunks) load(c + kSplit, nw, ns);
    uint32_t bq[kTiles][8][2];
#pragma unroll
    for (int j = 0; j < kTiles; ++j) {
      const uint32_t codes[8] = {wq[j][0].x, wq[j][0].y, wq[j][0].z, wq[j][0].w, wq[j][1].x, wq[j][1].y, wq[j][1].z, wq[j][1].w};
      const uint32_t sc = e8m0_bf16x2(sq[j]);
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const uint32_t q = codes[s];
        bq[j][s][0] = bf16x2_mul(uint32_t(lut[q & 0xff]) | (uint32_t(lut[(q >> 8) & 0xff]) << 16), sc);
        bq[j][s][1] = bf16x2_mul(uint32_t(lut[(q >> 16) & 0xff]) | (uint32_t(lut[q >> 24]) << 16), sc);
      }
      wq[j][0] = nw[j][0];
      wq[j][1] = nw[j][1];
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
    const int r = i / kTok, tok = i % kTok, row = f0 + r;
    if (tok >= m || row >= n) continue;
    float v = 0.0f;
#pragma unroll
    for (int q = 0; q < kSplit; ++q) v += red[q][r][tok];
    y[tok * ldy + row] = __bfloat16_as_ushort(__float2bfloat16(v));
  }
}

constexpr bool mx_cfg_fits(int tiles, int split, int mt) {
  return split * tiles * 8 * (16 * mt + 1) * 4 + 512 <= 48 * 1024;
}

// BF16 weights through the same tensor-core GEMV: thread (g, t) loads its 32 values of a 128-wide chunk (64 bytes) of
// weight row g, already the B fragments of the 8 k-steps; the rest as mxfp8_mma_kernel (K over kSplit warps, kTiles
// 8-row weight tiles per block, kMT 16-row tiles of tokens sharing them).
template <int kTiles, int kSplit, bool kHi, int kMT>
__global__ void __launch_bounds__(kSplit * 32) bf16_mma_kernel(const uint16_t* __restrict__ x, int64_t ldx,
                                                              const uint16_t* __restrict__ w, uint16_t* __restrict__ y,
                                                              int64_t ldy, int m, int n, int k) {
  constexpr int kTok = 16 * kMT;
  __shared__ float red[kSplit][kTiles * 8][kTok + 1];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int f0 = blockIdx.x * kTiles * 8;
  const uint16_t* wp[kTiles];
#pragma unroll
  for (int j = 0; j < kTiles; ++j) wp[j] = w + int64_t(min(f0 + 8 * j + g, n - 1)) * k + 32 * t;
  const int chunks = k >> 7;
  float acc[kTiles][kMT][4];
#pragma unroll
  for (int j = 0; j < kTiles; ++j)
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt) acc[j][mt][0] = acc[j][mt][1] = acc[j][mt][2] = acc[j][mt][3] = 0.0f;
  uint4 wq[kTiles][4];
  auto load = [&](int c, uint4 (*dw)[4]) {
#pragma unroll
    for (int j = 0; j < kTiles; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) dw[j][q] = ldg_stream(wp[j] + c * 128 + 8 * q);
  };
  if (warp < chunks) load(warp, wq);
  for (int c = warp; c < chunks; c += kSplit) {
    uint4 nw[kTiles][4];
    if (c + kSplit < chunks) load(c + kSplit, nw);
    uint32_t bq[kTiles][16];
#pragma unroll
    for (int j = 0; j < kTiles; ++j) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        bq[j][4 * q] = wq[j][q].x, bq[j][4 * q + 1] = wq[j][q].y, bq[j][4 * q + 2] = wq[j][q].z, bq[j][4 * q + 3] = wq[j][q].w;
        wq[j][q] = nw[j][q];
      }
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
          mma_bf16(acc[j][mt], xa[2 * s], xb[2 * s], xa[2 * s + 1], xb[2 * s + 1], bq[j][2 * s], bq[j][2 * s + 1]);
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
    const int r = i / kTok, tok = i % kTok, row = f0 + r;
    if (tok >= m || row >= n) continue;
    float v = 0.0f;
#pragma unroll
    for (int q = 0; q < kSplit; ++q) v += red[q][r][tok];
    y[tok * ldy + row] = __bfloat16_as_ushort(__float2bfloat16(v));
  }
}

template <int T, int S>
cudaError_t launch_bf16(const uint16_t* x, int64_t ldx, const uint16_t* w, uint16_t* y, int64_t ldy, int rows, int n,
                        int k, cudaStream_t stream) {
  const int blocks = (n + 8 * T - 1) / (8 * T), mt = (rows + 15) / 16;
  if (mt == 1 && rows <= 8)
    bf16_mma_kernel<T, S, false, 1><<<blocks, S * 32, 0, stream>>>(x, ldx, w, y, ldy, rows, n, k);
  else if (mt == 1)
    bf16_mma_kernel<T, S, true, 1><<<blocks, S * 32, 0, stream>>>(x, ldx, w, y, ldy, rows, n, k);
  else if (mt == 2)
    bf16_mma_kernel<T, S, true, 2><<<blocks, S * 32, 0, stream>>>(x, ldx, w, y, ldy, rows, n, k);
  else if constexpr (mx_cfg_fits(T, S, 4)) {
    if (mt == 3)
      bf16_mma_kernel<T, S, true, 3><<<blocks, S * 32, 0, stream>>>(x, ldx, w, y, ldy, rows, n, k);
    else
      bf16_mma_kernel<T, S, true, 4><<<blocks, S * 32, 0, stream>>>(x, ldx, w, y, ldy, rows, n, k);
  } else {
    return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

template <int T, int S>
cudaError_t launch_mx(const uint16_t* x, int64_t ldx, const uint8_t* w, const uint8_t* scale, uint16_t* y, int64_t ldy,
                      int rows, int n, int k, cudaStream_t stream) {
  const int blocks = (n + 8 * T - 1) / (8 * T), mt = (rows + 15) / 16;
  if (mt == 1 && rows <= 8)
    mxfp8_mma_kernel<T, S, false, 1><<<blocks, S * 32, 0, stream>>>(x, ldx, w, scale, y, ldy, rows, n, k);
  else if (mt == 1)
    mxfp8_mma_kernel<T, S, true, 1><<<blocks, S * 32, 0, stream>>>(x, ldx, w, scale, y, ldy, rows, n, k);
  else if (mt == 2)
    mxfp8_mma_kernel<T, S, true, 2><<<blocks, S * 32, 0, stream>>>(x, ldx, w, scale, y, ldy, rows, n, k);
  else if constexpr (mx_cfg_fits(T, S, 4)) {
    if (mt == 3)
      mxfp8_mma_kernel<T, S, true, 3><<<blocks, S * 32, 0, stream>>>(x, ldx, w, scale, y, ldy, rows, n, k);
    else
      mxfp8_mma_kernel<T, S, true, 4><<<blocks, S * 32, 0, stream>>>(x, ldx, w, scale, y, ldy, rows, n, k);
  } else {
    return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

__global__ void mxfp8_dequant_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ scale, int n, int k,
                                     uint16_t* __restrict__ out) {
  const int64_t vecs = int64_t(n) * (k >> 4);
  for (int64_t v = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; v < vecs; v += int64_t(gridDim.x) * blockDim.x) {
    const int64_t row = v / (k >> 4), col = (v % (k >> 4)) * 16;
    const uint4 q = *reinterpret_cast<const uint4*>(w + row * k + col);
    const uint32_t e = scale[row * (k >> 5) + col / 32];
    const float s = e == 0 ? exp2f(-127.0f) : __uint_as_float(e << 23);
    const uint32_t words[4] = {q.x, q.y, q.z, q.w};
    uint32_t o[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t wd = words[i >> 1] >> (16 * (i & 1));
      __nv_fp8_e4m3 a, b;
      a.__x = uint8_t(wd & 0xff);
      b.__x = uint8_t((wd >> 8) & 0xff);
      o[i] = uint32_t(__bfloat16_as_ushort(__float2bfloat16(float(a) * s))) |
             (uint32_t(__bfloat16_as_ushort(__float2bfloat16(float(b) * s))) << 16);
    }
    uint4* dst = reinterpret_cast<uint4*>(out + row * k + col);
    dst[0] = make_uint4(o[0], o[1], o[2], o[3]);
    dst[1] = make_uint4(o[4], o[5], o[6], o[7]);
  }
}

}  // namespace

extern "C" int blz_mxfp8_mma_default_cfg(int rows, int k, int* split, int* tiles) {
  const int mt = (rows + 15) / 16;
  *split = mt == 1 ? (rows <= 8 && k >= 5120 ? 8 : (k >= 2048 ? 4 : 2)) : 2;
  *tiles = mt == 1 ? 1 : (mt == 2 ? 2 : 4);
  return 0;
}

extern "C" int blz_mxfp8_mma_cfg_valid(int rows, int split, int tiles) {
  const int mt = (rows + 15) / 16;
  return (split == 2 || split == 4 || split == 8) && (tiles == 1 || tiles == 2 || tiles == 4) && mt >= 1 && mt <= 4 &&
         mx_cfg_fits(tiles, split, mt) && !(tiles == 4 && split == 8);
}

extern "C" cudaError_t blz_mxfp8_gemv_mma_cfg(const uint16_t* x, int64_t ldx, const uint8_t* w, const uint8_t* scale,
                                              uint16_t* y, int64_t ldy, int rows, int n, int k, int split, int tiles,
                                              cudaStream_t stream) {
  if (rows <= 0 || n <= 0) return cudaSuccess;
  if (k % 128 != 0 || rows > BLZ_MXFP8_MMA_MAX_ROWS) return cudaErrorInvalidValue;
  if (split == 0 || tiles == 0) blz_mxfp8_mma_default_cfg(rows, k, &split, &tiles);
  if (!blz_mxfp8_mma_cfg_valid(rows, split, tiles)) return cudaErrorInvalidValue;
#define BLZ_MX_S(T)                                                                          \
  switch (split) {                                                                           \
    case 2: return launch_mx<T, 2>(x, ldx, w, scale, y, ldy, rows, n, k, stream);            \
    case 4: return launch_mx<T, 4>(x, ldx, w, scale, y, ldy, rows, n, k, stream);            \
    default: return launch_mx<T, 8>(x, ldx, w, scale, y, ldy, rows, n, k, stream);           \
  }
  switch (tiles) {
    case 1: BLZ_MX_S(1)
    case 2: BLZ_MX_S(2)
    default:
      if (split == 2) return launch_mx<4, 2>(x, ldx, w, scale, y, ldy, rows, n, k, stream);
      return launch_mx<4, 4>(x, ldx, w, scale, y, ldy, rows, n, k, stream);
  }
#undef BLZ_MX_S
}

extern "C" cudaError_t blz_mxfp8_gemv_mma(const uint16_t* x, int64_t ldx, const uint8_t* w, const uint8_t* scale,
                                          uint16_t* y, int64_t ldy, int rows, int n, int k, cudaStream_t stream) {
  return blz_mxfp8_gemv_mma_cfg(x, ldx, w, scale, y, ldy, rows, n, k, 0, 0, stream);
}

extern "C" cudaError_t blz_bf16_gemv_mma_cfg(const uint16_t* x, int64_t ldx, const uint16_t* w, uint16_t* y,
                                             int64_t ldy, int rows, int n, int k, int split, int tiles,
                                             cudaStream_t stream) {
  if (rows <= 0 || n <= 0) return cudaSuccess;
  if (k % 128 != 0 || rows > BLZ_MXFP8_MMA_MAX_ROWS || ldx % 8 != 0 || reinterpret_cast<uintptr_t>(x) % 16 != 0 ||
      reinterpret_cast<uintptr_t>(w) % 16 != 0)
    return cudaErrorInvalidValue;
  if (split == 0 || tiles == 0) blz_mxfp8_mma_default_cfg(rows, k, &split, &tiles);
  if (!blz_mxfp8_mma_cfg_valid(rows, split, tiles)) return cudaErrorInvalidValue;
#define BLZ_BF_S(T)                                                                 \
  switch (split) {                                                                  \
    case 2: return launch_bf16<T, 2>(x, ldx, w, y, ldy, rows, n, k, stream);        \
    case 4: return launch_bf16<T, 4>(x, ldx, w, y, ldy, rows, n, k, stream);        \
    default: return launch_bf16<T, 8>(x, ldx, w, y, ldy, rows, n, k, stream);       \
  }
  switch (tiles) {
    case 1: BLZ_BF_S(1)
    case 2: BLZ_BF_S(2)
    default:
      if (split == 2) return launch_bf16<4, 2>(x, ldx, w, y, ldy, rows, n, k, stream);
      return launch_bf16<4, 4>(x, ldx, w, y, ldy, rows, n, k, stream);
  }
#undef BLZ_BF_S
}

extern "C" cudaError_t blz_mxfp8_weight_dequant(const uint8_t* w, const uint8_t* scale, int n, int k, uint16_t* out,
                                                cudaStream_t stream) {
  if (n <= 0) return cudaSuccess;
  if (k % 32 != 0) return cudaErrorInvalidValue;
  const int64_t vecs = int64_t(n) * (k >> 4);
  const int blocks = int((vecs + 255) / 256 < 65535 * 8 ? (vecs + 255) / 256 : 65535 * 8);
  mxfp8_dequant_kernel<<<blocks, 256, 0, stream>>>(w, scale, n, k, out);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_fp8_gemv_mma(const uint16_t* x, int64_t ldx, const uint8_t* w, const float* block_scale,
                                        int block_n, const float* alpha, uint16_t* y, int64_t ldy, int rows, int n,
                                        int k, cudaStream_t stream) {
  if (rows <= 0 || n <= 0) return cudaSuccess;
  if (k % 128 != 0 || rows > BLZ_FP8_MMA_MAX_ROWS || (block_scale && (block_n <= 0 || block_n % 8 != 0)))
    return cudaErrorInvalidValue;
  static const int forced = [] {
    const char* e = getenv("BLZ_MMA_SPLIT");
    return e ? atoi(e) : 0;
  }();
  const int split = forced ? forced : (rows <= 8 && k >= 5120 ? 8 : (k >= 2048 ? 4 : 2));
  const int blocks = (n + 7) / 8;
#define BLZ_FP8_LAUNCH(S)                                                                                         \
  {                                                                                                               \
    if (block_scale) {                                                                                            \
      if (rows > 8)                                                                                               \
        fp8_mma_kernel<S, true, true><<<blocks, S * 32, 0, stream>>>(x, ldx, w, block_scale, block_n, alpha, y,  \
                                                                     ldy, rows, n, k);                            \
      else                                                                                                        \
        fp8_mma_kernel<S, false, true><<<blocks, S * 32, 0, stream>>>(x, ldx, w, block_scale, block_n, alpha, y, \
                                                                      ldy, rows, n, k);                           \
    } else {                                                                                                      \
      if (rows > 8)                                                                                               \
        fp8_mma_kernel<S, true, false><<<blocks, S * 32, 0, stream>>>(x, ldx, w, nullptr, 1, alpha, y, ldy,      \
                                                                      rows, n, k);                                \
      else                                                                                                        \
        fp8_mma_kernel<S, false, false><<<blocks, S * 32, 0, stream>>>(x, ldx, w, nullptr, 1, alpha, y, ldy,     \
                                                                       rows, n, k);                               \
    }                                                                                                             \
  }
  switch (split) {
    case 2: BLZ_FP8_LAUNCH(2) break;
    case 8: BLZ_FP8_LAUNCH(8) break;
    default: BLZ_FP8_LAUNCH(4) break;
  }
#undef BLZ_FP8_LAUNCH
  return cudaGetLastError();
}

extern "C" cudaError_t blz_fp8_weight_dequant(const uint8_t* w, const float* block_scale, int block_n,
                                              const float* alpha, int n, int k, uint16_t* out, cudaStream_t stream) {
  if (n <= 0) return cudaSuccess;
  if (k % 128 != 0) return cudaErrorInvalidValue;
  const int64_t vecs = int64_t(n) * (k >> 4);
  const int blocks = int((vecs + 255) / 256 < 65535 * 8 ? (vecs + 255) / 256 : 65535 * 8);
  fp8_dequant_kernel<<<blocks, 256, 0, stream>>>(w, block_scale, block_n, alpha, n, k, out);
  return cudaGetLastError();
}
