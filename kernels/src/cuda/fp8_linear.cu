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

}  // namespace

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
