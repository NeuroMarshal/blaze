// NVFP4 W4A16 linear layers (blaze/nvfp4_linear.h).
//
// GEMV: one warp per R output rows; lane l reads 16-byte chunks (32 codes = two scale blocks) l, l + 32, ... of each
// row, so a warp load covers 512 contiguous bytes. Codes become floats through a 256-entry table (byte -> two
// values) in shared memory; each block's 16 products are summed in FP32 and multiplied by the block scale once.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "blaze/nvfp4_linear.h"

namespace {

constexpr int kWarps = 4;

__device__ __forceinline__ float e2m1_level(int idx) { return idx < 4 ? 0.5f * idx : (idx < 7 ? float(idx - 2) : 6.0f); }

__device__ __forceinline__ float e2m1(int code) {
  const float m = e2m1_level(code & 7);
  return (code & 8) ? -m : m;
}

__device__ __forceinline__ float e4m3(uint8_t v) {
  __nv_fp8_e4m3 f;
  f.__x = v;
  return float(f);
}

__device__ __forceinline__ uint4 ldg_stream(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
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

  for (int c = lane; c < chunks; c += 32) {
    uint4 wv[R];
    float s0[R], s1[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int row = min(row0 + r, n - 1);
      wv[r] = ldg_stream(w + row * wrow + c * 16);
      const uint16_t sc = __ldg(reinterpret_cast<const uint16_t*>(scale + row * srow + c * 2));
      s0[r] = e4m3(sc & 0xff);
      s1[r] = e4m3(sc >> 8);
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

extern "C" cudaError_t blz_nvfp4_weight_dequant(const uint8_t* w, const uint8_t* scale, const float* alpha, int n, int k,
                                         uint16_t* out, cudaStream_t stream) {
  if (n <= 0) return cudaSuccess;
  if (k % 32 != 0) return cudaErrorInvalidValue;
  const int64_t chunks = int64_t(n) * (k >> 5);
  const int blocks = int((chunks + 255) / 256 < 65535 * 8 ? (chunks + 255) / 256 : 65535 * 8);
  nvfp4_dequant_kernel<<<blocks, 256, 0, stream>>>(w, scale, alpha, n, k, out);
  return cudaGetLastError();
}
