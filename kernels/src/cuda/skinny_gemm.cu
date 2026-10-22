// Skinny 16-bit linear layers (see blaze/skinny_gemm.h): one block per output column, 128 threads striding K by
// 16-byte packs, FP32 sums per row reduced over the block.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "blaze/skinny_gemm.h"

namespace blaze {
namespace {

constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;

__device__ __forceinline__ void unpack8(const uint4& v, float* f, __nv_bfloat16) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 p = __bfloat1622float2(h[i]);
    f[2 * i] = p.x, f[2 * i + 1] = p.y;
  }
}

__device__ __forceinline__ void unpack8(const uint4& v, float* f, __half) {
  const __half2* h = reinterpret_cast<const __half2*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 p = __half22float2(h[i]);
    f[2 * i] = p.x, f[2 * i + 1] = p.y;
  }
}

__device__ __forceinline__ __nv_bfloat16 from_float(float v, __nv_bfloat16) { return __float2bfloat16_rn(v); }
__device__ __forceinline__ __half from_float(float v, __half) { return __float2half_rn(v); }

// grid: (n), block: kThreads
template <typename T>
__global__ void __launch_bounds__(kThreads) skinny_gemm_kernel(const T* __restrict__ x, int64_t x_stride,
                                                               const T* __restrict__ w, T* __restrict__ y, int rows,
                                                               int n, int k) {
  const int col = blockIdx.x, tid = threadIdx.x;
  float acc[BLZ_SKINNY_MAX_ROWS];
#pragma unroll
  for (int r = 0; r < BLZ_SKINNY_MAX_ROWS; ++r) acc[r] = 0.0f;
  const uint4* wrow = reinterpret_cast<const uint4*>(w + int64_t(col) * k);
  const int packs = k / 8;
  for (int p = tid; p < packs; p += kThreads) {
    float wf[8];
    unpack8(__ldg(wrow + p), wf, T{});
#pragma unroll
    for (int r = 0; r < BLZ_SKINNY_MAX_ROWS; ++r) {
      if (r >= rows) break;
      float xf[8];
      unpack8(__ldg(reinterpret_cast<const uint4*>(x + int64_t(r) * x_stride) + p), xf, T{});
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[r] = fmaf(xf[i], wf[i], acc[r]);
    }
  }
  __shared__ float partial[kWarps][BLZ_SKINNY_MAX_ROWS];
  const int warp = tid / 32, lane = tid % 32;
#pragma unroll
  for (int r = 0; r < BLZ_SKINNY_MAX_ROWS; ++r) {
    if (r >= rows) break;
    float v = acc[r];
#pragma unroll
    for (int o = 16; o > 0; o /= 2) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) partial[warp][r] = v;
  }
  __syncthreads();
  if (tid < rows) {
    float v = 0.0f;
#pragma unroll
    for (int i = 0; i < kWarps; ++i) v += partial[i][tid];
    y[int64_t(tid) * n + col] = from_float(v, T{});
  }
}

// grid: (n / 8), block: 256 (one warp per column); dynamic shared memory: rows * k bf16 (x)
template <int R, int KP>
__global__ void __launch_bounds__(256) gemv_wide_kernel(const __nv_bfloat16* __restrict__ x, int64_t x_stride,
                                                        const __nv_bfloat16* __restrict__ w,
                                                        __nv_bfloat16* __restrict__ y, int64_t y_stride, int n, int k) {
  extern __shared__ uint4 xs[];  // [R][k / 8]
  const int packs = k / 8;
  for (int i = threadIdx.x; i < R * packs; i += blockDim.x) {
    const int r = i / packs, p = i % packs;
    xs[i] = __ldg(reinterpret_cast<const uint4*>(x + int64_t(r) * x_stride) + p);
  }
  __syncthreads();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int col = blockIdx.x * 8 + warp;
  if (col >= n) return;
  const uint4* wr = reinterpret_cast<const uint4*>(w + int64_t(col) * k);
  float acc[R];
#pragma unroll
  for (int r = 0; r < R; ++r) acc[r] = 0.0f;
  // KP packs per lane per pass, all loaded before use
  for (int p0 = 0; p0 < packs; p0 += 32 * KP) {
    uint4 wv[KP];
#pragma unroll
    for (int j = 0; j < KP; ++j) {
      const int p = p0 + j * 32 + lane;
      wv[j] = p < packs ? __ldg(wr + p) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int j = 0; j < KP; ++j) {
      const int p = p0 + j * 32 + lane;
      if (p >= packs) break;
      float wf[8];
      unpack8(wv[j], wf, __nv_bfloat16{});
#pragma unroll
      for (int r = 0; r < R; ++r) {
        float xf[8];
        unpack8(xs[r * packs + p], xf, __nv_bfloat16{});
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[r] = fmaf(xf[i], wf[i], acc[r]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    float v = acc[r];
#pragma unroll
    for (int o = 16; o > 0; o /= 2) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) y[int64_t(r) * y_stride + col] = __float2bfloat16_rn(v);
  }
}

}  // namespace
}  // namespace blaze

extern "C" cudaError_t blz_gemv_wide(const void* x, int64_t x_stride, const void* w, void* y, int64_t y_stride,
                                     int rows, int n, int k, cudaStream_t stream) {
  if (rows < 1 || rows > BLZ_GEMV_WIDE_MAX_ROWS || k % 256 || k > BLZ_GEMV_WIDE_MAX_K || x_stride % 8 ||
      size_t(rows) * k * 2 > BLZ_GEMV_WIDE_MAX_SMEM) {
    return cudaErrorInvalidValue;
  }
  if (n == 0) return cudaSuccess;
  const dim3 grid((n + 7) / 8);
  const size_t smem = size_t(rows) * k * 2;
  auto xx = static_cast<const __nv_bfloat16*>(x);
  auto ww = static_cast<const __nv_bfloat16*>(w);
  auto yy = static_cast<__nv_bfloat16*>(y);
  auto launch = [&](auto kernel) {
    if (smem > 48 * 1024) cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
    kernel<<<grid, 256, smem, stream>>>(xx, x_stride, ww, yy, y_stride, n, k);
  };
  switch (rows) {
    case 1: launch(blaze::gemv_wide_kernel<1, 8>); break;
    case 2: launch(blaze::gemv_wide_kernel<2, 8>); break;
    case 3: launch(blaze::gemv_wide_kernel<3, 4>); break;
    default: launch(blaze::gemv_wide_kernel<4, 4>); break;
  }
  return cudaGetLastError();
}

extern "C" cudaError_t blz_skinny_gemm(const void* x, int64_t x_stride, const void* w, void* y, int rows, int n,
                                       int k, blz_half_t dtype, cudaStream_t stream) {
  if (rows < 0 || rows > BLZ_SKINNY_MAX_ROWS || n < 0 || k % 8 != 0 || x_stride % 8 != 0 ||
      reinterpret_cast<uintptr_t>(x) % 16 != 0 || reinterpret_cast<uintptr_t>(w) % 16 != 0)
    return cudaErrorInvalidValue;
  if (rows == 0 || n == 0) return cudaSuccess;
  if (dtype == BLZ_BF16)
    blaze::skinny_gemm_kernel<__nv_bfloat16><<<n, blaze::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x), x_stride, static_cast<const __nv_bfloat16*>(w),
        static_cast<__nv_bfloat16*>(y), rows, n, k);
  else
    blaze::skinny_gemm_kernel<__half><<<n, blaze::kThreads, 0, stream>>>(
        static_cast<const __half*>(x), x_stride, static_cast<const __half*>(w), static_cast<__half*>(y), rows, n, k);
  return cudaGetLastError();
}
