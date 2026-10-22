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

}  // namespace
}  // namespace blaze

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
