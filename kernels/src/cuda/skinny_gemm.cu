// Skinny 16-bit linear layers (see blaze/skinny_gemm.h): one block per CPB output columns, 128 threads striding K by
// 16-byte packs, FP32 sums per row reduced over the block, the rows in passes of BLZ_SKINNY_MAX_ROWS.

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

// grid: (ceil(n / CPB)), block: kThreads. The CPB columns of a block share every load of x; rows beyond
// BLZ_SKINNY_MAX_ROWS take further passes over the block's weight rows (from L1 / L2 after the first).
template <typename T, int CPB>
__global__ void __launch_bounds__(kThreads) skinny_gemm_kernel(const T* __restrict__ x, int64_t x_stride,
                                                               const T* __restrict__ w, T* __restrict__ y,
                                                               int64_t y_stride, int rows, int n, int k) {
  constexpr int R = BLZ_SKINNY_MAX_ROWS;
  const int col0 = blockIdx.x * CPB, tid = threadIdx.x;
  const int cols = min(CPB, n - col0);
  const int warp = tid / 32, lane = tid % 32;
  const int packs = k / 8;
  __shared__ float partial[kWarps][R][CPB];
  for (int r0 = 0; r0 < rows; r0 += R) {
    const int rc = min(rows - r0, R);
    float acc[R][CPB];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int c = 0; c < CPB; ++c) acc[r][c] = 0.0f;
    for (int p = tid; p < packs; p += kThreads) {
      float wf[CPB][8];
#pragma unroll
      for (int c = 0; c < CPB; ++c) {
        if (c < cols) {
          unpack8(__ldg(reinterpret_cast<const uint4*>(w + int64_t(col0 + c) * k) + p), wf[c], T{});
        } else {
#pragma unroll
          for (int i = 0; i < 8; ++i) wf[c][i] = 0.0f;
        }
      }
#pragma unroll
      for (int r = 0; r < R; ++r) {
        if (r >= rc) break;
        float xf[8];
        unpack8(__ldg(reinterpret_cast<const uint4*>(x + int64_t(r0 + r) * x_stride) + p), xf, T{});
#pragma unroll
        for (int c = 0; c < CPB; ++c)
#pragma unroll
          for (int i = 0; i < 8; ++i) acc[r][c] = fmaf(xf[i], wf[c][i], acc[r][c]);
      }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      if (r >= rc) break;
#pragma unroll
      for (int c = 0; c < CPB; ++c) {
        float v = acc[r][c];
#pragma unroll
        for (int o = 16; o > 0; o /= 2) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) partial[warp][r][c] = v;
      }
    }
    __syncthreads();
    for (int i = tid; i < rc * cols; i += kThreads) {
      const int r = i / cols, c = i % cols;
      float v = 0.0f;
#pragma unroll
      for (int j = 0; j < kWarps; ++j) v += partial[j][r][c];
      y[int64_t(r0 + r) * y_stride + col0 + c] = from_float(v, T{});
    }
    __syncthreads();
  }
}

template <typename T>
void launch_skinny(const T* x, int64_t x_stride, const T* w, T* y, int64_t y_stride, int rows, int n, int k, int cpb,
                   cudaStream_t stream) {
  if (cpb == 0) {
    // untuned: as many columns per block as keep about one block per SM or more (x is read once per block)
    int sms = 0, dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cpb = rows > 1 && n >= 4 * sms ? 4 : rows > 1 && n >= 2 * sms ? 2 : 1;
  }
  if (cpb == 4) {
    skinny_gemm_kernel<T, 4><<<(n + 3) / 4, kThreads, 0, stream>>>(x, x_stride, w, y, y_stride, rows, n, k);
  } else if (cpb == 2) {
    skinny_gemm_kernel<T, 2><<<(n + 1) / 2, kThreads, 0, stream>>>(x, x_stride, w, y, y_stride, rows, n, k);
  } else {
    skinny_gemm_kernel<T, 1><<<n, kThreads, 0, stream>>>(x, x_stride, w, y, y_stride, rows, n, k);
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

extern "C" cudaError_t blz_skinny_gemm_cfg(const void* x, int64_t x_stride, const void* w, void* y, int64_t y_stride,
                                           int rows, int n, int k, blz_half_t dtype, int cols_per_block,
                                           cudaStream_t stream) {
  if (rows < 0 || rows > BLZ_SKINNY_EX_MAX_ROWS || n < 0 || k % 8 != 0 || x_stride % 8 != 0 || y_stride < n ||
      reinterpret_cast<uintptr_t>(x) % 16 != 0 || reinterpret_cast<uintptr_t>(w) % 16 != 0 ||
      (cols_per_block != 0 && cols_per_block != 1 && cols_per_block != 2 && cols_per_block != 4))
    return cudaErrorInvalidValue;
  if (rows == 0 || n == 0) return cudaSuccess;
  if (dtype == BLZ_BF16)
    blaze::launch_skinny(static_cast<const __nv_bfloat16*>(x), x_stride, static_cast<const __nv_bfloat16*>(w),
                         static_cast<__nv_bfloat16*>(y), y_stride, rows, n, k, cols_per_block, stream);
  else
    blaze::launch_skinny(static_cast<const __half*>(x), x_stride, static_cast<const __half*>(w),
                         static_cast<__half*>(y), y_stride, rows, n, k, cols_per_block, stream);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_skinny_gemm_ex(const void* x, int64_t x_stride, const void* w, void* y, int64_t y_stride,
                                          int rows, int n, int k, blz_half_t dtype, cudaStream_t stream) {
  return blz_skinny_gemm_cfg(x, x_stride, w, y, y_stride, rows, n, k, dtype, 0, stream);
}

extern "C" cudaError_t blz_skinny_gemm(const void* x, int64_t x_stride, const void* w, void* y, int rows, int n,
                                       int k, blz_half_t dtype, cudaStream_t stream) {
  if (rows > BLZ_SKINNY_MAX_ROWS) return cudaErrorInvalidValue;
  return blz_skinny_gemm_ex(x, x_stride, w, y, n, rows, n, k, dtype, stream);
}
