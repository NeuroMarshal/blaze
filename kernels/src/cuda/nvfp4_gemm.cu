// NVFP4 GEMMs of many rows (blaze/nvfp4_gemm.h): the W4A4 route on the FP4 tensor cores of SM120a.
//
// Rows of x are quantized first (one block per row: its largest value sets an FP32 row scale s = amax / (6 * 448);
// each 16 values get the E4M3 scale amax16 / (6 s), rounded to nearest, and E2M1 codes of x / (scale * s), ties to
// even). The GEMM keeps 128 x 128 output tiles per CTA (8 warps of 64 x 32), streams 128-wide K tiles of both
// operands and their scales through three cp.async stages, and runs mma.sync m16n8k64 kind::mxf4nvf4 with the
// E4M3 block scales of both operands (scale_vec::4X: one scale per 16 values). FP32 accumulation; the epilogue
// multiplies the row scale of x and the weight's alpha and stores BF16.
//
// Fragment layouts (CUTLASS's SM120_16x8x64_TN_VS traits): thread (g = lane / 4, t = lane % 4) holds A rows g and
// g + 8, K bytes 4t..4t+3 and 16 + 4t..; B column g, the same K bytes; the scale of A for row g + 8 (lane & 1) or g,
// four K blocks in one 32-bit word; of B for column g. Shared rows are 80 bytes (64 + 16) apart: the 8 rows x 4 words
// of a fragment load fall in different banks.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "blaze/nvfp4.h"
#include "blaze/nvfp4_gemm.h"

namespace {

constexpr int kQuantThreads = 256;

__device__ __forceinline__ float bf16f(uint16_t b) { return __uint_as_float(uint32_t(b) << 16); }

__device__ __forceinline__ uint32_t e2m1_code(float v) {
  const uint32_t s = v < 0.0f ? 8u : 0u;
  return s | uint32_t(blaze::e2m1_index(fabsf(v)));
}

__global__ void __launch_bounds__(kQuantThreads) quant_rows_kernel(const uint16_t* __restrict__ x, int64_t ldx, int k,
                                                                   uint8_t* __restrict__ codes,
                                                                   uint8_t* __restrict__ scales,
                                                                   float* __restrict__ row_scale) {
  __shared__ float red[kQuantThreads / 32];
  const int r = blockIdx.x, tid = threadIdx.x;
  const uint16_t* row = x + int64_t(r) * ldx;
  float amax = 0.0f;
  for (int i = tid; i < k; i += kQuantThreads) amax = fmaxf(amax, fabsf(bf16f(row[i])));
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  if ((tid & 31) == 0) red[tid >> 5] = amax;
  __syncthreads();
  if (tid < 32) {
    float v = tid < kQuantThreads / 32 ? red[tid] : 0.0f;
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if (tid == 0) red[0] = v;
  }
  __syncthreads();
  const float s = red[0] > 0.0f ? red[0] / (6.0f * 448.0f) : 1.0f;
  if (tid == 0) row_scale[r] = s;
  const int blocks = k / 16;
  for (int b = tid; b < blocks; b += kQuantThreads) {
    float v[16], bmax = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      v[j] = bf16f(row[b * 16 + j]);
      bmax = fmaxf(bmax, fabsf(v[j]));
    }
    // four over six: the block's largest value at 6 or at 4 of the E2M1 grid (4: finer steps for the rest, the
    // largest exact), the one with the smaller squared error
    uint32_t lo = 0, hi = 0;
    __nv_fp8_e4m3 sc;
    float best = INFINITY;
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const __nv_fp8_e4m3 cand(bmax / ((c ? 4.0f : 6.0f) * s));  // round to nearest, saturating at 448
      const float bs = float(cand) * s;
      const float inv = bs > 0.0f ? 1.0f / bs : 0.0f;
      uint32_t l = 0, h = 0;
      float err = 0.0f;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const uint32_t code = e2m1_code(v[j] * inv);
        const float d = v[j] - blaze::e2m1_value(int(code)) * bs;
        err = fmaf(d, d, err);
        if (j < 8) l |= code << (4 * j);
        else h |= code << (4 * (j - 8));
      }
      if (err < best) best = err, lo = l, hi = h, sc = cand;
    }
    uint2* dst = reinterpret_cast<uint2*>(codes + int64_t(r) * (k / 2) + b * 8);
    *dst = make_uint2(lo, hi);
    scales[int64_t(r) * blocks + b] = sc.__x;
  }
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200 && defined(__CUDA_ARCH_FEAT_SM120_ALL)
#define BLZ_FP4_MMA 1
#endif

constexpr int BM = 128, BN = 128, BK = 128, kStages = 3, kRow = 80, kThreads = 256;
constexpr int kTileBytes = (BM + BN) * kRow + (BM + BN) * 8;
constexpr int kSmem = kStages * kTileBytes;

__device__ __forceinline__ void cp_async(void* dst, const void* src, int bytes, bool valid) {
  const uint32_t d = uint32_t(__cvta_generic_to_shared(dst));
  const int n = valid ? bytes : 0;
  if (bytes == 16)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(n));
  else
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(d), "l"(src), "r"(n));
}

__global__ void probe_kernel(int* out) {
#ifdef BLZ_FP4_MMA
  *out = 1;
#else
  *out = 0;
#endif
}

__global__ void __launch_bounds__(kThreads) w4a4_kernel(const uint8_t* __restrict__ xc, const uint8_t* __restrict__ xs,
                                                        const float* __restrict__ xrow, const uint8_t* __restrict__ wc,
                                                        const uint8_t* __restrict__ ws, const float* __restrict__ walpha,
                                                        uint16_t* __restrict__ y, int64_t ldy, int m, int n, int k) {
#ifdef BLZ_FP4_MMA
  extern __shared__ __align__(16) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 2, wn = warp & 3, g = lane >> 2, t = lane & 3;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int kc = k / 2, ks = k / 16, tiles = k / BK;
  auto stage = [&](int s) { return smem + s * kTileBytes; };
  auto load = [&](int kt, int s) {
    uint8_t* A = stage(s);
    uint8_t* B = A + BM * kRow;
    uint8_t* SA = B + BN * kRow;
    uint8_t* SB = SA + BM * 8;
#pragma unroll
    for (int c = tid; c < BM * 4; c += kThreads) {
      const int r = c >> 2, ch = c & 3;
      const bool va = m0 + r < m, vb = n0 + r < n;
      cp_async(A + r * kRow + ch * 16, xc + int64_t(va ? m0 + r : 0) * kc + kt * 64 + ch * 16, 16, va);
      cp_async(B + r * kRow + ch * 16, wc + int64_t(vb ? n0 + r : 0) * kc + kt * 64 + ch * 16, 16, vb);
    }
    if (tid < BM) {
      const bool v = m0 + tid < m;
      cp_async(SA + tid * 8, xs + int64_t(v ? m0 + tid : 0) * ks + kt * 8, 8, v);
    } else {
      const int r = tid - BM;
      const bool v = n0 + r < n;
      cp_async(SB + r * 8, ws + int64_t(v ? n0 + r : 0) * ks + kt * 8, 8, v);
    }
  };
  float acc[4][4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < tiles) load(s, s);
    asm volatile("cp.async.commit_group;\n");
  }
  for (int kt = 0; kt < tiles; ++kt) {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(kStages - 2));
    __syncthreads();
    const uint8_t* A = stage(kt % kStages);
    const uint8_t* B = A + BM * kRow;
    const uint8_t* SA = B + BN * kRow;
    const uint8_t* SB = SA + BM * 8;
#pragma unroll
    for (int h = 0; h < 2; ++h) {  // two 64-wide K steps of the 128-wide tile
      uint32_t b[4][2], sfb[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int c = wn * 32 + j * 8 + g;
        b[j][0] = *reinterpret_cast<const uint32_t*>(B + c * kRow + h * 32 + t * 4);
        b[j][1] = *reinterpret_cast<const uint32_t*>(B + c * kRow + h * 32 + 16 + t * 4);
        sfb[j] = *reinterpret_cast<const uint32_t*>(SB + c * 8 + h * 4);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int r = wm * 64 + i * 16 + g;
        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(A + r * kRow + h * 32 + t * 4);
        const uint32_t a1 = *reinterpret_cast<const uint32_t*>(A + (r + 8) * kRow + h * 32 + t * 4);
        const uint32_t a2 = *reinterpret_cast<const uint32_t*>(A + r * kRow + h * 32 + 16 + t * 4);
        const uint32_t a3 = *reinterpret_cast<const uint32_t*>(A + (r + 8) * kRow + h * 32 + 16 + t * 4);
        const uint32_t sfa = *reinterpret_cast<const uint32_t*>(SA + (r + (lane & 1) * 8) * 8 + h * 4);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          float* d = acc[i][j];
          asm volatile(
              "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
              "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
              : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
              : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b[j][0]), "r"(b[j][1]), "r"(sfa), "h"(uint16_t(0)),
                "h"(uint16_t(0)), "r"(sfb[j]), "h"(uint16_t(0)), "h"(uint16_t(0)));
        }
      }
    }
    const int next = kt + kStages - 1;
    if (next < tiles) load(next, next % kStages);
    asm volatile("cp.async.commit_group;\n");
  }
  asm volatile("cp.async.wait_group 0;\n");
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int r0 = m0 + wm * 64 + i * 16 + g;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int c = n0 + wn * 32 + j * 8 + 2 * t;
      if (c >= n) continue;
      const float a0 = walpha[c], a1 = c + 1 < n ? walpha[c + 1] : 0.0f;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int r = r0 + half * 8;
        if (r >= m) continue;
        const float s = xrow[r];
        const float v0 = acc[i][j][2 * half] * s * a0, v1 = acc[i][j][2 * half + 1] * s * a1;
        uint16_t* out = y + int64_t(r) * ldy + c;
        if (c + 1 < n && (ldy & 1) == 0) {
          const __nv_bfloat162 p = __floats2bfloat162_rn(v0, v1);
          *reinterpret_cast<__nv_bfloat162*>(out) = p;
        } else {
          // an odd row stride: no aligned pair
          out[0] = __bfloat16_as_ushort(__float2bfloat16(v0));
          if (c + 1 < n) out[1] = __bfloat16_as_ushort(__float2bfloat16(v1));
        }
      }
    }
  }
#endif
}

// W4A16, FP16 cores
//
// x as FP16 rows times 2^-shift (blz_nvfp4_f16_rows: max |x| <= 64), the weight decoded from its NVFP4 codes at the
// fragment loads (code times block scale times 2^-wshift: exact in FP16), mma.sync m16n8k16 with FP16 accumulation
// within each 128 values of K (partial sums below 128 * 64 * 6 < 65504) and FP32 across them; the epilogue gives
// back alpha and both powers of two. 128 x 128 tiles of 8 warps (64 x 32), 64-wide K tiles in three stages.

constexpr int FBM = 128, FBN = 128, FBK = 64, kFStages = 3;
constexpr int kARow = FBK * 2 + 16;  // bytes of an A row in shared memory (144: ldmatrix rows in distinct banks)
constexpr int kBRow = FBK / 2 + 16;  // bytes of a B code row (48)
constexpr int kFTile = FBM * kARow + FBN * kBRow + FBN * 4;
constexpr int kFSmem = kFStages * kFTile;

__device__ __forceinline__ void cp_async4(void* dst, const void* src, bool valid) {
  const uint32_t d = uint32_t(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(d), "l"(src), "r"(valid ? 4 : 0));
}

// Four E2M1 codes (16 bits, first code lowest) -> two FP16 pairs (as in nvfp4_linear.cu).
__device__ __forceinline__ void f16_codes(uint32_t q, uint32_t& w0, uint32_t& w1) {
  const uint32_t hi = __byte_perm(0x3E3C3800u, 0x46444240u, q & 0x7777u);
  w0 = __byte_perm(hi, 0u, 0x1404) | ((q & 0x8u) << 12) | ((q & 0x80u) << 24);
  w1 = __byte_perm(hi, 0u, 0x3424) | ((q & 0x800u) << 4) | ((q & 0x8000u) << 16);
}

__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  __half2 r = __hmul2(*reinterpret_cast<__half2*>(&a), *reinterpret_cast<__half2*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}

__global__ void __launch_bounds__(kThreads) w4a16_kernel(const uint16_t* __restrict__ x16, const int32_t* __restrict__ shift,
                                                         const uint8_t* __restrict__ wc, const uint8_t* __restrict__ ws,
                                                         const float* __restrict__ walpha, int wshift,
                                                         uint16_t* __restrict__ y, int64_t ldy, int m, int n, int k) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  extern __shared__ __align__(16) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 2, wn = warp & 3, g = lane >> 2, t = lane & 3;
  const int m0 = blockIdx.y * FBM, n0 = blockIdx.x * FBN;
  const int kc = k / 2, ks = k / 16, tiles = k / FBK;
  const float wscale = ldexpf(1.0f, -wshift);
  auto stage = [&](int s) { return smem + s * kFTile; };
  auto load = [&](int kt, int s) {
    uint8_t* A = stage(s);
    uint8_t* B = A + FBM * kARow;
    uint8_t* SB = B + FBN * kBRow;
#pragma unroll
    for (int c = tid; c < FBM * 8; c += kThreads) {  // 8 chunks of 16 bytes per A row
      const int r = c >> 3, ch = c & 7;
      const bool v = m0 + r < m;
      cp_async(A + r * kARow + ch * 16, x16 + int64_t(v ? m0 + r : 0) * k + kt * FBK + ch * 8, 16, v);
    }
    {
      const int r = tid >> 1, ch = tid & 1;  // 2 chunks per B row
      const bool v = n0 + r < n;
      cp_async(B + r * kBRow + ch * 16, wc + int64_t(v ? n0 + r : 0) * kc + kt * (FBK / 2) + ch * 16, 16, v);
    }
    if (tid < FBN) {
      const bool v = n0 + tid < n;
      cp_async4(SB + tid * 4, ws + int64_t(v ? n0 + tid : 0) * ks + kt * (FBK / 16), v);
    }
  };
  float acc[4][4][4];
  uint32_t h[4][4][2];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
      h[i][j][0] = h[i][j][1] = 0u;
    }
#pragma unroll
  for (int s = 0; s < kFStages - 1; ++s) {
    if (s < tiles) load(s, s);
    asm volatile("cp.async.commit_group;\n");
  }
  for (int kt = 0; kt < tiles; ++kt) {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(kFStages - 2));
    __syncthreads();
    const uint8_t* A = stage(kt % kFStages);
    const uint8_t* B = A + FBM * kARow;
    const uint8_t* SB = B + FBN * kBRow;
#pragma unroll
    for (int kk = 0; kk < FBK / 16; ++kk) {  // one 16-wide step = one block scale
      uint32_t b[4][2];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int c = wn * 32 + j * 8 + g;
        const uint8_t* row = B + c * kBRow + kk * 8;
        const uint32_t q = uint32_t(row[t]) | (uint32_t(row[t + 4]) << 8);
        uint32_t w0, w1;
        f16_codes(q, w0, w1);
        const __nv_fp8_e4m3 sc = *reinterpret_cast<const __nv_fp8_e4m3*>(SB + c * 4 + kk);
        const __half hs = __float2half_rn(float(sc) * wscale);
        const uint32_t s2 = uint32_t(__half_as_ushort(hs)) * 0x10001u;
        b[j][0] = hmul2(w0, s2);
        b[j][1] = hmul2(w1, s2);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int r = wm * 64 + i * 16 + (lane & 15);
        const uint32_t addr = uint32_t(__cvta_generic_to_shared(A + r * kARow + kk * 32 + (lane >> 4) * 16));
        uint32_t a0, a1, a2, a3;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                     : "r"(addr));
#pragma unroll
        for (int j = 0; j < 4; ++j)
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};\n"
                       : "+r"(h[i][j][0]), "+r"(h[i][j][1])
                       : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b[j][0]), "r"(b[j][1]));
      }
    }
    if ((kt & 1) || kt + 1 == tiles) {  // every 128 values of K: the FP16 sums into FP32
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h[i][j][e]));
            acc[i][j][2 * e] += f.x;
            acc[i][j][2 * e + 1] += f.y;
            h[i][j][e] = 0u;
          }
        }
    }
    const int next = kt + kFStages - 1;
    if (next < tiles) load(next, next % kFStages);
    asm volatile("cp.async.commit_group;\n");
  }
  asm volatile("cp.async.wait_group 0;\n");
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int r0 = m0 + wm * 64 + i * 16 + g;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int c = n0 + wn * 32 + j * 8 + 2 * t;
      if (c >= n) continue;
      const float a0 = walpha[c], a1 = c + 1 < n ? walpha[c + 1] : 0.0f;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int r = r0 + half * 8;
        if (r >= m) continue;
        const int e = shift[r] + wshift;
        const float v0 = ldexpf(acc[i][j][2 * half] * a0, e), v1 = ldexpf(acc[i][j][2 * half + 1] * a1, e);
        uint16_t* out = y + int64_t(r) * ldy + c;
        if (c + 1 < n && (ldy & 1) == 0) {
          *reinterpret_cast<__nv_bfloat162*>(out) = __floats2bfloat162_rn(v0, v1);
        } else {
          out[0] = __bfloat16_as_ushort(__float2bfloat16(v0));
          if (c + 1 < n) out[1] = __bfloat16_as_ushort(__float2bfloat16(v1));
        }
      }
    }
  }
#endif
}

}  // namespace

extern "C" cudaError_t blz_nvfp4_gemm_f16(const uint16_t* x16, const int32_t* shift, const uint8_t* w_codes,
                                          const uint8_t* w_scales, const float* w_alpha, int wshift, uint16_t* y,
                                          int64_t ldy, int m, int n, int k, cudaStream_t stream) {
  if (m <= 0 || n <= 0) return cudaSuccess;
  if (k % 128 != 0) return cudaErrorInvalidValue;
  static bool attr[16] = {};
  int dev = 0;
  cudaGetDevice(&dev);
  if (dev >= 16 || !attr[dev]) {
    cudaFuncSetAttribute(w4a16_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kFSmem);
    if (dev < 16) attr[dev] = true;
  }
  const dim3 grid((n + FBN - 1) / FBN, (m + FBM - 1) / FBM);
  w4a16_kernel<<<grid, kThreads, kFSmem, stream>>>(x16, shift, w_codes, w_scales, w_alpha, wshift, y, ldy, m, n, k);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_nvfp4_quant_rows(const uint16_t* x, int64_t ldx, int m, int k, uint8_t* codes, uint8_t* scales,
                                            float* row_scale, cudaStream_t stream) {
  if (m <= 0) return cudaSuccess;
  if (k % 16 != 0) return cudaErrorInvalidValue;
  quant_rows_kernel<<<m, kQuantThreads, 0, stream>>>(x, ldx, k, codes, scales, row_scale);
  return cudaGetLastError();
}

extern "C" int blz_nvfp4_gemm_w4a4_supported(void) {
  static int known[16] = {};  // 0 unknown, 1 no, 2 yes
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return 0;
  if (dev < 16 && known[dev]) return known[dev] == 2;
  int* d = nullptr;
  int v = 0;
  if (cudaMalloc(&d, 4) == cudaSuccess) {
    probe_kernel<<<1, 1>>>(d);
    if (cudaMemcpy(&v, d, 4, cudaMemcpyDeviceToHost) != cudaSuccess) v = 0;
    cudaFree(d);
  }
  cudaGetLastError();  // a build without this architecture: the launch fails, the route is off
  if (dev < 16) known[dev] = v ? 2 : 1;
  return v;
}

extern "C" cudaError_t blz_nvfp4_gemm_w4a4(const uint8_t* x_codes, const uint8_t* x_scales, const float* x_row_scale,
                                           const uint8_t* w_codes, const uint8_t* w_scales, const float* w_alpha,
                                           uint16_t* y, int64_t ldy, int m, int n, int k, cudaStream_t stream) {
  if (m <= 0 || n <= 0) return cudaSuccess;
  if (k % BK != 0 || !blz_nvfp4_gemm_w4a4_supported()) return cudaErrorNotSupported;
  static bool attr[16] = {};
  int dev = 0;
  cudaGetDevice(&dev);
  if (dev >= 16 || !attr[dev]) {
    cudaFuncSetAttribute(w4a4_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
    if (dev < 16) attr[dev] = true;
  }
  const dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
  w4a4_kernel<<<grid, kThreads, kSmem, stream>>>(x_codes, x_scales, x_row_scale, w_codes, w_scales, w_alpha, y, ldy, m, n,
                                                 k);
  return cudaGetLastError();
}
