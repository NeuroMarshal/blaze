// Building blocks of the shared-dequantization attention kernels (mha_prefill.cu, mha_verify.cu): cp.async copies,
// ldmatrix, f16 MMAs with FP32 or FP16 accumulation, and the NVFP4 -> FP16 conversions. Internal to src/cuda.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cstdint>

#include "blaze/mha.h"

namespace blaze {
namespace fa {

constexpr int kD = 256;             // head dim
constexpr int kStride = kD + 8;     // f16 per row of Q, K, V in shared memory (conflict-free ldmatrix)
constexpr int kRawRow = kD / 2;     // code bytes of a position and side
constexpr int kRawScale = kD / 16;  // scale bytes of a position and side
constexpr float kLog2e = 1.4426950408f;
// The running max of a row moves only when a score passes it by this much (log2 units), so O is rarely rescaled and
// P stays below 2^kSlack (FP32 accumulation; FP16 accumulation keeps the max exact).
constexpr float kSlack = 8.0f;

__device__ __forceinline__ void cp16(void* smem, const void* gmem, bool ok) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
// Until at most N of this thread's newest copy groups are in flight.
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const __half* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], const __half* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

__device__ __forceinline__ void mma_f16(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// FP16 accumulation: twice the MMA rate of FP32 accumulation on GeForce GPUs (RTX 4060: 58 against 29 TFLOPS).
__device__ __forceinline__ void mma_f16h(uint32_t d[2], const uint32_t a[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
               : "+r"(d[0]), "+r"(d[1])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void add_h2(float& x, float& y, uint32_t h) {
  const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
  x += f.x, y += f.y;
}

__device__ __forceinline__ uint32_t pack_f16(float lo, float hi) {
  const __half2 v = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}

// An E4M3 scale byte as f16x2 (both halves), exactly (cvt.rn.f16x2.e4m3x2 on SM89+).
__device__ __forceinline__ __half2 e4m3_h2(uint8_t b) {
  const __half2_raw r = __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t(uint16_t(b) * 0x0101u), __NV_E4M3);
  return *reinterpret_cast<const __half2*>(&r);
}

// Two E2M1 codes (one byte, the first in the low nibble) as f16x2.
__device__ __forceinline__ __half2 e2m1x2_to_f16x2(uint32_t byte) {
#if __CUDA_FP8_INTERNAL_CAN_RELY_ON_PTX_FOR_SHORTTYPESCVT__
  const __half2_raw r = __nv_cvt_fp4x2_to_halfraw2(__nv_fp4x2_storage_t(byte & 0xff), __NV_E2M1);
  return *reinterpret_cast<const __half2*>(&r);
#else
  // code s e1 e0 m into FP16 bits s << 15 | e1 e0 m << 9: times 2^14 that is the E2M1 value (0.5 a subnormal)
  const uint32_t bits = ((byte & 0x8u) << 12) | ((byte & 0x7u) << 9) | ((byte & 0x80u) << 24) | ((byte & 0x70u) << 21);
  return __hmul2(*reinterpret_cast<const __half2*>(&bits), __half2half2(__float2half_rn(16384.0f)));
#endif
}

// One NVFP4 block (8 code bytes, its E4M3 scale) as 16 f16 values into shared memory (exact: an E2M1 value times an
// E4M3 scale fits FP16).
__device__ __forceinline__ void dequant_block(const uint8_t* codes, uint8_t scale, __half* dst) {
  const uint2 w = *reinterpret_cast<const uint2*>(codes);
  const __half2 s2 = e4m3_h2(scale);
  uint4 lo, hi;
  uint32_t* l = reinterpret_cast<uint32_t*>(&lo);
  uint32_t* h = reinterpret_cast<uint32_t*>(&hi);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const __half2 a = __hmul2(e2m1x2_to_f16x2(w.x >> (8 * j)), s2), b = __hmul2(e2m1x2_to_f16x2(w.y >> (8 * j)), s2);
    l[j] = *reinterpret_cast<const uint32_t*>(&a);
    h[j] = *reinterpret_cast<const uint32_t*>(&b);
  }
  *reinterpret_cast<uint4*>(dst) = lo;
  *reinterpret_cast<uint4*>(dst + 8) = hi;
}

__device__ __forceinline__ int64_t index_at(blz_index_ptr p, int64_t i) {
  return p.type == BLZ_I64 ? __ldg(static_cast<const int64_t*>(p.ptr) + i)
                           : int64_t(__ldg(static_cast<const int32_t*>(p.ptr) + i));
}

// The shared-dequantization decode / verify kernel (mha_verify.cu) for the dense causal calls of mha_nvfp4.cu:
// the query tiles of a request (0: the call stays on the register-dequantization kernel) and the launch, which leaves
// `splits` parts per row for mha_nvfp4.cu's merge.
int verify_tiles(int q_heads, int kv_heads, int q_len, int* per_tile, int* mt);
cudaError_t verify_launch(const void* q, int64_t sq_tok, int64_t sq_head, int q_heads, blz_nvfp4_pool pool,
                          const int32_t* req_to_token, int64_t req_stride, blz_index_ptr req_indices,
                          blz_index_ptr kv_lens, int batch, int q_len, int splits, float scale, bool f16,
                          float* part_o, float* part_ml, cudaStream_t stream);

}  // namespace fa
}  // namespace blaze
