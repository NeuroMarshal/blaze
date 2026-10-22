// Building blocks of the NVFP4 attention kernels (qsa_prefill.cu, mha_nvfp4.cu): the tile geometry, bf16 MMA, and
// raw NVFP4 row shares dequantized on the way into shared memory. Internal to src/cuda.
#pragma once

#include <cuda_bf16.h>

#include "blaze/nvfp4.h"
#include "blaze/qsa.h"

namespace blaze {
namespace attn {

constexpr int kDim = 256;                 // QSA head dim
constexpr int kBlockN = 32;               // selected positions per block
constexpr int kWarps = 4;
constexpr int kThreads = 32 * kWarps;
constexpr int kKStride = kDim + 8;        // bf16 per K row in shared memory (padding: conflict-free fragments)
constexpr int kVStride = kBlockN + 8;     // bf16 per transposed V row
constexpr int kPStride = kBlockN + 8;     // bf16 per P row
constexpr float kLog2e = 1.4426950408f;

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ void mma_bf16(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// One thread's share of a block: values [d0, d0 + 32) of positions 2p and 2p + 1, K and V, as raw NVFP4 bytes
// (16 code bytes and 2 scale bytes per position and side).
struct Raw {
  uint4 k[2], v[2];
  uint32_t ks[2], vs[2];
};

__device__ __forceinline__ void load_raw(const blz_nvfp4_pool& pool, int32_t row, int d0, uint4& k, uint4& v,
                                         uint32_t& ks, uint32_t& vs) {
  if (row < 0) return;
  k = __ldg(reinterpret_cast<const uint4*>(pool.k_data + int64_t(row) * (kDim / 2) + d0 / 2));
  v = __ldg(reinterpret_cast<const uint4*>(pool.v_data + int64_t(row) * (kDim / 2) + d0 / 2));
  ks = __ldg(reinterpret_cast<const uint16_t*>(pool.k_scale + int64_t(row) * (kDim / kNvfp4Block) + d0 / kNvfp4Block));
  vs = __ldg(reinterpret_cast<const uint16_t*>(pool.v_scale + int64_t(row) * (kDim / kNvfp4Block) + d0 / kNvfp4Block));
}

// The 32 values of a share (codes c, scale bytes s2) as floats.
__device__ __forceinline__ void dequant32(const uint4& c, uint32_t s2, float g, float out[32]) {
  const uint32_t words[4] = {c.x, c.y, c.z, c.w};
  const float s0 = e4m3_to_float(uint8_t(s2)) * g, s1 = e4m3_to_float(uint8_t(s2 >> 8)) * g;
#pragma unroll
  for (int w = 0; w < 4; ++w)
#pragma unroll
    for (int j = 0; j < 8; ++j) out[8 * w + j] = e2m1_value((words[w] >> (4 * j)) & 15) * (w < 2 ? s0 : s1);
}

}  // namespace attn
}  // namespace blaze
