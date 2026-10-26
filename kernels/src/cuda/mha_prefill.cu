// Causal prefill attention straight from an NVFP4 KV pool, FlashAttention-2 style (blaze/mha.h,
// blz_nvfp4_mha_prefill).
//
// A CTA is 16 query tokens times G query heads of one KV head (all of the KV head's group, or a part of it when the
// group is larger: 8 = 2 x 4, 12 = 2 x 6): one warp a head, its 16 MMA rows the tokens. The
// keys come in tiles of 32 positions. A tile's raw codes and block scales (slots through req_to_token) go by cp.async
// into shared memory once for the CTA, are dequantized there once into FP16 (exact: an E2M1 value times an E4M3 scale
// fits FP16), and every warp reads its MMA fragments with ldmatrix (V transposed). Q, times scale x log2 e x the K
// global scale, sits in shared memory in FP16. S = Q K^T and O += P V run on m16n8k16 MMAs with FP32 accumulation and
// an online softmax with a lazy rescale (the running max moves only when a score passes it by kSlack, so O is rarely
// rescaled and P stays below 2^kSlack); P stays in registers as the A operand of P V. One CTA walks the whole visible
// context, so there is no split and no merge kernel. The next tile's raw bytes are in flight while the current tile is
// multiplied; the tiles before the CTA's first query position need no mask.
//
// The decode kernel (mha_nvfp4.cu) ran prefill before: 48 rows a CTA, every warp dequantizing the whole context in
// registers for its 16 rows, the context re-read by every tile of 8 query tokens.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <type_traits>

#include "blaze/mha.h"
#include "blaze/nvfp4.h"

namespace blaze {
namespace {

constexpr int kD = 256;               // head dim
constexpr int kTok = 16;              // query tokens per CTA: a warp's 16 MMA rows
constexpr int kN = 32;                // positions per key tile
constexpr int kStride = kD + 8;       // f16 per row of Q, K, V in shared memory (conflict-free ldmatrix)
constexpr int kRawRow = kD / 2;       // code bytes of a position and side
constexpr int kRawScale = kD / 16;    // scale bytes of a position and side
constexpr int kChunks = 2 * (kRawRow / 16 + 1);  // 16-byte copies of a position (K and V, codes and scales)
constexpr float kLog2e = 1.4426950408f;
constexpr float kSlack = 8.0f;

// Q [G][16][kStride], K and V [kN][kStride] in f16; raw K / V codes [kN][128] and scales [kN][16]
template <int G>
constexpr size_t smem_bytes() {
  return (size_t(G) * kTok + 2 * kN) * kStride * sizeof(__half) + size_t(2) * kN * (kRawRow + kRawScale);
}

__device__ __forceinline__ void cp16(void* smem, const void* gmem, bool ok) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::); }

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

// Bench-only cuts (BLAZE_PREFILL_ABLATE builds): no dequantization, no Q K^T, no P V.
constexpr int kNoDequant = 2, kNoQk = 4, kNoPv = 8;

// grid: (ceil(q_len / 16), kv_heads x group / G), block: 32 G, dynamic shared memory: smem_bytes<G>().
// F16: the MMAs accumulate in FP16 (Q K^T over 64 dims at a time, P V over one 16-position step, both summed in FP32)
// with an exact running max (no lazy slack: P <= 1, so a step of P V stays below 16 x 6 x 448 = 43008 < 65504).
template <int G, bool F16, int CUT = 0>
__global__ void __launch_bounds__(32 * G, 1)
    prefill_kernel(const __nv_bfloat16* __restrict__ q, int64_t sq_tok, int64_t sq_head, blz_nvfp4_pool pool,
                   const int32_t* __restrict__ r2t, int kv_len, int q_len, int group, float qscale,
                   __nv_bfloat16* __restrict__ out, int64_t so_tok, int64_t so_head) {
  extern __shared__ __align__(16) unsigned char smem[];
  __half* sq = reinterpret_cast<__half*>(smem);
  __half* sk = sq + G * kTok * kStride;
  __half* sv = sk + kN * kStride;
  uint8_t* rk = reinterpret_cast<uint8_t*>(sv + kN * kStride);
  uint8_t* rv = rk + kN * kRawRow;
  uint8_t* rks = rv + kN * kRawRow;
  uint8_t* rvs = rks + kN * kRawScale;

  const int parts = group / G, g = blockIdx.y / parts, part = blockIdx.y % parts;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, gr = lane / 4, c = lane % 4;
  const int qi0 = blockIdx.x * kTok, nq = min(kTok, q_len - qi0), head = g * group + part * G + warp;
  const int64_t base = int64_t(kv_len) - q_len;  // position of query token 0
  const int64_t vis_end = base + qi0 + nq;       // one past the last position a row of the CTA sees
  const int64_t free_end = base + qi0;           // positions below are visible to every row
  const int64_t heads = pool.heads;

  // This warp's Q rows: f16(q * scale * log2 e * K global scale), rows past the chunk zero.
  {
    const float m = qscale * __ldg(pool.k_g);
    __half* dst = sq + warp * kTok * kStride;
    for (int i = lane; i < kTok * (kD / 8); i += 32) {
      const int r = i / (kD / 8), d = (i % (kD / 8)) * 8;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (r < nq) {
        const uint4 raw = __ldg(reinterpret_cast<const uint4*>(q + int64_t(qi0 + r) * sq_tok + int64_t(head) * sq_head + d));
        const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&raw);
        uint32_t* h = reinterpret_cast<uint32_t*>(&v);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 f = __bfloat1622float2(b[j]);
          h[j] = pack_f16(f.x * m, f.y * m);
        }
      }
      *reinterpret_cast<uint4*>(dst + r * kStride + d) = v;
    }
  }

  // The raw bytes of one side (K or V) of the tile at p0: per position 8 copies of codes and 1 of scales; past the
  // CTA's last visible position, zeros.
  auto load_side = [&](int64_t p0, bool v) {
    constexpr int per = kChunks / 2;
    for (int i = threadIdx.x; i < kN * per; i += 32 * G) {
      const int pos = i / per, j = i % per;
      const int64_t p = p0 + pos;
      const bool ok = p < vis_end;
      const int64_t row = ok ? int64_t(__ldg(r2t + p)) * heads + g : 0;
      const uint8_t* src;
      uint8_t* dst;
      if (j < kRawRow / 16) {
        src = (v ? pool.v_data : pool.k_data) + row * kRawRow + 16 * j;
        dst = (v ? rv : rk) + pos * kRawRow + 16 * j;
      } else {
        src = (v ? pool.v_scale : pool.k_scale) + row * kRawScale;
        dst = (v ? rvs : rks) + pos * kRawScale;
      }
      cp16(dst, src, ok);
    }
    cp_commit();
  };

  // Unit u of one side's dequantization (u < kN * 16; a thread takes u = tid + 32 G i): the 16 values of a block
  // from its 8 code bytes times its E4M3 scale, into f16 K or V. Called between the MMAs of the other side's product,
  // so the tensor cores and the ALUs work at once.
  constexpr int kUnits = kN * (kD / 16);
  auto dequant_unit = [&](bool v, int i) {
    const int u = int(threadIdx.x) + 32 * G * i;
    if (u >= kUnits) return;
    const int pos = u / (kD / 16), blk = u % (kD / 16);
    const uint2 w = *reinterpret_cast<const uint2*>((v ? rv : rk) + pos * kRawRow + 8 * blk);
    const __half2 s2 = e4m3_h2((v ? rvs : rks)[pos * kRawScale + blk]);
    uint4 lo, hi;
    uint32_t* l = reinterpret_cast<uint32_t*>(&lo);
    uint32_t* h = reinterpret_cast<uint32_t*>(&hi);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const __half2 a = __hmul2(e2m1x2_to_f16x2(w.x >> (8 * j)), s2), b = __hmul2(e2m1x2_to_f16x2(w.y >> (8 * j)), s2);
      l[j] = *reinterpret_cast<const uint32_t*>(&a);
      h[j] = *reinterpret_cast<const uint32_t*>(&b);
    }
    __half* dst = (v ? sv : sk) + pos * kStride + 16 * blk;
    *reinterpret_cast<uint4*>(dst) = lo;
    *reinterpret_cast<uint4*>(dst + 8) = hi;
  };
  constexpr int kUnitSteps = (kUnits + 32 * G - 1) / (32 * G);  // units a thread takes per side and tile
  auto dequant_all = [&](bool v) {
#pragma unroll
    for (int i = 0; i < kUnitSteps; ++i) dequant_unit(v, i);
  };

  float o[kD / 8][4];
#pragma unroll
  for (int nt = 0; nt < kD / 8; ++nt) o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.0f;
  float mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.0f, 0.0f};
  // rows gr and gr + 8 (query tokens qi0 + gr, + 8) see positions <= lim; a row past the chunk none
  const int64_t lim[2] = {gr < nq ? base + qi0 + gr : -1, gr + 8 < nq ? base + qi0 + gr + 8 : -1};
  const __half* qw = sq + warp * kTok * kStride;

  // Two phases a tile, each the MMAs of one side next to the dequantization of the other (no barrier between):
  //   1: the next tile's K bytes in flight; S = Q K^T of this tile while its V goes into f16
  //   2: the next tile's V bytes in flight; O += P V of this tile while the next tile's K goes into f16
  load_side(0, false);
  load_side(0, true);
  cp_wait_all();
  __syncthreads();
  if constexpr (!(CUT & kNoDequant)) dequant_all(false);
  __syncthreads();
  for (int64_t p0 = 0; p0 < vis_end; p0 += kN) {
    const bool more = p0 + kN < vis_end;
    if (more) load_side(p0 + kN, false);
    // this tile's V units, spread over the k-steps of Q K^T
    auto dq_v = [&](int ks) {
      if constexpr (!(CUT & kNoDequant)) {
#pragma unroll
        for (int i = 0; i < kUnitSteps; ++i)
          if (ks == (i * (kD / 16)) / kUnitSteps) dequant_unit(true, i);
      }
    };

    // S = Q K^T: 16 rows x 32 positions (n8 tiles 4 np + 0..3 over positions 8 nt)
    float s[4][4] = {};
    if constexpr (!(CUT & kNoQk)) {
      if constexpr (F16) {
#pragma unroll
        for (int k4 = 0; k4 < kD / 64; ++k4) {
          uint32_t hs[4][2] = {};
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) {
            const int ks = 4 * k4 + kk;
            dq_v(ks);
            uint32_t a[4];
            ldsm4(a, qw + (lane & 15) * kStride + 16 * ks + (lane >> 4) * 8);
#pragma unroll
            for (int np = 0; np < 2; ++np) {
              uint32_t b[4];
              ldsm4(b, sk + (16 * np + (lane & 7) + ((lane >> 4) << 3)) * kStride + 16 * ks + ((lane >> 3) & 1) * 8);
              mma_f16h(hs[2 * np], a, b[0], b[1]);
              mma_f16h(hs[2 * np + 1], a, b[2], b[3]);
            }
          }
#pragma unroll
          for (int nt = 0; nt < 4; ++nt) add_h2(s[nt][0], s[nt][1], hs[nt][0]), add_h2(s[nt][2], s[nt][3], hs[nt][1]);
        }
      } else {
#pragma unroll
        for (int ks = 0; ks < kD / 16; ++ks) {
          dq_v(ks);
          uint32_t a[4];
          ldsm4(a, qw + (lane & 15) * kStride + 16 * ks + (lane >> 4) * 8);
#pragma unroll
          for (int np = 0; np < 2; ++np) {
            uint32_t b[4];
            ldsm4(b, sk + (16 * np + (lane & 7) + ((lane >> 4) << 3)) * kStride + 16 * ks + ((lane >> 3) & 1) * 8);
            mma_f16(s[2 * np], a, b[0], b[1]);
            mma_f16(s[2 * np + 1], a, b[2], b[3]);
          }
        }
      }
    } else {
#pragma unroll
      for (int ks = 0; ks < kD / 16; ++ks) dq_v(ks);
    }

    // Online softmax; this thread's positions p0 + 8 nt + 2c + (e & 1), rows gr (e < 2) and gr + 8.
    const bool masked = p0 + kN > free_end;
    float cm[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int h = e >> 1;
        if (masked && p0 + 8 * nt + 2 * c + (e & 1) > lim[h]) s[nt][e] = -INFINITY;
        cm[h] = fmaxf(cm[h], s[nt][e]);
      }
    float alpha[2];
    bool grow = false;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      cm[h] = fmaxf(cm[h], __shfl_xor_sync(0xffffffffu, cm[h], 1));
      cm[h] = fmaxf(cm[h], __shfl_xor_sync(0xffffffffu, cm[h], 2));
      alpha[h] = 1.0f;
      if (cm[h] > mrow[h] + (F16 ? 0.0f : kSlack) || (mrow[h] == -INFINITY && cm[h] != -INFINITY)) {
        alpha[h] = mrow[h] == -INFINITY ? 0.0f : exp2f(mrow[h] - cm[h]);
        mrow[h] = cm[h];
        grow = true;
      }
    }
    if (__any_sync(0xffffffffu, grow)) {
#pragma unroll
      for (int nt = 0; nt < kD / 8; ++nt) {
        o[nt][0] *= alpha[0], o[nt][1] *= alpha[0];
        o[nt][2] *= alpha[1], o[nt][3] *= alpha[1];
      }
    }
    // a row with nothing visible yet keeps its zero state (exp2(-inf - -inf) would be NaN)
    const float b0 = mrow[0] == -INFINITY ? 0.0f : mrow[0], b1 = mrow[1] == -INFINITY ? 0.0f : mrow[1];
    float sum[2] = {0.0f, 0.0f};
    uint32_t pa[2][4];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      // A fragment of k-step nt / 2 (positions 16 j ..): a0 / a1 rows gr / gr + 8 at 2c, a2 / a3 at 8 + 2c
      if constexpr (F16) {
        // exp2 of two scores in one f16x2 instruction (the running max is exact: the inputs are <= 0, P is f16 anyway)
        const __half2 e0 = h2exp2(__floats2half2_rn(s[nt][0] - b0, s[nt][1] - b0));
        const __half2 e1 = h2exp2(__floats2half2_rn(s[nt][2] - b1, s[nt][3] - b1));
        const float2 f0 = __half22float2(e0), f1 = __half22float2(e1);
        sum[0] += f0.x + f0.y, sum[1] += f1.x + f1.y;
        pa[nt >> 1][(nt & 1) * 2 + 0] = *reinterpret_cast<const uint32_t*>(&e0);
        pa[nt >> 1][(nt & 1) * 2 + 1] = *reinterpret_cast<const uint32_t*>(&e1);
      } else {
        const float p00 = exp2f(s[nt][0] - b0), p01 = exp2f(s[nt][1] - b0);
        const float p10 = exp2f(s[nt][2] - b1), p11 = exp2f(s[nt][3] - b1);
        sum[0] += p00 + p01, sum[1] += p10 + p11;
        pa[nt >> 1][(nt & 1) * 2 + 0] = pack_f16(p00, p01);
        pa[nt >> 1][(nt & 1) * 2 + 1] = pack_f16(p10, p11);
      }
    }
    lrow[0] = lrow[0] * alpha[0] + sum[0];
    lrow[1] = lrow[1] * alpha[1] + sum[1];
    cp_wait_all();
    __syncthreads();  // this tile's V in f16, the next tile's K bytes in; every warp is done with this tile's K
    if (more) load_side(p0 + kN, true);
    // the next tile's K units, spread over the steps of P V
    auto dq_k = [&](int step) {
      if constexpr (!(CUT & kNoDequant)) {
        if (more) {
#pragma unroll
          for (int i = 0; i < kUnitSteps; ++i)
            if (step == (i * (kD / 8)) / kUnitSteps) dequant_unit(false, i);
        }
      }
    };

    // O += P V: k-steps j over positions 16 j .., n8 tiles 2 dp, 2 dp + 1 over dims 16 dp ..
    if constexpr (!(CUT & kNoPv)) {
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int dp = 0; dp < kD / 16; ++dp) {
          dq_k(j * (kD / 16) + dp);
          uint32_t b[4];
          ldsm4t(b, sv + (16 * j + (lane & 7) + ((lane >> 3) & 1) * 8) * kStride + 16 * dp + (lane >> 4) * 8);
          if constexpr (F16) {
            uint32_t h0[2] = {0, 0}, h1[2] = {0, 0};
            mma_f16h(h0, pa[j], b[0], b[1]);
            mma_f16h(h1, pa[j], b[2], b[3]);
            add_h2(o[2 * dp][0], o[2 * dp][1], h0[0]), add_h2(o[2 * dp][2], o[2 * dp][3], h0[1]);
            add_h2(o[2 * dp + 1][0], o[2 * dp + 1][1], h1[0]), add_h2(o[2 * dp + 1][2], o[2 * dp + 1][3], h1[1]);
          } else {
            mma_f16(o[2 * dp], pa[j], b[0], b[1]);
            mma_f16(o[2 * dp + 1], pa[j], b[2], b[3]);
          }
        }
    } else {
#pragma unroll
      for (int step = 0; step < kD / 8; ++step) dq_k(step);
    }
    cp_wait_all();
    __syncthreads();  // the next tile's K in f16 and its V bytes in; every warp is done with this tile's V
  }

  // The rows' outputs: O / l times the V global scale.
  const float vg = __ldg(pool.v_g);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    float l = lrow[h];
    l += __shfl_xor_sync(0xffffffffu, l, 1);
    l += __shfl_xor_sync(0xffffffffu, l, 2);
    const int r = gr + 8 * h;
    if (r >= nq) continue;
    const float inv = l > 0.0f ? vg / l : 0.0f;
    __nv_bfloat16* orow = out + int64_t(qi0 + r) * so_tok + int64_t(head) * so_head;
#pragma unroll
    for (int nt = 0; nt < kD / 8; ++nt)
      *reinterpret_cast<__nv_bfloat162*>(orow + 8 * nt + 2 * c) = __floats2bfloat162_rn(o[nt][2 * h] * inv, o[nt][2 * h + 1] * inv);
  }
}

template <int G, bool F16, int CUT = 0>
cudaError_t launch(const void* q, int64_t sq_tok, int64_t sq_head, blz_nvfp4_pool pool, const int32_t* r2t, int kv_len,
                   int q_len, int group, float scale, void* out, int64_t so_tok, int64_t so_head, cudaStream_t stream) {
  constexpr size_t smem = smem_bytes<G>();
  cudaError_t err = cudaFuncSetAttribute(prefill_kernel<G, F16, CUT>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
  if (err != cudaSuccess) return err;
  const dim3 grid(static_cast<unsigned>((q_len + kTok - 1) / kTok), static_cast<unsigned>(pool.heads * (group / G)));
  prefill_kernel<G, F16, CUT><<<grid, 32 * G, smem, stream>>>(static_cast<const __nv_bfloat16*>(q), sq_tok, sq_head, pool, r2t,
                                                    kv_len, q_len, group, scale * kLog2e,
                                                    static_cast<__nv_bfloat16*>(out), so_tok, so_head);
  return cudaGetLastError();
}

}  // namespace
}  // namespace blaze

extern "C" int blz_nvfp4_mha_prefill_supported(void) { return 1; }

namespace {
// The query heads a CTA takes of a group: the largest of 6, 4, 2, 1 that divides it.
int heads_per_cta(int group) { return group % 6 == 0 ? 6 : group % 4 == 0 ? 4 : group % 2 == 0 ? 2 : 1; }
}  // namespace

extern "C" int blz_nvfp4_mha_prefill_group_ok(int q_heads, int kv_heads) {
  return q_heads > 0 && kv_heads > 0 && q_heads % kv_heads == 0 && q_heads / kv_heads <= 64;
}

extern "C" cudaError_t blz_nvfp4_mha_prefill(const void* q, int64_t q_stride_tok, int64_t q_stride_head, int q_heads,
                                             blz_nvfp4_pool pool, const int32_t* req_to_token, int64_t req_stride,
                                             int64_t req, int kv_len, int q_len, float scale, int f16_accumulate,
                                             blz_half_t dtype, void* out, int64_t out_stride_tok,
                                             int64_t out_stride_head, cudaStream_t stream) {
  if (dtype != BLZ_BF16 || pool.format != BLZ_KV_NVFP4 || pool.dim != blaze::kD || q_len < 1 || kv_len < q_len ||
      !blz_nvfp4_mha_prefill_group_ok(q_heads, pool.heads) || q_stride_tok % 8 || q_stride_head % 8 || out_stride_tok % 2 ||
      out_stride_head % 2 || reinterpret_cast<uintptr_t>(q) % 16 || reinterpret_cast<uintptr_t>(out) % 4)
    return cudaErrorNotSupported;
  const int32_t* r2t = req_to_token + req * req_stride;
  const int group = q_heads / pool.heads;
  auto go = [&](auto g, auto f16) {
    return blaze::launch<decltype(g)::value, decltype(f16)::value>(q, q_stride_tok, q_stride_head, pool, r2t, kv_len, q_len,
                                                                   group, scale, out, out_stride_tok, out_stride_head, stream);
  };
  auto by_group = [&](auto f16) {
    switch (heads_per_cta(group)) {
      case 1: return go(std::integral_constant<int, 1>{}, f16);
      case 2: return go(std::integral_constant<int, 2>{}, f16);
      case 4: return go(std::integral_constant<int, 4>{}, f16);
      default: return go(std::integral_constant<int, 6>{}, f16);
    }
  };
  return f16_accumulate ? by_group(std::true_type{}) : by_group(std::false_type{});
}

#ifdef BLAZE_PREFILL_ABLATE
// Bench builds only: the G = 6 kernel with parts cut (kNoDequant | kNoQk | kNoPv).
template <int CUT, bool F16>
cudaError_t blz_prefill_ablate(const void* q, int64_t sq_tok, int64_t sq_head, blz_nvfp4_pool pool, const int32_t* r2t,
                               int kv_len, int q_len, float scale, void* out, int64_t so_tok, int64_t so_head) {
  return blaze::launch<6, F16, CUT>(q, sq_tok, sq_head, pool, r2t, kv_len, q_len, 6, scale, out, so_tok, so_head, nullptr);
}
#endif
