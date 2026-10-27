// Decode / verify attention over an NVFP4 KV pool with the dequantization shared by the CTA (the dense causal calls
// of blz_nvfp4_mha_decode with more than one MMA tile of rows: a verify of drafts, a short prefill chunk).
//
// The register kernel (mha_nvfp4.cu) gives every warp its own 16 rows and has each warp convert the codes it
// multiplies, so a verify of 8 tokens x GQA 6 (3 tiles) converts the context three times, and 9 tokens took two
// CTAs that each read the whole context. Here a CTA is all the rows of a request's query tiles (up to 5 MMA tiles,
// row r = token r / group, head r % group, as the register kernel) for one KV head and one split of the context.
// Key tiles of 32 positions go by cp.async into a ring of NS raw stages (codes and block scales through req_to_token,
// NS - 1 tiles in flight), each is converted once into FP16 shared memory by all threads (K of the next tile during
// P V, V of this tile during Q K^T), and every warp reads its MMA fragments with ldmatrix. The online softmax, the
// lazy rescale and the FP16-accumulation mode are those of the prefill kernel (mha_prefill.cu). Each CTA leaves its
// split's unnormalized output, running max (log2 units) and sum; mha_nvfp4.cu merges the splits.
//
// The pair kernel gives a tile of 16 rows two warps: warp 2 i + h multiplies Q K^T over dims 128 h .. 128 h + 127 of
// tile i, the pair adds its partial scores through shared memory (both warps then hold the same scores, running max
// and sum), and the warp accumulates P V into the same dims. O is 64 registers a thread instead of 128 and Q (128 dims
// of a warp's rows) stays in registers, so twice the warps run (2 MT a CTA, one CTA an SM) with registers to spare for
// their loads in flight, and the raw ring is five stages deep.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "fa_tile.cuh"

namespace blaze {
namespace fa {
namespace {

constexpr int kN = 32;                               // positions per key tile
constexpr int kPosRaw = 2 * (kRawRow + kRawScale);   // raw bytes of a position (K and V, codes and scales)
constexpr int kCopies = kPosRaw / 16;                // 16-byte copies of a position
constexpr int kMaxTiles = 5;                         // MMA tiles of rows per CTA

// raw stages of MT tiles: as many as fit beside Q and the FP16 K / V tile in 99 KB
constexpr int stages(int mt) { return mt <= 3 ? 4 : mt == 4 ? 3 : 2; }

// Q [MT 16][kStride], K and V [kN][kStride] in f16; NS raw stages of kN positions (K codes, V codes, K scales,
// V scales)
template <int MT>
constexpr size_t smem_bytes() {
  return (size_t(MT) * 16 + 2 * kN) * kStride * sizeof(__half) + size_t(stages(MT)) * kN * kPosRaw;
}

// grid: (batch x tiles, kv_heads, splits), block: 32 MT, dynamic shared memory: smem_bytes<MT>().
template <int MT, bool F16>
__global__ void __launch_bounds__(32 * MT, 1)
    verify_kernel(const __nv_bfloat16* __restrict__ q, int64_t sq_tok, int64_t sq_head, int group, int per_tile,
                  int tiles, int q_heads, blz_nvfp4_pool pool, const int32_t* __restrict__ req_to_token,
                  int64_t req_stride, blz_index_ptr req_indices, blz_index_ptr kv_lens, int q_len, int splits,
                  float qscale, float* __restrict__ part_o, float* __restrict__ part_ml) {
  constexpr int kThreads = 32 * MT, kRows = 16 * MT, NS = stages(MT);
  extern __shared__ __align__(16) unsigned char smem[];
  __half* sq = reinterpret_cast<__half*>(smem);
  __half* sk = sq + kRows * kStride;
  __half* sv = sk + kN * kStride;
  uint8_t* ring = reinterpret_cast<uint8_t*>(sv + kN * kStride);

  const int b = blockIdx.x / tiles, qt = blockIdx.x % tiles, g = blockIdx.y, s = blockIdx.z;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, gr = lane / 4, c = lane % 4;
  const int qi0 = qt * per_tile, nq = min(per_tile, q_len - qi0), valid_rows = nq * group;
  const int64_t kv_len = index_at(kv_lens, b), req = index_at(req_indices, b);
  const int32_t* r2t = req_to_token + req * req_stride;
  const int64_t base = kv_len - q_len;           // position of query token 0
  const int64_t vis_end = base + qi0 + nq;       // one past the last position a row of the CTA sees
  const int64_t free_end = base + qi0;           // positions below are visible to every row
  const int64_t all_tiles = (vis_end + kN - 1) / kN, per_split = (all_tiles + splits - 1) / splits;
  const int64_t t_begin = min(all_tiles, int64_t(s) * per_split), t_end = min(all_tiles, t_begin + per_split);
  const int64_t heads = pool.heads;

  // Q of the CTA's rows: f16(q * scale * log2 e * K global scale), rows past the tile zero.
  {
    const float m = qscale * __ldg(pool.k_g);
    for (int i = threadIdx.x; i < kRows * (kD / 8); i += kThreads) {
      const int r = i / (kD / 8), d = (i % (kD / 8)) * 8;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (r < valid_rows) {
        const __nv_bfloat16* src =
            q + (int64_t(b) * q_len + qi0 + r / group) * sq_tok + int64_t(g * group + r % group) * sq_head + d;
        const uint4 raw = __ldg(reinterpret_cast<const uint4*>(src));
        const __nv_bfloat162* bf = reinterpret_cast<const __nv_bfloat162*>(&raw);
        uint32_t* h = reinterpret_cast<uint32_t*>(&v);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 f = __bfloat1622float2(bf[j]);
          h[j] = pack_f16(f.x * m, f.y * m);
        }
      }
      *reinterpret_cast<uint4*>(sq + r * kStride + d) = v;
    }
  }

  // The raw bytes of key tile t into stage t % NS (zeros past the CTA's last visible position); one copy group a
  // tile, empty past the split, so that the group count stays uniform.
  auto load_tile = [&](int64_t t) {
    if (t < t_end) {
      uint8_t* st = ring + int(t % NS) * (kN * kPosRaw);
      for (int i = threadIdx.x; i < kN * kCopies; i += kThreads) {
        const int pos = i / kCopies, j = i % kCopies, side = j / (kCopies / 2), jj = j % (kCopies / 2);
        const int64_t p = t * kN + pos;
        const bool ok = p < vis_end;
        const int64_t row = ok ? int64_t(__ldg(r2t + p)) * heads + g : 0;
        const uint8_t* src;
        uint8_t* dst;
        if (jj < kRawRow / 16) {
          src = (side ? pool.v_data : pool.k_data) + row * kRawRow + 16 * jj;
          dst = st + side * (kN * kRawRow) + pos * kRawRow + 16 * jj;
        } else {
          src = (side ? pool.v_scale : pool.k_scale) + row * kRawScale;
          dst = st + 2 * kN * kRawRow + side * (kN * kRawScale) + pos * kRawScale;
        }
        cp16(dst, src, ok);
      }
    }
    cp_commit();
  };

  // Unit i of a thread (u = tid + 32 MT i < kN x 16): one block of one position of a side, stage st into f16.
  constexpr int kUnits = kN * (kD / 16);
  constexpr int kUnitSteps = (kUnits + kThreads - 1) / kThreads;
  auto dequant_unit = [&](int64_t t, bool v, int i) {
    const int u = int(threadIdx.x) + kThreads * i;
    if (kUnits % kThreads != 0 && u >= kUnits) return;
    const int pos = u / (kD / 16), blk = u % (kD / 16);
    const uint8_t* st = ring + int(t % NS) * (kN * kPosRaw);
    const uint8_t* codes = st + (v ? kN * kRawRow : 0) + pos * kRawRow + 8 * blk;
    const uint8_t scale = st[2 * kN * kRawRow + (v ? kN * kRawScale : 0) + pos * kRawScale + blk];
    dequant_block(codes, scale, (v ? sv : sk) + pos * kStride + 16 * blk);
  };

  float o[kD / 8][4];
#pragma unroll
  for (int nt = 0; nt < kD / 8; ++nt) o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.0f;
  float mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.0f, 0.0f};
  // rows 16 warp + gr and + 8 see positions <= lim; a row past the tile none
  int64_t lim[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int r = 16 * warp + gr + 8 * h;
    lim[h] = r < valid_rows ? base + qi0 + r / group : -1;
  }
  const __half* qw = sq + warp * 16 * kStride;

  // NS - 1 tiles in flight; the first tile's K into f16
#pragma unroll
  for (int i = 0; i < NS - 1; ++i) load_tile(t_begin + i);
  cp_wait<NS - 2>();
  __syncthreads();
  if (t_begin < t_end) {
#pragma unroll
    for (int i = 0; i < kUnitSteps; ++i) dequant_unit(t_begin, false, i);
  }
  __syncthreads();
  for (int64_t t = t_begin; t < t_end; ++t) {
    // into the stage of tile t - 1, free since the last barrier
    load_tile(t + NS - 1);
    const bool more = t + 1 < t_end;
    auto dq_v = [&](int ks) {
#pragma unroll
      for (int i = 0; i < kUnitSteps; ++i)
        if (ks == (i * (kD / 16)) / kUnitSteps) dequant_unit(t, true, i);
    };

    // S = Q K^T: 16 rows x 32 positions (n8 tiles 2 np, 2 np + 1 over positions 16 np ..)
    float sc[4][4] = {};
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
            uint32_t bb[4];
            ldsm4(bb, sk + (16 * np + (lane & 7) + ((lane >> 4) << 3)) * kStride + 16 * ks + ((lane >> 3) & 1) * 8);
            mma_f16h(hs[2 * np], a, bb[0], bb[1]);
            mma_f16h(hs[2 * np + 1], a, bb[2], bb[3]);
          }
        }
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) add_h2(sc[nt][0], sc[nt][1], hs[nt][0]), add_h2(sc[nt][2], sc[nt][3], hs[nt][1]);
      }
    } else {
#pragma unroll
      for (int ks = 0; ks < kD / 16; ++ks) {
        dq_v(ks);
        uint32_t a[4];
        ldsm4(a, qw + (lane & 15) * kStride + 16 * ks + (lane >> 4) * 8);
#pragma unroll
        for (int np = 0; np < 2; ++np) {
          uint32_t bb[4];
          ldsm4(bb, sk + (16 * np + (lane & 7) + ((lane >> 4) << 3)) * kStride + 16 * ks + ((lane >> 3) & 1) * 8);
          mma_f16(sc[2 * np], a, bb[0], bb[1]);
          mma_f16(sc[2 * np + 1], a, bb[2], bb[3]);
        }
      }
    }

    // Online softmax; this thread's positions 32 t + 8 nt + 2c + (e & 1), rows gr (e < 2) and gr + 8.
    const int64_t p0 = t * kN;
    const bool masked = p0 + kN > free_end;
    float cm[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int h = e >> 1;
        if (masked && p0 + 8 * nt + 2 * c + (e & 1) > lim[h]) sc[nt][e] = -INFINITY;
        cm[h] = fmaxf(cm[h], sc[nt][e]);
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
      if constexpr (F16) {
        const __half2 e0 = h2exp2(__floats2half2_rn(sc[nt][0] - b0, sc[nt][1] - b0));
        const __half2 e1 = h2exp2(__floats2half2_rn(sc[nt][2] - b1, sc[nt][3] - b1));
        const float2 f0 = __half22float2(e0), f1 = __half22float2(e1);
        sum[0] += f0.x + f0.y, sum[1] += f1.x + f1.y;
        pa[nt >> 1][(nt & 1) * 2 + 0] = *reinterpret_cast<const uint32_t*>(&e0);
        pa[nt >> 1][(nt & 1) * 2 + 1] = *reinterpret_cast<const uint32_t*>(&e1);
      } else {
        const float p00 = exp2f(sc[nt][0] - b0), p01 = exp2f(sc[nt][1] - b0);
        const float p10 = exp2f(sc[nt][2] - b1), p11 = exp2f(sc[nt][3] - b1);
        sum[0] += p00 + p01, sum[1] += p10 + p11;
        pa[nt >> 1][(nt & 1) * 2 + 0] = pack_f16(p00, p01);
        pa[nt >> 1][(nt & 1) * 2 + 1] = pack_f16(p10, p11);
      }
    }
    lrow[0] = lrow[0] * alpha[0] + sum[0];
    lrow[1] = lrow[1] * alpha[1] + sum[1];
    cp_wait<NS - 2>();
    __syncthreads();  // this tile's V in f16 and the next tile's bytes in; every warp is done with this tile's K
    auto dq_k = [&](int step) {
      if (more) {
#pragma unroll
        for (int i = 0; i < kUnitSteps; ++i)
          if (step == (i * (kD / 8)) / kUnitSteps) dequant_unit(t + 1, false, i);
      }
    };

    // O += P V: k-steps j over positions 16 j .., n8 tiles 2 dp, 2 dp + 1 over dims 16 dp ..
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int dp = 0; dp < kD / 16; ++dp) {
        dq_k(j * (kD / 16) + dp);
        uint32_t bb[4];
        ldsm4t(bb, sv + (16 * j + (lane & 7) + ((lane >> 3) & 1) * 8) * kStride + 16 * dp + (lane >> 4) * 8);
        if constexpr (F16) {
          uint32_t h0[2] = {0, 0}, h1[2] = {0, 0};
          mma_f16h(h0, pa[j], bb[0], bb[1]);
          mma_f16h(h1, pa[j], bb[2], bb[3]);
          add_h2(o[2 * dp][0], o[2 * dp][1], h0[0]), add_h2(o[2 * dp][2], o[2 * dp][3], h0[1]);
          add_h2(o[2 * dp + 1][0], o[2 * dp + 1][1], h1[0]), add_h2(o[2 * dp + 1][2], o[2 * dp + 1][3], h1[1]);
        } else {
          mma_f16(o[2 * dp], pa[j], bb[0], bb[1]);
          mma_f16(o[2 * dp + 1], pa[j], bb[2], bb[3]);
        }
      }
    __syncthreads();  // the next tile's K in f16; every warp is done with this tile's V
  }
  cp_wait<0>();

  // This split's partial results of the valid rows: O times the V global scale, the running max and sum.
  const float vg = __ldg(pool.v_g);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    float l = lrow[h];
    l += __shfl_xor_sync(0xffffffffu, l, 1);
    l += __shfl_xor_sync(0xffffffffu, l, 2);
    const int r = 16 * warp + gr + 8 * h;
    if (r >= valid_rows) continue;
    const int64_t tok = int64_t(b) * q_len + qi0 + r / group, head = int64_t(g) * group + r % group;
    const int64_t idx = (tok * q_heads + head) * splits + s;
    float* orow = part_o + idx * kD;
#pragma unroll
    for (int nt = 0; nt < kD / 8; ++nt)
      *reinterpret_cast<float2*>(orow + 8 * nt + 2 * c) = make_float2(o[nt][2 * h] * vg, o[nt][2 * h + 1] * vg);
    if (c == 0) *reinterpret_cast<float2*>(part_ml + idx * 2) = make_float2(mrow[h], l);
  }
}

constexpr int kHalfD = kD / 2;
constexpr int kPairStages = 5;

// K and V [kN][kStride] in f16; the partial scores [2 MT warps][4 n8 tiles][32 lanes] float4; kPairStages raw stages
template <int MT>
constexpr size_t pair_smem_bytes() {
  return 2 * size_t(kN) * kStride * sizeof(__half) + size_t(2 * MT) * 4 * 32 * sizeof(float4) +
         size_t(kPairStages) * kN * kPosRaw;
}

// the two warps of a tile (named barrier 1 + tile; 0 is __syncthreads)
__device__ __forceinline__ void bar_pair(int id) { asm volatile("bar.sync %0, 64;\n" ::"r"(id) : "memory"); }

// grid: (batch x tiles, kv_heads, splits), block: 64 MT, dynamic shared memory: pair_smem_bytes<MT>().
template <int MT, bool F16>
__global__ void __launch_bounds__(64 * MT, 1)
    pair_kernel(const __nv_bfloat16* __restrict__ q, int64_t sq_tok, int64_t sq_head, int group, int per_tile,
                int tiles, int q_heads, blz_nvfp4_pool pool, const int32_t* __restrict__ req_to_token,
                int64_t req_stride, blz_index_ptr req_indices, blz_index_ptr kv_lens, int q_len, int splits,
                float qscale, float* __restrict__ part_o, float* __restrict__ part_ml) {
  constexpr int kThreads = 64 * MT, NS = kPairStages;
  extern __shared__ __align__(16) unsigned char smem[];
  __half* sk = reinterpret_cast<__half*>(smem);
  __half* sv = sk + kN * kStride;
  float4* sx = reinterpret_cast<float4*>(sv + kN * kStride);
  uint8_t* ring = reinterpret_cast<uint8_t*>(sx + 2 * MT * 4 * 32);

  const int b = blockIdx.x / tiles, qt = blockIdx.x % tiles, g = blockIdx.y, s = blockIdx.z;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, gr = lane / 4, c = lane % 4;
  const int tile = warp >> 1, d0 = (warp & 1) * kHalfD;
  const int qi0 = qt * per_tile, nq = min(per_tile, q_len - qi0), valid_rows = nq * group;
  const int64_t kv_len = index_at(kv_lens, b), req = index_at(req_indices, b);
  const int32_t* r2t = req_to_token + req * req_stride;
  const int64_t base = kv_len - q_len;
  const int64_t vis_end = base + qi0 + nq, free_end = base + qi0;
  const int64_t all_tiles = (vis_end + kN - 1) / kN, per_split = (all_tiles + splits - 1) / splits;
  const int64_t t_begin = min(all_tiles, int64_t(s) * per_split), t_end = min(all_tiles, t_begin + per_split);
  const int64_t heads = pool.heads;

  // Q fragments of the warp's rows and dims (the A operand of m16n8k16: rows gr / gr + 8, dims 2c / 2c + 8 of a
  // k-step), f16(q * scale * log2 e * K global scale); rows past the tile zero.
  uint32_t qa[kHalfD / 16][4];
  {
    const float m = qscale * __ldg(pool.k_g);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = 16 * tile + gr + 8 * h;
      const bool ok = r < valid_rows;
      const __nv_bfloat16* src =
          q + (int64_t(b) * q_len + qi0 + (ok ? r / group : 0)) * sq_tok + int64_t(g * group + r % group) * sq_head + d0 + 2 * c;
#pragma unroll
      for (int ks = 0; ks < kHalfD / 16; ++ks)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          uint32_t v = 0;
          if (ok) {
            const uint32_t raw = __ldg(reinterpret_cast<const unsigned int*>(src + 16 * ks + 8 * e));
            const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw));
            v = pack_f16(f.x * m, f.y * m);
          }
          qa[ks][h + 2 * e] = v;
        }
    }
  }

  auto load_tile = [&](int64_t t) {
    if (t < t_end) {
      uint8_t* st = ring + int(t % NS) * (kN * kPosRaw);
      for (int i = threadIdx.x; i < kN * kCopies; i += kThreads) {
        const int pos = i / kCopies, j = i % kCopies, side = j / (kCopies / 2), jj = j % (kCopies / 2);
        const int64_t p = t * kN + pos;
        const bool ok = p < vis_end;
        const int64_t row = ok ? int64_t(__ldg(r2t + p)) * heads + g : 0;
        const uint8_t* src;
        uint8_t* dst;
        if (jj < kRawRow / 16) {
          src = (side ? pool.v_data : pool.k_data) + row * kRawRow + 16 * jj;
          dst = st + side * (kN * kRawRow) + pos * kRawRow + 16 * jj;
        } else {
          src = (side ? pool.v_scale : pool.k_scale) + row * kRawScale;
          dst = st + 2 * kN * kRawRow + side * (kN * kRawScale) + pos * kRawScale;
        }
        cp16(dst, src, ok);
      }
    }
    cp_commit();
  };

  constexpr int kUnits = kN * (kD / 16);
  constexpr int kUnitSteps = (kUnits + kThreads - 1) / kThreads;
  auto dequant_unit = [&](int64_t t, bool v, int i) {
    const int u = int(threadIdx.x) + kThreads * i;
    if (kUnits % kThreads != 0 && u >= kUnits) return;
    const int pos = u / (kD / 16), blk = u % (kD / 16);
    const uint8_t* st = ring + int(t % NS) * (kN * kPosRaw);
    const uint8_t* codes = st + (v ? kN * kRawRow : 0) + pos * kRawRow + 8 * blk;
    const uint8_t scale = st[2 * kN * kRawRow + (v ? kN * kRawScale : 0) + pos * kRawScale + blk];
    dequant_block(codes, scale, (v ? sv : sk) + pos * kStride + 16 * blk);
  };

  float o[kHalfD / 8][4];
#pragma unroll
  for (int nt = 0; nt < kHalfD / 8; ++nt) o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.0f;
  float mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.0f, 0.0f};
  int64_t lim[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int r = 16 * tile + gr + 8 * h;
    lim[h] = r < valid_rows ? base + qi0 + r / group : -1;
  }
  float4* mine = sx + warp * 128 + lane;
  const float4* other = sx + (warp ^ 1) * 128 + lane;

#pragma unroll
  for (int i = 0; i < NS - 1; ++i) load_tile(t_begin + i);
  cp_wait<NS - 2>();
  __syncthreads();
  if (t_begin < t_end) {
#pragma unroll
    for (int i = 0; i < kUnitSteps; ++i) dequant_unit(t_begin, false, i);
  }
  __syncthreads();
  for (int64_t t = t_begin; t < t_end; ++t) {
    load_tile(t + NS - 1);
    const bool more = t + 1 < t_end;
    auto dq_v = [&](int ks) {
#pragma unroll
      for (int i = 0; i < kUnitSteps; ++i)
        if (ks == (i * (kHalfD / 16)) / kUnitSteps) dequant_unit(t, true, i);
    };

    // this warp's half of S = Q K^T: 16 rows x 32 positions over its 128 dims
    float sc[4][4] = {};
    if constexpr (F16) {
#pragma unroll
      for (int k4 = 0; k4 < kHalfD / 64; ++k4) {
        uint32_t hs[4][2] = {};
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          const int ks = 4 * k4 + kk;
          dq_v(ks);
#pragma unroll
          for (int np = 0; np < 2; ++np) {
            uint32_t bb[4];
            ldsm4(bb, sk + (16 * np + (lane & 7) + ((lane >> 4) << 3)) * kStride + d0 + 16 * ks + ((lane >> 3) & 1) * 8);
            mma_f16h(hs[2 * np], qa[ks], bb[0], bb[1]);
            mma_f16h(hs[2 * np + 1], qa[ks], bb[2], bb[3]);
          }
        }
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) add_h2(sc[nt][0], sc[nt][1], hs[nt][0]), add_h2(sc[nt][2], sc[nt][3], hs[nt][1]);
      }
    } else {
#pragma unroll
      for (int ks = 0; ks < kHalfD / 16; ++ks) {
        dq_v(ks);
#pragma unroll
        for (int np = 0; np < 2; ++np) {
          uint32_t bb[4];
          ldsm4(bb, sk + (16 * np + (lane & 7) + ((lane >> 4) << 3)) * kStride + d0 + 16 * ks + ((lane >> 3) & 1) * 8);
          mma_f16(sc[2 * np], qa[ks], bb[0], bb[1]);
          mma_f16(sc[2 * np + 1], qa[ks], bb[2], bb[3]);
        }
      }
    }
    // the pair's sum: both warps add the same two values (addition commutes exactly), so both see the same scores
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) mine[32 * nt] = make_float4(sc[nt][0], sc[nt][1], sc[nt][2], sc[nt][3]);
    bar_pair(1 + tile);
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const float4 x = other[32 * nt];
      sc[nt][0] += x.x, sc[nt][1] += x.y, sc[nt][2] += x.z, sc[nt][3] += x.w;
    }

    const int64_t p0 = t * kN;
    const bool masked = p0 + kN > free_end;
    float cm[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int h = e >> 1;
        if (masked && p0 + 8 * nt + 2 * c + (e & 1) > lim[h]) sc[nt][e] = -INFINITY;
        cm[h] = fmaxf(cm[h], sc[nt][e]);
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
      for (int nt = 0; nt < kHalfD / 8; ++nt) {
        o[nt][0] *= alpha[0], o[nt][1] *= alpha[0];
        o[nt][2] *= alpha[1], o[nt][3] *= alpha[1];
      }
    }
    const float b0 = mrow[0] == -INFINITY ? 0.0f : mrow[0], b1 = mrow[1] == -INFINITY ? 0.0f : mrow[1];
    float sum[2] = {0.0f, 0.0f};
    uint32_t pa[2][4];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      if constexpr (F16) {
        const __half2 e0 = h2exp2(__floats2half2_rn(sc[nt][0] - b0, sc[nt][1] - b0));
        const __half2 e1 = h2exp2(__floats2half2_rn(sc[nt][2] - b1, sc[nt][3] - b1));
        const float2 f0 = __half22float2(e0), f1 = __half22float2(e1);
        sum[0] += f0.x + f0.y, sum[1] += f1.x + f1.y;
        pa[nt >> 1][(nt & 1) * 2 + 0] = *reinterpret_cast<const uint32_t*>(&e0);
        pa[nt >> 1][(nt & 1) * 2 + 1] = *reinterpret_cast<const uint32_t*>(&e1);
      } else {
        const float p00 = exp2f(sc[nt][0] - b0), p01 = exp2f(sc[nt][1] - b0);
        const float p10 = exp2f(sc[nt][2] - b1), p11 = exp2f(sc[nt][3] - b1);
        sum[0] += p00 + p01, sum[1] += p10 + p11;
        pa[nt >> 1][(nt & 1) * 2 + 0] = pack_f16(p00, p01);
        pa[nt >> 1][(nt & 1) * 2 + 1] = pack_f16(p10, p11);
      }
    }
    lrow[0] = lrow[0] * alpha[0] + sum[0];
    lrow[1] = lrow[1] * alpha[1] + sum[1];
    cp_wait<NS - 2>();
    __syncthreads();  // this tile's V in f16 and the next tile's bytes in; every warp is done with K and the scores
    auto dq_k = [&](int step) {
      if (more) {
#pragma unroll
        for (int i = 0; i < kUnitSteps; ++i)
          if (step == (i * (kHalfD / 8)) / kUnitSteps) dequant_unit(t + 1, false, i);
      }
    };

    // O += P V into this warp's dims d0 ..
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int dp = 0; dp < kHalfD / 16; ++dp) {
        dq_k(j * (kHalfD / 16) + dp);
        uint32_t bb[4];
        ldsm4t(bb, sv + (16 * j + (lane & 7) + ((lane >> 3) & 1) * 8) * kStride + d0 + 16 * dp + (lane >> 4) * 8);
        if constexpr (F16) {
          uint32_t h0[2] = {0, 0}, h1[2] = {0, 0};
          mma_f16h(h0, pa[j], bb[0], bb[1]);
          mma_f16h(h1, pa[j], bb[2], bb[3]);
          add_h2(o[2 * dp][0], o[2 * dp][1], h0[0]), add_h2(o[2 * dp][2], o[2 * dp][3], h0[1]);
          add_h2(o[2 * dp + 1][0], o[2 * dp + 1][1], h1[0]), add_h2(o[2 * dp + 1][2], o[2 * dp + 1][3], h1[1]);
        } else {
          mma_f16(o[2 * dp], pa[j], bb[0], bb[1]);
          mma_f16(o[2 * dp + 1], pa[j], bb[2], bb[3]);
        }
      }
    __syncthreads();  // the next tile's K in f16; every warp is done with this tile's V
  }
  cp_wait<0>();

  const float vg = __ldg(pool.v_g);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    float l = lrow[h];
    l += __shfl_xor_sync(0xffffffffu, l, 1);
    l += __shfl_xor_sync(0xffffffffu, l, 2);
    const int r = 16 * tile + gr + 8 * h;
    if (r >= valid_rows) continue;
    const int64_t tok = int64_t(b) * q_len + qi0 + r / group, head = int64_t(g) * group + r % group;
    const int64_t idx = (tok * q_heads + head) * splits + s;
    float* orow = part_o + idx * kD + d0;
#pragma unroll
    for (int nt = 0; nt < kHalfD / 8; ++nt)
      *reinterpret_cast<float2*>(orow + 8 * nt + 2 * c) = make_float2(o[nt][2 * h] * vg, o[nt][2 * h + 1] * vg);
    if (d0 == 0 && c == 0) *reinterpret_cast<float2*>(part_ml + idx * 2) = make_float2(mrow[h], l);
  }
}

template <int MT, bool F16, bool Pair>
cudaError_t launch(dim3 grid, cudaStream_t stream, const void* q, int64_t sq_tok, int64_t sq_head, int group,
                   int per_tile, int tiles, int q_heads, blz_nvfp4_pool pool, const int32_t* r2t, int64_t req_stride,
                   blz_index_ptr req, blz_index_ptr lens, int q_len, int splits, float qscale, float* po, float* pml) {
  constexpr size_t smem = Pair ? pair_smem_bytes<MT>() : smem_bytes<MT>();
  static_assert(smem <= 99 * 1024, "the verify kernel's shared memory exceeds a CTA's 99 KB");
  constexpr auto kernel = Pair ? pair_kernel<MT, F16> : verify_kernel<MT, F16>;
  const cudaError_t err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
  if (err != cudaSuccess) return err;
  kernel<<<grid, (Pair ? 64 : 32) * MT, smem, stream>>>(static_cast<const __nv_bfloat16*>(q), sq_tok, sq_head, group,
                                                        per_tile, tiles, q_heads, pool, r2t, req_stride, req, lens,
                                                        q_len, splits, qscale, po, pml);
  return cudaGetLastError();
}

}  // namespace

int verify_tiles(int q_heads, int kv_heads, int q_len, bool pair, int* per_tile, int* mt) {
  if (kv_heads <= 0 || q_heads % kv_heads != 0 || q_len < 1) return 0;
  const int group = q_heads / kv_heads;
  // up to 3 MMA tiles of rows the register kernel takes them in one CTA and is faster than the one-warp-a-tile kernel
  // (RTX 4060, 8 tokens x GQA 6 over 524k: 4.5 against 5.4 ms; it runs one CTA of 3-5 warps an SM and waits on its
  // own latencies); the pair kernel takes every call past one tile
  if (int64_t(q_len) * group <= (pair ? 16 : 48) || group > 16 * kMaxTiles) return 0;
  // at most kMaxTiles MMA tiles a CTA, the tokens spread evenly over the CTAs
  const int max_tok = 16 * kMaxTiles / group;
  const int tiles = (q_len + max_tok - 1) / max_tok;
  const int pt = (q_len + tiles - 1) / tiles;
  if (per_tile) *per_tile = pt;
  if (mt) *mt = (pt * group + 15) / 16;
  return tiles;
}

cudaError_t verify_launch(const void* q, int64_t sq_tok, int64_t sq_head, int q_heads, blz_nvfp4_pool pool,
                          const int32_t* req_to_token, int64_t req_stride, blz_index_ptr req_indices,
                          blz_index_ptr kv_lens, int batch, int q_len, int splits, float scale, bool f16, bool pair,
                          float* part_o, float* part_ml, cudaStream_t stream) {
  int per_tile = 0, mt = 0;
  const int tiles = verify_tiles(q_heads, pool.heads, q_len, pair, &per_tile, &mt);
  if (tiles == 0) return cudaErrorNotSupported;
  const int group = q_heads / pool.heads;
  const dim3 grid(static_cast<unsigned>(batch * tiles), static_cast<unsigned>(pool.heads), static_cast<unsigned>(splits));
  auto go = [&](auto m, auto h, auto p) {
    return launch<decltype(m)::value, decltype(h)::value, decltype(p)::value>(
        grid, stream, q, sq_tok, sq_head, group, per_tile, tiles, q_heads, pool, req_to_token, req_stride, req_indices,
        kv_lens, q_len, splits, scale * kLog2e, part_o, part_ml);
  };
  auto by_mt = [&](auto h, auto p) {
    switch (mt) {
      case 1:
      case 2: return go(std::integral_constant<int, 2>{}, h, p);
      case 3: return go(std::integral_constant<int, 3>{}, h, p);
      case 4: return go(std::integral_constant<int, 4>{}, h, p);
      default: return go(std::integral_constant<int, 5>{}, h, p);
    }
  };
  auto by_f16 = [&](auto p) { return f16 ? by_mt(std::true_type{}, p) : by_mt(std::false_type{}, p); };
  return pair ? by_f16(std::true_type{}) : by_f16(std::false_type{});
}

}  // namespace fa
}  // namespace blaze
